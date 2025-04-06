# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class CocoEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import CocoEvalOpt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info


import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict, Counter
from loguru import logger
from tqdm import tqdm

import torch
import os
# Assuming these utilities are available in your environment
from yolox.utils import (
    gather,
    is_main_process,
    postprocess, # We'll need the modified version
    synchronize,
    time_synchronized,
    xyxy2xywh,
    postprocess_with_kpts
    # We might need per-class tables if desired, but typically not used for kpts
    # per_class_AP_table,
    # per_class_AR_table
)


# --- Make sure COCO API is installed ---
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("pycocotools not found. Please run `pip install pycocotools`")
    # Provide dummy classes or raise error to prevent crashing later
    COCO = None
    COCOeval = None
# -----------------------------------------

class CocoPoseEvaluator:
    """
    COCO OKS AP Evaluation class for Keypoints.
    Processes val2017 dataset results and evaluates using COCO API for keypoints.
    """

    def __init__(
        self,
        dataloader,
        img_size: tuple, # Expect (height, width)
        confthre: float,
        nmsthre: float,
        num_classes: int, # Should be 1 for person
        num_keypoints: int, # Number of keypoints
        testdev: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): Evaluate dataloader (should yield images, targets, img_info, ids).
                                     The dataset class within dataloader must have a 'coco' attribute
                                     (pycocotools COCO object) and 'class_ids' mapping.
            img_size (tuple): Image size (height, width) after preprocess.
            confthre (float): Confidence threshold for filtering detections.
            nmsthre (float): IoU threshold for NMS.
            num_classes (int): Number of object classes (expected to be 1 for person).
            num_keypoints (int): Number of keypoints per instance.
            testdev (bool): Evaluate on test-dev dataset (requires specific file naming).
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.testdev = testdev

        # Ensure pycocotools are available
        if COCO is None or COCOeval is None:
            raise ImportError("pycocotools is not installed or failed to import.")


    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False # test_size might be redundant if self.img_size is used
    ):
        """
        COCO Object Keypoint Similarity (OKS) based Average Precision (AP) Evaluation.

        Args:
            model: The model to evaluate.
            distributed (bool): Whether evaluation is distributed.
            half (bool): Whether to use half precision (FP16).
            trt_file (str, optional): Path to TensorRT engine file.
            decoder: Optional decoder module (if different from model's internal one).
            test_size: Optional test size override.
            return_outputs: Whether to return raw formatted outputs along with metrics.

        Returns:
            oks_ap50_95 (float): COCO OKS AP IoU=0.50:0.95
            oks_ap50 (float): COCO OKS AP IoU=0.50
            summary (str): String summary from COCOeval.
        """
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()

        ids = []
        data_list = [] # Stores results formatted for COCO API [{image_id, category_id, bbox, score, keypoints}, ...]
        output_data = defaultdict(dict) # Keep if raw output dict is needed
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        # --- TRT Model Handling (Optional) ---
        if trt_file is not None:
            # ... (TRT loading code as in original) ...
            pass # Placeholder for brevity

        # --- Evaluation Loop ---
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                # Note: test_size might be passed to the model if needed by its forward pass
                current_test_size = test_size if test_size is not None else self.img_size

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs) # Get raw model output

                # --- Optional Decoder ---
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                # If the model's forward already includes decoding (like YoloxPoseHead with decode_in_inference=True),
                # the decoder argument might be unnecessary or outputs might already be decoded.

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                # --- Postprocessing with Keypoints ---
                # Use the keypoint-aware postprocessing function
                outputs = postprocess_with_kpts(
                    outputs,
                    num_classes=self.num_classes,
                    num_keypoints=self.num_keypoints,
                    conf_thre=self.confthre,
                    nms_thre=self.nmsthre
                )
                # outputs is now a list [batch_size] of tensors
                # Each tensor: [num_dets, 7 + 3*Nk] -> [x1,y1,x2,y2, obj, cls_conf, cls_idx, kp_vis1..N, kp_x1,kp_y1..N]
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

                # --- Format results for COCO API ---
                data_list_batch = self.convert_to_coco_keypoint_format(
                    outputs, info_imgs, ids
                )
                data_list.extend(data_list_batch)
                #output_data.update(image_wise_data) # Update if needed

        # --- Gather results in distributed setting ---
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0) # Gather if kept
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data)) # Merge if kept
            torch.distributed.reduce(statistics, dst=0)

        # --- Perform COCO Evaluation ---
        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
             return eval_results, output_data
        return eval_results # Return tuple: (ap50_95, ap50, summary_str)

    def convert_to_coco_keypoint_format(self, outputs, info_imgs, ids):
        """
        Converts raw detections (including keypoints) to COCO keypoint result format.
        """
        data_list = []
        img_h_batch, img_w_batch = info_imgs[0], info_imgs[1] # Original height/width

        for i, output_per_image in enumerate(outputs):
            img_h = img_h_batch[i]
            img_w = img_w_batch[i]
            img_id = ids[i]

            if output_per_image is None:
                continue

            output_per_image = output_per_image.cpu()

            # Calculate scaling factor (from original image to network input size)
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))

            # Extract data columns
            bboxes_xyxy = output_per_image[:, 0:4]
            scores = output_per_image[:, 4] * output_per_image[:, 5] # obj * cls_conf
            cls_ids = output_per_image[:, 6]
            kpt_vis_scores = output_per_image[:, 7 : 7 + self.num_keypoints]
            # Keypoint coordinates are already decoded to input image scale by decode_outputs if used
            # Shape [N_dets, 2*Nk] -> Reshape for easier access
            kpt_coords = output_per_image[:, 7 + self.num_keypoints :].view(-1, self.num_keypoints, 2)

            # Scale boxes and keypoints back to original image dimensions
            bboxes_xyxy /= scale
            kpt_coords /= scale

            # Convert boxes to XYWH format for COCO
            bboxes_xywh = xyxy2xywh(bboxes_xyxy)

            num_dets = output_per_image.shape[0]
            for det_idx in range(num_dets):
                class_id = self.dataloader.dataset.class_ids[int(cls_ids[det_idx])]

                # Ensure class is 'person' (or category_id 1 in standard COCO)
                # Adjust this check if your dataset uses different IDs
                if class_id != 1:
                   # logger.warning(f"Skipping detection with non-person class_id {class_id}")
                    continue

                keypoints_coco_fmt = []
                kpts_det = kpt_coords[det_idx] # [Nk, 2]
                kpt_scores_det = kpt_vis_scores[det_idx] # [Nk]

                for kp_idx in range(self.num_keypoints):
                    x, y = kpts_det[kp_idx].tolist()
                    score_kpt = kpt_scores_det[kp_idx].item()

                    # COCO format requires x, y, v
                    # Here, 'v' represents visibility/confidence score.
                    # Using the raw confidence score is common.
                    # Handle cases where internal thresholding set coords to 0
                    # (We still need to output a prediction for COCO eval)
                    if abs(x) < 1e-6 and abs(y) < 1e-6 and score_kpt < 0.5: # Check if likely zeroed out
                        visibility = 0 # Treat as not labeled/visible if zeroed and low score
                    else:
                        # Option 1: Use score directly (might not perfectly match v=0,1,2 definition)
                        # visibility = score_kpt
                        # Option 2: Simple thresholding to map score -> v (adjust threshold as needed)
                        visibility = 2 if score_kpt > 0.1 else 0 # Example: map score > 0.1 to visible=2

                    keypoints_coco_fmt.extend([round(x, 2), round(y, 2), visibility])

                pred_data = {
                    "image_id": int(img_id),
                    "category_id": class_id, # Should be 1 for person
                    "bbox": bboxes_xywh[det_idx].numpy().tolist(),
                    "score": scores[det_idx].numpy().item(),
                    "keypoints": keypoints_coco_fmt,
                }
                data_list.append(pred_data)

        return data_list


    def evaluate_prediction(self, data_dict, statistics):
        """
        Evaluates the predictions stored in data_dict using COCOeval for keypoints.
        """
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluating keypoints in main process...")

        annType = ["segm", "bbox", "keypoints"]
        iouType = annType[2] # <-- Use "keypoints"

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )
        info = time_info + "\n"

        if len(data_dict) == 0:
            logger.warning("No detections found, returning zero AP.")
            return 0, 0, info + "\nNo detections found."


        # Load ground truth
        cocoGt = self.dataloader.dataset.coco
        if cocoGt is None:
             raise ValueError("Dataloader's dataset must have a 'coco' attribute (pycocotools COCO object).")

        # Load results (predictions)
        # Use temp file to load results into COCO API
        _, tmp = tempfile.mkstemp()
        try:
            with open(tmp, "w") as f:
                json.dump(data_dict, f)
            cocoDt = cocoGt.loadRes(tmp)
        except Exception as e:
            logger.error(f"Error loading prediction results into COCO API: {e}")
            try:
                 os.remove(tmp)
            except OSError:
                 pass
            return 0, 0, info + f"\nError loading results: {e}"
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        # --- Use COCOeval ---
        # Check for optimized version first
        try:
            from yolox.layers import CocoEvalOpt as COCOeval
            logger.info("Using optimized COCOeval (CocoEvalOpt).")
        except ImportError:
            from pycocotools.cocoeval import COCOeval
            logger.warning("Optimized COCOeval (CocoEvalOpt) not found. Using standard pycocotools COCOeval.")


        cocoEval = COCOeval(cocoGt, cocoDt, iouType) # Use iouType = "keypoints"

        # Configure parameters for keypoint evaluation if needed (usually defaults are fine)
        # Example: cocoEval.params.useSegm = 0 # Explicitly disable segmentation eval if maskrcnn used iouType
        cocoEval.params.catIds = [1] # Evaluate only for the 'person' category (ID 1)

        cocoEval.evaluate()
        cocoEval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize() # Prints the standard COCO keypoint AP summary

        info += redirect_string.getvalue()

        # Extract key metrics (OKS AP @ IoU=0.50:0.95 and OKS AP @ IoU=0.50)
        ap50_95 = cocoEval.stats[0]
        ap50 = cocoEval.stats[1]

        # Per-class tables are not standard/meaningful for keypoints
        # if self.per_class_AP: ...
        # if self.per_class_AR: ...

        return ap50_95, ap50, info