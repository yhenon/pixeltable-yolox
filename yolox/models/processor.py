from __future__ import annotations

from typing import Iterable, TypedDict, Union

import numpy as np
import torch
from PIL.Image import Image

from yolox import data, utils
from yolox.config import YoloxConfig
from typing import Iterable, List, Tuple, Optional
from dataclasses import dataclass, field


class YoloxProcessor:
    config: YoloxConfig

    def __init__(
        self,
        model_name_or_config: Union[str, YoloxConfig],
    ):
        if isinstance(model_name_or_config, str):
            self.config = YoloxConfig.get_named_config(model_name_or_config)
        elif isinstance(model_name_or_config, YoloxConfig):
            self.config = model_name_or_config
        else:
            raise ValueError("model_name_or_config must be a string or YoloxConfig")

    def __call__(self, inputs: Iterable[Image]) -> torch.Tensor:
        return self.__images_to_tensor(inputs)

    def __images_to_tensor(self, images: Iterable[Image]) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        _val_transform = data.ValTransform(legacy=False)
        for image in images:
            # image = normalize_image_mode(image)
            image_transform, _ = _val_transform(np.array(image), None, self.config.test_size)
            tensors.append(torch.from_numpy(image_transform))
        return torch.stack(tensors)

    def postprocess(self, images: Iterable[Image], tensor: torch.Tensor, threshold: float = 0.5) -> list[Detections]:
        outputs: list[torch.Tensor] = utils.postprocess(tensor, self.config.num_classes, threshold, self.config.nmsthre, class_agnostic=False)
        results: list[Detections] = []
        for i, image in enumerate(images):
            ratio = min(self.config.test_size[0] / image.height, self.config.test_size[1] / image.width)
            if outputs[i] is None:
                results.append(Detections(bboxes=[], scores=[], labels=[]))
            else:
                results.append(
                    Detections(
                        bboxes=[tuple((output[:4] / ratio).tolist()) for output in outputs[i]],
                        scores=[output[4].item() * output[5].item() for output in outputs[i]],
                        labels=[int(output[6]) for output in outputs[i]],
                    )
                )
        return results


class Detections(TypedDict):
    bboxes: list[tuple[float, float, float, float]]
    scores: list[float]
    labels: list[int]



@dataclass
class DetectionsWithKpts:
    bboxes: List[Tuple[float, float, float, float]] # (x1, y1, x2, y2)
    scores: List[float]
    labels: List[int]
    # Keypoints: List of detections, each detection has a list of N_kpt keypoints (x, y, score)
    keypoints: Optional[List[List[Tuple[float, float, float]]]] = field(default=None)

class YoloxProcessorWithKpts(YoloxProcessor):

    def postprocess(self, images: Iterable[Image.Image], tensor: torch.Tensor, threshold: float = 0.5) -> list[Detections]:
        """
        Postprocesses YOLOX output tensor to produce Detections objects,
        including bounding boxes, scores, labels, and keypoints.

        Args:
            images (Iterable[Image.Image]): Input images (needed for coordinate scaling).
            tensor (torch.Tensor): Raw model output tensor.
            threshold (float): Confidence threshold.

        Returns:
            list[Detections]: List of Detections objects for each image.
        """
        num_kpts = self.config.num_keypoints
        num_classes = 1 # As specified

        # Call the modified internal postprocessing function
        # Make sure 'utils.postprocess' now points to 'postprocess_with_kpts'
        # or replace the call directly.
        # Assuming the internal function is now named postprocess_with_kpts
        outputs: list[Optional[torch.Tensor]] = utils.postprocess_with_kpts(
            tensor,
            num_classes=num_classes,
            num_keypoints=num_kpts,
            conf_thre=threshold,
            nms_thre=self.config.nmsthre,
            class_agnostic=False # Choose based on your needs
        )

        results: list[Detections] = []
        for i, image in enumerate(images):
            # Calculate scaling ratio (from test_size back to original image size)
            # Assuming test_size = (height, width)
            img_h, img_w = image.height, image.width
            test_h, test_w = self.config.test_size
            ratio_h = test_h / img_h
            ratio_w = test_w / img_w
            # Use the smaller ratio to maintain aspect ratio, matching YOLOX preprocessing
            ratio = min(ratio_h, ratio_w)

            output_i = outputs[i] # Detections tensor for the i-th image

            if output_i is None or output_i.numel() == 0:
                 # Append empty Detections if no detections found
                 results.append(Detections(bboxes=[], scores=[], labels=[], keypoints=[]))
                 continue

            # Prepare lists to store results for this image
            bboxes_scaled = []
            scores_final = []
            labels_final = []
            keypoints_scaled = []

            # Indices in output_i tensor [N_dets, 7 + 3*N_kpt]
            # 0:4 -> bbox (x1, y1, x2, y2)
            # 4   -> obj_conf
            # 5   -> class_conf
            # 6   -> class_pred (label)
            # 7 : 7+N_kpt -> kpt_conf
            # 7+N_kpt : 7+3*N_kpt -> kpt_coords (x,y pairs)

            kpt_conf_start_idx = 7
            kpt_conf_end_idx = kpt_conf_start_idx + num_kpts
            kpt_coords_start_idx = kpt_conf_end_idx
            # kpt_coords_end_idx = kpt_coords_start_idx + 2 * num_kpts # This is the end index

            for detection in output_i:
                # Bounding Box: Extract and scale
                box = detection[:4]
                # Scale box coordinates back to original image size
                box_scaled = (box / ratio).tolist()
                bboxes_scaled.append(tuple(box_scaled)) # (x1, y1, x2, y2)

                # Score: obj_conf * class_conf
                score = detection[4].item() * detection[5].item()
                scores_final.append(score)

                # Label: class_pred
                label = int(detection[6].item())
                labels_final.append(label)

                # Keypoints: Extract, reshape, scale, and combine
                kpt_confs = detection[kpt_conf_start_idx : kpt_conf_end_idx]
                # Reshape coords from [2*N_kpt] to [N_kpt, 2]
                kpt_coords = detection[kpt_coords_start_idx:].view(num_kpts, 2)

                # Scale keypoint coordinates back to original image size
                kpt_coords_scaled = kpt_coords / ratio

                kpts_for_one_det = []
                for kp_idx in range(num_kpts):
                    x, y = kpt_coords_scaled[kp_idx].tolist()
                    kpt_score = kpt_confs[kp_idx].item()
                    kpts_for_one_det.append((x, y, kpt_score)) # (x, y, score/vis)

                keypoints_scaled.append(kpts_for_one_det)

            # Create Detections object for the image
            results.append(
                Detections(
                    bboxes=bboxes_scaled,
                    scores=scores_final,
                    labels=labels_final,
                    keypoints=keypoints_scaled,
                )
            )

        return results