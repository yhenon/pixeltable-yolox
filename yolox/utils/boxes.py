# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "postprocess_with_kpts",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "adjust_kpts_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

import torch
import torchvision

def postprocess_with_kpts(prediction, num_classes, num_keypoints, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """
    Postprocesses detections and keypoints from YOLOX-style output.

    Args:
        prediction (torch.Tensor): Raw model output tensor of shape
            [batch_size, num_anchors, 4(bbox) + 1(obj) + num_classes + N_kpt(conf) + 2*N_kpt(coords)].
            Expected bbox format is (cx, cy, w, h).
        num_classes (int): Number of object classes. Should be 1 for this adaptation.
        num_keypoints (int): Number of keypoints per instance.
        conf_thre (float): Confidence threshold for filtering detections.
        nms_thre (float): NMS IoU threshold.
        class_agnostic (bool): Whether to perform class-agnostic NMS.

    Returns:
        list[torch.Tensor | None]: A list where each element corresponds to an image
            in the batch. Each element is a tensor of shape [num_detections, 7 + 3*num_keypoints]
            containing (x1, y1, x2, y2, obj_conf, class_conf, class_pred, kpt_conf_1, ..., kpt_conf_N,
            kpt_x1, kpt_y1, ..., kpt_xN, kpt_yN), or None if no detections are found.
    """
    # Box coordinate conversion (cx, cy, w, h) -> (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    # Calculate indices for slicing
    bbox_end_idx = 4
    obj_idx = 4
    cls_start_idx = 5
    cls_end_idx = cls_start_idx + num_classes
    kpt_conf_start_idx = cls_end_idx
    kpt_conf_end_idx = kpt_conf_start_idx + num_keypoints
    kpt_coords_start_idx = kpt_conf_end_idx
    kpt_coords_end_idx = kpt_coords_start_idx + 2 * num_keypoints # Should match total dim size

    # Verify total dimension: 4(box) + 1(obj) + num_classes + N_kpt(conf) + 2*N_kpt(coords)
    expected_last_dim = 4 + 1 + num_classes + 3 * num_keypoints
    assert prediction.shape[-1] == expected_last_dim, \
        f"Prediction tensor last dim ({prediction.shape[-1]}) != expected ({expected_last_dim})"


    for i, image_pred in enumerate(prediction): # Process each image in the batch

        # If no predictions for this image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        # image_pred[:, cls_start_idx : cls_end_idx] -> [num_anchors, num_classes]
        class_conf, class_pred = torch.max(image_pred[:, cls_start_idx : cls_end_idx], 1, keepdim=True)

        # Filter by confidence threshold: obj_conf * class_conf >= conf_thre
        conf_mask = (image_pred[:, obj_idx] * class_conf.squeeze() >= conf_thre).squeeze()

        # Create candidate detections tensor including keypoints
        # Format: (x1, y1, x2, y2, obj_conf, class_conf, class_pred, kpt_conf_all, kpt_coords_all)
        detections = torch.cat((
            image_pred[:, :bbox_end_idx],      # Bbox (x1, y1, x2, y2)
            image_pred[:, obj_idx:obj_idx+1], # Objectness conf
            class_conf,                      # Class conf (max)
            class_pred.float(),              # Class pred index
            image_pred[:, kpt_conf_start_idx:kpt_conf_end_idx], # Keypoint confs
            image_pred[:, kpt_coords_start_idx:kpt_coords_end_idx] # Keypoint coords (x,y pairs)
        ), 1)
        # Shape: [num_anchors, 4 + 1 + 1 + 1 + N_kpt + 2*N_kpt] = [num_anchors, 7 + 3*N_kpt]

        detections = detections[conf_mask] # Apply confidence mask

        # If no detections remain after confidence filtering
        if not detections.size(0):
            continue

        # Perform Non-Maximum Suppression (NMS)
        # NMS input: boxes (0:4), scores (obj_conf * class_conf -> col 4 * col 5), classes (col 6)
        nms_boxes = detections[:, :4]
        nms_scores = detections[:, 4] * detections[:, 5]
        nms_classes = detections[:, 6] # Class index

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                nms_boxes,
                nms_scores,
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                nms_boxes,
                nms_scores,
                nms_classes,
                nms_thre,
            )

        # Select detections that survived NMS
        # Keep all columns (including keypoints) for the selected indices
        detections_after_nms = detections[nms_out_index]

        # Store results for this image
        if output[i] is None:
            output[i] = detections_after_nms
        else:
            # This part might be unnecessary if NMS is done correctly once
            # but included for consistency with original code if inputs were chunked.
            output[i] = torch.cat((output[i], detections_after_nms))

    return output

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox

def adjust_kpts_anns(keypoints, scale_ratio, padw, padh, w_max, h_max):
    if keypoints.size == 0:
        return keypoints

    num_kpts = keypoints.shape[1] // 3
    kpts_reshaped = keypoints.reshape(-1, 3) # Shape (N*K, 3)

    # Scale x, y coordinates
    kpts_reshaped[:, :2] *= scale_ratio
    # Apply padding offset
    kpts_reshaped[:, 0] += padw
    kpts_reshaped[:, 1] += padh

    # Clip coordinates
    kpts_reshaped[:, 0] = kpts_reshaped[:, 0].clip(0, w_max)
    kpts_reshaped[:, 1] = kpts_reshaped[:, 1].clip(0, h_max)

    return kpts_reshaped.reshape(-1, num_kpts * 3) # Reshape back


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes
