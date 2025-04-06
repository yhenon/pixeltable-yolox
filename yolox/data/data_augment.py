# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale

def apply_affine_to_keypoints(targets, target_size, M):
    # Targets shape: (N, 5 + K*3)
    num_gts = len(targets)
    if num_gts == 0:
        return targets

    # Infer number of keypoints K based on array shape
    num_keypoints = (targets.shape[1] - 5) // 3
    if num_keypoints <= 0: # No keypoints present
        return targets

    twidth, theight = target_size
    keypoints = targets[:, 5:].reshape(num_gts * num_keypoints, 3) # Shape: (N*K, 3)

    # Separate coords (x, y) and visibility (v)
    xy = keypoints[:, :2] # Shape: (N*K, 2)
    original_visibility = keypoints[:, 2:3].copy() # Shape: (N*K, 1), copy needed

    # Apply affine transformation to xy coordinates
    # Create homogeneous coordinates (add a column of 1s)
    xy_homogeneous = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1) # Shape: (N*K, 3)

    # Calculate UNCLIPPED transformed coordinates
    unclipped_transformed_xy = xy_homogeneous @ M.T # Shape: (N*K, 2)

    # --- Determine which keypoints are outside the target bounds BEFORE clipping ---
    # Check if the unclipped x or y coordinate is outside the image dimensions.
    # Need strict inequality for > bounds, as the boundary itself is inclusive.
    outside_x = (unclipped_transformed_xy[:, 0] < 0) | (unclipped_transformed_xy[:, 0] >= twidth)
    outside_y = (unclipped_transformed_xy[:, 1] < 0) | (unclipped_transformed_xy[:, 1] >= theight)
    is_outside = outside_x | outside_y # Shape: (N*K,) Boolean mask

    # --- Clip transformed keypoints to target image bounds ---
    # Perform clipping *after* checking the out-of-bounds status
    clipped_transformed_xy = unclipped_transformed_xy.copy() # Start with unclipped
    clipped_transformed_xy[:, 0] = clipped_transformed_xy[:, 0].clip(0, twidth - 1e-5) # Clip x slightly inside
    clipped_transformed_xy[:, 1] = clipped_transformed_xy[:, 1].clip(0, theight - 1e-5) # Clip y slightly inside
    # Using twidth-eps and theight-eps avoids points exactly at the boundary if preferred.
    # Simpler: clip(0, twidth) and clip(0, theight) as before is also fine. Let's revert to simpler clip.
    clipped_transformed_xy[:, 0] = unclipped_transformed_xy[:, 0].clip(0, twidth)
    clipped_transformed_xy[:, 1] = unclipped_transformed_xy[:, 1].clip(0, theight)


    # --- Update visibility ---
    # Set visibility to 0 if the keypoint was originally visible (v > 0)
    # AND its *unclipped* transformed location was outside the bounds.
    updated_visibility = original_visibility.copy() # Start with original visibility
    # Flatten original_visibility check for easier indexing with the 1D is_outside mask
    condition = (original_visibility.flatten() > 0) & is_outside
    updated_visibility[condition.reshape(-1, 1)] = 0 # Set visibility to 0 where condition is true

    # --- Reassemble keypoints ---
    # Use the CLIPPED coordinates and the UPDATED visibility
    transformed_keypoints = np.concatenate([clipped_transformed_xy, updated_visibility], axis=1) # Shape: (N*K, 3)

    # Reshape back and update the original targets array
    targets[:, 5:] = transformed_keypoints.reshape(num_gts, num_keypoints * 3)

    return targets


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if targets is not None and len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)
        # Then apply keypoint transformation (if keypoints exist)
        if targets.shape[1] > 5:
             targets = apply_affine_to_keypoints(targets, target_size, M)

    return img, targets


def _mirror(image, boxes, kpts, flip_indices=None, prob=0.5):
    """
    Args:
        image (numpy.ndarray): Input image.
        boxes (numpy.ndarray): Bounding boxes in xyxy format.
        kpts (numpy.ndarray): Keypoints in (x, y, visibility) format, shape (N, K*3).
                               Assumes K keypoints per instance.
        flip_indices (list[tuple[int, int]], optional): List of index pairs to swap
                                                      (e.g., [(left_eye_idx, right_eye_idx), ...]).
                                                      Indices are 0-based keypoint indices.
                                                      Defaults to None (no swapping).
        prob (float): Probability of flipping.
    Returns:
        tuple: Tuple containing the possibly flipped image, boxes, and keypoints.
    """
    _, width, _ = image.shape
    do_flip = random.random() < prob
    if do_flip:
        # Flip image
        image = image[:, ::-1]

        # Flip boxes
        boxes[:, 0::2] = width - boxes[:, 2::-2] # Flip x1, x2 by swapping and subtracting from width
        # Flip keypoints
        if kpts is not None and kpts.shape[0] > 0 and kpts.shape[1] > 0:
            # 1. Flip all x-coordinates
            kpts[:, 0::3] = width - kpts[:, 0::3]

            # 2. Swap left/right pairs if flip_indices are provided
            if flip_indices is not None:
                for left_idx, right_idx in flip_indices:
                    # Calculate column indices for x, y, v for left and right keypoints
                    left_cols = slice(left_idx * 3, left_idx * 3 + 3)
                    right_cols = slice(right_idx * 3, right_idx * 3 + 3)

                    # Create copies before swapping to avoid overwriting issues with numpy views
                    left_data_copy = kpts[:, left_cols].copy()
                    right_data_copy = kpts[:, right_cols].copy()

                    # Perform the swap
                    kpts[:, left_cols] = right_data_copy
                    kpts[:, right_cols] = left_data_copy

    return image, boxes, kpts


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes, _ = _mirror(image, boxes, None, None, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels

class TrainTransformPose:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, keypoint_flip_indices=None):
        """
        Args:
            max_labels (int): Maximum number of labels.
            flip_prob (float): Probability of horizontal flip.
            hsv_prob (float): Probability of HSV augmentation.
            keypoint_flip_indices (list[tuple[int, int]], optional):
                List of keypoint index pairs to swap during horizontal flip.
                Defaults to None. Example for COCO: [(1, 2), (3, 4), ...].
        """
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.keypoint_flip_indices = keypoint_flip_indices # Store the flip map

    def __call__(self, image, targets, input_dim):
        # targets shape: (num_instances, 5 + K*3) -> [x1, y1, x2, y2, label, xk1, yk1, vk1, xk2, yk2, vk2, ...]
        num_instances = targets.shape[0]
        num_kpts = 0
        if targets.shape[1] > 5:
             # Ensure integer division
            assert (targets.shape[1] - 5) % 3 == 0, "Keypoint columns must be a multiple of 3 (x, y, vis)"
            num_kpts = (targets.shape[1] - 5) // 3

        # Define the target shape for empty/padded labels
        target_width = 5 + num_kpts * 3

        boxes = targets[:, :4].copy() # xyxy
        labels = targets[:, 4].copy()
        kpts = None
        if num_kpts > 0:
            kpts = targets[:, 5:].copy() # Shape: (N, K*3)

        if num_instances == 0:
            # Handle case with no instances in the input
            padded_targets = np.zeros((self.max_labels, target_width), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            # No targets to scale, return empty padded targets
            return image, padded_targets

        # --- Backup original data ---
        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4] # xyxy
        labels_o = targets_o[:, 4]
        kpts_o = None
        if num_kpts > 0:
            kpts_o = targets_o[:, 5:].copy() # Shape: (N, K*3)
        # Convert backup boxes for potential fallback use
        boxes_o_cxcywh = xyxy2cxcywh(boxes_o.copy()) # Use a copy

        # --- Augmentations ---
        if random.random() < self.hsv_prob:
            augment_hsv(image) # Augment image in place

        # Mirror image, boxes (xyxy), and keypoints (xyv)
        # Pass the stored keypoint_flip_indices to _mirror
        image_t, boxes, kpts = _mirror(
            image, boxes, kpts, self.keypoint_flip_indices, self.flip_prob
        )
        height, width, _ = image_t.shape # Get potentially flipped image dimensions

        # Preprocess image (resize, pad) and get resize ratio
        image_t, r_ = preproc(image_t, input_dim)
        input_h, input_w = input_dim # Target height, width

        # --- Transform boxes and keypoints ---
        # Convert potentially flipped boxes [xyxy] to [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        # Scale boxes
        boxes *= r_

        # Scale keypoints
        if kpts is not None:
            kpts[:, 0::3] *= r_ # Scale x
            kpts[:, 1::3] *= r_ # Scale y
            # Clip keypoints to padded image boundaries
            kpts[:, 0::3] = np.clip(kpts[:, 0::3], 0, input_w)
            kpts[:, 1::3] = np.clip(kpts[:, 1::3], 0, input_h)

        # --- Filter instances based on transformed box size ---
        # Use scaled cxcywh boxes for filtering
        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1 # Filter based on w, h > 1 pixel
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        kpts_t = kpts[mask_b] if kpts is not None else None

        # --- Handle case where all instances are filtered out ---
        if len(boxes_t) == 0:
            # Preprocess the *original* image
            image_t, r_o = preproc(image_o, input_dim)
            # Scale the *original* backup boxes (already converted to cxcywh)
            boxes_o_cxcywh *= r_o
            boxes_t = boxes_o_cxcywh # Use scaled original boxes
            labels_t = labels_o     # Use original labels

            # Scale the *original* backup keypoints
            if kpts_o is not None:
                kpts_o[:, 0::3] *= r_o # Scale x
                kpts_o[:, 1::3] *= r_o # Scale y
                # Clip scaled original keypoints
                kpts_o[:, 0::3] = np.clip(kpts_o[:, 0::3], 0, input_w)
                kpts_o[:, 1::3] = np.clip(kpts_o[:, 1::3], 0, input_h)
                kpts_t = kpts_o # Use scaled original keypoints
            else:
                kpts_t = None

            # If original targets were also empty, boxes_t will be empty here.
            # The padding below handles this.

        # --- Assemble final targets ---
        labels_t = np.expand_dims(labels_t, 1) # Shape (Nf, 1)

        # Prepare keypoints for hstack (ensure correct shape even if None)
        if kpts_t is not None and kpts_t.shape[0] > 0:
            # Shape (Nf, K*3) - already in correct format
            pass
        elif num_kpts > 0: # Need placeholder if kpts exist but all instances filtered
             kpts_t = np.zeros((len(labels_t), num_kpts * 3), dtype=np.float32)
        else: # No keypoints in the dataset
            kpts_t = None # Will not be included in hstack

        # Stack labels, boxes, and keypoints (if they exist)
        if kpts_t is not None:
            targets_t = np.hstack((labels_t, boxes_t, kpts_t)) # Shape (Nf, 1 + 4 + K*3)
        else:
            targets_t = np.hstack((labels_t, boxes_t)) # Shape (Nf, 5)

        # --- Pad targets ---
        padded_labels = np.zeros((self.max_labels, target_width), dtype=np.float32)
        num_targets_to_keep = min(len(targets_t), self.max_labels)
        if num_targets_to_keep > 0:
            padded_labels[:num_targets_to_keep] = targets_t[:num_targets_to_keep]

        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels

class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
