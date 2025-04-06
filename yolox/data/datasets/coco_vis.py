import cv2
import numpy as np
import random
import torch

def visualize_output(image_tensor, padded_labels, kpt_color_map=None, vis_thresh=0.3, window_name="Augmented Output"):
    """
    Visualizes the output of TrainTransformPose using OpenCV.

    Args:
        image_tensor (np.ndarray): The transformed image tensor (C x H x W, float32).
        padded_labels (np.ndarray): The padded labels tensor (max_labels, 5 + K*3).
                                    Format: [label, cx, cy, w, h, xk1, yk1, vk1, ...]
        skeleton_connections (list[tuple[int, int]], optional):
            List of keypoint index pairs to connect for drawing skeletons.
            Defaults to None.
        kpt_color_map (list[tuple[int, int, int]], optional):
            List of BGR color tuples, one for each keypoint type.
            If None, uses a default palette.
        vis_thresh (float): Minimum visibility score to display a keypoint.
        window_name (str): Name for the OpenCV display window.
    """

    if type(image_tensor) == torch.Tensor:
        image_tensor = image_tensor.cpu().detach().numpy()
        padded_labels = padded_labels.cpu().detach().numpy()

    _COLORS = np.array([
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000,
    ]).astype(np.float32).reshape(-1, 3)

    # Example Skeleton Connections for COCO (17 keypoints)
    # Adjust this based on your keypoint definition
    # Each tuple represents a pair of keypoint indices to connect
    skeleton_connections = [
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Body
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        # Legs
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    # Example Skeleton Connections for a 5-point Face model
    # Indices: 0: L eye, 1: R eye, 2: Nose, 3: L mouth corner, 4: R mouth corner
    FACE_SKELETON = [
        (0, 1), (0, 2), (1, 2), (3, 2), (4, 2), (3, 4)
    ]

    def cxcywh_to_xyxy(box_cxcywh):
        """Converts bounding box from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
        cx, cy, w, h = box_cxcywh
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])

    # 1. Convert image tensor back to OpenCV format (H, W, C) and uint8
    # Check if image is C, H, W or H, W, C
    if image_tensor.shape[0] == 3 or image_tensor.shape[0] == 1: # Check channel dimension first
        img_display = image_tensor.transpose(1, 2, 0)
    else:
        img_display = image_tensor

    # Ensure it's contiguous
    img_display = np.ascontiguousarray(img_display)

    # Convert to uint8 - Assuming the preproc output is 0-255 float
    # If your preproc includes normalization, you'll need to reverse it here.
    if img_display.dtype == np.float32 or img_display.dtype == np.float64:
        # Clip just in case, though preproc usually doesn't exceed 255
        img_display = np.clip(img_display, 0, 255)
    img_display = img_display.astype(np.uint8)

    # If the image was single channel, convert it to BGR for color drawing
    if len(img_display.shape) == 2 or img_display.shape[2] == 1:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    # Make a copy to draw on, otherwise drawing modifies the original tensor view
    img_display = img_display.copy()

    # Determine number of keypoints K
    num_kpts = 0
    if padded_labels.shape[1] > 5:
        num_kpts = (padded_labels.shape[1] - 5) // 3

    # Prepare keypoint colors
    if kpt_color_map is None:
        # Use the default _COLORS, cycling if needed
        num_colors = len(_COLORS)
        kpt_colors = [_COLORS[i % num_colors] * 255 for i in range(num_kpts)]
        kpt_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kpt_colors] # Convert to int BGR tuples
    elif len(kpt_color_map) >= num_kpts:
        kpt_colors = kpt_color_map[:num_kpts]
    else:
        print(f"Warning: Provided kpt_color_map has {len(kpt_color_map)} colors, but {num_kpts} are needed. Using default.")
        num_colors = len(_COLORS)
        kpt_colors = [_COLORS[i % num_colors] * 255 for i in range(num_kpts)]
        kpt_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kpt_colors]

    # 2. Iterate through labels
    for label_data in padded_labels:
        label = int(label_data[0])
        box_cxcywh = label_data[1:5]
        kpts_flat = label_data[5:] # Shape (K*3,)

        # Check if it's a valid (non-padded) entry
        # A simple check: if w or h is near zero, it's likely padding
        if box_cxcywh[2] <= 1 or box_cxcywh[3] <= 1:
            continue

        # 3. Draw Bounding Box
        box_xyxy = cxcywh_to_xyxy(box_cxcywh)
        x1, y1, x2, y2 = map(int, box_xyxy)
        box_color = (0, 255, 0) # Green for boxes
        cv2.rectangle(img_display, (x1, y1), (x2, y2), box_color, 2)

        # Add label text (optional)
        # cv2.putText(img_display, f"Cls: {label}", (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # 4. Draw Keypoints
        if num_kpts > 0:
            kpts_xyv = kpts_flat.reshape(num_kpts, 3) # Shape (K, 3) -> [x, y, v]
            instance_kpt_coords = {} # Store coords for skeleton drawing

            for i in range(num_kpts):
                x, y, v = kpts_xyv[i]
                if v >= vis_thresh: # Only draw if visible enough
                    kpt_x, kpt_y = int(x), int(y)
                    color = kpt_colors[i]
                    cv2.circle(img_display, (kpt_x, kpt_y), radius=3, color=color, thickness=-1) # Filled circle
                    instance_kpt_coords[i] = (kpt_x, kpt_y) # Store visible coords

            # 5. Draw Skeleton
            if skeleton_connections is not None and len(instance_kpt_coords) > 1:
                for idx1, idx2 in skeleton_connections:
                    # Check if both keypoints for the connection are visible
                    if idx1 in instance_kpt_coords and idx2 in instance_kpt_coords:
                        pt1 = instance_kpt_coords[idx1]
                        pt2 = instance_kpt_coords[idx2]
                        # Use color of the first keypoint in the pair for the line
                        line_color = kpt_colors[idx1]
                        cv2.line(img_display, pt1, pt2, line_color, thickness=1)

    # 6. Display the image
    cv2.imshow(window_name, img_display)
    print(f"Displaying '{window_name}'. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name) # Close only this window
