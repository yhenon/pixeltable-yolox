import requests
from PIL import Image
from yolox.models import Yolox, YoloxProcessorWithKpts
from yolox.config import YoloxPose
import torch
import os

import cv2
import numpy as np
import torch # Keep for potential tensor input check, though less likely now
from PIL import Image # To handle PIL input images
from typing import List, Dict, Any, Optional, Tuple

# Keep the color palette and skeleton definition
_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556,
    0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300,
    0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000,
    0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000,
    0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500,
    0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500,
    0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500,
    1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000,
    0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000,
    0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000,
    1.000, 0.667, 1.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
    0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
    0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
    0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667,
    0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143,
    0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714,
    0.857, 0.857, 0.857, 1.000, 1.000, 1.000,
]).astype(np.float32).reshape(-1, 3)

# COCO Keypoint skeleton connections (indices are 0-based)
# Ensure this matches the order of your model's keypoint output
# 0: nose, 1: L eye, 2: R eye, 3: L ear, 4: R ear, 5: L shoulder, 6: R shoulder,
# 7: L elbow, 8: R elbow, 9: L wrist, 10: R wrist, 11: L hip, 12: R hip,
# 13: L knee, 14: R knee, 15: L ankle, 16: R ankle
COCO_SKELETON = [
    # Head (using eyes/nose instead of ears if ears aren't reliable)
    [1, 2], [0, 1], [0, 2], # [1, 3], [2, 4], # Optional ear connections
    # Body
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    # Legs
    [11, 13], [13, 15], [12, 14], [14, 16]
]

def visualize_predictions(
    image: Image.Image, # Expecting PIL image as input now
    predictions: List[Dict[str, Any]],
    kpt_color_map: Optional[List[Tuple[int, int, int]]] = None,
    skeleton_connections: Optional[List[Tuple[int, int]]] = COCO_SKELETON,
    kpt_vis_thresh: float = 0.3,
    box_score_thresh: float = 0.5, # Add threshold for detection score
    window_name: str = "Predicted Output"
):
    """
    Visualizes predicted detections (bboxes, labels, scores, keypoints) on an image.

    Args:
        image (PIL.Image.Image): The input image.
        predictions (List[Dict[str, Any]]): The output list from the postprocess function.
            Expected format: [{'bboxes': [...], 'scores': [...], 'labels': [...], 'keypoints': [[...],...]}]
        kpt_color_map (list[tuple[int, int, int]], optional):
            List of BGR color tuples, one for each keypoint type. Defaults to cycling _COLORS.
        skeleton_connections (list[tuple[int, int]], optional):
            List of keypoint index pairs to connect. Defaults to COCO_SKELETON.
        kpt_vis_thresh (float): Minimum score/visibility to display a keypoint.
        box_score_thresh (float): Minimum score to display a bounding box and its keypoints.
        window_name (str): Name for the OpenCV display window.
    """

    # 1. Convert PIL Image to OpenCV format (BGR, uint8)
    img_display = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_display = np.ascontiguousarray(img_display) # Ensure it's contiguous

    # Check if predictions list is valid and contains data
    if not predictions or not isinstance(predictions[0], dict):
        print("Warning: Invalid or empty predictions received.")
        # Display the raw image if no predictions
        cv2.imshow(window_name, img_display)
        print(f"Displaying '{window_name}' (no valid predictions). Press any key.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        return

    # Extract data for the single image (assuming batch size 1 in visualization)
    dets = predictions[0]
    bboxes = dets.get('bboxes', [])
    scores = dets.get('scores', [])
    labels = dets.get('labels', []) # Although likely all 0 for num_classes=1
    keypoints_all = dets.get('keypoints', []) # List of lists of (x, y, score) tuples

    if not scores: # Check if there are any detections
         print("No detections found in the prediction output.")
         # Display the raw image if no detections
         cv2.imshow(window_name, img_display)
         print(f"Displaying '{window_name}' (no detections). Press any key.")
         cv2.waitKey(0)
         cv2.destroyWindow(window_name)
         return

    # Determine number of keypoints K (if any detections have keypoints)
    num_kpts = 0
    if keypoints_all and keypoints_all[0]:
        num_kpts = len(keypoints_all[0])

    # Prepare keypoint colors
    if kpt_color_map is None:
        num_colors = len(_COLORS)
        kpt_colors = [_COLORS[i % num_colors] * 255 for i in range(num_kpts)]
        kpt_colors = [(int(c[2]), int(c[1]), int(c[0])) for c in kpt_colors] # Convert to int BGR tuples
    elif len(kpt_color_map) >= num_kpts:
        kpt_colors = kpt_color_map[:num_kpts] # Assuming input is BGR tuples
    else:
        print(f"Warning: Provided kpt_color_map has {len(kpt_color_map)} colors, but {num_kpts} are needed. Using default.")
        num_colors = len(_COLORS)
        kpt_colors = [_COLORS[i % num_colors] * 255 for i in range(num_kpts)]
        kpt_colors = [(int(c[2]), int(c[1]), int(c[0])) for c in kpt_colors]

    # 2. Iterate through detected instances
    num_detections = len(scores)
    for i in range(num_detections):
        score = scores[i]

        # --- Filter by detection score ---
        if score < box_score_thresh:
            continue

        # 3. Draw Bounding Box
        bbox = bboxes[i]
        x1, y1, x2, y2 = map(int, bbox)
        box_color = (0, 255, 0) # Green for boxes
        cv2.rectangle(img_display, (x1, y1), (x2, y2), box_color, 2)

        # Add score text (optional)
        label = labels[i] # Get label if needed
        text = f"Conf: {score:.2f}" #f"Cls: {label} Conf: {score:.2f}"
        cv2.putText(img_display, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # 4. Draw Keypoints for this instance
        if i < len(keypoints_all) and num_kpts > 0:
            instance_keypoints = keypoints_all[i] # List of (x, y, score) for this instance
            instance_kpt_coords = {} # Store coords for skeleton drawing {kp_index: (x, y)}

            for kp_idx, kpt_data in enumerate(instance_keypoints):
                kpt_x, kpt_y, kpt_score = kpt_data

                # --- Filter by keypoint visibility/score ---
                if kpt_score >= kpt_vis_thresh:
                    int_x, int_y = int(kpt_x), int(kpt_y)
                    color = kpt_colors[kp_idx]
                    cv2.circle(img_display, (int_x, int_y), radius=3, color=color, thickness=-1) # Filled circle
                    instance_kpt_coords[kp_idx] = (int_x, int_y) # Store visible coords

            # 5. Draw Skeleton for this instance
            if skeleton_connections is not None and len(instance_kpt_coords) > 1:
                for idx1, idx2 in skeleton_connections:
                    # Check if both keypoints for the connection are visible
                    if idx1 in instance_kpt_coords and idx2 in instance_kpt_coords:
                        pt1 = instance_kpt_coords[idx1]
                        pt2 = instance_kpt_coords[idx2]
                        # Use color of the first keypoint in the pair for the line? Or a fixed color?
                        # Using the first keypoint's color:
                        line_color = kpt_colors[idx1]
                        # Or a fixed skeleton color: line_color = (255, 128, 0) # Cyan-ish
                        cv2.line(img_display, pt1, pt2, line_color, thickness=1)

    # 6. Display the image
    cv2.imshow(window_name, img_display)
    print(f"Displaying '{window_name}'. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name) # Close only this window

model = YoloxPose().get_model()
processor = YoloxProcessorWithKpts("yolox_pose")
model.training = False
model.head.training = False

path = 'latest_ckpt.pth'
weights = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(weights['model'])

dirname = '../../datasets/coco/train2017'
for f in os.listdir(dirname):
    image = Image.open(os.path.join(dirname, f)).convert("RGB") 
    tensor = processor([image])
    output = model(tensor)
    result = processor.postprocess([image], output)
    print(result[0])
    if len(result[0]['bboxes']) > 0:
        visualize_predictions(image=image, predictions=result)