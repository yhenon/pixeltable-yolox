# Assuming previous imports and Dataset base class exist
# Need: random, cv2, numpy, get_local_rank (from yolox.utils), adjust_box_anns (if used, otherwise implement logic)
# Need: random_affine (modified version from above)
# Need: Dataset base class from .datasets_wrapper

# We might need to implement adjust_box_anns logic if it's not available
# Let's define a simple version for bbox adjustment logic used in mixup
def adjust_bboxes(bboxes, scale_ratio, padw, padh, target_w, target_h):
    """Adjusts bboxes by scaling and padding, then clips."""
    bboxes[:, :4] *= scale_ratio
    bboxes[:, 0] += padw
    bboxes[:, 1] += padh
    bboxes[:, 0] = bboxes[:, 0].clip(0, target_w)
    bboxes[:, 1] = bboxes[:, 1].clip(0, target_h)
    bboxes[:, 2] = bboxes[:, 2].clip(0, target_w)
    bboxes[:, 3] = bboxes[:, 3].clip(0, target_h)
    return bboxes

# New helper function to adjust keypoints in mixup
def adjust_keypoints(keypoints, scale_ratio, padw, padh, target_w, target_h, flip=False, flip_width=0):
    """Adjusts keypoints by scaling, padding, flipping, and clipping."""
    if keypoints.size == 0:
        return keypoints

    num_kpts = keypoints.shape[1] // 3
    kpts_reshaped = keypoints.reshape(-1, 3) # Shape (N*K, 3)

    # Scale x, y coordinates
    kpts_reshaped[:, :2] *= scale_ratio
    # Apply padding offset
    kpts_reshaped[:, 0] += padw
    kpts_reshaped[:, 1] += padh

    # Apply flipping if needed
    if flip:
        kpts_reshaped[:, 0] = flip_width - kpts_reshaped[:, 0]

    # Clip coordinates
    kpts_reshaped[:, 0] = kpts_reshaped[:, 0].clip(0, target_w)
    kpts_reshaped[:, 1] = kpts_reshaped[:, 1].clip(0, target_h)

    # Optional: Update visibility if clipped to boundary (similar to affine)
    # Keep original visibility for simplicity now

    return kpts_reshaped.reshape(-1, num_kpts * 3) # Reshape back


class MosaicDetection(Dataset):
    """
    Detection dataset wrapper that performs mosaic and mixup augmentations
    for datasets with bounding boxes and optional keypoints.
    """

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
        """
        Args:
            dataset(Dataset) : Pytorch dataset object (e.g., CocoDataset, CocoKeypointDataset).
                               Must have pull_item method returning (img, labels, img_info, img_id).
                               Labels format: [x1, y1, x2, y2, class_id, (optional) kpt1_x, kpt1_y, kpt1_v, ...]
            img_size (tuple): Target input dimension (height, width).
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func): preprocessing function applied at the end. Must handle keypoints if present.
            degrees (float): range for random rotation degrees.
            translate (float): range for random translation factor.
            mosaic_scale (tuple): range for mosaic random scaling.
            mixup_scale (tuple): range for mixup random scaling.
            shear (float): range for random shear degrees.
            enable_mixup (bool): enable mixup augmentation or not.
            mosaic_prob (float): probability of applying mosaic.
            mixup_prob (float): probability of applying mixup (after mosaic).
        """
        # Initialize Dataset properties (like input_dim if needed by decorator)
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        # Ensure img_size is stored correctly (e.g., as self.input_dim)
        # The base Dataset class might handle this, or set it explicitly:
        self.input_dim = img_size
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale # Renamed mosaic_scale to self.scale for consistency
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        # self.local_rank = get_local_rank() # Assuming get_local_rank() is available

    def __len__(self):
        return len(self._dataset)

    # Assuming Dataset.mosaic_getitem decorator handles switching between mosaic/normal
    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            # Use self.input_dim consistently
            input_h, input_w = self.input_dim[0], self.input_dim[1]

            # Mosaic center calculation
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # Get indices for the 4 mosaic images
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            mosaic_img = None # Initialize mosaic image

            for i_mosaic, index in enumerate(indices):
                # Use pull_item which should return img, label array, img_info, img_id
                # Ensure pull_item handles potential errors (e.g., missing data) gracefully
                pulled_data = self._dataset.pull_item(index)
                if pulled_data[0] is None: # Handle cases where pull_item failed
                    # Option: skip this image, try another? Or fail? Let's retry once.
                    retry_index = random.randint(0, len(self._dataset) - 1)
                    pulled_data = self._dataset.pull_item(retry_index)
                    if pulled_data[0] is None:
                         # If retry fails, maybe skip mosaic for this batch? Or raise error.
                         # For simplicity, let's continue, potentially with fewer than 4 images.
                         print(f"Warning: Failed to load image for mosaic index {index} and retry {retry_index}. Skipping.")
                         continue # Skip this sub-image

                img, _labels, _, _ = pulled_data # We don't need img_info, img_id here

                if img is None or _labels is None: # Check again after potential retry
                    print(f"Warning: img or _labels is None for mosaic index {index}. Skipping.")
                    continue

                h0, w0 = img.shape[:2]  # Original image size
                # Apply scaling to fit into the mosaic grid (proportional resize)
                scale = min(1. * input_h / h0, 1. * input_w / w0) # Use 1.0 for float division
                # Resize image to fit potentially smaller mosaic cell (avoids upsampling if scale > 1)
                # The original code seems to always scale based on input_dim, let's keep that.
                # scale = min(1. * input_h / h0, 1. * input_w / w0) # This could be < 1 or > 1
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                h, w = img.shape[:2] # Size after resizing

                # Initialize mosaic_img on first iteration
                if i_mosaic == 0:
                    # Assuming 3 channels, check img.shape if needed
                    c = img.shape[2] if len(img.shape) == 3 else 1
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                    # Handle grayscale images
                    if c == 1:
                         mosaic_img = np.squeeze(mosaic_img) # Remove channel dim if grayscale


                # Get coordinates for placing the scaled image patch into the large mosaic canvas
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                # Place the image patch
                # Ensure channel dimensions match if mixing color/gray (handle this if necessary)
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                # Calculate padding offset for labels
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                # --- Transform labels (bboxes and keypoints) ---
                labels = _labels.copy()
                if labels.size > 0:
                    # Adjust bounding boxes [x1, y1, x2, y2]
                    labels[:, 0] = scale * labels[:, 0] + padw # x1
                    labels[:, 1] = scale * labels[:, 1] + padh # y1
                    labels[:, 2] = scale * labels[:, 2] + padw # x2
                    labels[:, 3] = scale * labels[:, 3] + padh # y2

                    # Adjust keypoints [..., kpt_x, kpt_y, kpt_v, ...] (if they exist)
                    if labels.shape[1] > 5:
                        num_keypoints = (labels.shape[1] - 5) // 3
                        # Reshape for easier manipulation: (N, K, 3)
                        keypoints = labels[:, 5:].reshape(labels.shape[0], num_keypoints, 3)

                        # Mask for valid keypoints (v > 0) to avoid transforming non-existent ones
                        valid_kpt_mask = keypoints[:, :, 2] > 0

                        # Scale x, y coordinates where valid
                        keypoints[valid_kpt_mask, 0] = scale * keypoints[valid_kpt_mask, 0] + padw # x
                        keypoints[valid_kpt_mask, 1] = scale * keypoints[valid_kpt_mask, 1] + padh # y
                        # Visibility (v) remains unchanged

                        # Reshape back to flat format and update labels array
                        labels[:, 5:] = keypoints.reshape(labels.shape[0], num_keypoints * 3)

                mosaic_labels.append(labels)
            # End of mosaic loop

            # Check if mosaic image was created (might not if all pull_item failed)
            if mosaic_img is None:
                 print("Warning: Mosaic image could not be created. Falling back to single image.")
                 # Fallback to loading a single image without mosaic
                 # Needs self._dataset._input_dim potentially set?
                 self._dataset.input_dim = self.input_dim # Pass input_dim to dataset if needed
                 img, label, img_info, img_id = self._dataset.pull_item(idx)
                 if img is None or label is None: # Handle error loading fallback image
                     # Return dummy data or raise error
                     print(f"Error: Failed to load fallback image for index {idx}")
                     # Simplest: return None, let dataloader handle skipping
                     # Requires dataloader to handle None batches.
                     return None, None, None, None
                 img, label = self.preproc(img, label, self.input_dim)
                 return img, label, img_info, img_id


            # Concatenate labels from all 4 images
            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)

                # --- Clip labels to the large mosaic canvas (2*h, 2*w) ---
                # Clip bounding boxes
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0]) # x1
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1]) # y1
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2]) # x2
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3]) # y2

                # Clip keypoint coordinates (if they exist)
                if mosaic_labels.shape[1] > 5:
                    num_keypoints = (mosaic_labels.shape[1] - 5) // 3
                    keypoints = mosaic_labels[:, 5:].reshape(mosaic_labels.shape[0], num_keypoints, 3)

                    # Clip x coordinates
                    np.clip(keypoints[:, :, 0], 0, 2 * input_w, out=keypoints[:, :, 0])
                    # Clip y coordinates
                    np.clip(keypoints[:, :, 1], 0, 2 * input_h, out=keypoints[:, :, 1])
                    # Visibility remains unchanged during this clipping stage

                    mosaic_labels[:, 5:] = keypoints.reshape(mosaic_labels.shape[0], num_keypoints * 3)

                 # --- Filter out invalid boxes after clipping (e.g., zero width/height) ---
                 bboxes_wh = mosaic_labels[:, 2:4] - mosaic_labels[:, :2] # w = x2-x1, h = y2-y1
                 valid_indices = (bboxes_wh[:, 0] > 1) & (bboxes_wh[:, 1] > 1) # Keep if w > 1 and h > 1
                 mosaic_labels = mosaic_labels[valid_indices]


            # Apply random affine transformation to the large mosaic image and labels
            # Pass self.scale (renamed from mosaic_scale) to random_affine
            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h), # Target size is the final input size
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale, # Use the scale range defined for mosaic
                shear=self.shear,
            )

            # Apply MixUp augmentation
            if (
                self.enable_mixup
                and len(mosaic_labels) > 0 # Ensure there are labels to mix with
                and random.random() < self.mixup_prob
            ):
                # Pass input_dim correctly to mixup
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)


            # Final Preprocessing (e.g., normalization, tensor conversion)
            # self.preproc must handle the label format (bboxes + keypoints)
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            # Image info is less meaningful for mosaic/mixup, use final shape
            img_info = (mix_img.shape[1], mix_img.shape[0]) # Should be (height, width)? Check convention

            # Use the original index's img_id, though it's less relevant for mosaic
            # Need img_id from the original pull_item(idx) if required downstream
            # Let's get it from the first pull_item, assuming it succeeded.
            try:
                 # This assumes the first image (idx) was loaded successfully
                 _, _, _, first_img_id = self._dataset.pull_item(idx)
            except:
                 # Fallback if the primary image failed loading
                 first_img_id = np.array([-1]) # Placeholder ID


            return mix_img, padded_labels, img_info, first_img_id # Return img_id of the original index

        else: # Mosaic disabled or probability check failed
            # Load a single image and apply preproc
            self._dataset.input_dim = self.input_dim # Ensure dataset knows the target size
            img, label, img_info, img_id = self._dataset.pull_item(idx)

            if img is None or label is None:
                 print(f"Error: Failed to load single image for index {idx}")
                 # Handle error: return None or dummy data
                 return None, None, None, None

            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id


    def mixup(self, origin_img, origin_labels, input_dim):
        """Performs MixUp augmentation by overlaying a second image."""
        # Select a random second image and its labels
        jit_factor = random.uniform(*self.mixup_scale) # Scale factor for the second image
        FLIP = random.uniform(0, 1) > 0.5 # Random horizontal flip

        cp_labels = []
        # Ensure the selected second image has annotations
        attempts = 0
        max_attempts = 10 # Avoid infinite loop if no annotated images found
        while len(cp_labels) == 0 and attempts < max_attempts:
            cp_index = random.randint(0, self.__len__() - 1)
            # Use load_anno which should return the processed labels array directly
            # Check if load_anno returns the correct format (N, 5+K*3)
            anno_data = self._dataset.load_anno(cp_index)
            # load_anno might return just the label array, or maybe more. Adjust accordingly.
            # Assuming it returns the label array:
            cp_labels = anno_data if anno_data is not None else np.array([])
            attempts += 1

        if len(cp_labels) == 0:
             # If no annotated image found after attempts, return original image/labels
             print("Warning: Mixup could not find an image with annotations. Skipping mixup.")
             return origin_img, origin_labels

        # Load the second image
        # Use pull_item to get the image associated with cp_labels
        img2, _, _, _ = self._dataset.pull_item(cp_index)
        if img2 is None:
            print(f"Warning: Mixup failed to load image for index {cp_index}. Skipping mixup.")
            return origin_img, origin_labels

        # --- Prepare the second image (cp_img) ---
        h0, w0 = img2.shape[:2]
        # Scale the second image to fit within input_dim
        cp_scale_ratio = min(input_dim[0] / h0, input_dim[1] / w0)
        resized_img2 = cv2.resize(
            img2,
            (int(w0 * cp_scale_ratio), int(h0 * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        h_resized, w_resized = resized_img2.shape[:2]

        # Create a canvas of input_dim size, place resized_img2 on it
        # Determine background color based on image type (color/gray)
        if len(origin_img.shape) == 3:
            cp_img_canvas = np.full((input_dim[0], input_dim[1], 3), 114, dtype=np.uint8)
            cp_img_canvas[:h_resized, :w_resized, :] = resized_img2
        else: # Grayscale
            cp_img_canvas = np.full(input_dim, 114, dtype=np.uint8)
            cp_img_canvas[:h_resized, :w_resized] = resized_img2

        # Apply jittering (scaling) and flipping to the canvas
        cp_img_canvas = cv2.resize(
            cp_img_canvas,
            (int(cp_img_canvas.shape[1] * jit_factor), int(cp_img_canvas.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor # Update scale ratio to include jitter

        if FLIP:
            cp_img_canvas = cp_img_canvas[:, ::-1, :] if len(origin_img.shape) == 3 else cp_img_canvas[:, ::-1]

        origin_h, origin_w = cp_img_canvas.shape[:2] # Size after jitter/flip
        target_h, target_w = origin_img.shape[:2] # Size of the first image (mosaic output)

        # Create a padded canvas to place cp_img_canvas randomly
        # Use np.maximum for shape calculation for robustness
        pad_h = max(origin_h, target_h)
        pad_w = max(origin_w, target_w)
        padded_img = np.full((pad_h, pad_w, 3) if len(origin_img.shape) == 3 else (pad_h, pad_w), 114, dtype=np.uint8)

        padded_img[:origin_h, :origin_w] = cp_img_canvas

        # Randomly crop from the padded image to match target size
        y_offset = random.randint(0, pad_h - target_h) if pad_h > target_h else 0
        x_offset = random.randint(0, pad_w - target_w) if pad_w > target_w else 0
        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        # --- Transform labels of the second image (cp_labels) ---
        labels_cp = cp_labels.copy()
        bboxes_cp = labels_cp[:, :4]
        keypoints_cp = labels_cp[:, 5:] if labels_cp.shape[1] > 5 else np.array([])

        # Adjust bounding boxes: scale, (no pad here, offset happens later), flip, offset, clip
        # Step 1: Scale bboxes according to cp_scale_ratio
        bboxes_cp *= cp_scale_ratio
        # Step 2: Flip if needed (relative to origin_w, size *after* jitter/flip)
        if FLIP:
            bboxes_cp[:, 0::2] = origin_w - bboxes_cp[:, 0::2][:, ::-1] # Flip x1, x2
        # Step 3: Apply offset from random crop
        bboxes_cp[:, 0::2] -= x_offset # Adjust x coords
        bboxes_cp[:, 1::2] -= y_offset # Adjust y coords
        # Step 4: Clip to target dimensions (target_w, target_h)
        bboxes_cp[:, 0::2] = bboxes_cp[:, 0::2].clip(0, target_w)
        bboxes_cp[:, 1::2] = bboxes_cp[:, 1::2].clip(0, target_h)

        # Adjust keypoints: scale, flip, offset, clip
        if keypoints_cp.size > 0:
             num_kpts = keypoints_cp.shape[1] // 3
             kpts_reshaped = keypoints_cp.reshape(-1, 3) # Shape (N*K, 3)

             # Mask for valid keypoints before transformation
             valid_mask = kpts_reshaped[:, 2] > 0

             # Step 1: Scale x, y
             kpts_reshaped[valid_mask, :2] *= cp_scale_ratio
             # Step 2: Flip if needed
             if FLIP:
                 kpts_reshaped[valid_mask, 0] = origin_w - kpts_reshaped[valid_mask, 0]
             # Step 3: Apply offset
             kpts_reshaped[valid_mask, 0] -= x_offset
             kpts_reshaped[valid_mask, 1] -= y_offset
             # Step 4: Clip coordinates
             kpts_reshaped[:, 0] = kpts_reshaped[:, 0].clip(0, target_w)
             kpts_reshaped[:, 1] = kpts_reshaped[:, 1].clip(0, target_h)

             # Step 5: Update visibility: if a keypoint was visible (v>0) but is now clipped
             # outside the final target boundary, set its visibility to 0.
             # Check if clipped coordinates are *strictly* inside the bounds.
             inside_bounds_x = (kpts_reshaped[:, 0] > 0) & (kpts_reshaped[:, 0] < target_w)
             inside_bounds_y = (kpts_reshaped[:, 1] > 0) & (kpts_reshaped[:, 1] < target_h)
             is_inside = inside_bounds_x & inside_bounds_y

             # Update visibility: keep original visibility if inside, otherwise set to 0 if originally > 0
             new_visibility = np.where(is_inside, kpts_reshaped[:, 2], 0)
             kpts_reshaped[:, 2] = np.where(kpts_reshaped[:, 2] > 0, new_visibility, 0) # Only change if originally visible

             keypoints_cp = kpts_reshaped.reshape(-1, num_kpts * 3)


        # Filter labels_cp based on final bbox size after transformation
        bboxes_cp_wh = bboxes_cp[:, 2:4] - bboxes_cp[:, :2]
        valid_indices_cp = (bboxes_cp_wh[:, 0] > 1) & (bboxes_cp_wh[:, 1] > 1)

        # Reassemble valid transformed labels for the second image
        if np.any(valid_indices_cp): # Check if any valid labels remain
             valid_bboxes_cp = bboxes_cp[valid_indices_cp]
             valid_cls_cp = labels_cp[valid_indices_cp, 4:5] # Class ID column
             if keypoints_cp.size > 0:
                 valid_keypoints_cp = keypoints_cp[valid_indices_cp]
                 # Ensure keypoints array is 2D even if only one instance remains
                 if valid_keypoints_cp.ndim == 1:
                      valid_keypoints_cp = valid_keypoints_cp[np.newaxis, :]
                 labels_cp_transformed = np.hstack((valid_bboxes_cp, valid_cls_cp, valid_keypoints_cp))
             else:
                 labels_cp_transformed = np.hstack((valid_bboxes_cp, valid_cls_cp))

             # Combine origin_labels and the transformed cp labels
             # Ensure both have the same number of columns before vstack
             if origin_labels.shape[1] == labels_cp_transformed.shape[1]:
                 origin_labels = np.vstack((origin_labels, labels_cp_transformed))
             else:
                 # This case indicates a mismatch in keypoint presence between origin and mixup labels
                 # Handle this carefully. Option: only mixup if formats match, or pad.
                 # For now, let's just warn and skip appending if shapes mismatch.
                 print(f"Warning: Mixup label shape mismatch. Origin: {origin_labels.shape}, Mixup: {labels_cp_transformed.shape}. Skipping append.")

        # Mix the images (simple average)
        origin_img = origin_img.astype(np.float32)
        mixed_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return mixed_img.astype(np.uint8), origin_labels # Return uint8 image
