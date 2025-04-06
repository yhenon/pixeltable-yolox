# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class CocoDataset(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)

# --- Keep the original CocoDataset class ---
class CocoDataset(CacheDataset):
    """
    COCO dataset class. (Bounding Box version)
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name # Store name early for use in super().__init__

        annotation_path = os.path.join(self.data_dir, "annotations", self.json_file)
        print(f"Loading COCO annotations from: {annotation_path}")
        self.coco = COCO(annotation_path)
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.class_ids) # Load cats using sorted class_ids
        self._classes = tuple([c["name"] for c in self.cats])
        # Create mapping from category_id to contiguous category index (0, 1, 2...)
        self.cat_id_to_cls_idx = {cat_id: i for i, cat_id in enumerate(self.class_ids)}

        self.img_size = img_size
        self.preproc = preproc
        # Load annotations *after* essential members like coco, ids, cat_id_to_cls_idx are set
        self.annotations = self._load_coco_annotations()

        # Ensure annotations are loaded before creating path_filename
        if not self.annotations:
             print("Warning: No annotations loaded.")
             path_filename = []
        else:
             # Assuming the 4th element is filename, adjust if load_anno_from_ids changes
             path_filename = [os.path.join(self.name, anno[3]) for anno in self.annotations if anno is not None and len(anno)>3]


        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename, # Use the generated list
            cache=cache,
            cache_type=cache_type
        )
        print(f"Initialized CocoDataset with {self.num_imgs} images.")

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        print("Loading annotations for all images...")
        annotations = [self.load_anno_from_ids(_ids) for _ids in self.ids]
        print(f"Finished loading annotations. Found annotations for {len([a for a in annotations if a is not None and len(a[0]) > 0])} images.")
        return annotations


    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        # Filter out crowd annotations
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            # Basic check for bounding box existence and area
            if "bbox" not in obj or obj["area"] <= 0:
                continue

            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))

            # Ensure valid coordinates after clipping
            if x2 >= x1 and y2 >= y1:
                # Map the original COCO category id to our contiguous 0-based index
                cls_idx = self.cat_id_to_cls_idx.get(obj["category_id"])
                if cls_idx is None:
                    # print(f"Warning: Category ID {obj['category_id']} not found in self.cat_id_to_cls_idx mapping. Skipping object.")
                    continue # Skip if category is not in our list

                clean_bbox = [x1, y1, x2, y2]
                objs.append({
                    "clean_bbox": clean_bbox,
                    "cls_idx": cls_idx,
                    # Add other relevant fields if needed later
                })


        num_objs = len(objs)
        if num_objs == 0:
             # Return empty annotation if no valid objects found
             res = np.zeros((0, 5), dtype=np.float32)
        else:
            res = np.zeros((num_objs, 5), dtype=np.float32)
            for ix, obj_data in enumerate(objs):
                res[ix, 0:4] = obj_data["clean_bbox"]
                res[ix, 4] = obj_data["cls_idx"]

        # --- Resize annotations ---
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        # Only apply scaling if there are objects
        if num_objs > 0:
            res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        # Return format: (annotations, original_img_info, resized_img_info, filename)
        return (res, img_info, resized_info, file_name)


    def load_anno(self, index):
        # Returns the processed annotations (e.g., res array)
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        if img is None: return None # Handle case where image loading failed
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        # Check if annotations exist for this index
        if index >= len(self.annotations) or self.annotations[index] is None:
             print(f"Warning: No annotation found for index {index}. Cannot load image.")
             return None
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, self.name, file_name)

        try:
            img = cv2.imread(img_file)
            if img is None:
                 print(f"Warning: Failed to read image file: {img_file}")
                 # Optionally try to find the file in a different subdirectory if structure varies
                 # e.g., img_file = os.path.join(self.data_dir, 'images', self.name, file_name)
                 # img = cv2.imread(img_file)
            assert img is not None, f"file named {img_file} not found or failed to read"
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            return None

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        # Wraps load_resized_img with caching logic from parent
        return self.load_resized_img(index)

    def pull_item(self, index):
        # Check index validity
        if index >= len(self.ids):
             raise IndexError(f"Index {index} out of bounds for {len(self.ids)} images.")

        id_ = self.ids[index]

        # Check if annotation loaded correctly
        if index >= len(self.annotations) or self.annotations[index] is None:
             print(f"Warning: Annotation for index {index} (img_id {id_}) is missing or failed to load. Skipping item.")
             # Need to return something that __getitem__ can handle or raise error
             # Returning None might be problematic if downstream code expects tuples
             # Consider returning dummy data or raising a specific exception
             # For now, let's return None and handle in __getitem__ if needed
             return None, None, None, None # img, label, img_info, img_id

        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index) # This uses the cached reading

        if img is None:
            print(f"Warning: Failed to read image for index {index} (img_id {id_}). Skipping item.")
            return None, None, None, None

        # Return deepcopy of label to prevent modification issues if using caching/multiprocessing
        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.
        """
        pulled_data = self.pull_item(index)
        if pulled_data[0] is None:
             # Handle cases where pull_item failed (e.g., missing annotation/image)
             # Option 1: Return None (caller must handle)
             # return None
             # Option 2: Skip and get next item (requires modifying dataloader logic or raising specific exception)
             # raise SkipItemException(f"Failed to load item {index}")
             # Option 3: Return dummy data (might cause issues in training/eval)
             print(f"Warning: Skipping index {index} due to loading errors.")
             # For simplicity in this example, let's try getting the next item recursively.
             # Be cautious with recursion depth if many items fail.
             # A better approach might be to filter invalid indices during initialization.
             return self.__getitem__((index + 1) % len(self))


        img, target, img_info, img_id = pulled_data

        if self.preproc is not None:
            # Ensure target is not empty before preprocessing if preproc expects non-empty
            if target is not None and len(target) > 0:
                 img, target = self.preproc(img, target, self.img_size) # Use self.img_size here
            elif target is not None: # Target is empty array
                 img, target = self.preproc(img, target, self.img_size) # Preproc should handle empty target
            else: # Target is None (shouldn't happen with current pull_item logic)
                 print(f"Warning: Target is None for index {index}. Cannot apply preproc.")
                 # Handle appropriately, maybe return img, empty target?

        # Ensure target is a tensor if required by downstream code (like collate_fn)
        # import torch
        # if not isinstance(target, torch.Tensor):
        #    target = torch.from_numpy(target) if target is not None else torch.empty((0, 5))


        return img, target, img_info, img_id


# --- NEW CocoKeypointDataset class ---

class CocoKeypointDataset(CocoDataset):
    """
    COCO Keypoint dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="person_keypoints_train2017.json", # Default to keypoint annotations
        name="train2017",
        img_size=(640, 640), # Often larger for pose estimation
        preproc=None,
        cache=False,
        cache_type="ram",
        # Keypoint specific attributes (can be loaded from COCO metadata)
        num_keypoints=17, # Standard COCO
        keypoint_category_ids=None # Specify category IDs to load (e.g., [1] for 'person')
    ):
        """
        COCO Keypoint dataset initialization.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name (e.g., 'person_keypoints_train2017.json')
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (tuple): target image size after pre-processing
            preproc: data augmentation strategy (should handle keypoints)
            cache (bool): whether to cache images
            cache_type (str): 'ram' or 'disk'
            num_keypoints (int): Number of keypoints per instance.
            keypoint_category_ids (list[int], optional): List of category IDs to consider
                                                        (e.g., [1] for 'person'). If None, uses all
                                                        categories found in the JSON.
        """
        self.num_keypoints = num_keypoints
        self.keypoint_category_ids = keypoint_category_ids
        # Store keypoint info like names and skeleton if needed by preproc/visualization
        self.keypoint_names = None
        self.skeleton = None

        # Call parent __init__ AFTER setting keypoint specific attributes if they are needed
        # *before* _load_coco_annotations is called by the parent init.
        super().__init__(
            data_dir=data_dir,
            json_file=json_file,
            name=name,
            img_size=img_size,
            preproc=preproc,
            cache=cache,
            cache_type=cache_type,
        )

        # Load keypoint metadata (names, skeleton) from the COCO object if available
        # This assumes the keypoint JSON follows standard COCO format
        self._load_keypoint_metadata()

        # Filter self.annotations based on keypoint_category_ids if provided
        # Note: This filtering happens *after* loading all annotations.
        # It might be more efficient to filter inside load_anno_from_ids if memory is tight.
        if self.keypoint_category_ids is not None:
            print(f"Filtering annotations for category IDs: {self.keypoint_category_ids}")
            original_count = len(self.annotations)
            filtered_annotations = []
            valid_indices = [] # Keep track of indices corresponding to filtered annotations
            target_cat_indices = {self.cat_id_to_cls_idx[cat_id] for cat_id in self.keypoint_category_ids if cat_id in self.cat_id_to_cls_idx}

            for i, anno_tuple in enumerate(self.annotations):
                if anno_tuple is None: continue
                res, _, _, _ = anno_tuple
                # Check if any object in the image belongs to the target categories
                # Assumes class index is at the 4th position (0-4 bbox, 4 class)
                if res.shape[0] > 0 and np.any(np.isin(res[:, 4], list(target_cat_indices))):
                     filtered_annotations.append(anno_tuple)
                     valid_indices.append(i) # Store original index

            self.annotations = filtered_annotations
            # Important: Update self.ids and self.num_imgs to match filtered annotations
            original_ids = self.ids
            self.ids = [original_ids[i] for i in valid_indices]
            self.num_imgs = len(self.ids)
            # Update path_filename in the parent CacheDataset as well
            self.path_filename = [self.path_filename[i] for i in valid_indices]

            print(f"Filtered annotations: {original_count} -> {self.num_imgs}")


    def _load_keypoint_metadata(self):
        """Loads keypoint names and skeleton from COCO category info."""
        # Assuming keypoints are primarily for the 'person' category (id=1)
        # Or use the first category found that has keypoint info
        cat_ids_to_check = self.keypoint_category_ids if self.keypoint_category_ids else self.class_ids

        for cat_id in cat_ids_to_check:
            try:
                cat_info = self.coco.loadCats([cat_id])[0]
                
                if 'keypoints' in cat_info and 'skeleton' in cat_info:
                    self.keypoint_names = cat_info['keypoints']
                    self.skeleton = cat_info['skeleton']
                    # Verify number of keypoints matches
                    if len(self.keypoint_names) != self.num_keypoints:
                         print(f"Warning: Mismatch between provided num_keypoints ({self.num_keypoints}) "
                               f"and number of keypoint names in COCO metadata ({len(self.keypoint_names)}) "
                               f"for category {cat_id}. Using metadata length.")
                         self.num_keypoints = len(self.keypoint_names)
                    print(f"Loaded keypoint metadata for category '{cat_info['name']}' (ID: {cat_id})")
                    # Found metadata, stop searching (assuming one primary keypoint category)
                    break
            except Exception as e:
                print(f"Could not load category info for cat_id {cat_id}: {e}")

        if self.keypoint_names is None:
            print("Warning: Could not load keypoint names and skeleton from COCO metadata.")


    def load_anno_from_ids(self, id_):
        """
        Loads keypoint annotations for a given image ID.
        Overrides the method in CocoDataset.
        """
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            # --- Basic Filtering ---
            # Check if it has keypoints and bbox
            if "keypoints" not in obj or "bbox" not in obj:
                continue
            # Check if category is desired (if filtering is enabled)
            # Note: Filtering by self.keypoint_category_ids happens *after* loading now,
            # but we still need the cls_idx mapping.
            cls_idx = self.cat_id_to_cls_idx.get(obj["category_id"])
            if cls_idx is None:
                continue # Skip if category is not in the overall list

            # Check for minimum number of visible keypoints if needed (e.g., > 0)
            num_kpts = obj.get("num_keypoints", 0)
            if num_kpts <= 0:
                 continue

            # --- Bounding Box Processing (same as base class) ---
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))

            if obj["area"] <= 0 or x2 < x1 or y2 < y1:
                continue

            clean_bbox = [x1, y1, x2, y2]

            # --- Keypoint Processing ---
            keypoints = np.array(obj["keypoints"], dtype=np.float32) # Shape: (num_kpts * 3,)
            # Reshape to [num_kpts, 3] -> (x, y, visibility)
            try:
                keypoints = keypoints.reshape(-1, 3)
            except ValueError:
                print(f"Warning: Skipping object ID {obj['id']} in image {id_} due to malformed keypoints array (length {len(keypoints)}).")
                continue

            # Ensure the number of keypoints matches expected, pad if necessary (or skip)
            if keypoints.shape[0] != self.num_keypoints:
                 # This case should be rare with standard COCO but handle defensively
                 print(f"Warning: Object ID {obj['id']} in image {id_} has {keypoints.shape[0]} keypoints, expected {self.num_keypoints}. Skipping or padding needed.")
                 # Option 1: Skip
                 # continue
                 # Option 2: Pad (e.g., with zeros) - Be careful with this
                 # padded_kpts = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                 # count = min(keypoints.shape[0], self.num_keypoints)
                 # padded_kpts[:count, :] = keypoints[:count, :]
                 # keypoints = padded_kpts
                 # For now, let's skip to avoid potential issues downstream
                 continue


            objs.append({
                "clean_bbox": clean_bbox,
                "cls_idx": cls_idx,
                "keypoints": keypoints # Store as [N, 3] array
            })

        num_objs = len(objs)

        # --- Format Output ---
        # Create the result array: [bbox (4) + class_id (1) + keypoints (num_kpts * 3)]
        annotation_dim = 5 + self.num_keypoints * 3
        if num_objs == 0:
            res = np.zeros((0, annotation_dim), dtype=np.float32)
        else:
            res = np.zeros((num_objs, annotation_dim), dtype=np.float32)
            for ix, obj_data in enumerate(objs):
                res[ix, 0:4] = obj_data["clean_bbox"]
                res[ix, 4] = obj_data["cls_idx"]
                res[ix, 5:] = obj_data["keypoints"].reshape(-1) # Flatten keypoints

        # --- Resize annotations ---
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        if num_objs > 0:
            # Scale bounding boxes
            res[:, :4] *= r
            # Scale keypoint coordinates (x, y), leave visibility (v) unchanged
            # Keypoints start at index 5, step by 3 (x, y, v)
            res[:, 5::3] *= r  # Scale x coordinates
            res[:, 6::3] *= r  # Scale y coordinates
            # Visibility flags (res[:, 7::3]) remain unchanged

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        # Return format: (annotations, original_img_info, resized_img_info, filename)
        # Annotations shape: (num_objs, 5 + num_keypoints * 3)

        return (res, img_info, resized_info, file_name)

    # __len__, load_anno, load_resized_img, load_image, read_img, pull_item, __getitem__
    # are inherited from CocoDataset and should work correctly provided the
    # overridden load_anno_from_ids returns data in the expected tuple format,
    # and the preproc function is adapted to handle the new annotation format.

    # Optional: Override pull_item or __getitem__ if specific keypoint logic
    # is needed beyond what preproc handles. For example, if you need to pass
    # keypoint metadata (skeleton) to preproc.

    # def pull_item(self, index):
    #     # Example: If you needed to pass skeleton info
    #     img, target, img_info, img_id = super().pull_item(index)
    #     # Check if pull_item from parent succeeded
    #     if img is None:
    #         return None, None, None, None, None # Add extra None for skeleton
    #     return img, target, img_info, img_id, self.skeleton

    # def __getitem__(self, index):
        # # Example: Adapting for the modified pull_item above
        # pulled_data = self.pull_item(index)
        # if pulled_data[0] is None:
        #     # Handle skipping as before
        #     print(f"Warning: Skipping index {index} due to loading errors in keypoint dataset.")
        #     return self.__getitem__((index + 1) % len(self))

        # img, target, img_info, img_id, skeleton = pulled_data # Unpack skeleton

        # if self.preproc is not None:
        #     # Pass skeleton to preproc if it needs it
        #     img, target = self.preproc(img, target, self.img_size, skeleton=skeleton)
        # return img, target, img_info, img_id
