"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
"""

import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import MetadataCatalog

__all__ = ["COCOPanopticNewBaselineDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOSemanticNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        ignore_label
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.ignore_label = ignore_label

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "ignore_label": ignore_label,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image_ori = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image_ori)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt_ori = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt_ori = None

        if sem_seg_gt_ori is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        
        masks = []
        while len(masks) == 0:
            aug_input = T.AugInput(image_ori, sem_seg=sem_seg_gt_ori)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg

            # Pad image and segmentation label here!
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = image

            if sem_seg_gt is not None:
                dataset_dict["sem_seg"] = sem_seg_gt.long()

            if "annotations" in dataset_dict:
                raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

            # Prepare per-category binary masks
            if sem_seg_gt is not None:
                sem_seg_gt = sem_seg_gt.numpy()
                instances = Instances(image_shape)
                classes = np.unique(sem_seg_gt)
                # remove ignored region
                classes = classes[classes != self.ignore_label]
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

                masks = []
                for class_id in classes:
                    masks.append(sem_seg_gt == class_id)

                if len(masks) > 0:
                    masks_ = BitMasks(
                        torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                    )
                    instances.gt_masks = masks_.tensor

                dataset_dict["instances"] = instances


        return dataset_dict
