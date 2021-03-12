import numpy as np

from detectron2.engine import DefaultTrainer
import torch
from detectron2.projects.deeplab.build_solver import build_lr_scheduler as build_deeplab_lr_scheduler
import logging

import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader
import copy
from detectron2.data import detection_utils as utils
from typing import List, Union


def my_dataset_function(ikDataset):
    #returns a function that returns a list[dict] to be used as detectron2 dataset
    if "category_colors" in ikDataset["metadata"]:
        category_colors = {}
        for i,c in enumerate(ikDataset["metadata"]["category_colors"]):
            category_colors[i]=c
    else:
        category_colors = None
    def f():
        listDict = []
        for ikrecord in ikDataset["images"]:
            record={}
            record["id"] = ikrecord["id"]
            record["file_name"] =ikrecord["filename"]
            record["sem_seg_file_name"] = ikrecord["semantic_seg_masks_file"]
            record["height"]=ikrecord["height"]
            record["width"]=ikrecord["width"]
            if category_colors is not None:
                record["category_colors"]=category_colors
            listDict.append(record)
        return listDict
    return f


def rgb2mask(img, color2index):
    W = np.power(256, [[0], [1], [2]])
    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id == c] = color2index[tuple(img[img_id == c][0])]
        except:
            pass
    return mask


class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)

    def build_lr_scheduler(cls, cfg, optimizer):
        return build_deeplab_lr_scheduler(cfg, optimizer)

    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MyMapper(True, augmentations=[T.Resize(cfg.inputSize)], image_format="RGB"))


class MyMapper(DatasetMapper):
    def __init__(
            self,
            is_train: bool,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str):

        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            if "category_colors" in dataset_dict:
                sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "RGB")
                sem_seg_gt = rgb2mask(sem_seg_gt, dataset_dict["category_colors"])


            else:
                sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L")
                sem_seg_gt = sem_seg_gt.squeeze(2)

        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        return dataset_dict

