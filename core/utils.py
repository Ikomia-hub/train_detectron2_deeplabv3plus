import cv2
import os
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer,DefaultPredictor
import json
import torch


def my_dataset_function(ikDataset):
    #returns a function that returns a list[dict] to be used as detectron2 dataset
    if "category_colors" in ikDataset["metadata"]:
        category_colors = {}
        for i,c in enumerate(ikDataset["category_colors"]):
            category_colors[i]=c
    else:
        category_colors = None
    def f():
      img_dir = "/content/gdrive/MyDrive/apps/cwfid/"

      listDict = []
      for ikrecord in ikDataset["images"]:
          record={}
          record["id"] = ikrecord["id"]
          record["file_name"] =ikrecord["filename"]
          record["sem_seg_file_name"] = ikrecord["sem_seg"]
          record["height"]=ikrecord["height"]
          record["width"]=ikrecord["width"]
          if category_colors is not None:
            record["category_colors"]=category_colors

          listDict.append(record)
      return listDict
    return f
