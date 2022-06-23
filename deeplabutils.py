import numpy as np

from detectron2.engine import DefaultTrainer
import torch
from detectron2.projects.deeplab.build_solver import build_lr_scheduler as build_deeplab_lr_scheduler
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader
import copy
from detectron2.data import detection_utils as utils
from typing import List, Union
from detectron2.utils.events import EventStorage
import logging
from detectron2.evaluation import SemSegEvaluator
import PIL.Image as Image
from detectron2.utils.file_io import PathManager
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
import time
import datetime
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def my_dataset_function(ikDataset):
    # returns a function that returns a list[dict] to be used as detectron2 dataset
    if "category_colors" in ikDataset["metadata"]:
        category_colors = {}
        for i, c in enumerate(ikDataset["metadata"]["category_colors"]):
            category_colors[i] = c
    else:
        category_colors = None

    def f():
        possible_masks = ["semantic_seg_masks_file", "instance_seg_masks_file"]
        listDict = []
        for ikrecord in ikDataset["images"]:
            record = {}
            record["image_id"] = ikrecord["image_id"]
            record["file_name"] = ikrecord["filename"]
            for possible in possible_masks:
                if possible in ikrecord:
                    record["sem_seg_file_name"] = ikrecord[possible]
            record["height"] = ikrecord["height"]
            record["width"] = ikrecord["width"]
            if category_colors is not None:
                record["category_colors"] = {v: k for k, v in category_colors.items()}
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


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, trainer, train_process, patience):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.best_val_loss = np.inf
        self.patience = patience
        self.waiting = 0
        self.trainer = trainer
        self.train_process = train_process

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        tol = 1e10 - 4
        if mean_loss < self.best_val_loss + tol:
            self.best_val_loss = mean_loss
            self.waiting = 0
            print("Saving best model...")
            self.trainer.checkpointer.save("best_model")
            print("Model saved")
        self.waiting += 1

        self.trainer.storage.put_scalar('validation_loss', mean_loss)

        metrics_dict = {k: v[0] for k, v in self.trainer.storage.latest().items()}
        self.train_process.log_metrics(metrics_dict, self.trainer.iter)

        if self.waiting > self.patience and self.patience >= 0:
            self.trainer.run = False
        comm.synchronize()
        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


class MyTrainer(DefaultTrainer):
    def __init__(self, cfg, train_process):
        """
        Args:
            cfg (CfgNode):
        """
        self.run = True
        self.train_process = train_process
        super().__init__(cfg)

    def build_lr_scheduler(cls, cfg, optimizer):
        return build_deeplab_lr_scheduler(cfg, optimizer)

    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MyMapper(True, augmentations=[T.Resize(cfg.INPUT_SIZE)],
                                                                 image_format="RGB"))

    def build_evaluator(cfg, dataset_name):
        return MySemSegEvaluator(dataset_name, distributed=False, output_dir="eval",
                                 num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, ignore_label=255)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                MyMapper(False, augmentations=[T.Resize(self.cfg.INPUT_SIZE)], image_format="RGB")
            ),
            self,
            self.train_process,
            self.cfg.PATIENCE
        ))
        return hooks

    def build_writers(self):
        return []

    def train(self):

        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    if not (self.run):
                        break
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    self.train_process.emitStepProgress()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()


class MySemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(self, dataset_name, distributed, output_dir, num_classes, ignore_label):
        super().__init__(dataset_name, distributed=distributed, output_dir=output_dir, num_classes=num_classes,
                         ignore_label=ignore_label)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=int)
                if "category_colors" in input:
                    gt = rgb2mask(gt, input["category_colors"]).astype(dtype=np.uint8)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes+1) * pred.reshape(-1) + gt.reshape(-1), minlength=(self._num_classes+1) ** 2
            ).reshape((self._num_classes+1), (self._num_classes+1))

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


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
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt, dtype=torch.long)

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        return dataset_dict


def register_train_test(dataset_dict, metadata, train_ratio=0.66, seed=0):
    try:
        DatasetCatalog.remove("datasetTrain")
    except:
        pass
    try:
        DatasetCatalog.remove("datasetTest")
    except:
        pass
    try:
        MetadataCatalog.remove("datasetTrain")
    except:
        pass
    try:
        MetadataCatalog.remove("datasetTest")
    except:
        pass
    nb_input = len(dataset_dict)
    x = np.arange(nb_input)
    random.Random(seed).shuffle(x)
    idx_split = int(len(x) * train_ratio)
    DatasetCatalog.register("datasetTrain", my_dataset_function(
        {"images": np.array(dataset_dict)[x[:idx_split]], "metadata": metadata}))
    DatasetCatalog.register("datasetTest", my_dataset_function(
        {"images": np.array(dataset_dict)[x[idx_split:]], "metadata": metadata}))
    MetadataCatalog.get("datasetTrain").stuff_classes = [v for k, v in metadata["category_names"].items()]
    MetadataCatalog.get("datasetTest").stuff_classes = [v for k, v in metadata["category_names"].items()]

    """if ignoreValue is not None:
        MetadataCatalog.get("datasetTrain").ignore_label = ignoreValue"""
