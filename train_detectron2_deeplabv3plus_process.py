import copy
import os
from train_detectron2_deeplabv3plus import update_path, deeplabutils
from ikomia import core, dataprocess
from ikomia.dnn import datasetio, dnntrain
from ikomia.core.task import TaskParam
from detectron2.config import get_cfg, CfgNode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import launch
from datetime import datetime
import torch
import gc

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class TrainDeeplabv3plusParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["model_name"] = "DeepLabV3Plus"
        self.cfg["input_width"] = 800
        self.cfg["input_height"] = 800
        self.cfg["epochs"] = 1000
        self.cfg["classes"] = 2
        self.cfg["max_iter"] = 1000
        self.cfg["warmupFactor"] = 0.001
        self.cfg["warmupIters"] = 200
        self.cfg["polyLRFactor"] = 0.9
        self.cfg["polyLRConstantFactor"] = 0.0
        self.cfg["batch_size"] = 4
        self.cfg["resnetDepth"] = 50
        self.cfg["batchNorm"] = "BN"
        self.cfg["ignoreValue"] = None
        self.cfg["learning_rate"] = 0.02
        self.cfg["config"] = ""
        self.cfg["eval_period"] = 100
        self.cfg["earlyStopping"] = False
        self.cfg["patience"] = 10
        self.cfg["dataset_split_ratio"] = 90
        self.cfg["numGPU"] = 1
        self.cfg["output_folder"] = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["input_width"] = int(param_map["input_width"])
        self.cfg["input_height"] = int(param_map["input_height"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["classes"] = int(param_map["classes"])
        self.cfg["max_iter"] = int(param_map["max_iter"])
        self.cfg["warmupFactor"] = float(param_map["warmupFactor"])
        self.cfg["warmupIters"] = int(param_map["warmupIters"])
        self.cfg["polyLRFactor"] = float(param_map["polyLRFactor"])
        self.cfg["polyLRConstantFactor"] = float(param_map["polyLRConstantFactor"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["resnetDepth"] = int(param_map["resnetDepth"])
        self.cfg["batchNorm"] = param_map["batchNorm"]
        self.cfg["learning_rate"] = float(param_map["learning_rate"])
        self.cfg["config"] = param_map["config"]
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["earlyStopping"] = bool(param_map["earlyStopping"])
        self.cfg["patience"] = int(param_map["patience"])
        self.cfg["dataset_split_ratio"] = int(param_map["dataset_split_ratio"])
        self.cfg["numGPU"] = int(param_map["numGPU"])
        self.cfg["output_folder"] = param_map["output_folder"]


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class TrainDeeplabv3plus(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Create parameters class
        if param is None:
            self.set_param_object(TrainDeeplabv3plusParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.trainer = None
        self.enable_tensorboard(False)

    def run(self):
        # Core function of your process
        input = self.get_input(0)
        # Get parameters :
        param = self.get_param_object()

        if len(input.data["images"]) > 0:
            param.cfg["epochs"] = int(param.cfg["max_iter"] * param.cfg["batch_size"] / len(input.data["images"]))

            # complete class names if input dataset has no background class
            if not (input.has_bckgnd_class):
                tmp_dict = {0: "background"}
                for k, name in input.data["metadata"]["category_names"].items():
                    tmp_dict[k + 1] = name
                input.data["metadata"]["category_names"] = tmp_dict
                input.has_bckgnd_class = True

            param.cfg["classes"] = len(input.data["metadata"]["category_names"])

            # Call begin_task_run for initialization
            self.begin_task_run()

            if param.cfg["config"] == "":
                # Get default config
                cfg = get_cfg()

                # Add specific deeplab config
                add_deeplab_config(cfg)
                cfg.merge_from_file(os.path.dirname(os.path.realpath(__file__)) + "/model/configs/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")

                # Generic dataset names that will be used
                cfg.DATASETS.TRAIN = ("datasetTrain",)
                cfg.DATASETS.TEST = ("datasetTest",)
                cfg.SOLVER.MAX_ITER = param.cfg["max_iter"]
                cfg.SOLVER.WARMUP_FACTOR = 0.001
                cfg.SOLVER.WARMUP_ITERS = param.cfg["max_iter"] // 5
                cfg.SOLVER.POLY_LR_FACTOR = 0.9
                cfg.SOLVER.POLY_LR_CONSTANT_FACTOR = 0.0
                cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = param.cfg["classes"]
                cfg.SOLVER.BASE_LR = param.cfg["learning_rate"]
                cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
                cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
                cfg.SOLVER.IMS_PER_BATCH = param.cfg["batch_size"]
                cfg.DATALOADER.NUM_WORKERS = 0
                cfg.INPUT_SIZE = (param.cfg["input_width"], param.cfg["input_height"])
                cfg.TEST.EVAL_PERIOD = param.cfg["eval_period"]
                cfg.SPLIT_TRAIN_TEST = param.cfg["dataset_split_ratio"]
                cfg.SPLIT_TRAIN_TEST_SEED = -1
                cfg.MODEL.BACKBONE.FREEZE_AT = 5
                cfg.CLASS_NAMES = [name for k, name in input.data["metadata"]["category_names"].items()]

                if param.cfg["earlyStopping"]:
                    cfg.PATIENCE = param.cfg["patience"]
                else:
                    cfg.PATIENCE = -1

                if param.cfg["output_folder"] == "":
                    cfg.OUTPUT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/output"
                elif os.path.isdir(param.cfg["output_folder"]):
                    cfg.OUTPUT_DIR = param.cfg["output_folder"]
                else:
                    print("Incorrect output folder path")
            else:
                cfg = None
                with open(param.cfg["config"], 'r') as file:
                    cfg_data = file.read()
                    cfg = CfgNode.load_cfg(cfg_data)

            if cfg is not None:
                deeplabutils.register_train_test(input.data["images"], input.data["metadata"],
                                                 train_ratio=cfg.SPLIT_TRAIN_TEST / 100,
                                                 seed=cfg.SPLIT_TRAIN_TEST_SEED)

                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

                str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
                model_folder = cfg.OUTPUT_DIR + os.path.sep + str_datetime
                cfg.OUTPUT_DIR = model_folder

                if not os.path.isdir(model_folder):
                    os.mkdir(model_folder)

                cfg.OUTPUT_DIR = model_folder

                self.trainer = deeplabutils.MyTrainer(cfg, self)
                self.trainer.resume_or_load(resume=False)
                print("Starting training job...")
                launch(self.trainer.train, num_gpus_per_machine=1)
                print("Training job finished.")
                self.trainer = None
                gc.collect()
                torch.cuda.empty_cache()
                with open(cfg.OUTPUT_DIR+"/Detectron2_DeepLabV3Plus_Train_Config.yaml", 'w') as file:
                    file.write(cfg.dump())
            else:
                print("Error : can't load config file "+param.cfg["config"])

        # Call end_task_run to finalize process
        self.end_task_run()

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.get_param_object()
        if param is not None:
            return param.cfg["max_iter"]
        else:
            return 1

    def stop(self):
        super().stop()
        self.trainer.run = False

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class TrainDeeplabv3plusFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_detectron2_deeplabv3plus"
        self.info.short_description = "Training process for DeepLabv3+ model of Detectron2."
        self.info.description = "Implementation from Detectron2 (Facebook Research). " \
                                "This Ikomia plugin can train DeepLabV3+ model for semantic segmentation. " \
                                "Most common parameters are exposed in the settings window. For expert usage, " \
                                "it is also possible to select a custom configuration file." \
                                "To start your training:" \
                                "create a new workflow, " \
                                "add a task node loading your dataset in Ikomia format " \
                                "(consult the marketplace to check if a suitable dataset loader already exists), " \
                                "add this DeepLabV3+ train task, " \
                                "adjust parameters, " \
                                "and click apply to start the training. " \
                                "You are able to monitor your training runs through the MLflow dashboard."
        self.info.authors = "Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.2.0"
        self.info.icon_path = "icons/detectron2.png"
        self.info.article = "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
        self.info.journal = "ECCV 2018"
        self.info.year = 2018
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://detectron2.readthedocs.io/index.html"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "semantic, segmentation, detectron2, facebook, atrous, convolution, encoder, decoder"

    def create(self, param=None):
        # Create process object
        return TrainDeeplabv3plus(self.info.name, param)
