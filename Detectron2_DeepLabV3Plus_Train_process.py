from ikomia import core, dataprocess
import copy
from ikomia.dnn import dataset, datasetio
from detectron2.config import get_cfg
from ikomia.dnn import dnntrain
import deeplabutils
import cv2
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer,DefaultPredictor
import json
import torch
import os
from detectron2.projects.deeplab import add_deeplab_config
import numpy
# Your imports below

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainParam(dataprocess.CDnnTrainProcessParam):

    def __init__(self):
        dataprocess.CDnnTrainProcessParam.__init__(self)
        # Place default value initialization here
        self.cfg = get_cfg()
        self.inputSize=(512,512)
        self.numClasses=3
        self.maxIter = 100
        self.warmupFactor = 0.001
        self.warmupIters = 200
        self.polyLRFactor = 0.9
        self.polyLRConstantFactor = 0.0
        self.batchSize = 4
        self.ResnetDepth = 50
        self.freezeAt = 4
        self.batchNorm = "BN"
        self.ignoreValue = None
        self.baseLearningRate = 0.01
        self.trainer= None
        self.weights= None

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(paramMap["windowSize"])
        self.cfg = paramMap["cfg"]
        self.inputSize = paramMap["inputSize"]
        self.numClasses = paramMap["numClasses"]
        self.maxIter = paramMap["maxIter"]
        self.warmupFactor = paramMap["warmupFactor"]
        self.warmupIters = paramMap["warmupIters"]
        self.polyLRFactor = paramMap["polyLRFactor"]
        self.polyLRConstantFactor = paramMap["polyLRConstantFactor"]
        self.batchSize = paramMap["batchSize"]
        self.ResnetDepth = paramMap["ResnetDepth"]
        self.freezeAt = paramMap["freezeAt"]
        self.batchNorm = paramMap["BN"]
        self.ignoreValue = paramMap["ignoreValue"]

        pass

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        paramMap = core.ParamMap()
        paramMap["cfg"] = self.cfg
        paramMap["inputSize"] =self.inputSize
        paramMap["numClasses"] =self.numClasses
        paramMap["maxIter"] = self.maxIter
        paramMap["warmupFactor"] = self.warmupFactor
        paramMap["warmupIters"] = self.warmupIters
        paramMap["polyLRFactor"] = self.polyLRFactor
        paramMap["polyLRConstantFactor"] = self.polyLRConstantFactor
        paramMap["batchSize"] = self.batchSize
        paramMap["ResnetDepth"] = self.ResnetDepth
        paramMap["freezeAt"] = self.freezeAt
        paramMap["BN"] = self.batchNorm
        paramMap["ignoreValue"] = self.ignoreValue
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainProcess(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here
        # Example :  self.addInput(PyDataProcess.CImageProcessIO())
        #           self.addOutput(PyDataProcess.CImageProcessIO())
        self.addInput(datasetio.IkDatasetIO(dataprocess.DatasetFormat.OTHER))
        # Create parameters class
        if param is None:
            self.setParam(Detectron2_DeepLabV3Plus_TrainParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        # Examples :
        # Get input :
        input = self.getInput(0)

        # Get parameters :
        param = self.getParam()
        DatasetCatalog.clear()
        DatasetCatalog.register("datasetTrain", deeplabutils.my_dataset_function(input.data))

        if param.ignoreValue is not None:
            MetadataCatalog.get("datasetTrain").ignore_label = param.ignoreValue
        cfg = get_cfg()
        add_deeplab_config(cfg)
        cfg.merge_from_file(os.path.dirname(os.path.realpath(__file__))+"/model/configs/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
        cfg.DATASETS.TRAIN = ("datasetTrain",)
        cfg.DATASETS.TEST = ()
        cfg.SOLVER.MAX_ITER = 100
        cfg.SOLVER.WARMUP_FACTOR = 0.001
        cfg.SOLVER.WARMUP_ITERS = 40
        cfg.SOLVER.POLY_LR_FACTOR = 0.9
        cfg.SOLVER.POLY_LR_CONSTANT_FACTOR = 0.0
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
        cfg.SOLVER.BASE_LR = 0.02
        cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
        cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
        cfg.SOLVER.IMS_PER_BATCH=1
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.inputSize=param.inputSize

        cfg.OUTPUT_DIR = os.path.dirname(os.path.realpath(__file__))+"/output"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        if len(input.data["images"])>0:
            trainer = deeplabutils.MyTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
        # Get output :
        # output = self.getOutput(indexOfOutput)



        # Get image from input/output (numpy array):
        # srcImage = input.getImage()

        # Call to the process main routine
        # dstImage = ...

        # Set image of input/output (numpy array):
        # output.setImage(dstImage)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Detectron2_DeepLabV3Plus_Train"
        self.info.shortDescription = "your short description"
        self.info.description = "your description"
        self.info.authors = "Plugin authors"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return Detectron2_DeepLabV3Plus_TrainProcess(self.info.name, param)
