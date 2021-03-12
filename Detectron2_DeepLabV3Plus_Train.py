from ikomia import dataprocess
import Detectron2_DeepLabV3Plus_Train_process as processMod
import Detectron2_DeepLabV3Plus_Train_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_Train(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.Detectron2_DeepLabV3Plus_TrainProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.Detectron2_DeepLabV3Plus_TrainWidgetFactory()
