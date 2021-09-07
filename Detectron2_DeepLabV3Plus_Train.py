from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_Train(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from Detectron2_DeepLabV3Plus_Train.Detectron2_DeepLabV3Plus_Train_process import Detectron2_DeepLabV3Plus_TrainProcessFactory
        # Instantiate process object
        return Detectron2_DeepLabV3Plus_TrainProcessFactory()

    def getWidgetFactory(self):
        from Detectron2_DeepLabV3Plus_Train.Detectron2_DeepLabV3Plus_Train_widget import Detectron2_DeepLabV3Plus_TrainWidgetFactory
        # Instantiate associated widget object
        return Detectron2_DeepLabV3Plus_TrainWidgetFactory()
