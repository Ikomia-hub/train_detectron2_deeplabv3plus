from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from train_detectron2_deeplabv3plus.train_detectron2_deeplabv3plus_process import TrainDeeplabv3plusFactory
        # Instantiate process object
        return TrainDeeplabv3plusFactory()

    def getWidgetFactory(self):
        from train_detectron2_deeplabv3plus.train_detectron2_deeplabv3plus_widget import TrainDeeplabv3plusWidgetFactory
        # Instantiate associated widget object
        return TrainDeeplabv3plusWidgetFactory()
