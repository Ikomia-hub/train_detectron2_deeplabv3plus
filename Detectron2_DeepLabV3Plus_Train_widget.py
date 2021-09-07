from ikomia import utils, core, dataprocess
from Detectron2_DeepLabV3Plus_Train.Detectron2_DeepLabV3Plus_Train_process import Detectron2_DeepLabV3Plus_TrainParam
# PyQt GUI framework
from PyQt5.QtWidgets import *
from ikomia.utils import qtconversion
from ikomia.utils.pyqtutils import BrowseFileWidget


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = Detectron2_DeepLabV3Plus_TrainParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        inputSizeLabel = QLabel("Desired input size:")
        self.widthSpinBox = QSpinBox()
        self.widthSpinBox.setRange(16, 4096)
        self.widthSpinBox.setSingleStep(16)
        self.widthSpinBox.setValue(self.parameters.cfg["inputWidth"])

        self.heightSpinBox = QSpinBox()
        self.heightSpinBox.setRange(16, 4096)
        self.heightSpinBox.setSingleStep(16)
        self.heightSpinBox.setValue(self.parameters.cfg["inputHeight"])

        maxIterLabel = QLabel("Max iter:")
        self.maxIterSpinBox = QSpinBox()
        self.maxIterSpinBox.setRange(0, 2147483647)
        self.maxIterSpinBox.setSingleStep(1)
        self.maxIterSpinBox.setValue(self.parameters.cfg["maxIter"])

        batchSizeLabel = QLabel("Batch size:")
        self.batchSizeSpinBox = QSpinBox()
        self.batchSizeSpinBox.setRange(1, 2147483647)
        self.batchSizeSpinBox.setSingleStep(1)
        self.batchSizeSpinBox.setValue(self.parameters.cfg["batchSize"])

        splitTrainTestLabel = QLabel("Train test percentage:")
        self.splitTrainTestSpinBox = QSpinBox()
        self.splitTrainTestSpinBox.setRange(0,100)
        self.splitTrainTestSpinBox.setSingleStep(1)
        self.splitTrainTestSpinBox.setValue(self.parameters.cfg["splitTrainTest"])

        evalPeriodLabel = QLabel("Evaluation period:")
        self.evalPeriodSpinBox = QSpinBox()
        self.evalPeriodSpinBox.setRange(0,2147483647)
        self.evalPeriodSpinBox.setSingleStep(1)
        self.evalPeriodSpinBox.setValue(self.parameters.cfg["evalPeriod"])

        baseLearningRateLabel = QLabel("Base learning rate:")
        self.baseLearningRateSpinBox = QDoubleSpinBox()
        self.baseLearningRateSpinBox.setRange(0, 10)
        self.baseLearningRateSpinBox.setDecimals(4)
        self.baseLearningRateSpinBox.setSingleStep(0.0001)
        self.baseLearningRateSpinBox.setValue(self.parameters.cfg["learningRate"])

        resnetDepthLabel = QLabel("Resnet depth:")
        self.resnetDepthComboBox = QComboBox()
        depths = ["50", "101"]
        self.resnetDepthComboBox.addItems(depths)
        self.resnetDepthComboBox.setCurrentText(str(self.parameters.cfg["resnetDepth"]))

        earlyStoppingLabel = QLabel("Early stopping:")
        self.earlyStoppingCheckBox = QCheckBox()
        self.earlyStoppingCheckBox.setChecked(self.parameters.cfg["earlyStopping"])
        self.earlyStoppingCheckBox.clicked.connect(self.showPatienceSpinBox)

        self.patienceLabel = QLabel("Patience:")
        self.patienceLabel.hide()
        self.patienceSpinBox = QSpinBox()
        self.patienceSpinBox.setRange(0, 2147483647)
        self.patienceSpinBox.setSingleStep(1)
        self.patienceSpinBox.setValue(self.parameters.cfg["patience"])
        self.patienceSpinBox.hide()

        yaml_label = QLabel("Advanced YAML config:")
        self.yaml_browse_widget = BrowseFileWidget(path=self.parameters.cfg["expertModeCfg"],
                                               mode=QFileDialog.ExistingFile)

        self.numGPUQLabel = QLabel("Number of GPU:")
        self.numGPUSpinBox = QSpinBox()
        self.numGPUSpinBox.setRange(1, 16)
        self.numGPUSpinBox.setSingleStep(1)
        self.numGPUSpinBox.setValue(self.parameters.cfg["numGPU"])

        # Output folder
        output_label = QLabel("Output folder:")
        self.output_browse_widget = BrowseFileWidget(path=self.parameters.cfg["outputFolder"],
                                                     tooltip="Select folder",
                                                     mode=QFileDialog.Directory)

        # Set widget layout
        self.gridLayout.addWidget(inputSizeLabel, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.widthSpinBox, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.heightSpinBox, 0, 2, 1, 1)
        self.gridLayout.addWidget(maxIterLabel, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.maxIterSpinBox, 1, 1, 1, 2)
        self.gridLayout.addWidget(batchSizeLabel, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.batchSizeSpinBox, 2, 1, 1, 2)
        self.gridLayout.addWidget(splitTrainTestLabel, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.splitTrainTestSpinBox, 3, 1, 1, 2)
        self.gridLayout.addWidget(evalPeriodLabel, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.evalPeriodSpinBox, 4, 1, 1, 2)
        self.gridLayout.addWidget(baseLearningRateLabel, 5, 0, 1, 1)
        self.gridLayout.addWidget(self.baseLearningRateSpinBox, 5, 1, 1, 2)
        self.gridLayout.addWidget(resnetDepthLabel, 6, 0, 1, 2)
        self.gridLayout.addWidget(self.resnetDepthComboBox, 6, 1, 1, 2)
        self.gridLayout.addWidget(earlyStoppingLabel, 7, 0, 1, 1)
        self.gridLayout.addWidget(self.earlyStoppingCheckBox, 7, 1, 1, 2)
        self.gridLayout.addWidget(self.patienceLabel, 8, 0, 1, 1)
        self.gridLayout.addWidget(self.patienceSpinBox, 8, 1, 1, 2)
        self.gridLayout.addWidget(yaml_label, 9, 0, 1, 1)
        self.gridLayout.addWidget(self.yaml_browse_widget, 9, 1, 1, 2)
        self.gridLayout.addWidget(self.numGPUQLabel, 10, 0, 1, 1)
        self.gridLayout.addWidget(self.numGPUSpinBox, 10, 1, 1, 2)
        self.gridLayout.addWidget(output_label, 11, 0, 1, 1)
        self.gridLayout.addWidget(self.output_browse_widget, 11, 1, 1, 2)
        self.setLayout(layout_ptr)

    def showPatienceSpinBox(self):
        if self.earlyStoppingCheckBox.isChecked():
            self.patienceLabel.show()
            self.patienceSpinBox.show()
        else:
            self.patienceLabel.hide()
            self.patienceSpinBox.hide()

    def onApply(self):
        # Apply button clicked slot
        # Get parameters from widget
        w, h = self.widthSpinBox.value(),self.heightSpinBox.value()
        if w%16 == 0 and h%16 == 0:
            self.parameters.cfg["inputWidth"] = w
            self.parameters.cfg["inputHeight"] = h
        else:
            self.parameters.cfg["inputWidth"] = w//16*16
            self.parameters.cfg["inputHeight"] = h//16*16
            print("Width and Height must be multiples of 16, they have been changed to "+str(self.parameters.inputSize))

        self.parameters.cfg["maxIter"] = self.maxIterSpinBox.value()
        self.parameters.cfg["batchSize"] = self.batchSizeSpinBox.value()
        self.parameters.cfg["splitTrainTest"] = self.splitTrainTestSpinBox.value()
        self.parameters.cfg["evalPeriod"] = self.evalPeriodSpinBox.value()
        self.parameters.cfg["learningRate"] = self.baseLearningRateSpinBox.value()
        self.parameters.cfg["resnetDepth"] = str(self.resnetDepthComboBox.currentText())
        self.parameters.cfg["earlyStopping"] = self.earlyStoppingCheckBox.isChecked()
        self.parameters.cfg["patience"] = self.patienceSpinBox.value()
        self.parameters.cfg["expertModeCfg"] = self.yaml_browse_widget.path
        self.parameters.cfg["numGPU"] = self.numGPUSpinBox.value()
        self.parameters.cfg["outputFolder"] = self.output_browse_widget.path

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class Detectron2_DeepLabV3Plus_TrainWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "Detectron2_DeepLabV3Plus_Train"

    def create(self, param):
        # Create widget object
        return Detectron2_DeepLabV3Plus_TrainWidget(param, None)
