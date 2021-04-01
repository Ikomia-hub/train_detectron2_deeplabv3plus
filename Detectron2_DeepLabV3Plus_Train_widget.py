from ikomia import utils, core, dataprocess
import Detectron2_DeepLabV3Plus_Train_process as processMod
import sys
#PyQt GUI framework
from PyQt5.QtWidgets import *
from ikomia.utils.pyqtutils import BrowseFileWidget


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------


class Detectron2_DeepLabV3Plus_TrainWidget(core.CProtocolTaskWidget):

    def __init__(self, param, parent):
        core.CProtocolTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = processMod.Detectron2_DeepLabV3Plus_TrainParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = utils.PyQtToQt(self.gridLayout)

        inputSizeLabel = QLabel("Desired input size:")
        self.widthSpinBox = QSpinBox()
        self.widthSpinBox.setRange(16, 4096)
        self.widthSpinBox.setSingleStep(16)
        self.widthSpinBox.setValue(self.parameters.inputSize[0])

        self.heightSpinBox = QSpinBox()
        self.heightSpinBox.setRange(16, 4096)
        self.heightSpinBox.setSingleStep(16)
        self.heightSpinBox.setValue(self.parameters.inputSize[1])

        maxIterLabel = QLabel("Max iter:")
        self.maxIterSpinBox = QSpinBox()
        self.maxIterSpinBox.setRange(0, 2147483647)
        self.maxIterSpinBox.setSingleStep(1)
        self.maxIterSpinBox.setValue(self.parameters.maxIter)

        batchSizeLabel = QLabel("Batch size:")
        self.batchSizeSpinBox = QSpinBox()
        self.batchSizeSpinBox.setRange(1, 2147483647)
        self.batchSizeSpinBox.setSingleStep(1)
        self.batchSizeSpinBox.setValue(self.parameters.batch_size)

        splitTrainTestLabel = QLabel("Train test percentage:")
        self.splitTrainTestSpinBox = QSpinBox()
        self.splitTrainTestSpinBox.setRange(0,100)
        self.splitTrainTestSpinBox.setSingleStep(1)
        self.splitTrainTestSpinBox.setValue(self.parameters.splitTrainTest)

        evalPeriodLabel = QLabel("Evaluation period:")
        self.evalPeriodSpinBox = QSpinBox()
        self.evalPeriodSpinBox.setRange(0,2147483647)
        self.evalPeriodSpinBox.setSingleStep(1)
        self.evalPeriodSpinBox.setValue(self.parameters.evalPeriod)

        baseLearningRateLabel = QLabel("Base learning rate:")
        self.baseLearningRateSpinBox = QDoubleSpinBox()
        self.baseLearningRateSpinBox.setRange(0, 10)
        self.baseLearningRateSpinBox.setDecimals(4)
        self.baseLearningRateSpinBox.setSingleStep(0.0001)
        self.baseLearningRateSpinBox.setValue(self.parameters.learning_rate)

        resnetDepthLabel = QLabel("Resnet depth:")
        self.resnetDepthComboBox = QComboBox()
        depths = ["50","101"]
        self.resnetDepthComboBox.addItems(depths)
        self.resnetDepthComboBox.setCurrentText(str(self.parameters.resnetDepth))

        earlyStoppingLabel= QLabel("Early stopping:")
        self.earlyStoppingCheckBox = QCheckBox()
        self.earlyStoppingCheckBox.setChecked(self.parameters.earlyStopping)
        self.earlyStoppingCheckBox.clicked.connect(self.showPatienceSpinBox)

        self.patienceLabel= QLabel("Patience:")
        self.patienceLabel.hide()
        self.patienceSpinBox = QSpinBox()
        self.patienceSpinBox.setRange(0,2147483647)
        self.patienceSpinBox.setSingleStep(1)
        self.patienceSpinBox.setValue(self.parameters.patience)
        self.patienceSpinBox.hide()

        qlabel = QLabel("Advanced YAML config:")
        self.qbrowse_widget = BrowseFileWidget(path=self.parameters.expertModecfg, mode=QFileDialog.ExistingFile)

        self.numGPUQLabel = QLabel("Number of GPU:")
        self.numGPUSpinBox = QSpinBox()
        self.numGPUSpinBox.setRange(1,16)
        self.numGPUSpinBox.setSingleStep(1)
        self.numGPUSpinBox.setValue(self.parameters.numGPU)

        # Set widget layout

        self.gridLayout.addWidget(inputSizeLabel,0,0,1,1)
        self.gridLayout.addWidget(self.widthSpinBox,0,1,1,1)
        self.gridLayout.addWidget(self.heightSpinBox,0,2,1,1)
        self.gridLayout.addWidget(maxIterLabel,1,0,1,1)
        self.gridLayout.addWidget(self.maxIterSpinBox,1,1,1,2)
        self.gridLayout.addWidget(batchSizeLabel,2,0,1,1)
        self.gridLayout.addWidget(self.batchSizeSpinBox,2,1,1,2)
        self.gridLayout.addWidget(splitTrainTestLabel,3,0,1,1)
        self.gridLayout.addWidget(self.splitTrainTestSpinBox,3,1,1,2)
        self.gridLayout.addWidget(evalPeriodLabel,4,0,1,1)
        self.gridLayout.addWidget(self.evalPeriodSpinBox,4,1,1,2)
        self.gridLayout.addWidget(baseLearningRateLabel,5,0,1,1)
        self.gridLayout.addWidget(self.baseLearningRateSpinBox,5,1,1,2)
        self.gridLayout.addWidget(resnetDepthLabel,6,0,1,2)
        self.gridLayout.addWidget(self.resnetDepthComboBox,6,1,1,1)
        self.gridLayout.addWidget(earlyStoppingLabel,7,0,1,1)
        self.gridLayout.addWidget(self.earlyStoppingCheckBox,7,1,1,2)
        self.gridLayout.addWidget(self.patienceLabel, 8, 0, 1, 1)
        self.gridLayout.addWidget(self.patienceSpinBox, 8, 1, 1, 2)
        self.gridLayout.addWidget(qlabel, 9, 0,1,1)
        self.gridLayout.addWidget(self.qbrowse_widget, 9, 1,1,2)
        self.gridLayout.addWidget(self.numGPUQLabel,10,0,1,1)
        self.gridLayout.addWidget(self.numGPUSpinBox,10,1,1,2)

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
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        w,h = self.widthSpinBox.value(),self.heightSpinBox.value()
        if w%16==0 and h%16 ==0:
            self.parameters.inputSize = ( w,h )
        else:
            self.parameters.inputSize = (w//16*16, h//16*16)
            print("Width and Height must be multiples of 16, they have been changed to "+str(self.parameters.inputSize))
        self.parameters.maxIter = self.maxIterSpinBox.value()
        self.parameters.batch_size = self.batchSizeSpinBox.value()
        self.parameters.splitTrainTest =self.splitTrainTestSpinBox.value()
        self.parameters.evalPeriod = self.evalPeriodSpinBox.value()
        self.parameters.learning_rate = self.baseLearningRateSpinBox.value()
        self.parameters.resnetDepth = str(self.resnetDepthComboBox.currentText())
        self.parameters.earlyStopping = self.earlyStoppingCheckBox.isChecked()
        self.parameters.patience = self.patienceSpinBox.value()
        self.parameters.expertModecfg= self.qbrowse_widget.qedit_file.text()
        self.parameters.numGPU = self.numGPUSpinBox.value()

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
