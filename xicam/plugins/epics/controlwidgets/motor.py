from PySide.QtGui import *
from PySide.QtCore import *
from PySide.QtUiTools import QUiLoader
from modpkgs import pyside_dynamic
import pyqtgraph as pg
from functools import partial

from py4syn.epics.MotorClass import Motor
import simplewidgets
from modpkgs import guiinvoker

class motorwidget(QWidget):

    sigTargetChanged = Signal()
    sigStatusChanged = Signal(int)

    def __init__(self,motordevice):
        '''
        Parameters
        ----------
        motordevice :   Motor

        '''
        super(motorwidget, self).__init__()
        pyside_dynamic.loadUi('gui/motor.ui', self)
        self.motordevice = motordevice
        self.joylayout.addWidget(pg.JoystickButton())

        self.targetSpinBox = simplewidgets.placeHolderSpinBox(self.targetGo)

        self._updateRange()

        self._setTarget(self.motordevice.getValue())

        self.targetHSlider.valueChanged.connect(self._setTarget)
        self.targetVSlider.valueChanged.connect(self._setTarget)
        self.targetDial.valueChanged.connect(self._setTarget)
        self.targetLineEditLayout.insertWidget(0, self.targetSpinBox)
        #self.targetSpinBox.returnPressed.connect(lambda: self._setTarget(self.targetSpinBox.text()))
        self.targetGo.clicked.connect(lambda: self._setTarget(self.targetSpinBox.text()))
        self.stopButton.clicked.connect(self.stop)


        motordevice.motor.add_callback('RBV',self._setCurrentValue)
        motordevice.motor.add_callback('DMOV',partial(guiinvoker.invoke_in_main_thread,self._updateStatus))

        self.sigStatusChanged.connect(self._updateStatus)



        self._setCurrentValue(self.motordevice.getValue())

        self.targetHSlider.update()
        self.targetVSlider.update()



    def _setTarget(self,value):
        value=float(value)
        self.motordevice.setValue(value)
        self.targetHValue.setNum(value)
        self.targetVValue.setNum(value)
        self.targetDialValue.setNum(value)
        self.targetHSlider.setValue(value)
        self.targetVSlider.setValue(value)
        self.targetDial.setValue(value)
        self.targetSpinBox.setValue(value)



    def _updateRange(self):
        min = self.motordevice.getLowLimitValue()
        max = self.motordevice.getHighLimitValue()
        self.targetHSlider.setRange(min,max)
        self.targetVSlider.setRange(min,max)
        self.targetDial.setRange(min,max)
        self.currentHSlider.setRange(min,max)
        self.currentVSlider.setRange(min,max)
        self.currentDial.setRange(min,max)

        self.targetHMax.setText(unicode(max))
        self.targetVMax.setText(unicode(max))
        self.targetHMin.setText(unicode(min))
        self.targetVMin.setText(unicode(min))
        self.currentHMax.setText(unicode(max))
        self.currentVMax.setText(unicode(max))
        self.currentHMin.setText(unicode(min))
        self.currentVMin.setText(unicode(min))

    def _setCurrentValue(self,value=None,**kwargs):
        self.currentHSlider.setValue(value)
        self.currentVSlider.setValue(value)
        self.currentDial.setValue(value)
        self.currentLineEdit.setText(unicode(value))

        self.currentHValue.setNum(value)
        self.currentVValue.setNum(value)
        self.currentDialValue.setNum(value)

    def _updateStatus(self,value,**kwargs):
        self.progressBar.setRange(0,value)
        self.progressBar.setValue(value)

        if value==0:
            status = 'Moving to position...'
        elif value==1:
            status = 'Ready...'
        else:
            status = 'Error: Unknown status!'

        self.statusLabel.setText(status)

    def stop(self):
        self._setTarget(self.motordevice.getValue())
        self.motordevice.stop()
