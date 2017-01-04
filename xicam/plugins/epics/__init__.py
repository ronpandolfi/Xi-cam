from xicam.plugins import base
from py4syn.utils.motor import createMotor
from py4syn.epics.PilatusClass import Pilatus
import py4syn
from PySide.QtGui import *
from PySide.QtCore import *
import pyqtgraph as pg
import numpy as np
import controlwidgets
from modpkgs import guiinvoker
import time
from functools import partial

PVlist = None
detectors = dict()
devices = dict()

class EpicsPlugin(base.plugin):
    name = 'Epics'

    def __init__(self, *args, **kwargs):
        global PVlist

        self.rightwidget = None

        self.leftwidget = PVlist = QListWidget()

        self.centerwidget = advancedPythonWidget = controlwidgets.pythontools.advancedPythonWidget()
        self.rightwidget = self.itemStack = QStackedWidget()
        beamlinemodel = controlwidgets.beamlinemodel.beamlinemodel(PVlist, self.itemStack)
        advancedPythonWidget.addTab(beamlinemodel,'Beamline Model')
        self.bottomwidget = self.plot = pg.PlotWidget()
        self.curves = dict()
        self.timer = pg.QtCore.QTimer()

        # Device Items
        #PVMotorItem('Motor1', 'rp:m1')
        #PVMotorItem('Motor2', 'rp:m2')
        PilatusItem('Pilatus', '531PIL1:cam1')


        # Init curves
        for name, motor in py4syn.mtrDB.iteritems():
            self.curves[name] = self.plot.plot([motor.getValue()])
            motor.motor.add_callback('RBV', partial(self.update, motor.getMnemonic()))

        devices={}
        devices.update(detectors)
        devices.update(py4syn.mtrDB)
        advancedPythonWidget.push(devices)
        advancedPythonWidget.push({'quickSnap':self.quickSnap})

        super(EpicsPlugin, self).__init__(*args,**kwargs)

    @staticmethod
    def quickSnap(expTime, name):
        devices['Pilatus'].setImageName(name)
        devices['Pilatus'].setCountTime(expTime)
        devices['Pilatus'].startCount()

    def update(self,motorMnemonic,value,**kwargs):
        curve = self.curves[motorMnemonic]
        x, y = curve.getData()
        y = y[-99:]
        y = np.append(y,value)
        x = x[-99:]
        t = time.time()
        x = np.append(x,t)
        guiinvoker.invoke_in_main_thread(curve.setData,x,y)
        guiinvoker.invoke_in_main_thread(self.plot.setXRange,t-10,t)

class PVItem(QListWidgetItem):
    def __init__(self, mne='', pvName=''):
        super(PVItem, self).__init__(mne)
        global PVlist
        PVlist.addItem(self)

    def showWidget(self):
        self.itemStack.addWidget(self.widget)
        self.itemStack.setCurrentWidget(self.widget)


class PVMotorItem(PVItem):
    def __init__(self,pvName='',mne=''):
        super(PVMotorItem, self).__init__(pvName,mne)
        createMotor(pvName, mne)
        self.device = py4syn.mtrDB[pvName]
        self.widget = controlwidgets.motor.motorwidget(self.device)

        self.device.motor.add_callback('DMOV',partial(guiinvoker.invoke_in_main_thread,self._updatestatus))

    def _updatestatus(self,value,**kwargs):
        if value == 0:
            self.setBackground(QColor('orange'))
            #self.setForeground(QColor('cyan'))
        elif value == 1:
            self.setBackground(QListWidgetItem().background())
            #self.setForeground(QListWidgetItem().foreground())


class MotorControl(QWidget):
    pass

class PilatusItem(QListWidgetItem):
    # TODO: consider releasing the camera when not in use?

    def __init__(self, mne, pvName):
        super(PilatusItem, self).__init__(mne)
        global PVlist, detectors
        PVlist.addItem(self)
        self.device = Pilatus(mne,pvName)
        detectors[mne]=self.device

    def capture(self,filepath='',exposure=.1):
        self.device.setImageName(filepath)
        self.device.setCountTime(exposure)
        self.device.startCount()
        self.device.wait()
        self.device.stopCount()

    def _updatestatus(self,value,**kwargs):
        self.device.add_callback('DMOV', partial(guiinvoker.invoke_in_main_thread, self._updatestatus))

