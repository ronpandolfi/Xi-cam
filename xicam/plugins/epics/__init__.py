from xicam.plugins import base
from py4syn.epics.MotorClass import Motor
from py4syn.utils.motor import createMotor
import py4syn
from PySide.QtGui import *
from PySide.QtCore import *
import pyqtgraph as pg
import numpy as np


# Overload for Py2App
def new_load_qt(api_options):
    from PySide import QtCore, QtGui, QtSvg

    return QtCore, QtGui, QtSvg, 'pyside'


from qtconsole import qt_loaders

qt_loaders.load_qt = new_load_qt

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

import qdarkstyle

PVlist = None

class EpicsPlugin(base.plugin):
    name = 'Epics'

    def __init__(self, *args, **kwargs):
        global PVlist

        self.rightwidget = None

        self.leftwidget = PVlist = QListWidget()

        style = (qdarkstyle.load_stylesheet())

        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel
        kernel.gui = 'qt4'

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()

        control = RichJupyterWidget()
        control.kernel_manager = kernel_manager
        control.kernel_client = kernel_client
        control.exit_requested.connect(stop)
        control.style_sheet = style
        control.syntax_style = u'monokai'
        control.set_default_style(colors='Linux')

        self.centerwidget = control

        plot = pg.PlotWidget()
        self.curves = dict()
        self.bottomwidget=plot
        self.timer = pg.QtCore.QTimer()



        PVMotorItem('m1', 'rp:m1')
        PVMotorItem('m2', 'rp:m2')

        for name, motor in py4syn.mtrDB.iteritems():
            self.curves[name] = plot.plot([motor.getValue()])

        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        kernel.shell.push(py4syn.mtrDB)

        super(EpicsPlugin, self).__init__(*args,**kwargs)

    def update(self):
        for name, curve in self.curves.iteritems():
            x, y = curve.getData()
            y = y[-99:]
            y = np.append(y, py4syn.mtrDB[name].getValue())
            curve.setData(y)

class PVItem(QListWidgetItem):
    def __init__(self, pvName='', mne=''):
        super(PVItem, self).__init__(pvName)
        global PVlist
        PVlist.addItem(self)


class PVMotorItem(PVItem):
    def __init__(self, pvName='', mne=''):
        super(PVMotorItem, self).__init__(pvName, mne)
        self.device = createMotor(pvName, mne)


class MotorControl(QWidget):
    pass
