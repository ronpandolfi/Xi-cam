from __future__ import unicode_literals
from PySide import QtGui, QtCore
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from pyFAI import calibrant
from collections import OrderedDict
from pipeline import calibration


class calibrationpanel(ParameterTree):
    algorithms = OrderedDict(
        [('Fourier Autocorrelation', calibration.fourierAutocorrelation),
         ('2D Ricker Wavelet', calibration.rickerWavelets),
         ('DPDAK Refinement', calibration.dpdakRefine)])
    sigCalibrate = QtCore.Signal(object, str)
    sigSimulateCalibrant=QtCore.Signal(str)
    def __init__(self):
        super(calibrationpanel, self).__init__()

        self.autoCalibrateAction = pTypes.ActionParameter(name='Auto Calibrate')
        self.autoCalibrateAction.sigActivated.connect(self.calibrate)

        calibrants = sorted(calibrant.ALL_CALIBRANTS.all.keys())
        self.calibrant = pTypes.ListParameter(name='Calibrant Material', values=calibrants)

        self.autoCalibrateMethod = pTypes.ListParameter(name='Algorithm', values=self.algorithms)

        self.overlayAction = pTypes.ActionParameter(name='Simulate Calibrant')
        self.overlayAction.sigActivated.connect(self.simulatecalibrant)

        self.setParameters(pTypes.GroupParameter(name='Calibration', children=[self.autoCalibrateAction, self.calibrant,
                                                                               self.autoCalibrateMethod,
                                                                               self.overlayAction]),
                           showTop=False)

    def calibrate(self):
        self.sigCalibrate.emit(self.autoCalibrateMethod.value(), self.calibrant.value())

    def simulatecalibrant(self):
        self.sigSimulateCalibrant.emit(self.calibrant.value())