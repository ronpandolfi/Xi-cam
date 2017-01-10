# --coding: utf-8 --
import pickle
import pyFAI
from pyFAI import geometry
from PySide import QtGui
from pyqtgraph.parametertree import Parameter
import numpy as np
import yaml
from pipeline import pathtools
import os
from pipeline import msg


class settingstracker(object):
    settingspath = os.path.join(pathtools.user_config_dir, 'settings.yml')

    def __init__(self):
        self.settings = dict()
        if os.path.isfile(self.settingspath):
            with open(self.settingspath,'r') as stream:
                try:
                    self.settings = yaml.load(stream)
                except yaml.YAMLError as exc:
                    msg.logMessage(exc, msg.WARNING)
        if not self.settings: self.settings=dict()



    def write(self):

        if not os.path.exists(pathtools.user_config_dir):
            os.makedirs(pathtools.user_config_dir)
        with open(self.settingspath,'w') as stream:
            try:
                stream.write(yaml.dump(self.settings))
            except yaml.YAMLError as exc:
                print exc

    def __getitem__(self, item):
        try:
            return self.settings[item]
        except KeyError:
            return None

    def __setitem__(self, key, value):
        self.settings[key]=value
        self.write()

settings=settingstracker()

class PyFAIGeometry(pyFAI.geometry.Geometry):
    def set_fit2d(self,
                  wavelength,
                  distance,
                  center_x,
                  center_y,
                  tilt,
                  rotation):
        self.set_wavelength(wavelength * 1e-10)
        self.setFit2D(distance, center_x, center_y, tilt, rotation)

    def get_fit2d(self):
        param_dict = self.getFit2D()
        return [self.get_wavelength() * 1e10,
                param_dict['directDist'],
                param_dict['centerX'],
                param_dict['centerY'],
                param_dict['tilt'],
                param_dict['tiltPlanRotation']]


class experiment(Parameter):
    def __init__(self, path=None):

        self.imageshape = (1475, 1709)

        if path is None:  # If not loading an exeriment from file
            # Build an empty experiment tree
            config = [{'name': 'Name', 'type': 'str', 'value': 'New Experiment'},
                      {'name': 'Detector', 'type': 'str', 'value': 'Unknown'},
                      {'name': 'Pixel Size X', 'type': 'float', 'value': 0, 'siPrefix': True, 'suffix': 'm',
                       'step': 1e-6},
                      {'name': 'Pixel Size Y', 'type': 'float', 'value': 0, 'siPrefix': True, 'suffix': 'm',
                       'step': 1e-6},
                      {'name': 'Center X', 'type': 'float', 'value': 0, 'suffix': ' px'},
                      {'name': 'Center Y', 'type': 'float', 'value': 0, 'suffix': ' px'},
                      {'name': 'Detector Distance', 'type': 'float', 'value': 1, 'siPrefix': True, 'suffix': 'm',
                       'step': 1e-3},
                      {'name': 'Detector Tilt', 'type': 'float', 'value': 0, 'siPrefix': False, 'suffix': u'°',
                       'step': 1e-1},
                      {'name': 'Detector Rotation', 'type': 'float', 'value': 0, 'siPrefix': False, 'suffix': u'°',
                       'step': 1e-1},
                      {'name': 'Energy', 'type': 'float', 'value': 10000, 'siPrefix': True, 'suffix': 'eV'},
                      {'name': 'Wavelength', 'type': 'float', 'value': 1, 'siPrefix': True, 'suffix': 'm'},
                      # {'name': 'View Mask', 'type': 'action'},
                      {'name': 'Incidence Angle (GIXS)', 'type': 'float', 'value': 0.1, 'suffix': u'°'},
                      {'name': 'Notes', 'type': 'text', 'value': ''}]
            super(experiment, self).__init__(name='Experiment Properties', type='group', children=config)

            # Wire up the energy and wavelength parameters to fire events on change (so they always match)
            EnergyParam = self.param('Energy')
            WavelengthParam = self.param('Wavelength')
            EnergyParam.sigValueChanged.connect(self.EnergyChanged)
            WavelengthParam.sigValueChanged.connect(self.WavelengthChanged)

            # Add tilt style dialog
            tilt = self.param('Detector Tilt')
            rot = self.param('Detector Rotation')
            self.tiltStyleMenu = QtGui.QMenu()
            grp = QtGui.QActionGroup(self)
            self.fit2dstyle = QtGui.QAction('Use Fit2D style rot/tilt', self.tiltStyleMenu)
            self.wxdiffstyle = QtGui.QAction('Use WxDiff style rot/tilt', self.tiltStyleMenu)
            self.fit2dstyle.setCheckable(True)
            self.wxdiffstyle.setCheckable(True)
            self.fit2dstyle.setActionGroup(grp)
            self.wxdiffstyle.setActionGroup(grp)
            self.fit2dstyle.setChecked(True)
            self.fit2dstyle.triggered.connect(self.resetUnits)
            self.wxdiffstyle.triggered.connect(self.resetUnits)

            self.tiltStyleMenu.addActions([self.fit2dstyle, self.wxdiffstyle])

            # Start with a null mask
            self._mask = None

            self.EnergyChanged()
        else:
            # Load the experiment from file
            with open(path, 'r') as f:
                self.config = pickle.load(f)

        self.headermap = {'Beam Energy': 'Beam Energy',
                          'Sample Alpha Stage': 'Sample Alpha Stage',
                          'Detector Vertical': 'Detector Vertical',
                          'Detector Horizontal': 'Detector Horizontal',
                          'I1 AI': 'I1 AI',
                          'Timeline Axis': None}

    # Make the mask accessible as a property
    @property
    def mask(self):
        """I'm the 'mask' property."""
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @mask.deleter
    def mask(self):
        del self._mask

    def addtomask(self, maskedarea):
        # If the mask is empty, set the mask to the new masked area
        if self._mask is None or self._mask.shape != maskedarea.shape:
            self._mask = maskedarea.astype(np.int)
        else:  # Otherwise, bitwise or it with the current mask
            # print(self.experiment.mask,maskedarea)
            self._mask = np.bitwise_and(self._mask, maskedarea.astype(np.int))

        # try:  # hack for mask export
        #     from fabio import edfimage
        #
        #     edf = edfimage.edfimage(np.rot90(self.mask))
        #     edf.write('mask.edf')
        # except Exception:
        #     pass

    def EnergyChanged(self):
        # Make Energy and Wavelength match
        self.param('Wavelength').setValue(1.239842e-6 / self.param('Energy').value(),
                                          blockSignal=self.WavelengthChanged)

    def WavelengthChanged(self):
        # Make Energy and Wavelength match
        self.param('Energy').setValue(1.239842e-6 / self.param('Wavelength').value(), blockSignal=self.EnergyChanged)

    def save(self):
        # Save the experiment .....
        with open(self.getvalue('Name') + '.exp', 'w') as f:
            pickle.dump(self.saveState(), f)
        with open(self.getvalue('Name') + '.expmask', 'w') as f:
            np.save(f, self.mask)

    def getvalue(self, name):
        # Return the value of the named child
        return self.child(name).value()

    def setvalue(self, name, value):
        # Set the value of the named child
        self.child(name).setValue(value)

    @property
    def center(self):
        return self.getvalue('Center X'), self.getvalue('Center Y')

    @center.setter
    def center(self, cen):
        self.setvalue('Center X', cen[0])
        self.setvalue('Center Y', cen[1])

    def getAI(self):
        """
        :rtype : pyFAI.AzimuthalIntegrator
        """
        # print(self.getDetector().MAX_SHAPE)
        AI = pyFAI.AzimuthalIntegrator(
            wavelength=self.getvalue('Wavelength'))
        #                                dist=self.getvalue('Detector Distance'),
        #                                poni1=self.getvalue('Pixel Size X') * (self.getvalue('Center Y')),
        #                                poni2=self.getvalue('Pixel Size Y') * (self.getvalue('Center X')),
        #                                rot1=0,
        #                                rot2=0,
        #                                rot3=0,
        #                                pixel1=self.getvalue('Pixel Size Y'),
        #                                pixel2=self.getvalue('Pixel Size X'),
        #                                detector=self.getDetector(),
        if self.fit2dstyle.isChecked():
            AI.setFit2D(self.getvalue('Detector Distance') * 1000.,
                        self.getvalue('Center X'),
                        self.getvalue('Center Y'),
                        self.getvalue('Detector Tilt'),
                        360. - self.getvalue('Detector Rotation'),
                        self.getvalue('Pixel Size Y') * 1.e6,
                        self.getvalue('Pixel Size X') * 1.e6)
        elif self.wxdiffstyle.isChecked():
            AI.setFit2D(self.getvalue('Detector Distance') * 1000.,
                        self.getvalue('Center X'),
                        self.getvalue('Center Y'),
                        self.getvalue('Detector Tilt') / 2. / np.pi * 360.,
                        360. - (2 * np.pi - self.getvalue('Detector Rotation')) / 2. / np.pi * 360.,
                        self.getvalue('Pixel Size Y') * 1.e6,
                        self.getvalue('Pixel Size X') * 1.e6)
        AI.set_wavelength(self.getvalue('Wavelength'))
        # print AI
        return AI

        # def getGeometry(self):
        #     """
        #     :rtype : pyFAI.Geometry
        #     """
        #     # print(self.getDetector().MAX_SHAPE)
        #     geo = PyFAIGeometry(dist=self.getvalue('Detector Distance'),
        #                         poni1=self.getvalue('Pixel Size X') * (self.getvalue('Center Y')),
        #                         poni2=self.getvalue('Pixel Size Y') * (self.getvalue('Center X')),
        #                         rot1=0,
        #                         rot2=0,
        #                         rot3=0,
        #                         pixel1=self.getvalue('Pixel Size Y'),
        #                         pixel2=self.getvalue('Pixel Size X'),
        #                         detector=self.getDetector(),
        #                         wavelength=self.getvalue('Wavelength'))
        # geo = PyFAIGeometry(wavelength=self.getvalue('Wavelength'))
        # geo.setFit2D(self.getvalue('Detector Distance'),
        #             self.getvalue('Center Y'),
        #             self.getvalue('Center X'),
        #             self.getvalue('Detector Tilt'),
        #             360.-self.getvalue('Detector Rotation'),
        #             self.getvalue('Pixel Size Y')*1.e6,
        #             self.getvalue('Pixel Size X')*1.e6)
        # print AI

        # return geo

    def getDetector(self):
        key = self.getvalue('Detector')
        if key in pyFAI.detectors.ALL_DETECTORS:
            return pyFAI.detectors.ALL_DETECTORS[self.getvalue('Detector')]()

    def edit(self):
        pass



        # edit the data
        # config['key3'] = 'value3'

        # write it back to the file
        # with open('config.json', 'w') as f:
        #    json.dump(config, f)

    def iscalibrated(self):
        return (self.getvalue('Pixel Size X') > 0) and (self.getvalue('Pixel Size Y') > 0) and (
            self.getvalue('Detector Distance') > 0)

    def setHeaderMap(self, xikey, headerkey):
        self.headermap[xikey] = headerkey

    def mapHeader(self, xikey):
        return self.headermap[xikey]

    @property
    def qcorners(self):
        pass
        # bl=(0,0)
        # br=(self.imageshape[0],0)
        # tl=(0,self.imageshape[1])
        # tr=self.imageshape
        # self.getGeometry()
        # from pyFAI import geometry
        # l=geometry.Geometry.qFunction()
        #     qFunction(d1, d2, param=None, path='cython')

    def resetUnits(self):
        tilt = self.param('Detector Tilt')
        rot = self.param('Detector Rotation')
        if self.fit2dstyle.isChecked():
            tilt.setOpts(**{'suffix': u'°'})
            rot.setOpts(**{'suffix': u'°'})
        elif self.wxdiffstyle.isChecked():
            rot.setOpts(**{'suffix': ' rad'})
            tilt.setOpts(**{'suffix': ' rad'})


activeExperiment = None


def activate():
    global activeExperiment
    activeExperiment = experiment()
