from hipies.plugins import base
from PySide import QtGui
import os
from hipies import xglobals

import json
from PySide.QtUiTools import QUiLoader
from PySide import QtGui
from PySide import QtCore
import yaml
from collectionsmod import UnsortableOrderedDict
import ui
import featuremanager
import display
import customwidgets


class plugin(base.plugin):
    name = 'HipGISAXS'

    def __init__(self, *args, **kwargs):


        self.leftwidget, self.centerwidget, self.rightwidget = ui.load()


        # INIT FORMS
        self.computationForm = None
        self._detectorForm = None
        self._scatteringForm = None


        # SETUP FEATURES
        featuremanager.layout = ui.leftwidget.featuresList
        featuremanager.load()




        # INIT EXPERIMENT
        self.newExperiment()



        # WIREUP CONTROLS
        self.leftwidget.addFeatureButton.clicked.connect(featuremanager.addLayer)
        self.leftwidget.addSubstrateButton.clicked.connect(featuremanager.addSubstrate)
        self.leftwidget.addParticleButton.clicked.connect(featuremanager.addParticle)
        self.leftwidget.showScatteringButton.clicked.connect(self.showScattering)
        self.leftwidget.showComputationButton.clicked.connect(self.showComputation)
        self.leftwidget.showDetectorButton.clicked.connect(self.showDetector)
        self.leftwidget.addParticleButton.setMenu(ui.particlemenu)
        self.leftwidget.runLocal.clicked.connect(self.runLocal)


        # SETUP DISPLAY
        display.load()
        self.centerwidget.addWidget(display.viewWidget)

        super(plugin, self).__init__(*args, **kwargs)


    def newExperiment(self):
        pass
        # featuremanager.clearFeatures()
        # featuremanager.addSubstrate()
        # featuremanager.addLayer()

    def showFeature(self, index):
        self.showForm(index.internalPointer().form)
        print 'clicked:', index.row(), index.parent().internalPointer()


    def showComputation(self):
        if self.computationForm is None:
            self.computationForm = featuremanager.loadform('gui/guiComputation.ui')
        self.showForm(self.computationForm)

    def showScattering(self):
        self.showForm(self.scatteringForm.form)

    def showDetector(self):
        self.showForm(self.detectorForm.form)

    def showForm(self, form):
        self.rightwidget.addWidget(form)
        self.rightwidget.setCurrentWidget(form)

    @property
    def detectorForm(self):
        if self._detectorForm is None:
            self._detectorForm = customwidgets.detector()
        return self._detectorForm

    @property
    def scatteringForm(self):
        if self._scatteringForm is None:
            self._scatteringForm = customwidgets.scattering()
        return self._scatteringForm

    def runLocal(self):
        shapes = []
        layers = []

        shapes = [feature.toDict() for feature in featuremanager.features if type(feature) is customwidgets.particle]
        layers = [feature.toDict() for feature in featuremanager.features if
                  type(feature) is customwidgets.layer or customwidgets.substrate]
        unitcells = [feature.structure.toUnitCellDict() for feature in featuremanager.features if
                     type(feature) is customwidgets.particle]
        structures = [feature.structure.toStructureDict() for feature in featuremanager.features if
                      type(feature) is customwidgets.particle]

        out = {'hipGisaxsInput': UnsortableOrderedDict([('shapes', shapes),
                                                        ('unitcells', unitcells),
                                                        ('layers', layers),
                                                        ('structures', structures),
                                                        ('instrumentation', self.detectorForm.toDict()),
                                                        ('computation', self.scatteringForm.toDict())])}
        with open('test.json', 'w') as outfile:
            json.dump(out, outfile, indent=4)

        with open('test.yml', 'w') as outfile:
            yaml.dump(out, outfile, indent=4)

        print yaml.dump(out, indent=4)


class mainwindow():
    def __init__(self, app):
        self.app = app






