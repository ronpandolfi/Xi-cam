import json
from PySide.QtUiTools import QUiLoader
from PySide import QtGui
from PySide import QtCore
import yaml
from modpkgs.collectionsmod import UnsortableOrderedDict
import ui
import featuremanager
import display
import customwidgets


class mainwindow():
    def __init__(self, app):
        self.app = app

        ui.loadUi()

        # INIT FORMS
        self.computationForm = None
        self._detectorForm = None
        self._scatteringForm = None

        # STYLE
        self.app.setStyle('Plastique')
        with open('xicam/gui/style.stylesheet', 'r') as f:
            self.app.setStyleSheet(f.read())


        # SETUP FEATURES
        featuremanager.layout = ui.mainwindow.featuresList
        featuremanager.load()




        # INIT EXPERIMENT
        self.newExperiment()



        # WIREUP CONTROLS
        ui.mainwindow.addFeatureButton.clicked.connect(featuremanager.addLayer)
        ui.mainwindow.addSubstrateButton.clicked.connect(featuremanager.addSubstrate)
        ui.mainwindow.addParticleButton.clicked.connect(featuremanager.addParticle)
        ui.mainwindow.showScatteringButton.clicked.connect(self.showScattering)
        ui.mainwindow.showComp50utationButton.clicked.connect(self.showComputation)
        ui.mainwindow.showDetectorButton.clicked.connect(self.showDetector)
        ui.mainwindow.addParticleButton.setMenu(ui.particlemenu)
        ui.mainwindow.runLocal.clicked.connect(self.runLocal)


        # SETUP DISPLAY
        display.load()
        ui.mainwindow.latticeplaceholder.addWidget(display.viewWidget)


        # END
        ui.mainwindow.show()
        ui.mainwindow.raise_()


    def newExperiment(self):
        pass
        # featuremanager.clearFeatures()
        # featuremanager.addSubstrate()
        # featuremanager.addLayer()

    def showFeature(self, index):
        self.showForm(index.internalPointer().form)


    def showComputation(self):
        if self.computationForm is None:
            self.computationForm = featuremanager.loadform('xicam/gui/guiComputation.ui')
        self.showForm(self.computationForm)

    def showScattering(self):
        self.showForm(self.scatteringForm.form)

    def showDetector(self):
        self.showForm(self.detectorForm.form)

    def showForm(self, form):
        ui.mainwindow.featureWidgetHolder.addWidget(form)
        ui.mainwindow.featureWidgetHolder.setCurrentWidget(form)

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

    # def runLocal(self):
    #     shapes = []
    #     layers = []
    #
    #     shapes = [feature.toDict() for feature in featuremanager.features if type(feature) is customwidgets.particle]
    #     layers = [feature.toDict() for feature in featuremanager.features if
    #               type(feature) is customwidgets.layer or customwidgets.substrate]
    #     unitcells = [feature.structure.toUnitCellDict() for feature in featuremanager.features if
    #                  type(feature) is customwidgets.particle]
    #     structures = [feature.structure.toStructureDict() for feature in featuremanager.features if
    #                   type(feature) is customwidgets.particle]
    #
    #     out = {'hipGisaxsInput': UnsortableOrderedDict([('shapes', shapes),
    #                                                     ('unitcells', unitcells),
    #                                                     ('layers', layers),
    #                                                     ('structures', structures),
    #                                                     ('instrumentation', self.detectorForm.toDict()),
    #                                                     ('computation', self.scatteringForm.toDict())])}
    #     with open('test.json', 'w') as outfile:
    #         json.dump(out, outfile, indent=4)
    #
    #     with open('test.yml', 'w') as outfile:
    #         yaml.dump(out, outfile, indent=4)
    #
    #     print yaml.dump(out, indent=4)




