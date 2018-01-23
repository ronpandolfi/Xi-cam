import __future__
import os, sys, time
import simulation, fitting, cdsaxs, data_treatment, custom_widget
import pyqtgraph as pg
import numpy as np
from xicam.plugins import base, widgets
from xicam import threads
from modpkgs import guiinvoker
from functools import partial
from PySide import QtGui, QtCore
import multiprocessing
import deap.base as deap_base
from deap import creator
from scipy import interpolate
import fabio

creator.create('FitnessMin', deap_base.Fitness, weights=(-1.0,))  # want to minimize fitness
creator.create('Individual', list, fitness=creator.FitnessMin)

class plugin(base.plugin):
    name = "CDSAXS"

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.datatreatmenttree = pg.parametertree.ParameterTree()
        self.materialstree = pg.parametertree.ParameterTree()
        self.rightmodes = [(self.datatreatmenttree, QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.Desktop)),
                           (self.materialstree, QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.Computer))]
        self.topwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabClose)
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.bottomwidget = custom_widget.CDLineProfileWidget()

        # Setup parametertree

        self.datatreatmentparam = pg.parametertree.Parameter.create(name='params', type='group', children=[
            {'name': 'Experimental input', 'type': 'group', 'children': [
                {'name': 'Start angle', 'type': 'float', 'value': 0},
                {'name': 'End angle', 'type': 'float', 'value': 0},
                {'name': 'Angle step', 'type': 'float', 'value': 0},
                {'name': 'Line pitch', 'type': 'float', 'value': 86},
                {'name': 'Data treatment', 'type': 'action'}]},
            {'name': 'Initial line Profile', 'type': 'group', 'children': [
                {'name': 'Number trapezoid', 'type': 'float', 'value': 5},
                {'name': 'Heigth', 'type': 'float', 'value': 20},
                {'name': 'Linewidth', 'type': 'float', 'value': 50},
                {'name': 'Sidewall angle', 'type': 'float', 'value': 90},
                {'name': 'Simulation', 'type': 'action'}]},
            {'name': 'Fit output', 'type': 'group', 'children': [
                {'name': 'num trap fit', 'type': 'float', 'readonly': True},
                {'name': 'H fit', 'type': 'float', 'readonly': True},
                {'name': 'w0 fit', 'type': 'float', 'readonly': True},
                {'name': 'Beta fit', 'type': 'float', 'readonly': True},
                {'name': 'f_val', 'type': 'float', 'readonly': True}]}])
        self.datatreatmenttree.setParameters(self.datatreatmentparam, showTop=False)

        self.materialparam = pg.parametertree.Parameter.create(name='params', type='group', children=[
            {'name': 'Material', 'type': 'group', 'children': [
                {'name': 'Substrate thickness', 'type': 'float', 'value': 27 * 10 ** -6},
                {'name': 'Substrate attenuation', 'type': 'float', 'value': 200 * 10 ** -6}]}])
        self.materialstree.setParameters(self.materialparam, showTop=False)

        from .trapezoidparameter import TrapezoidAnglesWidgetParameter
        trap = TrapezoidAnglesWidgetParameter(name='Test Trapezoids')
        self.datatreatmentparam.addChild(trap)

        self.datatreatmentparam.param('Experimental input', 'Data treatment').sigActivated.connect(self.datatreatment)
        self.datatreatmentparam.param('Initial line Profile', 'Simulation').sigActivated.connect(self.fit)

        super(plugin, self).__init__(*args, **kwargs)

    def update_model(self, widget):
        guiinvoker.invoke_in_main_thread(self.bottomwidget.plotLineProfile, *widget.modelParameters)

    def update_right_widget(self):
        guiinvoker.invoke_in_main_thread(self.datatreatmentparam.param('Fit output', 'num trap fit').setValue, self.Num_trap)
        guiinvoker.invoke_in_main_thread(self.datatreatmentparam.param('Fit output', 'H fit').setValue, self.H)
        guiinvoker.invoke_in_main_thread(self.datatreatmentparam.param('Fit output', 'w0 fit').setValue, self.w0)
        guiinvoker.invoke_in_main_thread(self.datatreatmentparam.param('Fit output', 'Beta fit').setValue, self.Beta1)
        guiinvoker.invoke_in_main_thread(self.datatreatmentparam.param('Fit output', 'f_val').setValue, self.fval)

    def openfiles(self, files, operation=None, operationname=None):
        self.activate()
        if type(files) is not list:
            files = [files]

        widget = widgets.OOMTabItem(itemclass=CDSAXSWidget, src=files, operation=operation, operationname=operationname,
                                    plotwidget=self.bottomwidget, toolbar=self.toolbar)
        self.centerwidget.addTab(widget, os.path.basename(files[0]))
        self.centerwidget.setCurrentWidget(widget)

        fitrunnable = threads.RunnableMethod(data_treatment.readheader, method_args=(files,),
                                             callback_slot=self.update_experimental_input)
        threads.add_to_queue(fitrunnable)

    def update_experimental_input(self, files, angle_start, angle_end, angle_step):
        guiinvoker.invoke_in_main_thread(self.datatreatmentparam.param('Experimental input', 'Start angle').setValue,
                                         angle_start)
        guiinvoker.invoke_in_main_thread(self.datatreatmentparam.param('Experimental input', 'End angle').setValue,
                                         angle_end)
        guiinvoker.invoke_in_main_thread(self.datatreatmentparam.param('Experimental input', 'Angle step').setValue,
                                         angle_step)
        self.files = files

    def datatreatment(self):
        Phi_min, Phi_max, Phi_step, Pitch = self.datatreatmentparam['Experimental input', 'Start angle'], self.datatreatmentparam['Experimental input', 'End angle'], self.datatreatmentparam[
            'Experimental input', 'Angle step'], self.datatreatmentparam['Experimental input', 'Line pitch']
        substratethickness, substrateattenuation = self.materialparam['Material', 'Substrate thickness'], self.materialparam['Material', 'Substrate attenuation']
        fitrunnable = threads.RunnableMethod(data_treatment.loadRAW, method_args=(self.files, Phi_min, Phi_max, Phi_step, Pitch, substratethickness, substrateattenuation), callback_slot=self.diplay_experimentaldata)
        threads.add_to_queue(fitrunnable)

    def diplay_experimentaldata(self, data, img, qx, qz, I):
        self.qx = qx
        self.qz = qz
        self.I = I
        self.getCurrentTab().CDRawWidget.setImage(data)
        self.getCurrentTab().CDCartoWidget.setImage(img)
        self.getCurrentTab().update_profile_ini(qz, I)

    def fit(self):
        activeSet = self.getCurrentTab()
        activeSet.setCurrentWidget(activeSet.CDModelWidget)
        self.DW, self.I0, self.Bkg = 0.11, 3.0, 1 # default
        self.H, self.w0, self.Beta1, self.Num_trap = self.datatreatmentparam['Initial line Profile', 'Heigth'], self.datatreatmentparam['Initial line Profile', 'Linewidth'], self.datatreatmentparam['Initial line Profile', 'Sidewall angle'], \
                                 self.datatreatmentparam['Initial line Profile', 'Number trapezoid']

        self.best_param = [self.DW, self.I0, self.Bkg + self.H, self.w0] + [int(self.Beta1) for i in range(0, self.Num_trap, 1)]
        self.Ifit = data_treatment.SL_model1(self.qx, self.qz, self.best_param)
        self.diplay_fitteddata(self.Ifit, self.best_param)

        fitrunnable1 = threads.RunnableMethod(data_treatment.fitting_cmaes, method_args=(self.qx, self.qz, self.I, self.H, self.w0, self.Beta1, self.Num_trap), callback_slot=self.diplay_fitteddata)
        #fitrunnable2 = threads.RunnableMethod(data_treatment.fitting_mcmc, method_args=(self.qx, self.qz, self.I, self.H, self.w0, self.Beta1, self.Num_trap))
        threads.add_to_queue(fitrunnable1)
        self.update_right_widget()
        #threads.add_to_queue(fitrunnable2)

    def diplay_fitteddata(self, Ifit, best_param, fval = None):
        if fval is None:
            self.fval= 0
        else:
            self.Ifit = Ifit
            self.best_param = [best_param[0], best_param[1], best_param[2], best_param[3], best_param[4]] + [best_param[5:]]
            self.fval = fval

        self.update_right_widget()
        self.getCurrentTab().update_profile(self.qz, self.I, self.Ifit)

    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()
        self.getCurrentTab().sigDrawModel.connect(self.update_model)
        self.getCurrentTab().sigDrawParam.connect(self.update_right_widget)
        self.getCurrentTab().sigDrawParam.connect(self.update_experimental_input)

    def tabClose(self, index):
        self.centerwidget.widget(index).deleteLater()

    def getCurrentTab(self):
        if self.centerwidget.currentWidget() is None: return None
        if not hasattr(self.centerwidget.currentWidget(), 'widget'): return None
        return self.centerwidget.currentWidget().widget

class CDSAXSWidget(QtGui.QTabWidget):
    sigDrawModel = QtCore.Signal(object)
    sigDrawParam = QtCore.Signal(object)

    def __init__(self, src, *args, **kwargs):
        super(CDSAXSWidget, self).__init__()

        self.CDRawWidget = custom_widget.CDRawWidget()
        self.CDCartoWidget = custom_widget.CDCartoWidget()
        self.CDModelWidget = custom_widget.CDModelWidget()

        self.addTab(self.CDRawWidget, 'RAW')
        self.addTab(self.CDCartoWidget, 'Cartography')
        self.addTab(self.CDModelWidget, 'Model')

        self.setTabPosition(self.South)
        self.setTabShape(self.Triangular)

    def update_model(self):
        self.sigDrawModel.emit(self)

    def update_right_widget(self):
        self.sigDrawParam.emit(self)

    @property
    def modelParameters(self):
        # h,w,langle,rangle=None
        # 3,4,...
        return self.best_corr[3], self.best_corr[4], self.best_corr[5:]

    def update_profile_ini(self, qz, I):
        """
        Display the experimental peak intensity rescaled
        """
        for order in range(0, len(I), 1):
            I[order] -= min(I[order])
            I[order] /= max(I[order])
            I[order] += order + 1

            guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders[order].setData, qz[order], np.log(I[order]))

    def update_profile(self, qz, I, I_fit):
        """
        Display the simulated/experimental peak intensity rescaled
        """
        for order in range(0, len(I), 1):
            I[order] -= min(I[order])
            I[order] /= max(I[order])
            I[order] += order + 1

            I_fit[order] -= min(I_fit[order])
            I_fit[order] /= max(I_fit[order])
            I_fit[order] += order + 1

            guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders[order].setData, qz[order], np.log(I[order]))
            guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders1[order].setData, qz[order], np.log(I_fit[order]))