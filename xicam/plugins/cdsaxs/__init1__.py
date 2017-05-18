
import __future__
import os, sys, time
import simulation, fitting, cdsaxs
from scipy.fftpack import *
from  numpy.fft import *
from scipy.signal import resample
import platform
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
from pipeline import loader, hig, msg
import numpy as np
from xicam.plugins import base, widgets
import subprocess
from xicam import threads, ROI
from modpkgs import guiinvoker
from operator import itemgetter

from PySide import QtGui, QtCore
from xicam import debugtools

from pipeline.spacegroups import spacegroupwidget
from xicam import config

from xicam.widgets.calibrationpanel import calibrationpanel
from pipeline import integration, center_approx

import astropy

from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors, Consts
from pyevolve import Statistics
from pyevolve import DBAdapters
import pyevolve

from random import randrange
import cPickle as pickle
import psutil
import multiprocessing
from collections import deque
from itertools import repeat
#import pandas as pd
import emcee
import deap.base as deap_base
from deap import creator, tools
from deap import cma as cmaes

class plugin(base.plugin):
    name = "CDSAXS"

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.rightwidget = self.parametertree = pg.parametertree.ParameterTree()
        self.topwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabClose)
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.bottomwidget = pg.ImageView()

        # Setup parametertree
        self.param = pg.parametertree.Parameter.create(name='params', type='group', children=[
                                                                                        {'name' : 'test', 'type' : 'group', 'children' : [
                                                                                        {'name':'Phi_min','type':'float'},
                                                                                        {'name':'Phi_max','type':'float'},
                                                                                        {'name':'Phi_step','type':'float'},
                                                                                        {'name':'H','type':'float'},
                                                                                        {'name':'w0','type':'float'},
                                                                                        {'name':'Beta','type': 'float'},
                                                                                        {'name':'Num_trap','type': 'float'},
                                                                                        {'name':'Run1', 'type': 'action'}]},
                                                                                        {'name' :'test1', 'type' : 'group', 'children' : [
                                                                                        {'name':'H_fit','type':'float', 'readonly': True},
                                                                                        {'name':'w0_fit','type':'float', 'readonly': True},
                                                                                        {'name':'Beta_fit','type':'float', 'readonly': True},
                                                                                        {'name':'f_val','type':'float', 'readonly': True}]}])

        self.parametertree.setParameters(self.param,showTop=False)
        self.param.param('test', 'Run1').sigActivated.connect(self.fit)

        super(plugin, self).__init__(*args, **kwargs)

    def update_model(self,widget):
        guiinvoker.invoke_in_main_thread(self.bottomwidget.setImage,widget.modelImage)

    def update_right_widget(self,widget):
        H, LL, beta, f_val = widget.modelParameter
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'H_fit').setValue, H)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'w0_fit').setValue, LL)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'Beta_fit').setValue, beta)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'f_val').setValue, f_val)

    def fit(self):
        activeSet = self.getCurrentTab()
        activeSet.setCurrentWidget(activeSet.CDModelWidget)
        H, w0, Beta1, Num_trap = self.param['test', 'H'], self.param['test', 'w0'], self.param['test', 'Beta'], self.param['test', 'Num_trap']
        fitrunnable = threads.RunnableMethod(self.getCurrentTab().fitting_test1,method_args=(H, w0, Beta1, Num_trap))
        threads.add_to_queue(fitrunnable)

    def openfiles(self, files, operation=None, operationname=None):
        self.activate()
        if type(files) is not list:
            files = [files]
        widget = widgets.OOMTabItem(itemclass=CDSAXSWidget, src=files, operation=operation,
                                    operationname=operationname, plotwidget=self.bottomwidget,
                                    toolbar=self.toolbar)

        self.centerwidget.addTab(widget, os.path.basename(files[0]))
        self.centerwidget.setCurrentWidget(widget)

        Phi_min, Phi_max, Phi_step = self.param['test', 'Phi_min'], self.param['test', 'Phi_max'], self.param['test', 'Phi_step']
        fitrunnable = threads.RunnableMethod(self.getCurrentTab().loadRAW,method_args=(Phi_min, Phi_max, Phi_step))
        threads.add_to_queue(fitrunnable)

    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()
        self.getCurrentTab().sigDrawModel.connect(self.update_model)
        self.getCurrentTab().sigDrawParam.connect(self.update_right_widget)

    def tabClose(self,index):
        self.centerwidget.widget(index).deleteLater()

    def getCurrentTab(self):
        if self.centerwidget.currentWidget() is None: return None
        if not hasattr(self.centerwidget.currentWidget(),'widget'): return None
        return self.centerwidget.currentWidget().widget

class CDSAXSWidget(QtGui.QTabWidget):
    sigDrawModel = QtCore.Signal(object)
    sigDrawParam = QtCore.Signal(object)

    def __init__(self, src, *args, **kwargs):
        super(CDSAXSWidget, self).__init__()

        self.CDRawWidget = CDRawWidget()
        self.CDCartoWidget = CDCartoWidget()
        self.CDModelWidget = CDModelWidget()

        self.addTab(self.CDRawWidget, 'RAW')
        self.addTab(self.CDCartoWidget, 'Cartography')
        self.addTab(self.CDModelWidget, 'Model')

        self.setTabPosition(self.South)
        self.setTabShape(self.Triangular)

        self.src=src

        #self.loadRAW()

    def loadRAW(self, Phi_min = -45 , Phi_max = 45, Phi_step = 1):

        #center = center_approx.center_approx(loader.loadimage(self.src[0]))
        center = [552, 225]    #Center CMS beamline
        #center = [730, 488]     #Center 733
        maxR, minR = 250, 0
        nb_pixel = maxR - minR
        maxy, miny = -40, 40

        data = []
        angles = []
        profiles = []
        x, y = np.indices(loader.loadimage(self.src[0]).shape)
        x_1, y_1 = x - center[0], y - center[1]
        rmincut = (y_1 > minR)
        rmaxcut = (y_1 < maxR)
        thetamincut = (x_1 > -10)
        thetamaxcut = (x_1 < 10)
        cutmask = (rmincut * rmaxcut * thetamincut * thetamaxcut).T


        #Use for 733
        '''
        for file in self.src:
            img = np.rot90(loader.loadimage(file), 1)
            self.wavelength = 0.12398       # not contained in the header of 733
            #wavelength = loader.loadparas(file)['Beamline Energy']
            self.SDD = 4        # not contained in the header of 733
            #SDD = loader.loadparas(file)['Beamline Energy']
            self.pixel_size = 172 * 10**-6
            #pixel_size = loader.loadparas(file)['title']
            I_0 = loader.loadparas(file)['Izero']
            angle = loader.loadparas(file)['Sample Rotation Stage']

            profile = np.sum(cutmask * img, axis=1)
            profiles.append(profile[center[1] + minR: center[1] + maxR])
            data.append(img)
            angles.append(np.float(angle))

        angles, data, profiles = zip(*sorted(zip(angles, data, profiles)))
        Phi_min, Phi_max = min(angles), max(angles)
        Phi_step = (max(angles) - min(angles))/(len(angles)-1)

        print(Phi_min, Phi_step)
        '''

        #Use for CMS beamline
        for file in self.src:
            img = np.rot90(loader.loadimage(file), 1)
            profile = np.sum(cutmask*img, axis=1)
            #profile = cutmask * img
            profiles.append(profile[center[1] + minR: center[1] + maxR ])
            #profiles.append(profile[center[1]:])
            data.append(img)
        self.pixel_size, self.SDD, self.wavelength = 172 * 10 **-6, 4., 0.095372   # CMS beamline

        data = np.stack(data)
        data = np.log(data - data.min()+1.)

        self.maskimage = pg.ImageItem(opacity=.25)
        self.CDRawWidget.view.addItem(self.maskimage)
        invmask = 1-cutmask
        self.maskimage.setImage(np.dstack((invmask, np.zeros_like(invmask), np.zeros_like(invmask), invmask)).astype(np.float), opacity=.5)
        self.CDRawWidget.setImage(data)

        #1Conversion to q spqce
        self.QxyiData = cdsaxs.generate_carto(profiles, nb_pixel, Phi_min, Phi_step, self.pixel_size, self.SDD, self.wavelength, center[0])

        #Intensity correction
        substratethickness, substrateattenuation = 750 * 10 **-6, 200 * 10 **-6
        self.QxyiDatacor = cdsaxs.correc_Iexp(self.QxyiData, substratethickness, substrateattenuation)

        #interpolation and Plotting carthography
        sampling_size = (400, 400)
        #img = cdsaxs.inter_carto(self.QxyiData)
        self.img, qk_shift= cdsaxs.interpolation(self.QxyiDatacor, sampling_size)
        np.save('/Users/guillaumefreychet/Desktop/carto', self.img)
        self.CDCartoWidget.setImage(self.img)

        #Definition of the variables
        self.q = []
        self.Qxexp = []
        self.Qxfit = []
        self.Q__Z = []

        #Extraction of nb of profile + their positions
        profile_carto = np.mean(np.nan_to_num(self.img), axis=1)
        self.q, self.Qxexp, self.Q__Z = cdsaxs.find_peaks(profile_carto, self.QxyiDatacor, self.wavelength, nb_pixel, self.pixel_size, self.SDD)

        self.update_profile_ini()
        self.SL_model1(10, 40, np.array([95]))

    def fitting_test1(self, H=10, LL=20, Beta1=70, Num_trap=5, DW=0.1, I0=1, Bk=0):  # these are simp not fittingp
        self.number_trapezoid = int(Num_trap)

        #algorithm = 'cmaes'
        #algorithm = 'mcmc'
        algorithm = 'pyevolve'

        if algorithm == 'cmaes':
            self.fix_fitness = self.fix_fitness_cmaes
            self.residual(np.zeros(len(initiale_value)), test='True')
            self.cmaes(sigma=100, ngen=50, popsize=100, mu=10, N=len(initiale_value), restarts=0, verbose=False, tolhistfun=5e-5, ftarget=None)
        elif algorithm == 'mcmc':
            self.fix_fitness = self.fix_fitness_mcmc
            self.residual(np.zeros(len(initiale_value)), test='True')
            self.mcmc(N=len(initiale_value), sigma=100, nsteps=100, nwalkers=4, use_mh='MH', parallel=False, seed=None, verbose=True)

        elif algorithm == 'pyevolve':
            self.fitting_test(H = 10, LL = 20, Beta1 = 92)

    def fitting_test(self, H = 10, LL = 20, Beta1 = 92):

        Beta= np.full((self.number_trapezoid), Beta1)
        DW, I0, Bkg = 10, 10, 10

        initiale_value = []
        initiale_value.append(int(DW)), initiale_value.append(int(I0)), initiale_value.append(int(Bkg))
        initiale_value.append(int(H)), initiale_value.append(int(LL))

        for i in Beta:
            initiale_value.append(int(i))

        self.residual(initiale_value)
        self.update_right_widget()

        self.best_score = 20
        self.num_param = int(len(initiale_value))

        genome = G1DList.G1DList(self.num_param)
        genome.setParams(rangemin=0, rangemax=1000)
        genome.evaluator.set(self.residual)

        ga = GSimpleGA.GSimpleGA(genome)
        ga.selector.set(Selectors.GRouletteWheel)
        ga.setGenerations(300)
        ga.setMinimax(Consts.minimaxType["minimize"])

        #ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
        #sqlite_adapter = DBAdapters.DBSQLite(identify="ex1", resetDB=True)
        #ga.setDBAdapter(sqlite_adapter)

        ga.stepCallback.set(self.evolve_callback)
        ga.evolve()

        print ga.bestIndividual()
        best = ga.bestIndividual()
        print(best.genomeList, best.score)

        self.residual(best.genomeList, 'True')
        self.update_model()
        self.modelParameter = 5 + 0.02 * best.genomeList[3], 20 + 0.04 * best.genomeList[4], 70 + 0.04 * \
                              best.genomeList[5], best.score
        self.update_right_widget()

    def evolve_callback(self, ga_engine):
        generation = ga_engine.getCurrentGeneration()
        if generation % 10 == 0:
            print "Current generation: %d" % (generation,)
            best = ga_engine.bestIndividual()
            if best.score < self.best_score:
                self.residual(best.genomeList, 'True')
                self.update_profile('True')
                self.modelParameter = 5 + 0.02 * best.genomeList[3], 20 + 0.04 * best.genomeList[4], 70 + 0.04 * best.genomeList[5], best.score
                H, LL = 5 + 0.02 * best.genomeList[3], 20 + 0.04 * best.genomeList[4]
                Beta = np.zeros((self.num_param - 5))
                for i in range(0, self.number_trapezoid, 1):
                    Beta[i] = 70 + 0.04 * best.genomeList[5 + i]
                Obj = np.rot90(simulation.multipyramid(H, LL, Beta, 500, 500), 3)
                self.modelImage = Obj[150: 350, 350: 500]
                self.update_model()
                self.update_right_widget()
                self.best_score = best.score

    #Need to be test to stop number points / goodnes of the fit/ convergence of the residual
    def ConvergenceCriteria(self, ga_engine):
        """ Terminate the evolution when the population have converged
        Example:
            ga_engine.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
        """
        pop = ga_engine.getPopulation()
        return pop[0] == pop[len(pop) - 1]

    def FitnessStatsCriteria(self, ga_engine):
        """ Terminate the evoltion based on the fitness stats
        Example:
            ga_engine.terminationCriteria.set(GSimpleGA.FitnessStatsCriteria)
        """
        stats = ga_engine.getStatistics()
        if stats["fitMax"] == stats["fitMin"]:
            if stats["fitAve"] == stats["fitMax"]:
                return True
        return False

    def update_model(self):
        self.sigDrawModel.emit(self)

    def update_right_widget(self):
        self.sigDrawParam.emit(self)

    def update_profile_ini(self):
        for order in range(0, len(self.Qxexp), 1):
            self.Qxexp[order] -= min(self.Qxexp[order])
            self.Qxexp[order] /= max(self.Qxexp[order])
            self.Qxexp[order] += order + 1

            guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders[order].setData, self.Q__Z[order], np.log(self.Qxexp[order]))


    def update_profile(self, plot = 'False'):
        for order in range(0, len(self.Qxexp), 1):
            self.Qxexp[order] -= min(self.Qxexp[order])
            self.Qxexp[order] /= max(self.Qxexp[order])
            self.Qxexp[order] += order + 1

            self.Qxfit[order] -= min(self.Qxfit[order])
            self.Qxfit[order] /= max(self.Qxfit[order])
            self.Qxfit[order] += order + 1


            if plot == 'True':
                guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders[order].setData, self.Q__Z[order], np.log(self.Qxexp[order]))
                guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders1[order].setData, self.Q__Z[order], np.log(self.Qxfit[order]))

    #@debugtools.timeit
    def residual(self, p, test = 'False', plot_mode=False):
        DW = 0.0001 * p[0]
        I0 = 0.01 * p[1]
        Bkg = 0.01 * p[2]
        H = 5 + 0.02 * p[3]
        LL = 20 + 0.04 * p[4]
        Beta = []

        for i in range(5, len(p), 1):
            Beta.append(70 + 0.04 * p[i])
        Beta = np.array(Beta)

        self.Qxfit = self.SL_model1(H, LL, Beta, DW, I0, Bkg)
        self.update_profile(test)

        res = 0

        for i in range(0, len(self.Qxexp), 1):
            res += fitting.log_error(self.Qxexp[i], self.Qxfit[i])

        return res

    def SL_model1(self, H, LL, Beta, DW_factor=0, I0=1, Bk=0):
        qy = self.Q__Z
        qz = self.q
        langle = np.deg2rad(np.asarray(Beta))
        rangle = np.deg2rad(np.asarray(Beta))
        self.Qxfit = []
        self.Qxfitc = []
        for i in range(len(qz)):
            self.Qxfit.append(simulation.stacked_trapezoids(qz[i], qy[i], 0, LL, H, langle, rangle))

        self.Qxfitc = fitting.corrections_DWI0Bk(self.Qxfit, DW_factor, I0, Bk, self.q, self.Q__Z)
        return self.Qxfitc

class CDRawWidget(pg.ImageView):
    pass

class CDCartoWidget(pg.ImageView):
    pass

class CDModelWidget(pg.PlotWidget):
    def __init__(self):
        super(CDModelWidget, self).__init__()
        self.addLegend()
        self.orders = []
        self.orders1 = []
        for i,color in enumerate('gyrbcmkgyr'):
            self.orders.append(self.plot([],pen=pg.mkPen(color, width=2), name = 'Order ' + str(i)))
            self.orders1.append(self.plot([], pen=pg.mkPen(color, width=2, style=QtCore.Qt.DashLine), name='Order ' + str(i)))

class CDProfileWidget(pg.ImageView):
    pass

