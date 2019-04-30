import __future__
import os, sys, time
from xicam.plugins.cdsaxs import simulation, fitting, cdsaxs
from scipy.fftpack import *
from numpy.fft import *
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
import matplotlib.pyplot as plt

from PySide import QtGui, QtCore

from random import randrange
# import cPickle as pickle
import psutil
import multiprocessing
from collections import deque
from itertools import repeat
import emcee
import deap.base as deap_base
from deap import creator, tools
from deap import cma as cmaes

creator.create('FitnessMin', deap_base.Fitness, weights=(-1.0,))  # want to minimize fitness
creator.create('Individual', list, fitness=creator.FitnessMin)

exp_data = None
Qxfit = None
Q__z = None
q = None


def residual(p, test='False', plot_mode=False):
    simp = fittingp_to_simp(p)
    if simp is None:
        return fix_fitness_cmaes(np.inf)
    DW = simp[0]
    I0 = simp[1]
    Bkg = simp[2]
    H = simp[3]
    LL = simp[4]
    Beta = simp[5:]

    Beta = np.array(Beta)

    print(H, LL, Beta)

    Qxfit = SL_model1(H, LL, Beta, DW, I0, Bkg)

    res = 0
    for i in range(0, len(exp_data), 1):
        res += fitting.log_error(exp_data[i], Qxfit[i])

    return fix_fitness_cmaes(res)


def fittingp_to_simp(fittingp):
    # DW, I0, Bk, H, LL, *Beta[5] = simp
    # values assume initial fittingp centered at 0 and std. dev. of 100
    # multiples = np.array([0.001, 0.1, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    multiples = [0.0001, 0.01, 0.01, 0.02, 0.03] + [0.04] * (len(adds) - 5)
    simp = np.asarray(multiples) * np.asarray(fittingp) + adds
    if np.any(simp[:5] < 0):
        return None
    if np.any(simp[5:] < 0) or np.any(simp[5:] > 180):
        return None
    return simp


def SL_model1(H, LL, Beta, DW_factor=0, I0=1, Bk=0):
    qy = Q__z
    qz = q
    langle = np.deg2rad(np.asarray(Beta))
    rangle = np.deg2rad(np.asarray(Beta))
    Qxfit = []
    Qxfitc = []
    for i in range(len(qz)):
        Qxfit.append(simulation.stacked_trapezoids(qz[i], qy[i], 0, LL, H, langle, rangle))

    Qxfitc = fitting.corrections_DWI0Bk(Qxfit, DW_factor, I0, Bk, q, Q__z)
    return Qxfitc


def fix_fitness_cmaes(fitness):
    """cmaes accepts the individuals with the lowest fitness, doesn't matter degree to which they are lower"""

    return fitness


def fix_fitness_mcmc(fitness):
    """
    Metropolis-Hastings criterion: acceptance probability equal to ratio between P(new)/P(old)
    where P is proportional to probability distribution we want to find
    for our case we assume that probability of our parameters being the best is proportional to a Gaussian centered at fitness=0
    where fitness can be log, abs, squared error, etc.
    emcee expects the fitness function to return ln(P(new)), P(old) is auto-calculated
    """
    c = 1e-1  # empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual, higher c increases acceptance rate
    return -fitness / c
    # return -0.5 * fitness ** 2 / c ** 2


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
            {'name': 'test', 'type': 'group', 'children': [
                {'name': 'Phi_min', 'type': 'float'},
                {'name': 'Phi_max', 'type': 'float'},
                {'name': 'Phi_step', 'type': 'float'},
                {'name': 'H', 'type': 'float'},
                {'name': 'w0', 'type': 'float'},
                {'name': 'Beta', 'type': 'float'},
                {'name': 'Num_trap', 'type': 'float'},
                {'name': 'Run1', 'type': 'action'}]},
            {'name': 'test1', 'type': 'group', 'children': [
                {'name': 'H_fit', 'type': 'float', 'readonly': True},
                {'name': 'w0_fit', 'type': 'float', 'readonly': True},
                {'name': 'Beta_fit', 'type': 'float', 'readonly': True},
                {'name': 'f_val', 'type': 'float', 'readonly': True}]}])

        self.parametertree.setParameters(self.param, showTop=False)
        self.param.param('test', 'Run1').sigActivated.connect(self.fit)

        super(plugin, self).__init__(*args, **kwargs)

    def update_model(self, widget):
        guiinvoker.invoke_in_main_thread(self.bottomwidget.setImage, widget.modelImage)

    def update_right_widget(self, widget):
        H, LL, beta, f_val = widget.modelParameter
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'H_fit').setValue, H)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'w0_fit').setValue, LL)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'Beta_fit').setValue, beta)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'f_val').setValue, f_val)

    def fit(self):
        activeSet = self.getCurrentTab()
        activeSet.setCurrentWidget(activeSet.CDModelWidget)
        H, w0, Beta1, Num_trap = self.param['test', 'H'], self.param['test', 'w0'], self.param['test', 'Beta'], \
                                 self.param['test', 'Num_trap']
        fitrunnable = threads.RunnableMethod(self.getCurrentTab().fitting_test1, method_args=(H, w0, Beta1, Num_trap))
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

        Phi_min, Phi_max, Phi_step = self.param['test', 'Phi_min'], self.param['test', 'Phi_max'], self.param[
            'test', 'Phi_step']
        fitrunnable = threads.RunnableMethod(self.getCurrentTab().loadRAW, method_args=(Phi_min, Phi_max, Phi_step))
        threads.add_to_queue(fitrunnable)

    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()
        self.getCurrentTab().sigDrawModel.connect(self.update_model)
        self.getCurrentTab().sigDrawParam.connect(self.update_right_widget)

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

        self.CDRawWidget = CDRawWidget()
        self.CDCartoWidget = CDCartoWidget()
        self.CDModelWidget = CDModelWidget()

        self.addTab(self.CDRawWidget, 'RAW')
        self.addTab(self.CDCartoWidget, 'Cartography')
        self.addTab(self.CDModelWidget, 'Model')

        self.setTabPosition(self.South)
        self.setTabShape(self.Triangular)

        self.src = src

        # self.loadRAW()

    def loadRAW(self, Phi_min=-45, Phi_max=45, Phi_step=1):
        center = [552, 225]
        maxR, minR = 250, 0
        nb_pixel = maxR - minR
        maxy, miny = -40, 40

        data = []
        profiles = []
        x, y = np.indices(loader.loadimage(self.src[0]).shape)
        x_1, y_1 = x - center[0], y - center[1]
        rmincut = (y_1 > minR)
        rmaxcut = (y_1 < maxR)
        thetamincut = (x_1 > -10)
        thetamaxcut = (x_1 < 10)
        cutmask = (rmincut * rmaxcut * thetamincut * thetamaxcut).T

        for file in self.src:
            img = np.rot90(loader.loadimage(file), 1)
            profile = np.sum(cutmask * img, axis=1)
            # profile = cutmask * img
            profiles.append(profile[center[1] + minR: center[1] + maxR])
            # profiles.append(profile[center[1]:])
            data.append(img)

        data = np.stack(data)
        data = np.log(data - data.min() + 1.)

        self.maskimage = pg.ImageItem(opacity=.25)
        self.CDRawWidget.view.addItem(self.maskimage)
        invmask = 1 - cutmask
        self.maskimage.setImage(
            np.dstack((invmask, np.zeros_like(invmask), np.zeros_like(invmask), invmask)).astype(np.float), opacity=.5)
        self.CDRawWidget.setImage(data)

        # 1Conversion to q spqce
        pixel_size, sample_detector_distance, wavelength = 172 * 10 ** -6, 5., 0.09184
        self.QxyiData = cdsaxs.generate_carto(profiles, nb_pixel, Phi_min, Phi_step, pixel_size,
                                              sample_detector_distance, wavelength, center[0])

        # Intensity correction
        substratethickness, substrateattenuation = 700 * 10 ** -6, 200 * 10 ** -6
        self.QxyiDatacor = cdsaxs.correc_Iexp(self.QxyiData, substratethickness, substrateattenuation)

        # interpolation and Plotting carthography
        sampling_size = (400, 400)
        # img = cdsaxs.inter_carto(self.QxyiData)
        self.img, qk_shift = cdsaxs.interpolation(self.QxyiDatacor, sampling_size)
        np.save('/Users/guillaumefreychet/Desktop/carto_cxro.npy', self.img)
        self.CDCartoWidget.setImage(self.img)

        # Definition of the variables
        self.q = []
        self.Qxexp = []
        self.Qxfit = []
        self.Q__Z = []

        # Extraction of nb of profile + their positions
        profile_carto = np.mean(np.nan_to_num(self.img), axis=1)
        self.q, self.Qxexp, self.Q__Z = cdsaxs.find_peaks(profile_carto, self.QxyiDatacor, wavelength, nb_pixel,
                                                          pixel_size, sample_detector_distance)

        np.save('/Users/guillaumefreychet/Desktop/test', self.Qxexp)
        self.update_profile_ini()
        self.SL_model1(10, 40, np.array([95]))
        self.maxres = 0

    def fitting_test1(self, H=10, LL=20, Beta1=70, Num_trap=5, DW=0.1, I0=1, Bk=0):  # these are simp not fittingp

        self.number_trapezoid = int(Num_trap)
        beta = []
        for i in range(0, self.number_trapezoid, 1):
            beta.append(Beta1)
        Beta = np.array(beta)

        initiale_value = [DW, I0, Bk]
        initiale_value.append(int(H))
        initiale_value.append(int(LL))

        for i in Beta:
            initiale_value.append(int(i))

        self.adds = np.asarray(initiale_value)

        # set globals
        global q, Q__z, exp_data, adds, Qxfit

        adds = self.adds
        exp_data = self.Qxexp
        q = self.q
        Q__z = self.Q__Z

        self.fix_fitness = fix_fitness_cmaes
        self.residual(np.zeros(len(initiale_value)), test='True')
        # self.cmaes(sigma=200, ngen=200, popsize=100, mu=10, N=len(initiale_value), restarts=0, verbose=False, tolhistfun=5e-5, ftarget=None)
        self.cmaes(sigma=100, ngen=100, popsize=100, mu=10, N=len(initiale_value), restarts=0, verbose=False,
                   tolhistfun=5e-5, ftarget=None)
        self.residual(self.best_uncorr, test='False')
        self.residual(self.best_uncorr, test='True')

        print(self.best_uncorr)
        print(self.best_corr)

        print('OK')
        # initiale_value1 = [self.best_corr[0], self.best_corr[1], self.best_corr[2], self.best_corr[3], self.best_corr[4], self.best_corr[5], self.best_corr[6], self.best_corr[7], self.best_corr[8], self.best_corr[9]]
        # self.adds = np.asarray(initiale_value1)
        '''
        self.fix_fitness = fix_fitness_mcmc
        self.mcmc(N=len(self.best_corr), sigma=1000, nsteps=1000, nwalkers=100, use_mh='MH', parallel=True, seed=None,
                  verbose=True)
        #self.residual(self.best_corr, test='True')
        print('Done')
        '''
        '''
        elif algorithm == 'mcmc':
            self.fix_fitness = self.fix_fitness_mcmc
            self.residual(np.zeros(len(initiale_value)), test='True')
            self.mcmc(N=len(initiale_value), sigma=100, nsteps=100, nwalkers=4, use_mh='MH', parallel=False, seed=None, verbose=True)
        '''

    def update_model(self):
        self.sigDrawModel.emit(self)

    def update_right_widget(self):
        self.sigDrawParam.emit(self)

    def update_profile_ini(self):
        for order in range(0, len(self.Qxexp), 1):
            self.Qxexp[order] -= min(self.Qxexp[order])
            self.Qxexp[order] /= max(self.Qxexp[order])
            self.Qxexp[order] += order + 1

            guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders[order].setData, self.Q__Z[order],
                                             np.log(self.Qxexp[order]))

    def update_profile(self, plot='False'):
        for order in range(0, len(self.Qxexp), 1):
            self.Qxexp[order] -= min(self.Qxexp[order])
            self.Qxexp[order] /= max(self.Qxexp[order])
            self.Qxexp[order] += order + 1

            self.Qxfit[order] -= min(self.Qxfit[order])
            self.Qxfit[order] /= max(self.Qxfit[order])
            self.Qxfit[order] += order + 1

            if plot == 'True':
                guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders[order].setData, self.Q__Z[order],
                                                 np.log(self.Qxexp[order]))
                guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders1[order].setData, self.Q__Z[order],
                                                 np.log(self.Qxfit[order]))

    # @debugtools.timeit
    def residual(self, p, test='False', plot_mode=False):
        simp = fittingp_to_simp(p)
        if simp is None:
            return fix_fitness_cmaes(np.inf)
        DW = simp[0]
        I0 = simp[1]
        Bkg = simp[2]
        H = simp[3]
        LL = simp[4]
        Beta = simp[5:]

        Beta = np.array(Beta)

        self.Qxfit = SL_model1(H, LL, Beta, DW, I0, Bkg)
        self.update_profile(test)

        res = 0

        for i in range(0, len(self.Qxexp), 1):
            res += fitting.log_error(self.Qxexp[i], self.Qxfit[i])

        return fix_fitness_cmaes(res)

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

    @staticmethod
    def fix_fitness_cmaes(fitness):
        """cmaes accepts the individuals with the lowest fitness, doesn't matter degree to which they are lower"""
        return fitness,

    def fittingp_to_simp(self, fittingp):
        # DW, I0, Bk, H, LL, *Beta[5] = simp
        # values assume initial fittingp centered at 0 and std. dev. of 100
        multiples = [0.0001, 0.01, 0.01, 0.02, 0.03] + [0.04] * (len(self.adds) - 5)
        simp = np.asarray(multiples) * np.asarray(fittingp) + self.adds
        if np.any(simp[:5] < 0):
            return None
        if np.any(simp[5:] < 0) or np.any(simp[5:] > 180):
            return None
        return simp

    def cmaes(self, sigma, ngen, popsize, mu, N, restarts, verbose, tolhistfun, ftarget, restart_from_best=False):
        """Modified from deap/algorithms.py to return population_list instead of final population and use additional termination criteria

        Returns:
            population_list: list of (list of individuals (lists), length popsize), length ngen
            logbook: list of dicts, length ngen, contains stats for each generation
        """
        toolbox = deap_base.Toolbox()
        toolbox.register('evaluate', self.residual)
        # parallel = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(parallel)
        # toolbox.register('map', pool.map)
        # last_time = time.perf_counter()
        process = psutil.Process()
        print('{} CPUs in node'.format(multiprocessing.cpu_count()))
        print('pid:{}'.format(os.getpid()))
        print(psutil.virtual_memory())
        halloffame = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', lambda x: np.mean(np.asarray(x)[np.isfinite(x)]) if np.asarray(x)[np.isfinite(
            x)].size != 0 else None)
        stats.register('std', lambda x: np.std(np.asarray(x)[np.isfinite(x)]) if np.asarray(x)[np.isfinite(
            x)].size != 0 else None)
        stats.register('min', lambda x: np.min(np.asarray(x)[np.isfinite(x)]) if np.asarray(x)[np.isfinite(
            x)].size != 0 else None)
        stats.register('max', lambda x: np.max(np.asarray(x)[np.isfinite(x)]) if np.asarray(x)[np.isfinite(
            x)].size != 0 else None)
        stats.register('fin', lambda x: np.sum(np.isfinite(x)) / np.size(x))
        # stats.register('cumtime', lambda x: time.perf_counter() - last_time)
        stats.register('rss_MB', lambda x: process.memory_info().rss / 1048576)
        stats.register('vms_MB', lambda x: process.memory_info().vms / 1048576)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        population_list = []
        kwargs = {'lambda_': popsize if popsize is not None else int(4 + 3 * np.log(N))}
        if mu is not None:
            kwargs['mu'] = mu
        initial_individual = [0] * N
        morestats = {}
        morestats['sigma_gen'] = []
        morestats['axis_ratio'] = []  # ratio of min and max scaling at each generation
        morestats['diagD'] = []  # scaling of each parameter at each generation (eigenvalues of covariance matrix)
        morestats['ps'] = []
        allbreak = False
        checkpoint_num = 0

        for restart in range(restarts + 1):
            if allbreak:
                break
            if restart != 0:
                kwargs['lambda_'] *= 2
                print('Doubled popsize')
                if restart_from_best:
                    initial_individual = halloffame[0]
            # type of strategy: (parents, children) = (mu/mu_w, popsize), selection takes place among offspring only
            strategy = cmaes.Strategy(centroid=initial_individual, sigma=sigma, **kwargs)
            # The CMA-ES One Plus Lambda algorithm takes a initialized parent as argument
            #    parent = creator.Individual(initial_individual)
            #    parent.fitness.values = toolbox.evaluate(parent)
            #    strategy = cmaes.StrategyOnePlusLambda(parent=parent, sigma=sigma, lambda_=popsize)
            toolbox.register('generate', strategy.generate, creator.Individual)
            toolbox.register('update', strategy.update)

            last_best_fitnesses = deque(maxlen=10 + int(np.ceil(30 * N / kwargs['lambda_'])))
            cur_gen = 0
            # fewer generations when popsize is doubled (unless fixed ngen is specified)
            ngen_ = ngen if ngen is not None else int(100 + 50 * (N + 3) ** 2 / kwargs['lambda_'] ** 0.5)
            while cur_gen < ngen_:
                cur_gen += 1
                sys.stdout.flush()
                # Generate a new population
                population = toolbox.generate()
                population_list.append(population)
                # Evaluate the individuals
                fitnesses = toolbox.map(toolbox.evaluate, population)
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = (fit,)  # tuple of length 1

                halloffame.update(population)
                # if cur_gen % 10 == 0:  # print best every 10 generations
                #     best = np.copy(halloffame[0])
                #     for i in range(len(best)):
                #         best[i] = self.scaling[i][1] * halloffame[0][i] + self.correction[i]
                #     print(*['{0}:{1:.3g}'.format(i, j) for i, j in zip(self.labels, best)], sep=', ')

                # Update the strategy with the evaluated individuals
                toolbox.update(population)

                record = stats.compile(population) if stats is not None else {}
                logbook.record(gen=cur_gen, nevals=len(population), **record)
                if verbose:
                    print(logbook.stream)
                morestats['sigma_gen'].append(strategy.sigma)
                morestats['axis_ratio'].append(max(strategy.diagD) ** 2 / min(strategy.diagD) ** 2)
                morestats['diagD'].append(strategy.diagD ** 2)
                morestats['ps'].append(strategy.ps)

                last_best_fitnesses.append(record['min'])
                if (ftarget is not None) and record['min'] <= ftarget:
                    print('Iteration terminated due to ftarget criterion after {} gens'.format(cur_gen))
                    allbreak = True
                    break
                if (tolhistfun is not None) and (len(last_best_fitnesses) == last_best_fitnesses.maxlen) and (
                        max(last_best_fitnesses) - min(last_best_fitnesses) < tolhistfun):
                    print('Iteration terminated due to tolhistfun criterion after {} gens'.format(cur_gen))
                    break
                if os.path.exists('break'):
                    print('Iteration terminated due to user after {} gens'.format(cur_gen))
                    break
                if os.path.exists('allbreak'):
                    print('Iteration terminated due to user after {} gens'.format(cur_gen))
                    allbreak = True
                    break
                if os.path.exists('checkpoint{}'.format(checkpoint_num)):
                    # saves current state of self as pickle and continues
                    self.logbook, self.morestats, self.strategy = logbook, morestats, strategy

                    self.minfitness_each_gen = self.logbook.select('min')
                    self.best_uncorr = halloffame[0]
                    self.best_fitness = self.best_uncorr.fitness.values[0]

                    # simulate best individual one more time to print fitness and save sim_list
                    # optionally plot sim and exp, plot trapezoids of best individual
                    # self.fitness_individual(self.best_uncorr, plot_on=False, print_fitness=True)
                    residual(self.Qxexp, self.q, self.Q__Z)(self.best_uncorr)

                    # make population dataframe, order of rows is first generation for all children, then second generation for all children...
                    # make and print best individual series
                    population_array = np.array(
                        [list(individual) for generation in population_list for individual in generation])
                    fitness_array = np.array(
                        [individual.fitness.values[0] for generation in population_list for individual in generation])
                    self.make_population_frame_best(population_array, fitness_array)
                    print(self.best)
                    filename = 'checkpoint{}.pickle'.format(checkpoint_num)
                    with open(filename, 'wb') as f:
                        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                        print('saved to ' + os.path.join(os.getcwd(), filename))
                    checkpoint_num += 1
            else:
                print('Iteration terminated due to ngen criterion after {} gens'.format(cur_gen))

        # pool.close()
        self.logbook = logbook
        self.morestats = morestats
        self.strategy = strategy

        self.minfitness_each_gen = self.logbook.select('min')
        self.best_uncorr = halloffame[0]  # np.abs(halloffame[0])
        self.best_fitness = halloffame[0].fitness.values[0]
        self.best_corr = fittingp_to_simp(self.best_uncorr)
        self.residual(self.best_corr, test='True')
        print(self.best_corr, self.best_fitness)
        # make population dataframe, order of rows is first generation for all children, then second generation for all children...
        self.population_array = np.array(
            [list(individual) for generation in population_list for individual in generation])
        self.population_array = self.fittingp_to_simp(self.population_array)
        self.fitness_array = np.array(
            [individual.fitness.values[0] for generation in population_list for individual in generation])
        # self.population_frame = pd.DataFrame(np.column_stack((self.population_array, self.fitness_array)))

    def mcmc(self, N, sigma, nsteps, nwalkers, use_mh=False, parallel=True, seed=None, verbose=True):
        """Fit with emcee package's implementation of MCMC algorithm and place into instance of self
        Calls fitness_individual many times, then calls make_population_frame_best

        Attributes:
            best_uncorr: best uncorrected individual
            best_fitness: scalar
            minfitness_each_gen: length ngen
            sampler: instance of emcee.Sampler with detailed output of algorithm

        Args:
            self: instance of Run
            sigma: array or scalar, initial standard deviation for each parameter
            nsteps: number of steps
            nwalkers: number of walkers
            use_mh: True for Metropolis-Hastings proposal and ensemble sampler, False for ensemble sampler, 'MH' for Metropolis-Hastings proposal and sampler
            parallel: False for no parallel, True for cpu_count() processes, or int to specify number of processes, or 'scoop' for cluster
            plot_on: whether to plot fitness, best trapezoids and sim and exp scattering
            seed: seed for random number generator
        """

        def do_verbose(i, sampler):
            if (i % 100) == 0:
                print(i)
                if hasattr(sampler, 'acceptance_fraction'):
                    print('Acceptance fraction: ' + str(np.mean(sampler.acceptance_fraction)))
                else:
                    print('Acceptance fraction: ' + str(np.mean([sampler.acceptance_fraction for sampler in sampler])))
                sys.stdout.flush()
            if (i % 1000) == 0:
                process = psutil.Process()
                # print('time elapsed: {} min'.format((time.perf_counter() - last_time) / 60))
                print('rss_MB: {}'.format(process.memory_info().rss / 1048576))
                print('vms_MB: {}'.format(process.memory_info().vms / 1048576))

        def get_sampler(a):
            walker_num, N, sigma, nsteps, residual, verbose = a
            cov = np.identity(N) * sigma ** 2
            sampler = emcee.MHSampler(cov.copy(), cov.shape[0], residual, args=[False, False])
            for i, _ in enumerate(sampler.sample(np.zeros(N), None, None, iterations=nsteps)):
                if verbose and (walker_num == 0):
                    do_verbose(i, sampler)
            return sampler

        '''
        # set globals
        global adds
        adds = self.adds
        '''

        c = 1e-1  # empirical factor to modify mcmc acceptance rate, makes printed fitness different than actual, higher c increases acceptance rate
        # last_time = time.perf_counter()
        print('{} CPUs in node'.format(multiprocessing.cpu_count()))
        print('pid:{}'.format(os.getpid()))
        print(psutil.virtual_memory())
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        if seed is None:
            seed = randrange(2 ** 32)
        self.seed = seed
        np.random.seed(seed)
        self.fix_fitness = fix_fitness_mcmc

        if hasattr(sigma, '__len__'):
            self.sigma = sigma
        else:
            self.sigma = [sigma] * N

        if parallel == 'scoop':
            if use_mh != 'MH':
                raise NotImplementedError
            self.parallel = multiprocessing.cpu_count()
            from scoop import futures
            map_MH = futures.map
        elif parallel is True:
            self.parallel = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(self.parallel)
            map_MH = pool.map

        self.parallel = 1
        map_MH = map

        if use_mh == 'MH':
            samplers = list(map_MH(get_sampler, zip(range(nwalkers), repeat(N), self.sigma, repeat(nsteps),
                                                    repeat(residual), repeat(verbose))))
            chain = np.dstack(sampler.chain for sampler in samplers)
            s = chain.shape
            flatchain = np.transpose(chain, axes=[0, 2, 1]).reshape(s[0] * s[2], s[1])
            lnprobability = np.vstack(sampler.lnprobability for sampler in samplers)
            flatlnprobability = lnprobability.transpose().flatten()
            self.minfitness_each_gen = np.min(-lnprobability * c, axis=0)
        else:
            print('{} parameters'.format(N))
            if use_mh:
                individuals = [np.zeros(N) for _ in range(nwalkers)]
                mh_proposal = emcee.utils.MH_proposal_axisaligned(self.sigma)
                sampler = emcee.EnsembleSampler(
                    nwalkers, N, residual, args=[False, False], threads=self.parallel)
                for i, _ in enumerate(
                        sampler.sample(individuals, None, None, iterations=nsteps, mh_proposal=mh_proposal)):
                    if verbose:
                        do_verbose(i, sampler)
            else:
                individuals = [[np.random.normal(loc=0, scale=s) for s in self.sigma] for _ in range(nwalkers)]
                sampler = emcee.EnsembleSampler(
                    nwalkers, N, residual, args=[False, False], threads=self.parallel)
                for i, _ in enumerate(sampler.sample(individuals, None, None, iterations=nsteps)):
                    if verbose:
                        do_verbose(i, sampler)
            s = sampler.chain.shape
            flatchain = np.transpose(sampler.chain, axes=[1, 0, 2]).reshape(s[0] * s[1], s[2])
            flatlnprobability = sampler.lnprobability.transpose().flatten()
            self.minfitness_each_gen = np.min(-sampler.lnprobability * c, axis=0)

        if 'pool' in locals():
            pool.close()

        # flatchain has shape (nwalkers * nsteps, N)
        # flatlnprobability has shape (nwalkers * nsteps,)
        # flatchain and flatlnprobability list first step of all walkers, then second step of all walkers...

        # sampler.flatchain and sampler.flatlnprobability (made by package) list all steps of first walker, then all steps of second walker...
        # but we don't want that

        flatfitness = -flatlnprobability * c
        best_index = np.argmin(flatfitness)
        self.best_fitness = flatfitness[best_index]
        self.best_uncorr = flatchain[best_index]
        self.best_corr = self.fittingp_to_simp(self.best_uncorr)
        self.residual(self.best_uncorr, test='True')
        # can't make sampler attribute before run_mcmc, pickling error
        self.sampler = samplers if use_mh == 'MH' else sampler
        self.population_array = self.fittingp_to_simp(flatchain)
        '''
        self.population_frame = pd.DataFrame(np.column_stack((self.population_array, flatfitness)))
        gen_start = 0
        gen_stop = len(flatfitness)
        gen_step = 1
        popsize = int(self.population_frame.shape[0] / len(flatfitness))
        index = []
        for i in range(gen_start, gen_stop, gen_step):
            index.extend(list(range(i * popsize, (i + 1) * popsize)))
        resampled_frame = self.population_frame.iloc[index]
        self.stats = resampled_frame.describe()
        self.stats.to_csv('C:/Users/cdl/Desktop/test.csv')
        '''


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
        for i, color in enumerate('gyrbcmkgyr'):
            self.orders.append(self.plot([], pen=pg.mkPen(color), name='Order ' + str(i)))
            self.orders1.append(self.plot([], pen=pg.mkPen(color), name='Order ' + str(i)))


class CDProfileWidget(pg.ImageView):
    pass
