import __future__
import os, sys, time
import cdrsoxs, fitting, simulation
import pyqtgraph as pg
import numpy as np
from xicam.plugins import base, widgets
from xicam import threads
from modpkgs import guiinvoker
from functools import partial
from PySide import QtGui, QtCore
import multiprocessing
import deap.base as deap_base
import deap.base as deap_base
from deap import creator
from scipy import interpolate

creator.create('FitnessMin', deap_base.Fitness, weights=(-1.0,))  # want to minimize fitness
creator.create('Individual', list, fitness=creator.FitnessMin)

class plugin(base.plugin):
    name = "CDSOXS"

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.rightwidget = self.parametertree = pg.parametertree.ParameterTree()
        self.topwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabClose)
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.bottomwidget = CDLineProfileWidget()

        # Setup parametertree

        self.param = pg.parametertree.Parameter.create(name='params', type='group', children=[
            {'name': 'User_input', 'type': 'group', 'children': [
                {'name': '1st_pic', 'type': 'float'},
                {'name': 'Pitch', 'type': 'float'},
                {'name': 'Num_trap', 'type': 'float'},
                {'name': 'H', 'type': 'float'},
                {'name': 'w0', 'type': 'float'},
                {'name': 'Beta', 'type': 'float'},
                {'name': 'Simulation', 'type': 'action'}]},
            {'name': 'Fit_output', 'type': 'group', 'children': [
                {'name': 'H_fit', 'type': 'float', 'readonly': True},
                {'name': 'w0_fit', 'type': 'float', 'readonly': True},
                {'name': 'Beta_fit', 'type': 'float', 'readonly': True},
                {'name': 'f_val', 'type': 'float', 'readonly': True}]}])

        self.parametertree.setParameters(self.param, showTop=False)
        self.param.param('User_input', 'Simulation').sigActivated.connect(self.fit)

        super(plugin, self).__init__(*args, **kwargs)

    def update_model(self, widget):
        guiinvoker.invoke_in_main_thread(self.bottomwidget.plotLineProfile, *widget.modelParameters)

    def update_right_widget(self, widget):
        H, LL, beta, f_val = widget.modelParameter
        guiinvoker.invoke_in_main_thread(self.param.param('Fit_output', 'H_fit').setValue, H)
        guiinvoker.invoke_in_main_thread(self.param.param('Fit_output', 'w0_fit').setValue, LL)
        guiinvoker.invoke_in_main_thread(self.param.param('Fit_output', 'Beta_fit').setValue, beta)
        guiinvoker.invoke_in_main_thread(self.param.param('Fit_output', 'f_val').setValue, f_val)

    def fit(self):
        activeSet = self.getCurrentTab()
        activeSet.setCurrentWidget(activeSet.CDModelWidget)
        H, w0, Beta1, Num_trap = self.param['User_input', 'H'], self.param['User_input', 'w0'], self.param['User_input', 'Beta'], \
                                 self.param['User_input', 'Num_trap']
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

        fst_pic, Pitch = self.param['User_input', '1st_pic'], self.param['User_input', 'Pitch']
        fitrunnable = threads.RunnableMethod(self.getCurrentTab().loadRAW, method_args=(fst_pic, Pitch))
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

    def loadRAW(self, fst_pic = 50, Pitch = 100):
        """ This function is launched when the user select the data. 4 parameters have to be entered manually by the user (Phi, ..., pitch) => to change when angles are contained in the header
        Here this fucntion will process all the data treatment in order to dislay the experimental raw data, the qx,qz cartography and the peak intensity profile

        Parameters
        ----------
        Phi_min, Phi_max, Phi_step (float32): first/last/step angles (to be turned into default value)
        Pitch (float32): pitch

        Returns
        -------
        Display the experimental raw data (as a stack of image in CDRawWidget), the (qx,qz) cartography (in the CDCartoWidget) and the peak intensity profile (in the CDModelWidget)

        """
        #substratethickness, substrateattenuation = 700 * 10 ** -6, 200 * 10 ** -6

        #11012 beamlline : Reading of the detector ???
        pixel_size, sample_detector_distance, wavelength = 27 * 10 ** -6, 0.15, 2.36
        substratethickness, substrateattenuation = 200 * 10 ** -9, 0.5 * 10 ** -3

        self.qx, self.qz, self.I = [], [], []
        self.Qx, self.Qz, self.In = [],[],[]

        file = [val for val in self.src]

        #find a smart way to calculate q-pitch : Find theta = 0 => procedure doen in test.....
        q_pitch = np.abs(2. * np.pi / Pitch)

        # Parallelization
        pool = multiprocessing.Pool()
        func = partial(cdrsoxs.test, substratethickness, substrateattenuation, Pitch, q_pitch, fst_pic)
        a = file
        #b = [list(elem) for elem in a]
        #I_cor, img1, q_x, q_z, Qxexp, Q__Z, I_peaks = zip(*pool.map(func, b))
        I_cor, img1, q_x, q_z, Qxexp, Q__Z, I_peaks = zip(*map(func, a))
        np.save('/Users/guillaumefreychet/Desktop/i_ini.npy', I_peaks)


        print(np.shape(I_peaks))
        pool.close()

        data = np.stack(img1)
        data = np.log(data - data.min() + 1.)
        self.CDRawWidget.setImage(data)

        I_peaks = [np.array(I_peaks)[:,i] for i in range(len(np.array(I_peaks)[0]))]

        threshold = max(map(max, np.array(I_peaks)))[0] /10000.
        column_max = map(max, I_peaks)
        ind = np.where(np.array([item for sublist in column_max for item in sublist]) > threshold)

        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        for i in ind[0]:
            self.qx.append(np.array([item for sublist in np.array(Qxexp)[:, i] for item in np.array(sublist)]))
            self.qz.append(np.array([item for sublist in np.array(Q__Z)[:, i] for item in np.array(sublist)]))
            self.I.append(np.array([item for sublist in np.array(I_peaks)[i, :] for item in np.array(sublist)]))
            #y = np.array([item for sublist in np.array(I_peaks)[i, :] for item in np.array(sublist)])

            #nans, x = nan_helper(y)
            #y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            #self.I.append(y)

        #unorganized file
        for i in range (0, len(self.Qx), 1):
            self.qx.append(np.array([item for item in zip(*sorted(zip(self.Qx[i], self.Qz[i], self.In[i]), key = lambda x: x[1]))[0]]))
            self.qz.append(np.array([item for item in zip(*sorted(zip(self.Qx[i], self.Qz[i], self.In[i]), key = lambda x: x[1]))[1]]))
            self.I.append(np.array([item for item in zip(*sorted(zip(self.Qx[i], self.Qz[i], self.In[i]), key = lambda x: x[1]))[2]]))

        np.save('/Users/guillaumefreychet/Desktop/qx.npy', self.qx)
        np.save('/Users/guillaumefreychet/Desktop/qz.npy', self.qz)
        np.save('/Users/guillaumefreychet/Desktop/i.npy', self.I)

        sampling_size = (400, 400)
        qx_carto = np.array([item for sublist in q_x for item in sublist])
        qz_carto = np.array([item for sublist in q_z for item in sublist])
        profiles = np.array([item for sublist in I_cor for item in sublist])

        self.img = cdrsoxs.interpolation(qx_carto, qz_carto, profiles, sampling_size)
        #Change interpolation

        #grid_x, grid_z = np.mgrid[0:200, 0:200]
        #grid_x = grid_x/200.*(max(qx_carto)-min(qx_carto))+ min(qx_carto)
        #grid_z = grid_z/200.*(max(qz_carto)-min(qz_carto)) + min(qz_carto)
        #self.img = interpolate.griddata(np.stack([qx_carto,qz_carto]).T,profiles,(grid_x,grid_z),method='linear', fill_value = 0)

        self.CDCartoWidget.setImage(self.img)

        #Display the experimental profiles
        self.update_profile_ini()
        self.maxres = 0

    def fitting_test1(self, H=10, LL=20, Beta=70, Num_trap=5, DW=0.11, I0=3, Bkg=1):  # these are simp not fittingp
        """ This function is launched when the user click on 'Simulation' button. 4 initial parameterd for the fit have to be entered manually by the user (height, linewidth, anle, number of trapezoid). The fitting algorythm is run with this function

        Parameters
        ----------
        H, LL, Beta (float32): Height, Linewidth and sidewall angle
        Num_trap (int): Number of trapezoid to descibe the line profile

        Returns
        -------
        Reach the best combination of parameters and display the simulated peak intensity, as well as the line profile

        """
        self.number_trapezoid = int(Num_trap)
        initiale_value = [DW, I0, Bkg, int(H), int(LL)] + [int(Beta) for i in range(0, self.number_trapezoid,1)]

        #self.fix_fitness = fitting.fix_fitness_cmaes
        #fix_fitness = fix_fitness_cmaes
        self.best_corr = initiale_value
        self.Qxfit = self.SL_model1(H, LL, np.array([int(Beta) for i in range(0, self.number_trapezoid,1)]), DW, I0, Bkg)
        self.update_profile()
        self.update_model()

        self.best_corr, best_fitness = fitting.cmaes(data=self.I, qx=self.qx, qz=self.qz, initial_guess=np.asarray(initiale_value), sigma=200, ngen=200, popsize=100, mu=10, N=len(initiale_value), restarts=0, verbose=False, tolhistfun=5e-5, ftarget=None)
        self.Qxfit = self.SL_model1(self.best_corr[3], self.best_corr[4], np.array([self.best_corr[i+5] for i in range(0, self.number_trapezoid,1)]), self.best_corr[0], self.best_corr[1], self.best_corr[2])
        print('angle', self.best_corr[5:])
        print('H', self.best_corr[4])
        print('LW', self.best_corr[3])
        self.update_profile()
        self.update_model()

        #To try
        #fitting.mcmc(data=self.I, qx=self.qx, qz=self.qz, initial_guess=np.asarray(self.best_corr[3:]), N=len(self.best_corr[3:]), sigma=1000, nsteps=1000, nwalkers=100, use_mh='MH', parallel=True, seed=None, verbose=True)
        #self.update_profile()
        #self.update_model()

        print('OK')

    def SL_model1(self, H, LL, Beta, DW=0.11, I0=3, Bkg=3):
        """ Calculate the peak intensity profile for the best set of parameters

        Parameters
        ----------
        H, LL, Beta (float32): Height, Linewidth and sidewall angle
        DW, I0, Bkg (float32): Debye-Waller, Incident intensity, Noise level

        Returns
        -------
        Qxfitc (list of float32) : Simulated peak intensity
        """
        langle = np.deg2rad(np.asarray(Beta))
        rangle = np.deg2rad(np.asarray(Beta))
        Qxfit = []
        for i in range(len(self.qz)):
            ff_core = simulation.stacked_trapezoids(self.qx[i], self.qz[i], 0, LL, H, langle, rangle)
            Qxfit.append(ff_core)
        Qxfitc = fitting.corrections_DWI0Bk(Qxfit, DW, I0, Bkg, self.qx, self.qz)
        return Qxfitc

    def update_model(self):
        self.sigDrawModel.emit(self)

    def update_right_widget(self):
        self.sigDrawParam.emit(self)

    @property
    def modelParameters(self):
        # h,w,langle,rangle=None
        # 3,4,...
        return self.best_corr[3], self.best_corr[4], self.best_corr[5:]

    def update_profile_ini(self):
        """
        Display the experimental peak intensity rescaled
        """
        for order in range(0, len(self.I), 1):
            self.I[order] -= min(self.I[order])
            self.I[order] /= max(self.I[order])
            self.I[order] += order + 1

            guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders[order].setData, self.qz[order], np.log(self.I[order]))
            #np.save('/Users/guillaumefreychet/Desktop/I_all.npy'%order, self.I)
            #np.save('/Users/guillaumefreychet/Desktop/q_all.npy' %order, self.qz)

    def update_profile(self):
        """
        Display the simulated/experimental peak intensity rescaled
        """
        for order in range(0, len(self.I), 1):
            self.I[order] -= min(self.I[order])
            self.I[order] /= max(self.I[order])
            self.I[order] += order + 1

            self.Qxfit[order] -= min(self.Qxfit[order])
            self.Qxfit[order] /= max(self.Qxfit[order])
            self.Qxfit[order] += order + 1

            guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders[order].setData, self.qz[order],
                                             np.log(self.I[order]))
            guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders1[order].setData, self.qz[order],
                                             np.log(self.Qxfit[order]))

class CDRawWidget(pg.ImageView):
    pass

class CDCartoWidget(pg.ImageView):
    def __init__(self):
        self.plotitem = pg.PlotItem()
        super(CDCartoWidget, self).__init__(view=self.plotitem)

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

class CDLineProfileWidget(pg.PlotWidget):
    
    def __init__(self):
        super(CDLineProfileWidget, self).__init__()
        self.setAspectLocked(True)

    def plotLineProfile(self, h, w, langle, rangle=None):
        self.clear()
        x,y = self.profile(h, w, langle, rangle)
        self.plot(x,y)

    @staticmethod
    def profile(h, w, langle, rangle=None):
        if isinstance(langle, list):
            langle = np.array(langle)

        if not isinstance(langle, np.ndarray):
            raise TypeError('Angles must be a numpy.ndarray or list')

        langle = np.deg2rad(langle)
        if rangle is None:
            rangle = langle
        else:
            rangle = np.deg2rad(rangle)

        n = len(langle)
        x = np.zeros(2 * (n + 1), dtype=np.float_)
        y = np.zeros_like(x)

        dxl = np.cumsum(h / np.tan(langle))
        dxr = np.cumsum(h / np.tan(rangle))[::-1]
        x[0] = -0.5 * w
        x[-1] = 0.5 * w
        x[1:n + 1] = x[0] + dxl
        x[n + 1:-1] = x[-1] - dxr

        y[1:n + 1] = np.arange(1, n + 1) * h
        y[n + 1:-1] = np.arange(1, n + 1)[::-1] * h
        return x, y