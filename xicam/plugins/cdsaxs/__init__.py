import __future__
import os, sys, time
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import leastsq, minimize
from scipy.fftpack import *
from astropy import *
from  numpy.fft import *
from scipy.signal import resample
import platform
from fabio import edfimage
from xicam.plugins import base
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
from fabio import tifimage
from pipeline import loader, hig, msg
import numpy as np
from xicam.plugins import base
from xicam.plugins import widgets
import subprocess
import xicam.RmcView as rmc
from xicam import threads
from modpkgs import guiinvoker

from PySide import QtGui, QtCore
from xicam import debugtools

from pipeline.spacegroups import spacegroupwidget
from xicam import config

from xicam.widgets.calibrationpanel import calibrationpanel
from xicam import ROI
from pipeline import integration, center_approx

from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Statistics
from pyevolve import DBAdapters
import pyevolve


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
                                                                                        {'name': 'Run1', 'type': 'action'}]},
                                                                                        {'name' : 'test1', 'type' : 'group', 'children' : [
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
        H, w0, Beta1 = self.param['test', 'H'], self.param['test', 'w0'], self.param['test', 'Beta']
        fitrunnable = threads.RunnableMethod(self.getCurrentTab().fitting_test,method_args=(H, w0, Beta1))
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
        # build arc cut mask
        #center = center_approx.center_approx(loader.loadimage(self.src[0]))
        '''
        center = [552, 225]
        maxR, minR = 250, 0
        nb_pixel = maxR - minR
        maxChi, minChi = 93, 87

        data=[]
        profiles=[]
        x, y = np.indices(loader.loadimage(self.src[0]).shape)
        x_1, y_1 = x - center[0], y - center[1]
        r = np.sqrt(x_1 ** 2 + y_1 ** 2)
        theta = np.degrees(np.arctan2(y_1,x_1))
        rmincut=(r>minR)
        rmaxcut=(r<maxR)
        thetamincut=(theta>minChi)
        thetamaxcut=(theta<maxChi)
        cutmask = (rmincut*rmaxcut*thetamincut*thetamaxcut).T
        '''

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

        i = 0

        for file in self.src:
            '''
            im = Image.fromarray(loader.loadimage(file))
            plt.imshow(im)
            plt.savefig('img%.3d_2.png'%i)
            plt.close()
            i += 1
            '''
            img = np.rot90(loader.loadimage(file), 1)
            profile = np.sum(cutmask*img, axis=1)

            #profile = cutmask * img
            profiles.append(profile[center[1] + minR: center[1] + maxR ])
            #profiles.append(profile[center[1]:])
            data.append(img)
        #plt.imshow(img)
        #plt.show()

        data = np.stack(data)
        data = np.log(data - data.min()+1.)

        self.maskimage = pg.ImageItem(opacity=.25)
        self.CDRawWidget.view.addItem(self.maskimage)
        invmask = 1-cutmask
        self.maskimage.setImage(np.dstack((invmask, np.zeros_like(invmask), np.zeros_like(invmask), invmask)).astype(np.float), opacity=.5)
        self.CDRawWidget.setImage(data)

        #1D cut from experimental data
        pixel_size, sample_detector_distance, wavelength = 172 * 10 **-6, 5., 0.09184
        #pixel_size, sample_detector_distance, wavelength = 172 * 10 **-6, 5., 4.42



        self.QxyiData = self.generate_carto(profiles, nb_pixel, Phi_min, Phi_step, pixel_size, sample_detector_distance, wavelength, center[0])
        self.QxyiDatacor = self.correc_Iexp(self.QxyiData)
        #np.save('qxyi_cxro.npy', self.QxyiData)

        #interpolation
        sampling_size = (400, 400)
        #img = self.inter_carto(self.QxyiData)
        self.img, qk_shift= self.interpolation(self.QxyiDatacor, sampling_size)

        #Extraction of nb of profile + their positions
        profile_carto = np.mean(np.nan_to_num(self.img), axis=1)

        Int1 = np.amax(profile_carto)
        ind1 = np.int(np.where(profile_carto == Int1)[0])

        pos_gauss = np.linspace(ind1 - 20, ind1 + 20, 41, dtype=np.int32)

        g_init = models.Gaussian1D(amplitude=Int1, mean=ind1, stddev=1.)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, pos_gauss, profile_carto[pos_gauss])
        ind1 = g.mean.value

        limit_ampli = 0.01 * Int1

        ind = []
        ind.append(ind1)
        cnt = 0
        i=2
        finish = 'False'

        while ((i+1) * ind[0] < np.shape(profile_carto)[0]) and (finish != 'True'):
            ind_imp1 = i * ind[0]
            pos_gauss1 = np.linspace(ind_imp1 - 20, ind_imp1 + 20, 41, dtype=np.int32)
            g_init_1 = models.Gaussian1D(amplitude = limit_ampli, mean = ind_imp1, stddev = 1.)
            g_1 = fit_g(g_init_1, pos_gauss1, profile_carto[pos_gauss1])
            ind_imp1 = g_1.mean.value
            if g_1.amplitude.value < limit_ampli and (i+1) * ind[0] < np.shape(profile_carto)[0]:
                ind_imp2 = (i + 1) * ind[0]
                pos_gauss2 = np.linspace(ind_imp2 - 20, ind_imp2 + 20, 41, dtype=np.int32)
                g_init_2 = models.Gaussian1D(amplitude = limit_ampli, mean = ind_imp2, stddev = 1.)
                g_2 = fit_g(g_init_2, pos_gauss2, profile_carto[pos_gauss2])
                ind_imp2 = g_2.mean.value

                if g_2.amplitude < limit_ampli:
                    finish = 'True'

                else:
                    ind.append(ind_imp1)
                    ind.append(ind_imp2)
                    i = i + 2

            elif g_1.amplitude.value < limit_ampli and (i+1) * ind[0] > np.shape(profile_carto)[0]:
                finish = 'True'

            else:
                ind.append(ind_imp1)
                i = i +1

        #print(len(ind))
        self.q = []
        self.Qxexp = []
        self.Qxfit = []
        self.Q__Z = []
        self.Q__Z1 = []
        self.Position = []
        self.maxres = 0
        #self.phiMax = np.radians(30)
        self.imageshape = (500, 500)

        for i in range(0, len(ind),1):
            if i != 6:
                self.q.append((4 * np.pi / wavelength) * np.sin(np.arctan((((ind[i] * 500 / 400) * nb_pixel / 500) * (pixel_size / sample_detector_distance)))))
                self.Position.append(int(np.floor(i / (2 * np.pi / (0.5 * self.imageshape[0])))))
                self.get_exp_values(self.QxyiData, self.q[i])
                #print(self.Position[i])
                #self.Qxexp.append(np.sum(np.nan_to_num(self.img[ind[i] - 3 : ind[i] + 3 , : ]), axis = 0))
                #self.Qxexp.append(np.nan_to_num(self.img[ind[i], :]))

        self.update_profile_ini()

        phi_max, phi_min = Phi_max,  Phi_min

        #Definition of the fitting mask
        center_1 = [38, 0]
        maxR_1, minR_1 = 80, 0
        nb_pixel_1 = maxR_1 - minR_1
        maxChi_1, minChi_1 = 90 + 0.5 * phi_max, 90 + 0.5 * phi_min
        Chi_sym = max(abs(0.5 * phi_max), abs(0.5 * phi_min))
        maxChi_sym, minChi_sym = 90 + Chi_sym, 90 - Chi_sym

        x, y = np.indices((79,79))
        x_1_1, y_1_1 = x - center_1[0], y - center_1[1]
        r_1 = np.sqrt(x_1_1 ** 2 + y_1_1 ** 2)
        theta_1 = np.degrees(np.arctan2(y_1_1,x_1_1))
        rmincut_1, rmaxcut_1 =(r_1>minR_1), (r_1<maxR_1)
        thetamincut_1, thetamaxcut_1=(theta_1>minChi_1), (theta_1<maxChi_1)
        thetamincut_sym, thetamaxcut_sym = (theta_1 > minChi_sym), (theta_1 < maxChi_sym)
        self.cutmask = (rmincut_1*rmaxcut_1*thetamincut_1*thetamaxcut_1).T
        self.cutmask_sym = (rmincut_1 * rmaxcut_1 * thetamincut_sym * thetamaxcut_sym).T

        self.SL_model(100, 40, np.array([95]))

    def generate_carto(self, profiles, nb_pixel, Phi_min, Phi_step, pixel_size, sample_detector_distance, wavelength, center_x):
        nv = np.zeros([np.shape(profiles)[1],4], dtype=np.float32)
        QxyiData = np.zeros([np.shape(profiles)[1],4], dtype=np.float32)
        for i in range(0, np.shape(profiles)[0], 1):
            phi = np.radians(Phi_min + i * Phi_step)
            q = [0] * np.shape(profiles)[1]
            qx = [0] * np.shape(profiles)[1]
            qz = [0] * np.shape(profiles)[1]
            for j in range(0, np.shape(profiles)[1], 1):
                q[j] = (4 * np.pi / wavelength) * np.arctan(j * nb_pixel / np.shape(profiles)[1] * pixel_size / sample_detector_distance)
                qx[j] = q[j] * np.cos(phi + 2 * np.arcsin(q[j] * wavelength/ (4 * np.pi)))
                qz[j] = q[j] * np.sin(phi + 2 * np.arcsin(q[j] * wavelength/ (4 * np.pi)))
            nv[:, 0] = qx
            nv[:, 1] = qz
            nv[:, 2] = profiles[i]
            nv[:, 3] = phi
            QxyiData = np.vstack((QxyiData, nv))
        return QxyiData

    # Correction of the footprint and substrate attenuation
    # Add sample size/sample attenuation and polarization
    def correc_Iexp(self, Qxyi):
        footprintcorr = 'True'
        abscorr = 'True'
        samplesizecorr = 'False'
        substratethickness, substrateattenuation = 700 * 10 **-6, 200 * 10 **-6
        fwhm, sample_size = 1, 1
        for i in range(0, len(Qxyi[0]), 1):
            footprintfactor = np.cos(Qxyi[i,3]) if footprintcorr else 1
            absfactor = np.exp(-substratethickness * substrateattenuation * (1 - 1 / Qxyi[i,3])) if abscorr else 1
            Qxyi[i, 2] *= absfactor * footprintfactor
        return Qxyi

    def inter_carto(self, qxyi):
        #Reverse map carthography
        qxyi = qxyi[1167:]
        qy = qxyi[:, 0]
        qz = qxyi[:, 1]
        val = qxyi[:, 2]
        angles = qxyi[:, 3]

        # get angles
        a, i, n = np.unique(angles, return_index=True, return_counts=True)
        inds = dict(zip(a, i))
        if np.sum(n - n[0]) > 0:
            raise ValueError('One of the angles has too many Intensity values')

        # calculate number of columns and rows
        ncol = n[0]
        amax = np.abs(a).max()
        nrow = np.int(ncol * np.sin(np.deg2rad(amax)) + 1)
        nrow = 2 * nrow + 1

        # setup output image
        img = np.zeros((nrow, ncol))
        u, v = np.indices((nrow, ncol))
        u -= nrow // 2

        # temp = np.round(np.rad2deg(np.arctan2(u, v)), decimals=1)
        angle = np.round(2 * np.rad2deg(np.arctan2(u, v))) / 2
        radius = np.round(np.sqrt(u ** 2 + v ** 2)).astype(int)

        it = np.nditer(img, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            y, x = it.multi_index
            ang = angle[y, x]
            rad = min(radius[y, x], ncol - 1)
            if ang in inds:
                i = inds[ang] + rad
                it[0] = val[i]
            it.iternext()

        #log_possible = np.where(img!='nan')
        #img[log_possible] = np.log(img[log_possible] - img[log_possible].min() + 1.)
        #self.CDCartoWidget.setImage(np.log(img))
        return img


    def interpolation(self, qxyi, sampling_size=(400, 400)):
        roi_loc = (sampling_size[0] / 2., sampling_size[1] / 2.)
        roi_size = 400
        img = np.zeros((roi_size, roi_size))

        qj = np.floor(
            ((qxyi[:, 0] - qxyi[:, 0].min()) / (qxyi[:, 0] - qxyi[:, 0].min()).max()) * (sampling_size[0] - 1)).astype(
            np.int32)
        qk = np.floor(((qxyi[:, 1].ravel() - qxyi[:, 1].min()) / (qxyi[:, 1] - qxyi[:, 1].min()).max()) * (
            sampling_size[1] - 1)).astype(np.int32)
        I = qxyi[:, 2].ravel()

        # Area of the cartography to interpolate
        selected_idx = []
        assert qj.size == qk.size, 'uncorrect size for q{x,y} index vector'
        for i in xrange(qj.size):
            if -qk[i] / 2 < qj[i] and qj[i] <= roi_loc[0] + roi_size and roi_loc[1] - roi_size < qk[i] and qk[i] <= roi_loc[1] + roi_size :
                selected_idx.append(i)


        qj_shifted = qj[selected_idx] - qj[selected_idx].min()
        qk_shifted = qk[selected_idx] - qk[selected_idx].min()

        print(qxyi[:, 1].min())
        print(qxyi[:, 1].max())

        Isel = I[selected_idx]
        for i, isel in enumerate(Isel):
            img[qj_shifted[i], qk_shifted[i]] += isel
        '''
        plt.imshow((img))
        plt.show()
        '''
        qk_shift =  qk[selected_idx].min()

        to_fill = []
        to_fill = np.array(np.where(img == 0)).T

        interp_from = np.where(img != 0)
        origin = (roi_size / 2)

        interpolator = LinearNDInterpolator(interp_from, img[interp_from])

        cpt = 0


        #for p in to_fill:
        #    img[p[0], p[1]] += interpolator(p[0], p[1])

        '''
        for p in to_fill:
            if abs((p[1] - origin) / 2) >= (p[0]):
                continue
            try:
                img[p[0], p[1]] += interpolator(p[0], p[1])
            except ValueError:
                cpt += 1
                pass
        '''
        log_possible = np.where(img!='nan')
        img[log_possible] = np.log(img[log_possible] - img[log_possible].min() + 1.)
        self.CDCartoWidget.setImage(img)
        return img, qk_shift

    def get_exp_values(self,qxyi, cut_val):
        print(cut_val)
        delta = 0.002
        dtype = [('qx', np.float32), ('qy', np.float32), ('i', np.float32)]
        Sqxyi = []
        for v in qxyi:
            qx, qy, i, phi = v
            Sqxyi.append((qx, qy, i))
        Qi = np.array(Sqxyi, dtype)
        SQi = np.sort(Qi, order='qy')

        binf, bsup = cut_val - delta, cut_val + delta
        idx = np.where((SQi['qx'] > binf) * (SQi['qx'] < bsup))  # selection contraints by qy vals
        SQi['i'][idx]

        self.Qxexp.append(SQi['i'][idx])
        self.Q__Z.append(SQi['qy'][idx])

        return SQi['i'][idx]

    def fitting_test(self, H = 10, LL = 20, Beta1 = 2):
        ts = time.time()

        Beta2, Beta3, Beta4, Beta5 = Beta1, Beta1, Beta1, Beta1
        # Beta = np.array([Beta1])

        Beta = np.array([Beta1, Beta2, Beta3, Beta4, Beta5])

        initiale_value = []
        initiale_value.append(int(H))
        initiale_value.append(int(LL))

        for i in Beta:
            initiale_value.append(int(i))

        lower_bnds, upper_bnds = [], []
        for i in initiale_value:
            lower_bnds.append(int(i - 10))
            upper_bnds.append(int(i + 10))

        phi_min = -45
        phi_max = 45

        # Definition of the fitting mask
        center_1 = [39, 0]
        maxR_1, minR_1 = 80, 0
        nb_pixel_1 = maxR_1 - minR_1
        maxChi_1, minChi_1 = 90 + 0.5 * phi_max, 90 + 0.5 * phi_min
        Chi_sym = max(abs(0.5 * phi_max), abs(0.5 * phi_min))
        maxChi_sym, minChi_sym = 90 + Chi_sym, 90 - Chi_sym

        x, y = np.indices((79, 79))
        x_1_1, y_1_1 = x - center_1[0], y - center_1[1]
        r_1 = np.sqrt(x_1_1 ** 2 + y_1_1 ** 2)
        theta_1 = np.degrees(np.arctan2(y_1_1, x_1_1))
        rmincut_1, rmaxcut_1 = (r_1 > minR_1), (r_1 < maxR_1)
        thetamincut_1, thetamaxcut_1 = (theta_1 > minChi_1), (theta_1 < maxChi_1)
        thetamincut_sym, thetamaxcut_sym = (theta_1 > minChi_sym), (theta_1 < maxChi_sym)
        self.cutmask = (rmincut_1 * rmaxcut_1 * thetamincut_1 * thetamaxcut_1).T
        self.cutmask_sym = (rmincut_1 * rmaxcut_1 * thetamincut_sym * thetamaxcut_sym).T

        self.residual(initiale_value)
        self.update_right_widget()

        self.best_score = 0
        #pyevolve.logEnable()
        genome = G1DList.G1DList(7)
        genome.setParams(rangemin=0, rangemax=1000)
        genome.evaluator.set(self.residual)

        ga = GSimpleGA.GSimpleGA(genome)
        ga.selector.set(Selectors.GRouletteWheel)
        ga.setGenerations(30)

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
        self.modelParameter = 5 + 0.02 * best.genomeList[0], 20 + 0.04 * best.genomeList[1], 70 + 0.04 * \
                              best.genomeList[2], best.score
        self.update_right_widget()

        #xopt, fopt = pso(self.residual, lower_bnds, upper_bnds)

        # opt = minimize(self.residual, initiale_value, bounds=bndes, method='L-BFGS-B', options={'disp': True, 'eps': (1, 1, 0.3), 'ftol': 1e-9})


        te = time.time()
        print(te - ts)

        #print(xopt, fopt)

        #self.residual(xopt)

        # print(opt.message)

    def evolve_callback(self, ga_engine):
        generation = ga_engine.getCurrentGeneration()
        if generation % 1 == 0:
            print "Current generation: %d" % (generation,)
            best = ga_engine.bestIndividual()
            if best.score > self.best_score:
                self.residual(best.genomeList, 'True')
                #print(best.score, best.genomeList)
                self.update_model()
                self.update_profile('True')
                self.modelParameter = 5 + 0.02 * best.genomeList[0], 20 + 0.04 * best.genomeList[1], 70 + 0.04 * best.genomeList[2], best.score
                self.update_right_widget()
                self.best_score = best.score

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
                guiinvoker.invoke_in_main_thread(self.CDModelWidget.orders1[order].setData, self.Q__Z1[order], np.log(self.Qxfit[order]))

    #@debugtools.timeit
    def residual(self, p, test = 'False', plot_mode=False):
        H = 5 + 0.02 * p[0]
        LL = 20 + 0.04 * p[1]
        Beta = []

        for i in range(2, len(p), 1):
            Beta.append(70 + 0.04 * p[i])

        Beta = np.array(Beta)

        self.Qxfit = self.SL_model(H, LL, Beta)

        for i in range(0, len(self.Qxexp), 1):
            self.Qxexp[i], self.Qxfit[i] = self.resize_iset(self.Qxexp[i], self.Qxfit[i])

        self.update_profile(test)
        #self.update_model()


        res = 0
        res_min = 1000

        for i in range(0, len(self.Qxexp), 1):
            res += np.sqrt(sum((self.Qxfit[i] - self.Qxexp[i])**2) / sum((self.Qxexp[i])**2))

        #self.modelParameter = H, LL, Beta[0], 1 / res
        #self.update_right_widget()

        #self.update_profile()
        #self.update_right_widget()
        #self.modelParameter = H, LL, Beta[0], 1/res
        #self.update_model()


        '''
        #Plot only better res
        if res < res_min :
            res_min = res
            self.modelParameter = H, LL, Beta[0], res
            self.update_model()
            self.update_profile()
            self.update_right_widget()
        '''
        self.maxres = max(self.maxres, 1/res)
        return 1/res

    #@debugtools.timeit
    def SL_model(self, H, LL, Beta, plot_mode=False):
        pitch, nbligne = 100, 1
        I = []
        I = self.Fitlignes(pitch, Beta, LL, H, nbligne)

        # Fitting qx cut
        Tailleimagex = 500
        self.Qxfit = []
        self.Q__Z1 = []
        pixel = []
        q_z = np.zeros((len(self.q), 1000))
        j = 0
        for i in self.q:
            Position = np.floor(1.5 * i / (2 * np.pi / (0.5 * Tailleimagex)))
            ind_min, ind_max, ind_zero = self.Qxcut(I, Position, i)
            self.Qxexp[j], self.Qxfit[j] = self.resize_yset(self.Qxexp[j], self.Qxfit[j], ind_min, ind_max)
            pixel.append(np.linspace(ind_min, ind_max, len(self.Qxexp[j])))
            for k in range(0, len(pixel[j]), 1):
                q_z[j, k] = 2.5 * i * np.sin(np.arctan((pixel[j][k]- int(ind_zero))/Position)) + 0.00001

            self.Q__Z1.append(np.trim_zeros(q_z[j]))
            j = j + 1

        return self.Qxfit

    def SL_model1(self, H, LL, Beta, plot_mode=False):
        qy = np.asarray()
        self.Q__Z
        self.q

        qy = np.linspace(-1, 1, 128)
        qz = np.linspace(0, 2, 3)
        qy, qz = np.meshgrid(qy, qz)
        langle = np.deg2rad(np.repeat(80, 5))
        rangle = np.deg2rad(np.repeat(80, 5))
        ff = stacked_trapezoid(qy, qz, -25, 25, 10, langle, rangle)
        img = np.absolute(ff) ** 2
        plt.plot(np.log(img[0, :]))
        plt.plot(np.log(img[1, :]))
        plt.plot(np.log(img[2, :]))
        plt.show()


    # Generation of the form factor for the line profile generated with the fonction ligne1
    def Fitlignes(self, pitch, Beta, LL, H, nbligne, Taille_image=(600, 600)):
        # assert pitch >= Largeurligne+2*H*abs(np.tan(beta)), 'uncorrect desription of lines'

        Obj = self.multipyramid(H, LL, Beta, 500, 500)
        #Obj = self.multipyramid(H, LL, 90 + np.degrees(Beta1),90 + np.degrees(Beta2), 90 + np.degrees(Beta3), 90 + np.degrees(Beta4), 90 + np.degrees(Beta5), 600, 600)
        Obj_plot = np.rot90(Obj,3)
        self.modelImage = Obj_plot[150 : 350, 350 : 500]

        #I = np.random.poisson(abs(fftshift(fftn(np.rot90(Obj,3)))) ** 2)
        I = (abs(fftshift(fftn(np.rot90(Obj,3)))) ** 2)

        '''
        Dynamic = I.max()
        II = np.zeros(I.shape, dtype='float64')
        III = np.zeros(I.shape, dtype='int64')
        II = (I * Dynamic) / I.max()
        III = np.int64((II >= 1) * II)
        '''
        return I


    def trapezoid_form_factor(qy, qz, y1, y2, langle, rangle, h):
        m1 = np.tan(langle)
        m2 = np.tan(np.pi - rangle)
        t1 = qy + m1 * qz
        t2 = qy + m2 * qz
        with np.errstate(divide='ignore'):
            t3 = m1 * np.exp(-1j * qy * y1) * (1 - np.exp(-1j * h / m1 * t1)) / t1
            t4 = m2 * np.exp(-1j * qy * y2) * (1 - np.exp(-1j * h / m2 * t2)) / t2
            ff = (t4 - t3) / qy
        return ff

    def stacked_trapezoids(self, qy, qz, y1, y2, height, langle, rangle=None):
        if not isinstance(langle, np.ndarray):
            raise TypeError('anlges should be array')
        if rangle is not None:
            if not langle.size == rangle.size:
                raise ValueError('both angle array are not of same size')
        else:
            rangle = langle

        ff = np.zeros(self, qy.shape, dtype=np.complex)
        # loop over all the angles
        for i in range(langle.size):
            shift = height * i
            left, right = langle[i], rangle[i]
            ff += self.trapezoid_form_factor(qy, qz, y1, y2, left, right, height) * np.exp(-1j * shift * qz)
            m1 = np.tan(left)
            m2 = np.tan(np.pi - right)
            y1 += height / m1
            y2 += height / m2

        return np.absolute(ff) ** 2

    def multipyramid(self, h, w, a, nx, ny):
        if nx % 2 == 1:
            nx += 1

        n2 = nx / 2
        x0 = w / 2
        y0 = 0

        if not type(a) is np.ndarray:
            raise TypeError('Side-wall angle must be numpy array for multipyramid')

        # setup output array
        img = np.zeros((ny, n2))
        y, x = np.mgrid[0:ny, 0:n2]

        a = np.deg2rad(a)
        for i in range(a.size):
            A = np.sin(np.pi - a[i])
            B = -np.cos(np.pi - a[i])
            C = -(A * x0 + B * y0)
            d = A * x + B * y + C

            # update (x0, y0)
            y0 = (i + 1) * h
            x0 = -(B * y0 + C) / A

            # update image
            mask = np.logical_and(y >= i * h, y < (i + 1) * h)
            mask = np.logical_and(d < 0, mask)
            img[mask] = 1

        return np.hstack((np.fliplr(img), img))

    # Function doing the 1D cut along qx of the simulated signal along Position1, Position2, Position3
    def Qxcut(self, I, Position, qx, Taille_image=(500, 500), phimax=np.radians(27)):
        roisizex, roisizey = np.int(1 / (2 * np.pi) * Taille_image[0]), np.int(1 / (2 * np.pi) * Taille_image[1])
        phimin = np.radians(-90)
        phimax = np.radians(90)
        phi = max(phimin, phimax)
        center_x, center_y = Taille_image[0] / 2, Taille_image[1] / 2
        originx, originy = 0, (roisizey / 2) - 0.5

        Iroi= np.zeros([roisizex + 1, roisizey + 1])

        Iroi = I[center_x : center_x + roisizex, center_y + 1 - roisizey / 2 : center_y + roisizey + 1 - roisizey / 2]

        Iroi1 = Iroi * self.cutmask
        Iroi *= self.cutmask_sym
        I1 = np.sum(Iroi[int(Position) - 1:int(Position) + 1, :], axis=0)
        I2 = np.sum(Iroi1[int(Position) - 1:int(Position) + 1, :], axis=0)

        if len(np.where(I1!= 0)[0]) == 0:
            if len(np.where(I2 != 0)[0]) == 0:
                ind_min, ind_max, ind_zero = 0, len(I1) - 1, int(0.5 * len(I1))
            else:
                ind_min, ind_max, ind_zero = (np.where(I2!= 0)[0][0] - 0), (np.where(I2!= 0)[0][-1] - 0), np.floor(0.5 * len(I1)) - 1
        else:
            ind_min, ind_max, ind_zero = (np.where(I2 != 0)[0][0] - np.where(I1 != 0)[0][0]), (np.where(I2 != 0)[0][-1] - np.where(I1 != 0)[0][0]), np.floor(0.5 * len(I1)) - np.where(I1 != 0)[0][0] - 1

        for i in range(0, len(I1), 1):
            phi = np.arctan(np.float(i - np.ceil(0.5 * len(I1))) / np.float(Position))
            pathlength = 1/ np.cos(phi)
            I1[i] *= np.exp(-pathlength)

        self.Qxfit.append(np.trim_zeros(I1))
        return ind_min, ind_max, ind_zero

    # Rescale  the experimental and simulated data in qy
    def resize_yset(self, data0, data1, ind_min, ind_max):
        max_size = len(data0)
        a = len(data1)
        data1 = resample(data1, int(max_size * (np.float(len(data1)) / (np.float(int(ind_max - ind_min))))))
        ind_min1 = ind_min * (np.float(len(data1)) / (np.float(a)))
        ind_max1 = np.ceil(ind_max * (np.float(len(data1)) / (np.float(a))))

        return data0, data1[ind_min1 : ind_max1 ]


    def resize_iset(self, data0, data1):
        data1 = data1 * (max(data0) / max(data1))
        #ind = np.where(data1 < min(data0))
        #data1[ind] = min(data0)
        return data0, data1

    # Correction by the Debye-Waller()/Poisson_noise/Background_noise (+ constant)/I_rescale( * constant for all profiles)
    # parameter use in the fit : DW_factor
    def correc_Isim(self, Qxfit, Qzfit, DW_factor, I_res, Back_noise):
        for i in Qxfit:
            Ifit *= np.exp(-(i ** 2 + Qzfit ** 2) * DW_factor ** 2)
        Ifit *= I_res
        Ifit += Back_noise

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
            self.orders.append(self.plot([],pen=pg.mkPen(color), name = 'Order ' + str(i)))
            self.orders1.append(self.plot([], pen=pg.mkPen(color), name='Order ' + str(i)))

class CDProfileWidget(pg.ImageView):
    pass