import __future__
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import leastsq, minimize
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

from pipeline.spacegroups import spacegroupwidget
from xicam import config
import fabio

from xicam.widgets.calibrationpanel import calibrationpanel

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
        Phi_min, Phi_max, Phi_step = self.param['test', 'Phi_min'], self.param['test', 'Phi_max'], self.param['test', 'Phi_step']

        self.param.param('test', 'Run1').sigActivated.connect(self.fit)

        super(plugin, self).__init__(*args, **kwargs)

    def update_model(self,widget):
        guiinvoker.invoke_in_main_thread(self.bottomwidget.setImage,widget.modelImage)

    def update_right_widget(self,widget):
        H, LL, beta, f_val = widget.modelParameter
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'H_fit').setValue,H)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'w0_fit').setValue, LL)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'Beta_fit').setValue, beta)
        guiinvoker.invoke_in_main_thread(self.param.param('test1', 'f_val').setValue, f_val)

    def fit(self):
        activeSet = self.getCurrentTab()
        activeSet.setCurrentWidget(activeSet.CDModelWidget)
        H, w0, Beta = self.param['test', 'H'], self.param['test', 'w0'], self.param['test', 'Beta']
        fitrunnable = threads.RunnableMethod(self.getCurrentTab().fitting_test,method_args=(H,w0,Beta))
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

        self.loadRAW()
        self.main()

    def loadRAW(self):
        #data = np.stack([np.rot90(loader.loadimage(file),2) for file in self.src])
        data = np.stack([np.transpose(loader.loadimage(file)) for file in self.src])
        data = np.log(data - data.min()+1.)

        self.CDRawWidget.setImage(data)

    def main(self):
        prefix = 'contacta1_hs104_'
        serie1 = '060_'
        wavelength = 0.072932
        sampling_size = (400,400)
        phi_min, phi_max, phi_step = -30, 30, 1
        center_x, center_y = 248, 1668  # Position of the direct beam

        #I_norm = self.tiff('/Users/guillaumefreychet/GitHub/Xi-cam/xicam/plugins/cdsaxs/contacta1_hs104_060_0000.tif')
        #self.qxyi = np.zeros([len(I_norm[0]), 3], dtype=np.float32)
        #self.qxyi = self.data_fusion_2D(self.qxyi, prefix, wavelength, serie1, phi_min, phi_max, phi_step, center_x, center_y)
        self.qxyi = np.load('/Users/guillaumefreychet/Desktop/QxyiData_MemA.npy')
        self.interpolation(self.qxyi, sampling_size)
        #self.fitting_test(self.qxyi, 300, 35, 2)

    def tiff(self, tif_name):
        I = plt.imread(tif_name)
        I_norm = I
        return I_norm

    def data_fusion_2D(self, QxyiData, prefix, wavelength, serie, phi_min, phi_max, phi_step, center_x, center_y):
        i = 0
        I_norm = self.tiff(
            '/Users/guillaumefreychet/GitHub/Xi-cam/xicam/plugins/cdsaxs/contacta1_hs104_060_' + '%.4d.tif' % i)
        nv = np.zeros([len(I_norm[0])- center_x, 3], dtype=np.float32)
        I_cut = np.zeros(len(I_norm[0]) - center_x, dtype=np.int)
        for phi in range(phi_min, phi_max, phi_step):
            I_norm = self.tiff(
                '/Users/guillaumefreychet/GitHub/Xi-cam/xicam/plugins/cdsaxs/contacta1_hs104_060_' + '%.4d.tif' % i)
            I_cut = self.cut_and_sum(I_norm, center_x, center_y)
            nv[:, 2] = I_cut[:]
            phi_rad = np.radians(phi)
            nv[:, 0], nv[:, 1] = self.calcul_q(len(I_cut), phi_rad, wavelength)
            QxyiData = np.vstack((QxyiData, nv))
            i = i + 1
        return QxyiData

    def cut_and_sum(self, data, center_x, center_y):
        data_cut_sum = np.zeros(len(data[0]), dtype=np.int)
        delta = 10
        data_cut_sum = np.sum(data[center_x - np.int(delta / 2): center_y + np.int(delta / 2), center_x : ], axis=0)
        return data_cut_sum

    def indexfunc(self, index_I_cut):
        indexe = np.zeros(index_I_cut, dtype=np.int)
        for i in range(1, index_I_cut, 1):
            indexe[i] = i
        return indexe

    def calcul_q(self, index_max, phi, lamda):
        indexes = self.indexfunc(index_max)
        taille_pixel = 79.4 * 10 ** -6
        distance_ech_det = 4.54
        q = (4 * np.pi / lamda) * np.sin(np.tan(indexes[:] * taille_pixel / distance_ech_det))
        qx = q * np.cos(phi)
        qz = q * np.sin(phi)
        return qx, qz

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
            #if -qk[i] / 2 < qj[i] and qj[i] <= roi_loc[0] + roi_size and roi_loc[1] - roi_size < qk[i] and qk[i] <= \roi_loc[1] + roi_size:
            selected_idx.append(i)

        qj_shifted = qj[selected_idx] - qj[selected_idx].min()
        qk_shifted = qk[selected_idx] - qk[selected_idx].min()
        Isel = I[selected_idx]
        for i, isel in enumerate(Isel):
            img[qj_shifted[i], qk_shifted[i]] += isel

        to_fill = []
        to_fill = np.array(np.where(img == 0)).T
        interp_from = np.where(img != 0)
        origin = (roi_size / 2)
        interpolator = LinearNDInterpolator(interp_from, img[interp_from])

        cpt = 0
        for p in to_fill:
            if abs((p[1] - origin) / 2) >= (p[0]):
                continue
            try:
                img[p[0], p[1]] += interpolator(p[0], p[1])
            except ValueError:
                cpt += 1
                pass

        log_possible = np.where(img!='nan')
        img[log_possible] = np.log(img[log_possible] - img[log_possible].min() + 1.)

        self.CDCartoWidget.setImage(img, levels = (100, 200000))

    def get_exp_values(self,qxyi):
        cut_val = 2 * 0.0625
        delta = 0.0005
        dtype = [('qx', np.float32), ('qy', np.float32), ('i', np.float32)]
        Sqxyi = []
        for v in qxyi:
            qx, qy, i = v
            Sqxyi.append((qx, qy, i))
        Qi = np.array(Sqxyi, dtype)
        SQi = np.sort(Qi, order='qy')

        binf_1, bsup_1 = cut_val - delta, cut_val + delta
        binf_2, bsup_2 = 2 * cut_val - delta, 2 * cut_val + delta
        binf_3, bsup_3 = 3 * cut_val - delta, 3 * cut_val + delta

        idx_1 = np.where((SQi['qx'] > binf_1) * (SQi['qx'] < bsup_1))  # selection contraints by qy vals
        idx_2 = np.where((SQi['qx'] > binf_2) * (SQi['qx'] < bsup_2))
        idx_3 = np.where((SQi['qx'] > binf_3) * (SQi['qx'] < bsup_3))

        return SQi['i'][idx_1], SQi['i'][idx_2], SQi['i'][idx_3]

    def fitting_test(self, H, w0, Beta):
        initial_value = (H, w0, Beta)
        bnds = ((250, 350), (20, 50), (0.5, 3.) )

        #self.Qxexp1, self.Qxexp2, self.Qxexp3 = self.get_exp_values(self.qxyi)
        self.update_right_widget()

        opt = minimize(self.residual, initial_value, bounds=bnds, method='L-BFGS-B',
                       options={'disp': True, 'eps': (10, 1, 0.1), 'ftol': 0.000000001})
        # print(opt.x)
        # print(opt.message)

    def update_model(self):
        self.sigDrawModel.emit(self)

    def update_right_widget(self):
        self.sigDrawParam.emit(self)

    def update_profile(self):
        guiinvoker.invoke_in_main_thread(self.CDModelWidget.order1.setData,np.log(self.Qxexp1))
        guiinvoker.invoke_in_main_thread(self.CDModelWidget.order2.setData,np.log(self.Qxexp2))
        guiinvoker.invoke_in_main_thread(self.CDModelWidget.order3.setData,np.log(self.Qxexp3))

        guiinvoker.invoke_in_main_thread(self.CDModelWidget.order4.setData, np.log(self.Qxfit1))
        guiinvoker.invoke_in_main_thread(self.CDModelWidget.order5.setData, np.log(self.Qxfit2))
        guiinvoker.invoke_in_main_thread(self.CDModelWidget.order6.setData, np.log(self.Qxfit3))

    def residual(self, p, plot_mode=False):
        H, LL, beta = p
        beta = np.radians(beta)
        self.Qxexp1, self.Qxexp2, self.Qxexp3  = self.get_exp_values(self.qxyi)
        self.Qxfit1, self.Qxfit2, self.Qxfit3 = self.SL_model(H, LL, beta)

        # recalage en y
        self.Qxexp1, self.Qxfit1 = self.resize_yset(self.Qxexp1, self.Qxfit1)
        self.Qxexp2, self.Qxfit2 = self.resize_yset(self.Qxexp2, self.Qxfit2)
        self.Qxexp3, self.Qxfit3 = self.resize_yset(self.Qxexp3, self.Qxfit3)

        max_size = self.Qxexp3.size
        self.Qxexp1, self.Qxfit1 = self.centering(self.Qxexp1, self.Qxfit1, np.int(max_size))
        self.Qxexp2, self.Qxfit2 = self.centering(self.Qxexp2, self.Qxfit2, np.int(max_size))

        # recalage en intensite
        self.Qxexp1, self.Qxfit1 = self.resize_iset(self.Qxexp1, self.Qxfit1)
        self.Qxexp2, self.Qxfit2 = self.resize_iset(self.Qxexp2, self.Qxfit2)
        self.Qxexp3, self.Qxfit3 = self.resize_iset(self.Qxexp3, self.Qxfit3)

        self.update_profile()
        self.update_right_widget()
        #self.results()

        res = (sum(abs(self.Qxfit1 - self.Qxexp1)) + sum(abs(self.Qxfit2 - self.Qxexp2)) + sum(abs(self.Qxfit3 - self.Qxexp3))) / (
        sum(self.Qxexp1) + sum(self.Qxexp2)*(max(self.Qxfit1) / max(self.Qxfit2)) + sum(self.Qxexp3))

        self.modelParameter = H, LL, np.degrees(beta), res
        self.update_model()
        self.update_profile()
        self.update_right_widget()

        return res

    def SL_model(self, H, LL, beta, plot_mode=False):
        pitch, nbligne = 100, 1
        I = []
        I = self.Fitlignes(pitch, beta, LL, H, nbligne)

        # Fitting qx cut
        Tailleimagex = 600
        qref = 1.5 * 0.0628
        Position1 = np.floor(qref / (2 * np.pi / Tailleimagex))
        Position2 = np.floor(2 * qref / (2 * np.pi / Tailleimagex))
        Position3 = np.floor(3 * qref / (2 * np.pi / Tailleimagex))
        Iroi, Qxfit1, Qxfit2, Qxfit3 = self.Qxcut(I, Position1, Position2, Position3)

        return Qxfit1[Qxfit1.nonzero()[0]], Qxfit2[Qxfit2.nonzero()[0]], Qxfit3[Qxfit3.nonzero()[0]]


    # Generation of the form factor for the line profile generated with the fonction ligne1
    def Fitlignes(self, pitch, beta, LL, H, nbligne, Taille_image=(600, 600)):
        # assert pitch >= Largeurligne+2*H*abs(np.tan(beta)), 'uncorrect desription of lines'

        '''
        Tailleximage = Taille_image[0]
        Tailleyimage = Taille_image[1]
        Obj = np.zeros([Tailleximage, Tailleyimage])
        (a, b) = Obj.shape
        for a in range(0, int(nbligne * pitch), 1):
            for b in range(0, int(H), 1):
                x = a
                for c in range(0, nbligne, 1):
                    if x > pitch:
                        x = x - pitch
                    Obj[int(a + (Tailleximage / 2)), int(b + (Tailleyimage - H) / 2)] = self.ligne1(x, b, beta, LL, H, pitch)

        self.modelImage = Obj[250:350, 100:500]
        '''

        #'''
        Obj = self.pyramid(LL, H, 90 + np.degrees(beta), 600, 600)
        self.modelImage = Obj[950:1100, 800:1200]
        #'''

        I = np.random.poisson(abs(fftshift(fftn(np.rot90(Obj,1)))) ** 2)
        Dynamic = I.max()
        II = np.zeros(I.shape, dtype='float64')
        III = np.zeros(I.shape, dtype='int64')
        II = (I * Dynamic) / I.max()
        III = np.int64((II >= 1) * II)
        return III


    def pyramid(self, w, h, a, nx, ny):
        if nx % 2 == 1:
            nx = nx + 1

        # compute half and mirror it later
        n2 = nx / 2
        w2 = w / 2

        # setup arrays
        img = np.zeros((ny, n2))
        y, x = np.mgrid[0:ny, 0:n2]

        # equation of line for side of trapezium
        a = np.deg2rad(a)
        A = np.sin(np.pi - a)
        B = -np.cos(np.pi - a)
        C = - A * w2

        # calculate distance from line
        d = A * x + B * y + C
        img[np.logical_and(d < 0, y < h)] = 1
        return np.hstack((np.fliplr(img), img))


    # Simulation of 1 line pofil => move through NURBS
    def ligne1(self, x, y, beta, largeurligne, H, pitch):
        position1 = H * abs(np.tan(beta))
        if x == 0 and y == 0:
            return 1  # Def of 0
        elif x > 0 and x < position1 and y < abs(x / np.tan(beta)):
            return 1  # Def of the rising slope with the sidewall angle
        elif x >= (position1) and x < (largeurligne + position1) and y < H:
            return 1  # Def the top of the line
        elif x >= (largeurligne + position1) and x < (largeurligne + 2 * position1) and y < abs(
                        (largeurligne + 2 * position1 - x) / np.tan(beta)):
            return 1  # Def of the decreasing slope
        elif x >= (largeurligne + 2 * position1) and x < (pitch):
            return 0  # Def of space between 2 lines
        elif x >= pitch:
            return 0
        else:
            return 0

    # Function doing the 1D cut along qx of the simulated signal along Position1, Position2, Position3
    def Qxcut(self, I, Position1, Position2, Position3, Taille_image=(600, 600), phimax=np.radians(27)):
        roisizex = np.int(1 / (2 * np.pi) * Taille_image[0])
        roisizey = np.int(1 / (2 * np.pi) * Taille_image[1])
        phimax = np.radians(27)
        center_x = Taille_image[0] / 2
        center_y = Taille_image[1] / 2
        originx = 0
        originy = (roisizey / 2) - 0.5
        Iroi = np.zeros([roisizex+1, roisizey +1 ])
        for i in range(0, roisizex , 1):
            for j in range(1, roisizey +1 , 1):
                if (np.tan(phimax) * (originx - i) / 2) <= (originy - j) and (np.tan(phimax) * (originx - i) / 2) <= -(
                    originy - j) or (np.tan(phimax) * (originx - i) / 2) >= (originy - j) and (
                        np.tan(phimax) * (originx - i) / 2) >= -(originy - j):
                    Iroi[i, j] = (I[i + center_x, j + center_y - roisizey / 2])

        I1 = np.sum(Iroi[int(Position1) - 1:int(Position1) + 1, :], axis=0)
        I2 = np.sum(Iroi[int(Position2) - 1:int(Position2) + 1, :], axis=0)
        I3 = np.sum(Iroi[int(Position3) - 1:int(Position3) + 1, :], axis=0)
        return Iroi, I1, I2, I3

    # Rescale  the experimental and simulated data in qy
    def resize_yset(self, data0, data1):
        max_size = max(data0.size, data1.size)
        if max_size == data0.size:
            data1 = resample(data1, max_size)
        else:
            data0 = resample(data0, max_size)
        return data0, data1

    def centering(self, data0, data1, max_size):
        data0 = np.resize(data0, max_size)
        np.roll(data0, np.int((max_size-data0.size)/2))
        data1 = np.resize(data1, max_size)
        np.roll(data1, np.int((max_size - data1.size) / 2))
        return data0, data1

    # Rescale the experimental and simulated data in intensity
    def resize_iset(self, data0, data1):
        data1 = data1 * (max(data0) / max(data1))
        ind = np.where(data1 < min(data0))
        data1[ind] = min(data0)
        return data0, data1

class CDRawWidget(pg.ImageView):
    pass

class CDCartoWidget(pg.ImageView):
    pass

class CDModelWidget(pg.PlotWidget):
    def __init__(self):
        super(CDModelWidget, self).__init__()
        self.addLegend()
        self.order1 = self.plot([],pen=pg.mkPen('g'),name='Order 1')
        self.order2 = self.plot([],pen=pg.mkPen('y'),name='Order 2')
        self.order3 = self.plot([],pen=pg.mkPen('r'),name='Order 3')
        self.order4 = self.plot([], pen=pg.mkPen('g'), name='Order 4')
        self.order5 = self.plot([], pen=pg.mkPen('y'), name='Order 5')
        self.order6 = self.plot([], pen=pg.mkPen('r'), name='Order 6')

class CDProfileWidget(pg.ImageView):
    pass

'''
class LineEdit(QtGui.QLineEdit):
    def __init__(self, *args, **kwargs):
        super(LineEdit, self).__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setFrame(False)
'''