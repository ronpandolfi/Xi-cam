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
import widgets
import subprocess
import xicam.RmcView as rmc

from PySide import QtGui, QtCore

from pipeline.spacegroups import spacegroupwidget
from xicam import config
import fabio

from xicam.widgets.calibrationpanel import calibrationpanel


class plugin(base.plugin):
    name = "cdsaxs"

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.rightwidget = QtGui.QTabWidget()
        self.bottomwidget = QtGui.QTabWidget()
        self.topwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabClose)
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.rightwidget.setDocumentMode(True)
        self.rightwidget.setTabsClosable(True)
        self.rightwidget.tabCloseRequested.connect(self.tabClose)
        #self.rightwidget = None


        '''
        VBoxButton_1 = QtGui.QVBoxLayout()
        self.rightwidget.addWidget(VBoxButton_1, 'Phi_min')
        VBoxButton_2 = QtGui.QVBoxLayout()
        self.rightwidget.addTab(VBoxButton_2, 'Phi_max')
        VBoxButton_3 = QtGui.QVBoxLayout()
        self.rightwidget.addTab(VBoxButton_3, 'Phi_step')

        VerticalLayout = QtGui.QVBoxLayout()
        self.rightwidget.addTab(VerticalLayout, 'test1')
        funcButton_3 = QtGui.QPushButton("test")
        self.rightwidget.addTab(funcButton_3, 'test')
        '''


        super(plugin, self).__init__(*args, **kwargs)

        funcButton_1 = QtGui.QPushButton("Run interpol")
        # self.centerwidget.addTab(funcButton_1, 'interpol')
        funcButton_1.clicked.connect(self.main)

        funcButton_2 = QtGui.QPushButton("Run fit")
        # self.rightwidget.addTab(funcButton_2, 'fit')
        funcButton_2.clicked.connect(self.fitting)

    def openfiles(self, files, operation=None, operationname=None):
        self.activate()
        if type(files) is not list:
            files = [files]
        print files
        widget = widgets.OOMTabItem(itemclass=CDSAXSWidget, src=files, operation=operation,
                                    operationname=operationname, plotwidget=self.bottomwidget,
                                    toolbar=self.toolbar)
        self.centerwidget.addTab(widget, os.path.basename(files[0]))
        self.centerwidget.setCurrentWidget(widget)

    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()

    def tabClose(self,index):
        self.centerwidget.widget(index).deleteLater()

    def main(self):
        prefix = 'contacta1_hs104_'
        serie1 = '060_'
        wavelength = 0.072932
        sampling_size = (400,400)
        phi_min, phi_max, phi_step = -30, 30, 1
        center_x, center_y = 248, 1668  # Position of the direct beam

        I_norm = self.tiff('/Users/guillaumefreychet/GitHub/Xi-cam/xicam/plugins/cdsaxs/contacta1_hs104_060_0000.tif')
        qxyi = np.zeros([len(I_norm[0]), 3], dtype=np.float32)
        qxyi = self.data_fusion_2D(qxyi, prefix, wavelength, serie1, phi_min, phi_max, phi_step, center_x, center_y)

        self.interpolation(qxyi, sampling_size)

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

        plt.imshow(img, interpolation='none', norm=LogNorm(vmin=100, vmax=1000000));
        plt.show()
        return img



    def get_exp_values(self):
        cut_val = 0.1256
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
        binf_3, bsup_3 = 2 * cut_val - delta, 2 * cut_val + delta

        idx_1 = np.where((SQi['qx'] > binf_1) * (SQi['qx'] < bsup_1))  # selection contraints by qy vals
        idx_2 = np.where((SQi['qx'] > binf_2) * (SQi['qx'] < bsup_2))
        idx_3 = np.where((SQi['qx'] > binf_3) * (SQi['qx'] < bsup_3))

        plt.plot(SQi['qy'][idx_1], SQi['i'][idx_1])
        plt.plot(SQi['qy'][idx_2], SQi['i'][idx_2])
        plt.plot(SQi['qy'][idx_3], SQi['i'][idx_3])
        plt.yscale('log')
        plt.show()

        return SQi['i'][idx_1], SQi['i'][idx_2], SQi['i'][idx_2]



    def fitting(self):
        print 'Function under maintenance'
        self.get_exp_values()
        # initial_value => load by the user
        # Could be  implement in function of the precision needed
        initial_value = (320, 35, 2)
        bnds = ((305, 320), (32, 37), (0.5, 3.))
        opt = minimize(self.residual(qxyi), initial_value, bounds=bnds, method='L-BFGS-B',
                       options={'disp': True, 'eps': (1, 0.2, 0.1), 'ftol': 0.00001})
        print(opt.x)
        print(opt.message)

    def residual(self, qxyi, p, plot_mode=False):
        H, LL, beta = p
        Qxexp1, Qxexp2, Qxexp3  = self.get_exp_values(qxyi)


        Qxfit1, Qxfit2, Qxfit3 = self.SL_model(H, LL, beta)

        # recalage en y
        Qxexp1, Qxfit1 = self.resize_yset(Qxexp1, Qxfit1)
        Qxexp2, Qxfit2 = self.resize_yset(Qxexp2, Qxfit2)
        Qxexp3, Qxfit3 = self.resize_yset(Qxexp3, Qxfit3)

        # recalage en intensite
        Qxexp1, Qxfit1 = self.resize_iset(Qxexp1, Qxfit1)
        Qxexp2, Qxfit2 = self.resize_iset(Qxexp2, Qxfit2)
        Qxexp3, Qxfit3 = self.resize_iset(Qxexp3, Qxfit3)

        # permet de visualiser la correspondance entre profil
        if plot_mode:
            plt.plot(Qxexp1)
            plt.plot(Qxexp2)
            plt.plot(Qxexp3)
            plt.plot(Qxfit1)
            plt.plot(Qxfit2)
            plt.plot(Qxfit3)
            plt.yscale('log')
            plt.xlim(0, 128)
            plt.show()
            sys.exit()

        # Calcul de la difference
        res = (sum(abs(Qxfit1 - Qxexp1)) + sum(abs(Qxfit2 - Qxexp2)) + sum(abs(Qxfit3 - Qxexp3))) / (
        sum(Qxexp1) + sum(Qxexp2) + sum(Qxexp3))
        print(p)
        print('fval : ', res)
        return res



    def SL_model(self, H=300, LL=35, beta=1, plot_mode=False):
        I = []
        pitch = 100  # To implement with the approach as an entry parameter
        nbligne = 1  # To implement with the approach as an entry parameter
        I = self.Fitlignes(H, LL, beta, pitch, nbligne)
        # Fitting qx cut
        Iroi = []
        Tailleimagex = 2000
        qref = 1.5 * 0.0628
        Position1 = np.floor(qref / (2 * np.pi / Tailleimagex))
        Position2 = np.floor(2 * qref / (2 * np.pi / Tailleimagex))
        Position3 = np.floor(3 * qref / (2 * np.pi / Tailleimagex))
        Iroi, Qxfit1, Qxfit2, Qxfit3 = self.Qxcut(I, Position1, Position2, Position3)

        if plot_mode:
            plt.plot(Qxfit1[Qxfit1.nonzero()[0]])
            plt.plot(Qxfit2[Qxfit2.nonzero()[0]])
            plt.plot(Qxfit3[Qxfit3.nonzero()[0]])
            plt.yscale('log')
            plt.show()

        return Qxfit1[Qxfit1.nonzero()[0]], Qxfit2[Qxfit2.nonzero()[0]], Qxfit3[Qxfit3.nonzero()[0]]

    # Generation of the form factor for the line profile generated with the fonction ligne1
    def Fitlignes(self, pitch, beta, Largeurligne, H, nbligne, Taille_image=(2000, 2000)):
        # assert pitch >= Largeurligne+2*H*abs(np.tan(beta)), 'uncorrect desription of lines'
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
                    Obj[a + (Tailleximage / 2), b + (Tailleyimage - H) / 2] = self.ligne1(x, b, beta, Largeurligne, H, pitch)
        I = np.random.poisson(abs(fftshift(fftn(Obj))) ** 2)
        Dynamic = I.max() / 1000
        II = np.zeros(I.shape, dtype='float64')
        III = np.zeros(I.shape, dtype='int64')
        II = (I * Dynamic) / I.max()
        III = np.int64((II >= 1) * II)
        return III

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
    def Qxcut(self, I, Position1, Position2, Position3, Taille_image=(2000, 2000), phimax=np.radians(27)):
        roisizex = np.int(1 / (2 * np.pi) * Taille_image[0])
        roisizey = np.int(1 / (2 * np.pi) * Taille_image[1])
        phimax = np.radians(27)
        center_x = Taille_image[0] / 2
        center_y = Taille_image[1] / 2
        originx = 0
        originy = roisizey / 2
        Iroi = np.zeros([roisizex, roisizey])
        for i in range(1, roisizex, 1):
            for j in range(1, roisizey, 1):
                if (np.tan(phimax) * (originx - i) / 2) <= (originy - j) and (np.tan(phimax) * (originx - i) / 2) <= -(
                    originy - j) or (np.tan(phimax) * (originx - i) / 2) >= (originy - j) and (
                        np.tan(phimax) * (originx - i) / 2) >= -(originy - j):
                    Iroi[i, j] = (I[i + center_x, j + center_y - roisizey / 2])

        I1 = np.sum(Iroi[Position1 - 1:Position1 + 1, :], axis=0)
        I2 = np.sum(Iroi[Position2 - 1:Position2 + 1, :], axis=0)
        I3 = np.sum(Iroi[Position3 - 1:Position3 + 1, :], axis=0)
        return Iroi, I1, I2, I3

    # Rescale  the experimental and simulated data in qy
    def resize_yset(data0, data1):
        max_size = max(data0.size, data1.size)
        if max_size == data0.size:
            data1 = self.resample(data1, max_size)
        else:
            data0 = self.resample(data0, max_size)
        return data0, data1

    # Rescale the experimental and simulated data in intensity
    def resize_iset(self, data0, data1):
        data1 = data1 * (max(data0) / max(data1))
        ind = np.where(data1 < min(data0))
        data1[ind] = min(data0)
        return data0, data1


class CDSAXSWidget(QtGui.QTabWidget):
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

    def loadRAW(self):
        data = np.stack([np.rot90(loader.loadimage(file),2) for file in self.src])
        data = np.log(data - data.min()+1.)

        self.CDRawWidget.setImage(data)


class CDRawWidget(pg.ImageView):
    pass

class CDCartoWidget(pg.ImageView):
    pass

class CDModelWidget(pg.PlotWidget):
    pass
