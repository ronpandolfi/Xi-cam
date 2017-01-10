#! /usr/bin/env python

import os
import h5py
import numpy as np
import pipeline
import viewer
from pipeline import loader
from PySide import QtCore, QtGui
from xicam import xglobals


class SPOT_H5DSet:
    def __init__(self, dat, txt):
        self.data = dat
        tmp = dat.attrs
        self.dim1 = tmp['dim2']
        self.dim2 = tmp['dim3']
        self.date = tmp['date']
        self.header = tmp['edfHeaderInBytes']
        self.text = txt

    @classmethod
    def calibration(cls, grp):
        if not isinstance(grp, h5py._hl.group.Group):
            raise TypeError('argument must of a h5py Group')

        keys = grp.keys()
        if not len(keys) == 2:
            raise ValueError('Wrong h5py Group passed')

        txt = None
        edf = None
        for key in keys:
            if str(key).endswith('edf'):
                edf = grp[key]
            if str(key).endswith('txt'):
                txt = grp[key]
            return cls(edf, txt)


class SPOT_H5Node:
    def __init__(self, grp):
        if not isinstance(grp, h5py._hl.group.Group):
            raise TypeError('argument must of a h5py Group')
        keys = grp.keys()
        if not len(keys) == 3:
            raise ValueError('Wrong h5py Group passed')

        txt = None
        edf = None
        for key in keys:
            if key == unicode('calibration'):
                self.calibration = SPOT_H5DSet.calibration(grp[key])
            if str(key).endswith('.edf'):
                edf = grp[key]
            if str(key).endswith('.txt'):
                txt = grp[key]
        if edf == None or txt == None:
            raise IOError('file not found')
        self.data = SPOT_H5DSet(edf, txt)


class SPOT_H5:
    def __init__(self, filename):
        if not os.path.splitext(filename)[1] == '.h5':
            raise IOError('File extension must be .h5')

        self.h5 = h5py.File(filename)
        self.metadata = self.h5.attrs
        self.isTiled = False
        node = self.h5.values()[0].values()[0]
        keys = node.keys()

        if len(keys) == 2 and unicode('high') in keys:
            self.isTiled = True
            self.high = SPOT_H5Node(node['high'])
            self.node = SPOT_H5Node(node['low'])
        elif len(keys) == 3 and unicode('calibration') in keys:
            self.node = SPOT_H5Node(node)

    def getCalibrationImg(self):
        data = self.node.calibration.data
        shape = data.shape[1:]
        return data.value.reshape(shape)

    def getCalibrationHeader(self):
        return self.node.calibration.header

    def getImage(self):
        data = self.node.data.data
        shape = data.shape[1:]
        return data.value.reshape(shape)

    def getHeader(self):
        return self.node.data.header

    def getHighImage(self):
        data = self.high.data.data
        shape = data.shape[1:]
        return data.value.reshape(shape)

    def getHighHeader(self):
        return self.high.data.header

    def getText(self):
        return self.node.data.text.value[0]

    def getCalibrationText(self):
        return self.node.calibration.text.value[0]


class SPOTViewerPlugin(viewer.ViewerPlugin):
    name = "SPOTH5"

    def openfiles(self, filenames):
        h5 = SPOT_H5(filenames[0])
        data = h5.getCalibrationImg()
        self.opendata(data=data)
        self.calibrate()



        # stich high and low
        if h5.isTiled:
            low_header = {'Detector Vertical': 0, 'Detector Horizontal': 0}
            high_header = {'Detector Vertical': 6.88, 'Detector Horizontal': 0}
            low_data = h5.getImage()
            high_data = h5.getHighImage()
            data = loader.loadstitched(None, None, low_data, high_data, low_header, high_header)
            self.opendata(data)


if __name__ == '__main__':
    foo = SPOT_H5('/Users/dkumar/Downloads/20151004_234826_PorRun5_283.h5')

    txt = foo.getText()
    print txt
    dat = foo.getCalibrationImg()
    # saveedf("AgB_000.edf", dat, hdr)
    
