# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13

@author: fangren

"""

import pyFAI
from PIL import Image
import numpy as np
from scipy import signal

def data_reduction(imageFullname, d_in_pixel, Rot, tilt, lamda, x0, y0, PP):
    # open MARCCD tiff image
    im = Image.open(imageFullname)
    # change image object into an array
    imArray = np.array(im)
    s = int(imArray.shape[0])
    imArray = signal.medfilt(imArray, kernel_size = 5)
    im.close()
    
    detector_mask = np.ones((s,s))*(imArray <= 0)
    pixelsize = 79    # measured in microns
    d = d_in_pixel*pixelsize*0.001  # measured in milimeters

    p = pyFAI.AzimuthalIntegrator(wavelength=lamda)
    p.setFit2D(d,x0,y0,tilt,Rot,pixelsize,pixelsize)
    cake,Q,chi = p.integrate2d(imArray,1000, 1000, mask = detector_mask, polarization_factor = PP)
    # the output unit for Q is angstrom-1
    Q = Q * 10e8
    chi = chi+90

    Qlist, IntAve = p.integrate1d(imArray, 1000, mask = detector_mask, polarization_factor = PP)

    # the output unit for Q is angstrom-1
    Qlist = Qlist * 10e8
    return Q, chi, cake, Qlist, IntAve

