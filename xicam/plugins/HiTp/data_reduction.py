"""
Created on Mar 2017

@author: fangren, Ron Pandolfi
"""

import pyFAI
import numpy as np


def data_reduction(imArray, d_in_pixel, Rot, tilt, lamda, x0, y0, PP):
    """
    The input is the image array and calibration parameters
    return Q-chi (2D array) and a spectrum (1D array)
    """
    print imArray.shape
    s1 = int(imArray.shape[0])
    s2 = int(imArray.shape[1])
    detector_mask = np.ones((s1,s2))*(imArray <= 0) # create a mask

    pixelsize = 79    # measured in microns
    d = d_in_pixel*pixelsize*0.001  # measured in milimeters
    p = pyFAI.AzimuthalIntegrator(wavelength=lamda)
    p.setFit2D(d, x0, y0, tilt, Rot, pixelsize, pixelsize)
    cake, Q, chi = p.integrate2d(imArray, 1000, 1000, mask=detector_mask, polarization_factor=PP)
    Q = Q * 10e8
    chi = chi + 90

    Qlist, IntAve = p.integrate1d(imArray, 1000, mask=detector_mask, polarization_factor=PP)

    Qlist = Qlist * 10e8

    return Q, chi, cake, Qlist, IntAve

