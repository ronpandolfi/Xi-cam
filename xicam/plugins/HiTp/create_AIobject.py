"""
Created on Mar 2017

@author: fangren, Ron Pandolfi
"""

import pyFAI
import numpy as np


def create_AIobject(d_in_pixel, detect_tilt_alpha_rad, detect_tilt_beta_rad, lamda, x0, y0):
    """
    The input is the image array and calibration parameters
    return Q-chi (2D array) and a spectrum (1D array)
    """

    # initializing parameters  # distance from sample to detector plane along beam direction in pixel space
    Rot = (np.pi * 2 - detect_tilt_alpha_rad) / (2 * np.pi) * 360  # detector rotation
    tilt = detect_tilt_beta_rad / (2 * np.pi) * 360  # detector tilt  # wavelength

    pixelsize = 79    # measured in microns
    d = d_in_pixel*pixelsize*0.001  # measured in milimeters
    p = pyFAI.AzimuthalIntegrator(wavelength=lamda)
    p.setFit2D(d, x0, y0, tilt, Rot, pixelsize, pixelsize)

    return p

