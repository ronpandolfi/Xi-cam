"""
Created on Feb 7 2017

@author: Fang Ren
Contributor: Yijin Liu
"""

from scipy.signal import medfilt, savgol_filter
import numpy as np
from scipy.optimize import curve_fit

def func(x, *params):
    """
    create a Gaussian fitted curve according to params
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def extract_SNR(IntAve):
    filter_window = 15
    IntAve_smoothed = savgol_filter(IntAve, filter_window, 2)
    noise = IntAve - IntAve_smoothed


    ## set initial parameters for Gaussian fit
    guess = [0, 5, 10]
    high = [0.5, 300, 1000]
    low = [-0.5, 0, 0.1]
    bins = np.arange(-100, 100, 0.5)

    # fit noise histogram
    n, bins = np.histogram(noise, bins= bins)
    popt, pcov = curve_fit(func, bins[:-1], n, p0=guess)
    slope = 9.16805348809
    intercept = 60.7206954077

    SNR = slope * np.log(1/popt[2]) + intercept
    return [['SNR', SNR]]