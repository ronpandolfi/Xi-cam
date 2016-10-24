# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 15:00:51 2016

@author: fangren
"""

import imp
peakdet = imp.load_source("peakdet", "peak_detection.py")

def extract_peak_num(Qlist, IntAve, criterion, index):
    """
    extract the peak numbers from 1D spectra
    """
    maxtab, mintab = peakdet.peakdet(IntAve, criterion)
    if len(maxtab) > 0:
        peaks = maxtab[:,0].astype(int)
        if Qlist[peaks[0]] < 1:
            peaks = peaks[1:]
        if peaks[-1] > 930:
            peaks = peaks[:-1]
    else: 
        peaks = []
    newRow = [index, len(peaks)]
    return newRow, peaks