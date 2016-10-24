# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 15:00:51 2016

@author: fangren
"""

from os.path import basename
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
from scipy.optimize import curve_fit

from slacxop import Operation

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

class extract_peak_num(Operation):
    """
    extract the peak numbers from 1D spectra
    """
    def __init__(self):
        input_names = ['Qlist', 'IntAve', 'criterion']
        output_names = ['peaks', 'peak_num']
        super(extract_peak_num, self).__init__(input_names, output_names)
        self.input_doc['Intensity'] = 'Integrated intensity averaged by pixels #'
        self.input_doc['Qlist'] = 'momentum transfer in a list'
        self.output_doc['peaks'] = 'list of peaks'
        self.output_doc['peak_num'] = 'peak numbers'
        self.categories = ['1D DATA PROCESSING']

    def run(self):
        maxtab, mintab = peakdet.peakdet(self.inputs['IntAve'], self.inputs['criterion'])
        if len(maxtab) > 0:
            peaks = maxtab[:,0].astype(int)
            if Qlist[peaks[0]] < 1:
                peaks = peaks[1:]
            if peaks[-1] > 930:
                peaks = peaks[:-1]
        else:
            peaks = []
            # save results to self.outputs
            self.outputs['peaks'] = peaks
            self.outputs['peak_num'] = peak_num
