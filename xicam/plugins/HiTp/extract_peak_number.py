# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 15:00:51 2016

@author: Ron Pandolfi (LBL), fangren
"""

from scipy.signal import find_peaks_cwt, general_gaussian, fftconvolve
import numpy as np


def extract_peak_num(Qlist, IntAve, a1 = 1, a2 = 20):
    """
    extract the peak numbers from 1D spectra
    """

    peaks = find_peaks_cwt(IntAve, np.arange(a1, a2, 0.05))
    peaks = peaks[1:-1]
    h = 15  # number of points skipped in finite differences

    peaks_accepted = []
    window = h

    filter = np.nan_to_num(np.sqrt(-(IntAve[2 * h:] - 2 * IntAve[h:-h] + IntAve[0:-2 * h])))
    for peak in peaks:
        # if Qlist[peak] < 3:
            filterwindow = filter[max(peak - h - window, 0):min(peak - h + window, len(filter))]
            spectrawindow = IntAve[max(peak - window, h):min(peak + window, len(filter))]

            try:
                if np.any(filterwindow > spectrawindow / 200):  # np.percentile(filter,85) is also a good threshold
                    peaks_accepted.append(peak)
            except ValueError:
                continue
        # else:
        #     peaks_accepted.append(peak)

    newRow = [len(peaks_accepted)]
    return newRow, peaks_accepted