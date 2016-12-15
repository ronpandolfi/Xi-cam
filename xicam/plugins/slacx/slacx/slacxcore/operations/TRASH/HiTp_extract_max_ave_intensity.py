# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 18:02:32 2016

@author: fangren
"""

import numpy as np

class extract_max_ave_intensity(IntAve, index):
    """
    extract the maximum intensity, average intensity, and a ratio of the two from data
    """
    Imax = np.max(IntAve)
    Iave = np.mean(IntAve)
    ratio = Imax/Iave
    newRow = [index, Imax, Iave, ratio]
    return newRow