"""
Created on Mar 2017

@author: fangren, Apurva Mehta
"""


import numpy as np

def extract_max_ave_intensity(IntAve):
    """
    extract the maximum intensity, average intensity, and a ratio of the two from data
    """
    Imax = np.max(IntAve)
    Iave = np.mean(IntAve)
    ratio = Imax/Iave
    newRow = [['Imax', Imax], ['Iave',Iave], ['Imax/Iave', ratio]]
    return newRow