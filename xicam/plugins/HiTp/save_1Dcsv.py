# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13

@author: fangren

"""

import numpy as np
import os.path

def save_1Dcsv(Qlist, IntAve, imageFilename, save_path):
    """
    Qlist and IntAve are data. They are two 1D arrays created by 1D integrate function. 
    The function takes the two arrays and write them into two columns in a csv file
    imageFilename has the fomart of "*_0100.tif", the 1D csv will have the format of "_0100_1D.csv"
    """
    data= np.concatenate(([Qlist], [IntAve]))
    np.savetxt(os.path.join(save_path, imageFilename[:-4]+'_1D.csv'), data.T, delimiter=',')

