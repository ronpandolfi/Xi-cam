# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:48:10 2016

@author: fangren
"""

import numpy as np
import os.path
import csv

def add_feature_to_master(attributes, save_path):
    """

    """
    # generate a folder to put processed files
    save_path = os.path.join(save_path, 'Processed')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    csvfile = os.path.join(save_path, 'attributes1.csv')
    # if os.path.exists(csvfile):
    #     attributes_saved = np.genfromtxt(csvfile, delimiter= ',')
    print attributes
    # attributes = np.concatenate(attributes_saved, attributes)
    np.savetxt(csvfile, attributes, delimiter=',')