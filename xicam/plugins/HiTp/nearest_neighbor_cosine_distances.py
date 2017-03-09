# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 14:03:38 2016

@author: Tri Duong, Fang Ren
"""

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os.path

def find_neighbour(index, scan_register, length_of_row):
    """
    return the neighbour directly to right of the desired index and neighbour directly underneath the desired index
    """
    register = scan_register[index-1]
    register_neighbour1 = register - 1
    register_neighbour2 = register - length_of_row
    try:
         index_neighbour1 = scan_register.index(register_neighbour1)+1
         index_neighbour2 = scan_register.index(register_neighbour2)+1
    except ValueError:
         index_neighbour1 = index
         index_neighbour2 = index
    return index_neighbour1, index_neighbour2

def file_index(index):
    """
    formatting the index of each file
    """
    if len(str(index)) == 1:
        return '000' + str(index)
    elif len(str(index)) == 2:
        return '00' + str(index)
    elif len(str(index)) == 3:
        return '0' + str(index)
    elif len(str(index)) == 4:
        return str(index)

def import_data(index, save_path, base_filename):
    data = np.genfromtxt(os.path.join(save_path, base_filename + file_index(index) + '_1D.csv'), delimiter= ',')
    neighbor = data[:,1]
    return neighbor


def nearst_neighbor_distance(index, Qlist, IntAve, folder_path, save_path, base_filename, num_of_smpls_on_wafer = 25):
    """
    concatenate all the csv files in a folder, return a 2D numpy array
    """
    master_file = os.path.join(folder_path, base_filename + 'scan1.csv')
    master_data = np.genfromtxt(master_file, delimiter=',', skip_header=1)
    scan_register = list(master_data[:,0])
    index_neighbor1, index_neighbor2 = find_neighbour(index, scan_register, num_of_smpls_on_wafer)
    neighbor1 = import_data(index_neighbor1, save_path, base_filename)
    neighbor2 = import_data(index_neighbor2, save_path, base_filename)
    # plt.plot(Qlist, IntAve)
    # plt.plot(Qlist, neighbor1)
    # plt.plot(Qlist, neighbor2)
    # plt.savefig(save_path + base_filename + file_index(index) + '_neighbors.png')
    # plt.close('all')
    distance_sum = distance.cosine(IntAve, neighbor1) + distance.cosine(IntAve, neighbor2)
    return [index, distance_sum]
