"""
Created on Mar 2017

@author: fangren
"""

import os.path
import time
import numpy as np
import random
import sys
from image_loader import load_image
from data_reduction import data_reduction
from save_Qchi import save_Qchi
from save_1Dplot import save_1Dplot
from save_1Dcsv import save_1Dcsv
from extract_metadata import extract_metadata
from extract_max_ave_intensity import extract_max_ave_intensity
from extract_peak_number import extract_peak_num
from add_feature_to_master import add_feature_to_master
from save_texture_plot_csv import save_texture_plot_csv
from extract_texture_extent import extract_texture_extent
from nearest_neighbor_cosine_distances import nearst_neighbor_distance
from extract_signal_to_noise_ratio import extract_SNR
from bckgrd_subtract import bckgrd_subtract
from peak_fitting_GLS import peak_fitting_GLS

def run(filepath, p,
            polarization, smpls_per_row,
            Imax_Iave_ratio_module,
            texture_module,
            signal_to_noise_module,
            neighbor_distance_module,
            add_feature_to_csv_module,
            background_subtract_module,
            peak_fitting_module):
    # split filepath into folder_path, filename, and index
    # for example, filepath = 'C:/Users/Sample1/Sample1_10_0001.tif', folder_path = 'C:/Users/Sample1/', filename = 'Sample1_10_0001.tif', index = '0001'
    folder_path, imageFilename = os.path.split(os.path.abspath(filepath))
    #folder_path = os.path.dirname(filepath)
    index = int(imageFilename[-8:-4])
    #print index

    # generate a folder to put processed files
    save_path = os.path.join(folder_path, 'Processed')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print 'processing image ' + filepath
    print("\r")

    try:
        # import image and convert it into an array
        imArray = load_image(filepath)

    except (OSError, IOError, IndexError, ValueError):
        # The image was being created but not complete yet
        print 'waiting for image', filepath + ' to be ready...'
        # wait 1 second and try again
        time.sleep(1)
        imArray = load_image(filepath)

    # data_reduction to generate Q-chi and 1D spectra, Q
    Q, chi, cake, Qlist, IntAve = data_reduction(imArray, p, polarization)

    attributes = [['scan_num', index]]

    # add metadata to master file
    if add_feature_to_csv_module:
        metadata = extract_metadata(filepath)
        attributes = np.concatenate((attributes, metadata))

    # save Qchi as a plot *.png and *.mat
    save_Qchi(Q, chi, cake, imageFilename, save_path)
    # save 1D spectra as a *.csv
    save_1Dcsv(Qlist, IntAve, imageFilename, save_path)
    # extract composition information if the information is available
    # extract the number of peaks in 1D spectra as attribute3 by default
    attribute3, peaks = extract_peak_num(Qlist, IntAve)
    attributes = np.concatenate((attributes, attribute3))

    # save 1D plot with detected peaks shown in the plot
    save_1Dplot(Qlist, IntAve, peaks, imageFilename, save_path)


    if Imax_Iave_ratio_module:
        # extract maximum/average intensity from 1D spectra as attribute1
        attribute1 = extract_max_ave_intensity(IntAve)
        attributes = np.concatenate((attributes, attribute1))


    if texture_module:
        # save 1D texture spectra as a plot (*.png) and *.csv
        Qlist_texture, texture = save_texture_plot_csv(Q, chi, cake, imageFilename, save_path)
        # extract texture square sum from the 1D texture spectra as attribute2
        attribute2 = extract_texture_extent(Qlist_texture, texture)
        attributes = np.concatenate((attributes, attribute2))

    if neighbor_distance_module:
        # extract neighbor distances as attribute4
        attribute4 = nearst_neighbor_distance(index, Qlist, IntAve, folder_path, save_path, imageFilename[:-8],
                                                    smpls_per_row)
        attributes = np.concatenate((attributes, attribute4))

    if signal_to_noise_module:
        # extract signal-to-noise ratio
        attribute5 = extract_SNR(IntAve)
        attributes = np.concatenate((attributes, attribute5))

    # print attributes

    if add_feature_to_csv_module:
        add_feature_to_master(attributes.T, save_path)

    if background_subtract_module:
        bckgrd_subtracted = bckgrd_subtract(imageFilename, save_path, Qlist, IntAve)

    if background_subtract_module and peak_fitting_module:
        peak_fitting_GLS(imageFilename, save_path, Qlist, bckgrd_subtracted, 5, 20)

