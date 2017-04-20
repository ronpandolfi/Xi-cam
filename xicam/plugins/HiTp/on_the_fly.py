# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13

@author: fangren

"""

import os.path
import time
import numpy as np
import random
import sys
from data_reduction import data_reduction
from save_Qchi import save_Qchi
from save_1Dplot import save_1Dplot
from save_1Dcsv import save_1Dcsv
from extract_max_ave_intensity import extract_max_ave_intensity
from extract_peak_number import extract_peak_num
from add_feature_to_master import add_feature_to_master
from save_texture_plot_csv import save_texture_plot_csv
from extract_texture_extent import extract_texture_extent
from nearest_neighbor_cosine_distances import nearst_neighbor_distance
from extract_signal_to_noise_ratio import extract_SNR

def run(filepath, csvpath, detect_dist_pix, detect_tilt_alpha_rad, detect_tilt_beta_rad, wavelength_A, bcenter_x_pix, bcenter_y_pix,
            polarization, smpls_per_row,
            Imax_Iave_ratio_module,
            texture_module,
            signal_to_noise_module,
            neighbor_distance_module,
            add_feature_to_csv_module):
    # split filepath into folder_path, filename, and index
    # for example, filepath = 'C:/Users/Sample1/Sample1_10_0001.tif', folder_path = 'C:/Users/Sample1/', filename = 'Sample1_10_0001.tif', index = '0001'
    folder_path, imageFilename = os.path.split(os.path.abspath(filepath))
    #folder_path = os.path.dirname(filepath)
    index = imageFilename[-8:-4]

    # generate a folder to put processed files
    save_path = os.path.join(folder_path, 'Processed')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # initializing parameters  # distance from sample to detector plane along beam direction in pixel space
    Rot = (np.pi * 2 - detect_tilt_alpha_rad) / (2 * np.pi) * 360  # detector rotation
    tilt = detect_tilt_beta_rad / (2 * np.pi) * 360  # detector tilt  # wavelength

    print 'processing image ' + filepath
    print("\r")
    while 1:
        try:
            attributes = [index]
            # data_reduction to generate Q-chi and 1D spectra, Q
            Q, chi, cake, Qlist, IntAve = data_reduction(filepath, detect_dist_pix, Rot, tilt, wavelength_A,
                                                                   bcenter_x_pix, bcenter_y_pix, polarization)
            # save Qchi as a plot *.png and *.mat
            save_Qchi(Q, chi, cake, imageFilename, save_path)
            # save 1D spectra as a *.csv
            save_1Dcsv(Qlist, IntAve, imageFilename, save_path)
            # extract composition information if the information is available
            # extract the number of peaks in 1D spectra as attribute3 by default
            attribute3, peaks = extract_peak_num(Qlist, IntAve, index)
            attributes.append(attribute3)

            # save 1D plot with detected peaks shown in the plot
            save_1Dplot(Qlist, IntAve, peaks, imageFilename, save_path)

            if Imax_Iave_ratio_module == True:
                # extract maximum/average intensity from 1D spectra as attribute1
                attribute1 = extract_max_ave_intensity(IntAve, index)
                attributes.append(attribute1)


            if texture_module == True:
                # save 1D texture spectra as a plot (*.png) and *.csv
                Qlist_texture, texture = save_texture_plot_csv(Q, chi, cake, imageFilename, save_path)
                # extract texture square sum from the 1D texture spectra as attribute2
                attribute2 = extract_texture_extent(Qlist_texture, texture, index)
                attributes.append(attribute2)

            if neighbor_distance_module == True:
                # extract neighbor distances as attribute4
                attribute4 = nearst_neighbor_distance(index, Qlist, IntAve, folder_path, save_path, csvpath,
                                                            smpls_per_row)
                attributes.append(attribute4)

            if signal_to_noise_module == True:
                # extract signal-to-noise ratio
                attribute5 = extract_SNR(index, IntAve)
                attributes.append(attribute5)
            break
        except (OSError, IOError):
            # The image was being created but not complete yet
            print 'waiting for image', filepath + ' to be ready...'
            time.sleep(1)





