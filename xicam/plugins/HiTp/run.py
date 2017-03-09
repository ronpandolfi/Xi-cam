# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13

@author: fangren

"""

import os.path
import time
import imp
import numpy as np
import random
import sys

# import modules
reduction = imp.load_source("data_reduction", "data_reduction_smooth.py")
Qchi = imp.load_source("save_Qchi", "save_Qchi.py")
oneDplot = imp.load_source("save_1Dplot", "save_1Dplot.py")
oneDcsv = imp.load_source("save_1Dcsv", "save_1Dcsv.py")
max_ave = imp.load_source("extract_max_ave_intensity", "extract_max_ave_intensity.py")
peak_num = imp.load_source("extract_peak_num", "extract_peak_number.py")
add_feature = imp.load_source("add_feature_to_master", "add_feature_to_master.py")
save_texture = imp.load_source("save_texture_plot_csv", "save_texture_plot_csv.py")
extract_texture = imp.load_source("extract_texture_extent", "extract_texture_extent.py")
neighbor = imp.load_source("nearest_neighbor_distance", "nearest_neighbor_cosine_distances.py")
SNR = imp.load_source("extract_SNR", "extract_signal_to_noise_ratio.py")



def run(filepath, csvpath, detect_dist_pix, detect_tilt_alpha_rad, detect_tilt_beta_rad, wavelength_A, bcenter_x_pix, bcenter_y_pix,
            polarization, smpls_per_row,
            Imax_Iave_ratio_module,
            texture_module,
            signal_to_noise_module,
            neighbor_distance_module,
            add_feature_to_csv_module):

    # initializing parameters  # distance from sample to detector plane along beam direction in pixel space
    Rot = (np.pi * 2 - detect_tilt_alpha_rad) / (2 * np.pi) * 360  # detector rotation
    tilt = detect_tilt_beta_rad / (2 * np.pi) * 360  # detector tilt  # wavelength

    # split filepath into folder_path, filename, and index
    # for example, filepath = 'C:/Users/Sample1/Sample1_10_0001.tif', folder_path = 'C:/Users/Sample1/', filename = 'Sample1_10_0001.tif', index = '0001'
    folder_path, imageFilename = os.path.split(os.path.abspath(filepath))
    index = imageFilename[-8:-4]

    # generate a folder to put processed files
    save_path = os.path.join(folder_path, 'Processed')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # # wait until an image is created
    # print 'waiting for image', filepath + ' to be created...'
    # print("\r")
    # sleep = 0
    # while not os.path.exists(filepath) and sleep < 1000:
    #     time.sleep(1)
    #     sleep += 1
    #     # print 'sleeping'
    # if sleep == 1000:
    #     sys.exit()

    print 'processing image ' + filepath
    print("\r")
    while (1):
        newRow = []
        try:
            # data_reduction to generate Q-chi and 1D spectra, Q
            Q, chi, cake, Qlist, IntAve = reduction.data_reduction(filepath, detect_dist_pix, Rot, tilt, wavelength_A,
                                                                   bcenter_x_pix, bcenter_y_pix, polarization)
            # save Qchi as a plot *.png and *.mat
            Qchi.save_Qchi(Q, chi, cake, imageFilename, save_path)
            # save 1D spectra as a *.csv
            oneDcsv.save_1Dcsv(Qlist, IntAve, imageFilename, save_path)
            # extract composition information if the information is available
            # extract the number of peaks in 1D spectra as attribute3 by default
            newRow3, peaks = peak_num.extract_peak_num(Qlist, IntAve, index)
            newRow.append(newRow3)

            # save 1D plot with detected peaks shown in the plot
            oneDplot.save_1Dplot(Qlist, IntAve, peaks, imageFilename, save_path)

            if Imax_Iave_ratio_module == 'on':
                # extract maximum/average intensity from 1D spectra as attribute1
                newRow1 = max_ave.extract_max_ave_intensity(IntAve, index)
                newRow.append(newRow1)


            if texture_module == 'on':
                # save 1D texture spectra as a plot (*.png) and *.csv
                Qlist_texture, texture = save_texture.save_texture_plot_csv(Q, chi, cake, imageFilename, save_path)
                # extract texture square sum from the 1D texture spectra as attribute2
                newRow2 = extract_texture.extract_texture_extent(Qlist_texture, texture, index)
                newRow.append(newRow2)

            if neighbor_distance_module == 'on':
                # extract neighbor distances as attribute4
                newRow4 = neighbor.nearst_neighbor_distance(index, Qlist, IntAve, folder_path, save_path, csvpath,
                                                            smpls_per_row)
                newRow.append(newRow4)

            if signal_to_noise_module == 'on':
                # extract signal-to-noise ratio
                newRow5 = SNR.extract_SNR(index, IntAve)
                newRow.append(newRow5)

            break
        except (OSError, IOError):
            # The image was being created but not complete yet
            print 'waiting for image', filepath + ' to be ready...'
            time.sleep(1)
            # sleep += 1

    print newRow
    return newRow




