# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:48:10 2016

@author: fangren
"""

import numpy as np
import os.path
import csv

def add_feature_to_master(features, base_filename, folder_path, save_path, master_index, index):
    """
    add a feature 'feature' to master meta data, feature is in the form of a ziped row
    """

    master_filename = os.path.join(folder_path, base_filename + 'scan1.csv')
    # print master_filename

    if  os.path.exists(master_filename):
        csv_input = open(master_filename, 'rb')
        reader = csv.reader(csv_input, delimiter=',')
        i = 0
        master_data = []
        for row in reader:
            if i == 0:
                line_for_specPlot = row
            elif i == 1:
                header = row
            else:
                master_data.append(row)
            i += 1
        csv_input.close

        # there are wired string like ' 4.38247e-' in the data, need to replace them with zero first.
        master_data = np.array(master_data)
        for i in range(len(master_data[:, 2])):
            if 'e' in master_data[i][2]:
                master_data[i][2] = 0

        for i in range(len(master_data[:, 1])):
            if 'e' in master_data[i][1]:
                master_data[i][1] = 0

        # change data array into float
        # master_data = master_data.astype(float)

        # for debugging
        # print type(header)
        # print type(features[0,:])

        header = header[:master_data.shape[1]] + list(features[0, :])

        num_of_scan_processed = features.shape[0]-1

        # for debugging
        # print header
        # print dimension
        # print master_data.shape[0], features.shape[0] - 1
        # print num_of_scan_processed
        # print master_data[(index-num_of_scan_processed):index, :].shape
        # print features[:num_of_scan_processed, :].shape

        master_data = np.concatenate((master_data[(index-num_of_scan_processed):index, :], features[1:num_of_scan_processed+1, :]), axis=1)

        # print master_data

        csv_output =  open(os.path.join(save_path, base_filename + master_index + 'master_csv.csv'), 'wb')
        writer = csv.writer(csv_output, delimiter=',')
        writer.writerow(line_for_specPlot)
        writer.writerow(header)
        for row in master_data:
            writer.writerow(row)
        csv_output.close

    else:
        csv_output = open(os.path.join(save_path, base_filename + master_index + 'master_csv.csv'), 'wb')
        writer = csv.writer(csv_output, delimiter=',')
        for row in features:
            writer.writerow(row)
        csv_output.close