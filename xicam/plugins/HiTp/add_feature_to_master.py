"""
Created on Mar 2017

@author: fangren
"""

import numpy as np
import os.path
import csv

def add_feature_to_master(attributes, save_path):
    """

    """
    csvfile = os.path.join(save_path, 'attributes.csv')
    if os.path.exists(csvfile):
        csv_input = open(csvfile, 'rb')
        reader = csv.reader(csv_input, delimiter=',')
        old_attributes = []
        i = 0
        for row in reader:
            if i == 0:
                header = row
            elif i >=1:
                old_attributes.append(row)
            i += 1
        csv_input.close

        #print attributes[1:,:], old_attributes

        attributes = np.concatenate((old_attributes, attributes[1:,:]))

        csv_output = open(os.path.join(save_path, 'attributes.csv'), 'wb')
        writer = csv.writer(csv_output, delimiter=',')
        writer.writerow(header)
        for row in attributes:
            writer.writerow(row)
        csv_output.close

    else:
        csv_output = open(os.path.join(save_path, 'attributes.csv'), 'wb')
        writer = csv.writer(csv_output, delimiter=',')
        for row in attributes:
            writer.writerow(row)
        csv_output.close
