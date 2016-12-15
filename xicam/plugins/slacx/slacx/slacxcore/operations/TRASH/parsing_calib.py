# -*- coding: utf-8 -*-
"""
Created on Nov 10

@author: fangren

"""
import numpy as np
#from os.path import join
from os import linesep

from slacxop import Operation
import optools

class ParseCalib(Operation):
    """Parse a .calib file for use with image_to_1D.

    Returns a dictionary with appropriate keys for image_to_1D."""

    def __init__(self):
        input_names = ['filename']
        output_names = ['parameters']
        super(ParseCalib, self).__init__(input_names, output_names)
        self.input_doc['filename'] = 'path to .calib file'
        self.input_src['filename'] = optools.fs_input
        self.output_doc['parameters'] = 'dictionary organized in image_to_1D style'
        # source & type
        self.input_src['filename'] = optools.fs_input
        self.categories = ['INPUT.WXDIFF']

    def run(self):
        self.outputs['parameters'] = parse_calib_dictionary(self.inputs['filename'])




#filename = 'C:\Research_FangRen\Data\Apr2016\Jae_samples\LaB6\\LaB6_11RE.calib'


def parse_calib_dictionary(filename):
    file=open(filename,'r')
    data = []
    with file as inputfile:
        for line in inputfile:
            data.append(line.strip().split(linesep))
    bcenter_x = float(data[6][0][10:])
    bcenter_y = float(data[7][0][10:])
    detector_dist = float(data[8][0][12:])
    detect_tilt_alpha = float(data[9][0][18:])
    detect_tilt_delta = float(data[10][0][18:])
    wavelength = float(data[11][0][11:])
    # Pack in dictionary with image_to_1D style keys
    parameters = {}
    parameters['x0_pixel'] = bcenter_x
    parameters['y0_pixel'] = bcenter_y
    parameters['d_pixel'] = detector_dist
    parameters['rotation_rad'] = detect_tilt_alpha
    parameters['tilt_rad'] = detect_tilt_delta
    parameters['lamda'] = wavelength
    return parameters
