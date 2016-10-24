# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 18:02:32 2016

@author: fangren
"""

import numpy as np
from slacxop import Operation

class extract_max_ave_intensity(Operation):
    """
    extract the maximum intensity, average intensity, and a ratio of the two from data
    """

    def __init__(self):
        input_names = ['IntAve']
        output_names = ['Imax', 'Iave', 'Imax_Iave_ratio']
        super(extract_max_ave_intensity, self).__init__(input_names, output_names)
        self.input_doc['IntAve'] = 'Integrated intensity averaged by pixels #'
        self.output_doc['Imax'] = 'maximum intensity from 1D spectrum'
        self.output_doc['Iave'] = 'average intensity from 1D spectrum'
        self.output_doc['ratio'] = 'ratio of maximum intensity and average intensity'
        self.categories = ['1D DATA PROCESSING']

    def run(self):
        Imax = np.max(self.inputs['IntAve'])
        Iave = np.mean(self.inputs['IntAve'])
        ratio = Imax/Iave
        # save results to self.outputs
        self.outputs['Imax'] = Imax
        self.outputs['Iave'] = Iave
        self.outputs['Imax_Iave_ratio'] = ratio
