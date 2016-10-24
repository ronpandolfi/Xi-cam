# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:52:16 2016

@author: fangren
"""

import numpy as np

from slacxop import Operation

class extract_texture_sum(Operation):
    def __init__(self):
        input_names = ['Qlist_texture','texture']
        output_names = ['NormSqrSum_texture']
        super(extract_texture_sum,self).__init__(input_names,output_names)
        self.input_doc['texture'] = '2d array representing Q-chi image'
        self.input_doc['Qlist_texture'] = 'momentum transfer'
        self.output_doc['NormSqrSum_texture'] = 'normalized square sum of texture in texture 1D spectrum'
        self.categories = ['2D DATA PROCESSING']

    def run(self):
        texture = np.array(self.inputs['texture'])
        texture = texture **2
        SqrSum = np.nansum(texture)
        NormSqrSum = SqrSum/float(len(self.inputs['Qlist_texture']))
        # save results to self.outputs
        self.outputs['NormSqrSum_texture'] = NormSqrSum
