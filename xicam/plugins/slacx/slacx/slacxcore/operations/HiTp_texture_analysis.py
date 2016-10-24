# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:18:45 2016

@author: Fang Ren
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

from slacxop import Operation

class texture_analysis(Operation):
    def __init__(self):
        input_names = ['cake','Q', 'chi']
        output_names = ['Qlist_texture', 'texture']
        super(texture_analysis,self).__init__(input_names,output_names)
        self.input_doc['cake'] = '2d array representing Q-chi image'
        self.input_doc['Q'] = 'momentum transfer'
        self.input_doc['chi'] = 'out of plane angle'
        self.output_doc['texture'] = '2d array representing Q-chi image'
        self.output_doc['Qlist_texture'] = 'momentum transfer'
        self.categories = ['2D DATA PROCESSING']

    def run(self):
        Q, chi = np.meshgrid(self.inputs['Q'], self.inputs['chi'])

        keep = (self.inputs['cake'] != 0)
        chi = chi*np.pi/180

        cake = keep.astype(np.int) * self.inputs['cake']

        IntSum = np.bincount((Q.ravel()*100).astype(int), cake.ravel().astype(int))
        count = np.bincount((Q.ravel()*100).astype(int), keep.ravel().astype(int))
        IntAve = list(np.array(IntSum)/np.array(count))

        textureSum = np.bincount((Q.ravel()*100).astype(int), (cake*np.cos(chi)).ravel())
        chiCount = np.bincount((Q.ravel()*100).astype(int), (keep*np.cos(chi)).ravel())

        texture = list(np.array(textureSum)/np.array(IntAve)/np.array(chiCount)-1)

        step = 0.01
        Qlen = int(np.max(Q)/step+1)
        Qlist_texture = [i*step for i in range(Qlen)]

        # save results to self.outputs
        self.outputs['Qlist_texture'] = Qlist_texture
        self.outputs['texture'] = texture
