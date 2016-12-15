# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 18:02:32 2016

@author: fangren
"""

import numpy as np

from ..slacxop import Operation
from .. import optools

class IntensityFeatures(Operation):
    """
    Extract the maximum intensity, average intensity, and a ratio of the two from data
    """

    def __init__(self):
        input_names = ['I_spectrum']
        output_names = ['Imax', 'Iave', 'Imax_Iave_ratio']
        super(IntensityFeatures, self).__init__(input_names, output_names)
        self.input_doc['I_spectrum'] = 'A 1d vector representing the intensity spectrum'
        self.input_src['I_spectrum'] = optools.wf_input
        self.output_doc['Imax'] = 'The maximum intensity '
        self.output_doc['Iave'] = 'The average intensity'
        self.output_doc['Imax_Iave_ratio'] = 'The ratio of maximum to average intensity'
        self.categories = ['PROCESSING']

    def run(self):
        Imax = np.max(self.inputs['I_spectrum'])
        Iave = np.mean(self.inputs['I_spectrum'])
        ratio = Imax/Iave
        self.outputs['Imax'] = Imax
        self.outputs['Iave'] = Iave
        self.outputs['Imax_Iave_ratio'] = ratio

class TextureFeatures(Operation):
    """
    Analyze the texture 
    """

    def __init__(self):
        input_names = ['q','chi','I']
        output_names = ['q_texture','texture','int_sqr_texture']
        super(TextureFeatures,self).__init__(input_names,output_names)
        self.input_doc['q'] = '1d array of momentum transfer values'
        self.input_doc['chi'] = '1d array of out-of-plane diffraction angles'
        self.input_doc['cake'] = '2d array representing intensities at q,chi points'
        self.input_src['q'] = optools.wf_input
        self.input_src['chi'] = optools.wf_input
        self.input_src['cake'] = optools.wf_input 
        self.output_doc['q_texture'] = 'q values at which the texture is analyzed'
        self.output_doc['texture'] = 'quantification of texture for each q_texture'
        self.output_doc['int_sqr_texture'] = 'integral over q of the texture squared'
        self.categories = ['PROCESSING']

    def run(self):
        q, chi = np.meshgrid(self.inputs['q'], self.inputs['chi']*np.pi/float(180))
        keep = (self.inputs['I'] != 0)
        I = keep.astype(int) * self.inputs['I']

        # TODO: This appears to be a binning operation.
        # Maybe the bin size should not be hard-coded. 
        I_sum = np.bincount((q.ravel()*100).astype(int), I.ravel().astype(int))
        count = np.bincount((q.ravel()*100).astype(int), keep.ravel().astype(int))
        I_ave = list(np.array(I_sum)/np.array(count))
        texsum = np.bincount((q.ravel()*100).astype(int), (I*np.cos(chi)).ravel())
        chi_count = np.bincount((q.ravel()*100).astype(int), (keep*np.cos(chi)).ravel())
        texture = list(np.array(texsum)/np.array(I_ave)/np.array(chi_count)-1)
        step = 0.01
        q_texture = np.arange(step,np.max(q)+step)
        tsqr_int = np.nansum(texture ** 2)/float(q_texture[-1]-q_texture[0])
        self.outputs['q_texture'] = q_texture
        self.outputs['texture'] = texture
        self.outputs['int_sqr_texture'] = tsqr_int 

class PeakFeatures(Operation):
    """
    Extract the locations and intensities of peaks from a 1D spectrum
    """
    def __init__(self):
        input_names = ['q', 'I', 'delta_I']
        output_names = ['q_pk', 'I_pk']
        super(PeakFeatures,self).__init__(input_names, output_names)
        self.input_doc['q'] = '1d vector for x-axis of spectrum, named q for momentum transfer vector'
        self.input_doc['I'] = '1d vector for spectral intensities at q values'
        self.input_doc['delta_I'] = str('Criterion for peak finding: point is a maximum '
            + 'that is more than delta-I larger than the next-lowest point')
        self.input_src['q'] = optools.wf_input
        self.input_src['I'] = optools.wf_input
        self.input_src['delta_I'] = optools.user_input
        self.input_type['delta_I'] = optools.float_type
        self.inputs['delta_I'] = 0.0
        self.output_doc['q_pk'] = 'q values of found peaks'
        self.output_doc['I_pk'] = 'intensities of found peaks'
        self.categories = ['PROCESSING']

    def run(self):
        maxtab, mintab = self.get_extrema(self.inputs['q'], self.inputs['I'], self.inputs['delta_I'])
        q_pk = maxtab[:,0]
        I_pk = maxtab[:,1] 
        # save results to self.outputs
        self.outputs['q_pk'] = q_pk 
        self.outputs['I_pk'] = I_pk 

    def get_extrema(x, y, delta):
        """Given vectors x and y, return an n-by-2 array of x,y pairs for the minima and maxima of y(x)"""
        maxtab = []
        mintab = []
        x = np.array(x)
        y = np.array(y)
        y_min, y_max = np.Inf, -np.Inf
        x_min, x_max = np.NaN, np.NaN
        for i in arange(len(y)):
            if y[i] > y_max:
                x_max = x[i]
                y_max = y[i] 
            if y[i] < y_min:
                x_min = x[i]
                y_min = y[i] 
            lookformax = True
            if lookformax:
                if y[i] < y_max-delta:
                    maxtab.append((x_max, y_max))
                    y_min = y[i] 
                    x_min = x[i]
                    lookformax = False
            else:
                if y[i] > y_min+delta:
                    mintab.append((x_min, y_min))
                    y_max = y[i] 
                    x_max = x[i]
                    lookformax = True
        return np.array(maxtab), np.array(mintab)


