"""
Transform a raw image into Q-chi plot

Created on Tue Oct 4

@author: fangren
"""

import pyFAI
import numpy as np

from slacxop import Operation

class image_to_Qchi(Operation):
    """
    The input is a raw image and calibration parameters in WxDiff format
    return cake (2D array), Q, chi, Integrated intensity (1D array), Qlist
    """
    def __init__(self):
        input_names = ['image_data','d_pixel', 'rotation_rad', 'tilt_rad', 'lamda', 'x0_pixel', 'y0_pixel', 'PP', 'pixel_size']
        output_names = ['cake','Q','chi']
        super(image_to_Qchi,self).__init__(input_names,output_names)
        self.input_doc['image_data'] = '2d array representing intensity for each pixel'
        self.input_doc['d_pixel'] = 'detector to sample distance (in pixels) along x-ray direction (WxDiff)'
        self.input_doc['rotation_rad'] = 'rotation angle in radian (WxDiff)'
        self.input_doc['tilt_rad'] = 'tilt angle in radian (WxDiff)'
        self.input_doc['lamda'] = 'beam energy in angstrom (WxDiff)'
        self.input_doc['x0_pixel'] = 'beam center x in detector space (WxDiff)'
        self.input_doc['y0_pixel'] = 'beam center y in detector space (WxDiff)'
        self.input_doc['PP'] = 'polarization factor'
        self.input_doc['pixel_size'] = 'detector pixel size in microns'
        self.output_doc['cake'] = '2d array representing Q-chi image'
        self.output_doc['Q'] = 'momentum transfer'
        self.output_doc['chi'] = 'out of plane angle'
        self.categories = ['2D DATA PROCESSING']

    def run(self):
        """
        transform self.inputs['image_data'] and save as self.outputs['image_data']
        """
        imArray = self.inputs['image_data']

        # initialization parameters, change into Fit2D format
        Rot = (2*np.pi-self.inputs['rotation_rad'])/(2*np.pi)*360
        tilt = self.inputs['tilt_rad']/ (2 * np.pi) * 360
        lamda = self.inputs['lamda']
        x0 = self.inputs['x0_pixel']
        y0 = self.inputs['y0_pixel']
        PP = self.inputs['PP']
        pixelsize = self.inputs['pixel_size']
        d = self.inputs['d_pixel'] * pixelsize * 0.001

        s = int(imArray.shape[0])

        # define detector mask, alternatively, can be another input
        detector_mask = np.ones((s,s))*(imArray <= 0)

        p = pyFAI.AzimuthalIntegrator(wavelength=lamda)
        p.setFit2D(d,x0,y0,tilt,Rot,pixelsize,pixelsize)
        cake,Q,chi = p.integrate2d(imArray,1000, 1000, mask = detector_mask, polarization_factor = PP)
        Q = Q * 10e8
        chi = chi+90

        # save results to self.outputs
        self.outputs['cake'] = cake
        self.outputs['Q'] = Q
        self.outputs['chi'] = chi


