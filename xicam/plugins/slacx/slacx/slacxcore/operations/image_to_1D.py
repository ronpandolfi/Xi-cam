"""
Transform a raw image into Q-chi plot

Created on Tue Oct 4

@author: fangren
"""

import pyFAI
import numpy as np

from slacxop import Operation
import optools


class image_to_1D(Operation):
    """
    The input is a raw image and calibration parameters in WxDiff format
    return cake (2D array), Q, chi, Integrated intensity (1D array), Qlist
    """
    def __init__(self):
        input_names = ['image_data','d_pixel', 'rotation_rad', 'tilt_rad', 'lamda', 'x0_pixel', 'y0_pixel', 'PP', 'pixel_size']
        output_names = ['Intensity','Qlist']
        super(image_to_1D,self).__init__(input_names,output_names)
        # docstrings
        self.input_doc['image_data'] = '2d array representing intensity for each pixel'
        self.input_doc['d_pixel'] = 'detector to sample distance (in pixels) along x-ray direction (WxDiff)'
        self.input_doc['rotation_rad'] = 'rotation angle in radian (WxDiff)'
        self.input_doc['tilt_rad'] = 'tilt angle in radian (WxDiff)'
        self.input_doc['lamda'] = 'beam energy in angstrom (WxDiff)'
        self.input_doc['x0_pixel'] = 'beam center x in detector space (WxDiff)'
        self.input_doc['y0_pixel'] = 'beam center y in detector space (WxDiff)'
        self.input_doc['PP'] = 'polarization factor'
        self.input_doc['pixel_size'] = 'detector pixel size in microns'
        self.output_doc['Intensity'] = 'Integrated intensity averaged by pixels #'
        self.output_doc['Qlist'] = 'momentum transfer in a list'
        # source & type
        self.input_src['image_data'] = optools.wf_input
        self.input_src['d_pixel'] = optools.user_input
        self.input_src['rotation_rad'] = optools.user_input
        self.input_src['tilt_rad'] = optools.user_input
        self.input_src['lamda'] = optools.user_input
        self.input_src['x0_pixel'] = optools.user_input
        self.input_src['y0_pixel'] = optools.user_input
        self.input_src['PP'] = optools.user_input
        self.input_src['pixel_size'] = optools.user_input
        self.input_type['d_pixel'] = optools.float_type
        self.input_type['rotation_rad'] = optools.float_type
        self.input_type['tilt_rad'] = optools.float_type
        self.input_type['lamda'] = optools.float_type
        self.input_type['x0_pixel'] = optools.float_type
        self.input_type['y0_pixel'] = optools.float_type
        self.input_type['PP'] = optools.float_type
        self.input_type['pixel_size'] = optools.float_type

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
        # 1 for masked pixels, and 0 for valid pixels
        detector_mask = np.ones((s,s))*(imArray <= 0)

        p = pyFAI.AzimuthalIntegrator(wavelength=lamda)
        p.setFit2D(d,x0,y0,tilt,Rot,pixelsize,pixelsize)

        Qlist, IntAve = p.integrate1d(imArray, 1000, mask=detector_mask, polarization_factor=PP)
        Qlist = Qlist * 10e8

        # save results to self.outputs
        self.outputs['Intensity'] = IntAve
        self.outputs['Qlist'] = Qlist

class image_to_1D_simple(Operation):
    """
    The input is a raw image and calibration parameters in WxDiff format
    return cake (2D array), Q, chi, Integrated intensity (1D array), Qlist
    """
    def __init__(self):
        input_names = ['image_data', 'calib_file', 'PP', 'pixel_size']
        output_names = ['Intensity', 'Qlist']
        super(image_to_1D_simple,self).__init__(input_names,output_names)
        # docstrings
        self.input_doc['image_data'] = '2d array representing intensity for each pixel'
        self.input_doc['calib_file'] = 'detector to sample distance (in pixels) along x-ray direction (WxDiff)'
        self.input_doc['PP'] = 'polarization factor'
        self.input_doc['pixel_size'] = 'detector pixel size in microns'
        self.output_doc['Intensity'] = 'Integrated intensity averaged by pixels #'
        self.output_doc['Qlist'] = 'momentum transfer in a list'
        # source & type
        self.input_src['image_data'] = optools.wf_input
        self.input_src['calib_file'] = optools.fs_input
        self.input_src['PP'] = optools.user_input
        self.input_src['pixel_size'] = optools.user_input
        self.input_type['PP'] = optools.float_type
        self.input_type['pixel_size'] = optools.float_type
        self.categories = ['2D DATA PROCESSING']

    def run(self):
        """
        transform self.inputs['image_data'] and save as self.outputs['image_data']
        """
        print "image_to_1D_simple running"
        imArray = self.inputs['image_data']

        from parsing_calib import parse_calib_dictionary
        parameters = parse_calib_dictionary(self.inputs['calib_file'])
        print "parameters parsed"

        # initialization parameters, change into Fit2D format
        Rot = (2*np.pi-parameters['rotation_rad'])/(2*np.pi)*360
        tilt = parameters['tilt_rad']/ (2 * np.pi) * 360
        lamda = parameters['lamda']
        x0 = parameters['x0_pixel']
        y0 = parameters['y0_pixel']
        PP = self.inputs['PP']
        pixelsize = self.inputs['pixel_size']
        d = parameters['d_pixel'] * pixelsize * 0.001
        print "values set up"

        s = int(imArray.shape[0])
        # define detector mask, alternatively, can be another input
        # 1 for masked pixels, and 0 for valid pixels
        detector_mask = np.ones((s,s))*(imArray <= 0)
        print "mask chosen"

        p = pyFAI.AzimuthalIntegrator(wavelength=lamda)
        p.setFit2D(d,x0,y0,tilt,Rot,pixelsize,pixelsize)
        print "fit constructed"

        Qlist, IntAve = p.integrate1d(imArray, 1000, mask=detector_mask, polarization_factor=PP)
        Qlist = Qlist * 10e8
        print "foreground q, I made"
        print "q.shape, I.shape, q.dtype, I.dtype", Qlist.shape, IntAve.shape, Qlist.dtype, IntAve.dtype

        # save results to self.outputs
        self.outputs['Intensity'] = IntAve
        self.outputs['Qlist'] = Qlist

