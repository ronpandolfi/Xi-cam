__author__ = "Dinesh Kumar"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque",
               "Holden Parks", "Alexander Hexemer"]
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Alpha"

import numpy as np
import os
from pyqtgraph.parametertree import Parameter
from MSMcam.inout.ImageReader import ImageReader
from MSMcam.preprocessing.PreProcessor import PreProcessor
from MSMcam.segmentation.kmeans import kmeans
from MSMcam.segmentation.srm.pysrm import srm
from MSMcam.segmentation.pmrf import pmrf

class Workflow(object):
    def __init__(self):
        self.preproc_settings = {
            'DownsampleScale': 1, 'UseMedian': False, 'UseBilateral': False,
            'MedianMaskSize': 5, 'BilateralSigmaSpatial': 5, 'BilateralSigmaColor': 0.05
        }
        self.input_settings = {
            'InputType': 0, 'InputDir': None, 'Masked': False, 'GroundTruthExists': False, 
            'GroundTruthDir': None, 'FiberData': False, 'FirstSlice' : 0, 
            'LastSlice': 10, 'InMemory': True
        }

        self.segmentation_settings = {
            'Multiphase': False, 'NumClustersKMEANS': 3, 'NumClustersPMRF': 2,
            'QSRM': 32, 'RunThresh': False, 'ThreshAlg': 'Otsu',
            'RunPMRF': False, 'RunKMEANS': True, 'RunSRM': False, 'Invert': False
        }
        self.segmented = {'k-means': None, 'SRM': None, 'pMRF': None}

    def update_preproc_settings(self, params):
        if not isinstance(params, Parameter):
            raise TypeError('input must of an instance of pyqtgraph.parametertree.Parameter')
        self.preproc_settings['DownsampleScale'] = params.child('Downsample Scale').value()
        self.preproc_settings['UseMedian'] = params.child('Filter').child('Median').value()
        self.preproc_settings['MedianMaskSize'] = params.child('Filter').child('Median').child('Mask Size').value()
        self.preproc_settings['UseBilateral'] = params.child('Filter').child('Bilateral').value()
        self.preproc_settings['BilateralSigmaSpatial'] = params.child('Filter').child('Bilateral').child('Sigma Spatial').value()
        self.preproc_settings['BilateralSigmaColor'] = params.child('Filter').child('Bilateral').child('Sigma Color').value()
        
    def update_input_settings(self, params):
        if not isinstance(params, Parameter):
            raise TypeError('input must of an instance of pyqtgraph.parametertree.Parameter')
        self.input_settings['FiberData'] = params.child('Fiber Data').value()
        self.input_settings['FirstSlice'] = params.child('First Slice').value()
        self.input_settings['LastSlice'] = params.child('Last Slice').value()
        self.input_settings['InMemory'] = params.child('In Memory').value()


    def update_segmentaion_settings(self, params):
        if not isinstance(params, Parameter):
            raise TypeError('input must of an instance of pyqtgraph.parametertree.Parameter')
        nclusters = params.child('Segmentation').child('Clusters').value()
        if nclusters > 2: 
            self.segmentation_settings['Multiphase'] = True
        else:
            self.segmentation_settings['Multiphase'] = False
        self.segmentation_settings['Invert'] = params.child('Segmentation').child('Invert').value()
        self.segmentation_settings['QSRM'] = params.child('Segmentation').child('QSRM').value()
        self.segmentation_settings['RunKMEANS'] = params.child('Segmentation').child('k-means').value()
        self.segmentation_settings['NumClustersKMEANS'] = nclusters
        self.segmentation_settings['RunSRM'] = params.child('Segmentation').child('SRM').value()
        self.segmentation_settings['RunPMRF'] = params.child('Segmentation').child('PMRF').value()
        self.segmentation_settings['NumClustersPMRF'] = nclusters

    def filter(self, data):
        """
        TODO
        """
        if isinstance(data, np.ndarray):
            slc_a = 0
            slc_z = data.shape[0]
            if not slc_a == self.input_settings['FirstSlice']: 
                slc_a = self.input_settings['FristSlice']
            if not slc_z == self.input_settings['LastSlice']:
                slc_z = self.input_settings['LastSlice']
            preproc = PreProcessor(data[slc_a:slc_z,:,:], self.input_settings, self.preproc_settings, 0)
        else:
            self.input_settings['InputDir'] = os.path.dirname(data)
            reader =  ImageReader(self.input_settings['InputDir'], self.input_settings['InputType'], 
                self.input_settings['FirstSlice'], self.input_settings['LastSlice'],
                 False, self.input_settings['InMemory'], 0)
            reader.read()
            path = reader.getImageFilenames() 
            preproc = PreProcessor(path, self.input_settings, self.preproc_settings) 
        preproc.process()
        if self.input_settings['InMemory']:
            self.filtered,_ = preproc.getFiltered()
        else:
            _,self.filtered = preproc.getFiltered()

    def run(self):
        if self.segmentation_settings['RunKMEANS']:
            imgs =  kmeans.segment(self.filtered, self.input_settings, 
                    self.preproc_settings, self.segmentation_settings, 0)
            if self.input_settings['InMemory']:
                self.segmented['k-means'] = imgs[0]
            else:
                self.segmented['k-means'] = imgs[1]

        if self.segmentation_settings['RunSRM']:
            imgs = srm.segment(self.filtered, self.input_settings, 
                        self.preproc_settings, self.segmentation_settings, 0)
            if self.input_settings['InMemory']:
                self.segmented['SRM'] = imgs[0]
            else:
                self.segmented['SRM'] = imgs[1]
            
        if self.segmentation_settings['RunPMRF']:
            imgs = pmrf.segment(self.filtered, self.input_settings, 
                self.preproc_settings, self.segmentation_settings, 0)
            if self.input_settings['InMemory']:
                self.segmented['pMRF'] = imgs[0]
            else:
                self.segmented['pMRF'] = imgs[1]

    def writeConfig(self, filename):
        from configparser import ConfigParser
        from collections import OrderedDict

        c = ConfigParser()
        c.optionxform = str
        c.add_section('Input')
        c.add_section('Preprocess')
        c.add_section('Segmentation')
        c.add_section('Visualization')
        for key, val in self.input_settings.items():
            c.set('Input', key, str(val))

        for key, val in self.preproc_settings.items():
            c.set('Preprocess', key, str(val)) 
        
        for key, val in self.segmentation_settings.items():
            c.set('Segmentation', key, str(val)) 
        
        c['Visualization'] = { 'SavePlots': 'True', 'ViewPlots': 'False' }
        fp = open(filename, 'w')
        c.write(fp)
        fp.close()

    def apply_mask(self):
        """
        TODO
        """
        pass

