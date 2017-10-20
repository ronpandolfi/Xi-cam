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
from pyqtgraph.parametertree import Parameter
from MSMcam.preprocessing import PreProcessor
from MSMcam.segmentation import kmeans, srm, pmrf

class Workflow(object):
    def __init__(self):
        self.preproc_settings = {
            'DownsampleScale': 1, 'UseMedian': False, 'UseBilateral': False,
            'MedianMaskSize': 5, 'BilateralSigmaSpatial': 5, 'BilateralSigmaColor': 0.05
        }
        self.input_settings = {
            'InputType': 0, 'InputDir': None, 'Masked': False, 'GroundTruthExists': False, 
            'GroundTruthDir': None, 'FiberData': False, 'NumSlices' : 10, 'InMemory': True
        }

        self.segmentation_settings = {
            'Multiphase': False, 'NumClustersKMEANS': 3, 'NumClustersPMRF': 2,
            'RunPMRF': False, 'RunKMEANS': False, 'RunSRM': False, 'Invert': False
        }

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
        self.input_settings['NumSlices'] = params.child('No. of Slices').value()


    def update_segmentaion_settings(self, params):
        if not isinstance(params, Parameter):
            raise TypeError('input must of an instance of pyqtgraph.parametertree.Parameter')
        self.segmentation_settings['Multiphase'] = params.child('Segmentation').child('Multiphase').value()
        self.segmentation_settings['Invert'] = params.child('Segmentation').child('Multiphase').value()
        self.segmentation_settings['RunKMEANS'] = params.child('Segmentation').child('k-means').value()
        self.segmentation_settings['NumClustersKMEANS'] = params.child('Segmentation').child('k-means').child('Clusters').value()
        self.segmentation_settings['RunSRM'] = params.child('Segmentation').child('SRM').value()
        self.segmentation_settings['RunPMRF'] = params.child('Segmentation').child('PMRF').value()
        self.segmentation_settings['NumClustersPMRF'] = params.child('Segmentation').child('PMRF').child('Clusters').value()


    def run(self, images):
        preproc = PreProcessor(images, self.input_settings, self.preproc_settings)
        filtered,_ = preproc.getFiltered()
        segmented_images = {'kmeans': None, 'srm': None, 'pmrf': None}
        if segmentation_settings['RunKMEANS']:
            imgs,_ =  kmeans.segment(filtered, self.input_settings, self.preproc_settings, self.segmentation_settings)
            segmented_images['kmeans'] = imgs

        if segmentation_settings['RunSRM']:
            imgs,_ = srm.segment(filtered, self.input_settings, self.preproc_settings, self.segmentation_settings)
            segmented_images['srm'] = imgs
            
        if segmentation_settings['RunPMRF']:
            imgs,_ = srm.segment(filtered, self.input_settings, self.preproc_settings, self.segmentation_settings)
            segmented_images['pmrf'] = imgs
        
        return segmented_images


    def apply_mask(self):
        """
        TODO
        """
        pass
    def apply_filter(self):
        """
        TODO
        """
        pass
    def run_segementation(self):
        """
        TODO
        """
        pass
