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
import msmcam

class Workflow(object):
    def __init__(self):
        self.preproc_settings = {
                                'DownsampleScale': 1, 'UseMedian': False, 'UseBilateral': False,
                                'MedianMaskSize': 5, 'BilateralSigmaSpatial': 5,
                                'BilateralSigmaColor': 0.05
                                }
    def load_mask(self):

    def apply_mask(self):
        
        
    def apply_filter(self):

    def run_segementation(self):
        

    @property
    def currentFrame(self):
