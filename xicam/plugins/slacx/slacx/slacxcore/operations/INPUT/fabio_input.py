import os
import re

import numpy as np
import fabio

from ..slacxop import Operation

class LoadFabIO(Operation):
    """
    Takes a filesystem path and calls fabIO to load it. 
    """

    def __init__(self):
        input_names = ['path']
        output_names = ['image_data']
        super(LoadFabIO,self).__init__(input_names,output_names) 
        self.input_doc['path'] = 'string representing the path to a .tif image'
        self.output_doc['image_data'] = '2D array representing pixel values taken from the input file'
        self.categories = ['INPUT'] 
        
    def run(self):
        """
        Call on fabIO to extract image data
        """
        img_url = self.inputs['path']
        self.outputs['image_data'] = fabio.open(img_url).data

