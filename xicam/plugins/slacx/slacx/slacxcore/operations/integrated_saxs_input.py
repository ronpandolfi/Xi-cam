import os
import re

import numpy as np

from slacxop import Operation
import optools

class LoadIntegratedSaxsData(Operation):
    """
    Takes a filesystem path that points to a .dat file containing a header and four columns,
    (q,intensity,delta-q,delta-Intensity) produces an n-by-2 array of (q,intensity) data
    """

    def __init__(self):
        input_names = ['path']
        output_names = ['saxs_data']
        super(LoadIntegratedSaxsData,self).__init__(input_names,output_names) 
        # default behavior: load from filesystem
        self.input_src['path'] = optools.fs_input
        self.input_doc['path'] = 'string representing the path to the .dat file containing the integrated data'
        self.output_doc['saxs_data'] = 'n-by-4 array with only the numerical data'
        self.categories = ['INPUT'] 
        
    def run(self):
        """
        Read numerical data from the file
        """
        filepath = self.inputs['path']
        data = []
        content = open(filepath).readlines()
        for line in content:
            if line and not re.match('#',line):
                if len(line.split()) == 4:
                    data.append(np.array(line.split(),dtype=float))
        self.outputs['saxs_data'] = np.array(data)[:,0:2]


