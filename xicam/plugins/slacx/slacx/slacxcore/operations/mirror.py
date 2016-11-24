import numpy as np

from slacxop import Operation
import optools


class MirrorVertically(Operation):
    """Mirror an image, exchanging top and bottom.

    I.e., mirror an ndarray along axis = 1."""

    def __init__(self):
        input_names = ['image_in']
        output_names = ['image_out']
        super(MirrorVertically, self).__init__(input_names, output_names)
        self.input_doc['image_in'] = '2d ndarray'
        self.output_doc['image_out'] = '2d ndarray'
        # source & type
        self.input_src['image_in'] = optools.wf_input
        self.categories = ['2D DATA PROCESSING','MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['image_out'] = self.inputs['image_in'][:,::-1]



class MirrorHorizontally(Operation):
    """Mirror an image, exchanging left and right.

    I.e., mirror an ndarray along axis = 0.
    """

    def __init__(self):
        input_names = ['image_in']
        output_names = ['image_out']
        super(MirrorHorizontally, self).__init__(input_names, output_names)
        self.input_doc['image_in'] = '2d ndarray'
        self.output_doc['image_out'] = '2d ndarray'
        # source & type
        self.input_src['image_in'] = optools.wf_input
        self.categories = ['2D DATA PROCESSING','MISC.NDARRAY MANIPULATION']

    def run(self):
        self.outputs['image_out'] = self.inputs['image_in'][::-1,:]

