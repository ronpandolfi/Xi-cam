import numpy as np

from ..slacxop import Operation
from .. import optools


class MirrorVertical(Operation):
    """
    Mirror an image across a vertical plane,
    i.e., exchange indices along axis 1.
    """

    def __init__(self):
        input_names = ['image_in']
        output_names = ['image_out']
        super(MirrorVertical, self).__init__(input_names, output_names)
        self.input_doc['image_in'] = '2d array'
        self.output_doc['image_out'] = 'input array mirrored vertically'
        self.input_src['image_in'] = optools.wf_input
        self.categories = ['PROCESSING']

    def run(self):
        self.outputs['image_out'] = self.inputs['image_in'][:,::-1]

class MirrorHorizontal(Operation):
    """
    Mirror an image across a horizontal plane,
    i.e., exchange indices along axis 0.
    """

    def __init__(self):
        input_names = ['image_in']
        output_names = ['image_out']
        super(MirrorHorizontal, self).__init__(input_names, output_names)
        self.input_doc['image_in'] = '2d ndarray'
        self.output_doc['image_out'] = 'input array mirrored horizontally'
        self.input_src['image_in'] = optools.wf_input
        self.categories = ['PROCESSING']

    def run(self):
        self.outputs['image_out'] = self.inputs['image_in'][::-1,:]

