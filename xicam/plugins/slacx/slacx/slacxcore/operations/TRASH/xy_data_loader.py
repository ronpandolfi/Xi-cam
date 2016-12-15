import numpy as np

from ..slacxop import Operation
from .. import optools

class DummyXYData(Operation):
    """Load dummy x y arrays"""

    def __init__(self):
        input_names = []
        output_names = ['x','y']
        super(DummyXYData, self).__init__(input_names, output_names)
        self.output_doc['x'] = 'x values for testing'
        self.output_doc['y'] = 'y values for testing'
        self.categories = ['TESTS']

    def run(self):
        self.outputs['x'] = np.arange(10)
        self.outputs['y'] = np.sin(self.outputs['x'])


