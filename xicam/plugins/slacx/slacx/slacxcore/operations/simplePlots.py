#import numpy as np
from matplotlib import pyplot as plt

from slacxop import Operation
import optools


class SimplePlot(Operation):
    """Plot a vector against another vector."""

    def __init__(self):
        input_names = ['x', 'y']
        output_names = ['figure', 'axis']
        super(SimplePlot, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d ndarray; independent variable'
        self.input_doc['y'] = '1d ndarray; dependent variable'
        self.output_doc['figure'] = 'figure of *x* vs *y*; display this'
        self.output_doc['axis'] = 'axis object of *x* vs *y*; need this to modify plot further'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.categories = ['DISPLAY']

    def run(self):
        self.outputs['figure'], self.outputs['axis'] = simple_plot(self.inputs['x'], self.inputs['y'])

def simple_plot(x, y):
    fig, ax = plt.subplots(1)
    ax.plot(x, y)
    return fig, ax


class MPLFigFromXYData(Operation):

    def __init__(self):
        input_names = ['x', 'y']
        output_names = ['figure']
        super(MPLFigFromXYData, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d ndarray; independent variable'
        self.input_doc['y'] = '1d ndarray; dependent variable'
        self.output_doc['figure'] = 'figure of *x* vs *y*; display this'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.categories = ['DISPLAY.TESTS']

    def run(self):
        fig = plt.figure()
        plt.plot(self.inputs['x'],self.inputs['y'])
        self.outputs['figure'] = fig


