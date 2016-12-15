from matplotlib import pyplot as plt

from ..slacxop import Operation
from .. import optools

class SimplePlot(Operation):
    """Plot a 1d array against another 1d array."""

    def __init__(self):
        input_names = ['x', 'y']
        output_names = ['figure', 'axis']
        super(SimplePlot, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d array; independent variable'
        self.input_doc['y'] = '1d array; dependent variable'
        self.output_doc['figure'] = 'matplotlib.pyplot.Figure with a plot of y vs x'
        self.output_doc['axis'] = 'matplotlib.pyplot.Axes object for use in plot modification'
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.categories = ['DISPLAY']

    def run(self):
        self.outputs['figure'], self.outputs['axis'] = self.simple_plot(self.inputs['x'], self.inputs['y'])

    def simple_plot(x, y):
        fig, ax = plt.subplots(1)
        ax.plot(x, y)
        return fig, ax

class MPLFigFromXYData(Operation):

    def __init__(self):
        input_names = ['x', 'y']
        output_names = ['figure']
        super(MPLFigFromXYData, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d array; independent variable'
        self.input_doc['y'] = '1d array; dependent variable'
        self.output_doc['figure'] = 'matplotlib.pyplot.Figure with a plot of y vs x'
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.categories = ['DISPLAY']

    def run(self):
        fig = plt.figure()
        plt.plot(self.inputs['x'],self.inputs['y'])
        self.outputs['figure'] = fig


