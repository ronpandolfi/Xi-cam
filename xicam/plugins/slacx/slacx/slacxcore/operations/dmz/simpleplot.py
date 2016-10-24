from matplotlib import pyplot as plt

from core.operations.slacxop import Operation


class Plot(Operation):
    """Plot 1d vectors.

    Most arguments optional."""

    def __init__(self):
        input_names = ['x', 'y', 'axis', 'figure']
        output_names = ['axis', 'figure']
        super(Add, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d ndarray; independent variable'
        self.input_doc['y'] = '1d ndarray; dependent variable'
        self.input_doc['axis'] = 'preexisting axis on which to plot; default None'
        self.input_doc['figure'] = 'preexisting figure on which to plot; default None'
        self.output_doc['axis'] = ''
        self.output_doc['figure'] = ''

    def run(self):
        self.outputs['sum'] = self.inputs['augend'] + self.inputs['addend']
