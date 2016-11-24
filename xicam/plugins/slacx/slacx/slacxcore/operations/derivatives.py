#import numpy as np

from slacxop import Operation
import optools
#from smoothing import choose_m, choose_start_and_end, make_poly_matrices, polynomial


class DiscreteFirstDerivative(Operation):
    """Take the discrete analogue of the first derivative.

    Returns *slope* (the first derivative) and *new_x*, the coordinate values at which *slope* is best defined.
    *new_x* has one entry fewer than *x*."""

    def __init__(self):
        input_names = ['x', 'y']
        output_names = ['slope', 'new_x']
        super(DiscreteFirstDerivative, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d ndarray; independent variable'
        self.input_doc['y'] = '1d ndarray; dependent variable; same shape as *x*'
        self.output_doc['slope'] = 'rate of change of *y* with respect to *x*'
        self.output_doc['new_x'] = 'coordinates at which the slope is best defined'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.categories = ['1D DATA PROCESSING.DERIVATIVES']

    def run(self):
        self.outputs['slope'], self.outputs['new_x'] = discrete_first_derivative(self.inputs['x'], self.inputs['y'])


class DiscreteSecondDerivative(Operation):
    """Take the discrete analogue of the second derivative."""

    def __init__(self):
        input_names = ['x', 'y']
        output_names = ['curvature', 'new_x']
        super(DiscreteSecondDerivative, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d ndarray; independent variable'
        self.input_doc['y'] = '1d ndarray; dependent variable; same shape as *x*'
        self.output_doc['curvature'] = 'second derivative of *y* with respect to *x*'
        self.output_doc['new_x'] = 'coordinates at which the curvature is best defined'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.categories = ['1D DATA PROCESSING.DERIVATIVES']

    def run(self):
        self.outputs['curvature'], self.outputs['new_x'] = discrete_second_derivative(self.inputs['x'], self.inputs['y'])


def discrete_first_derivative(x, y):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    slope = dy/dx
    new_x = x[:-1] + 0.5 * dx
    return slope, new_x

def discrete_second_derivative(x, y):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    slope = dy / dx
    new_x = x + 0.5 * dx
    new_dx = new_x[1:] - new_x[:-1]
    dslope = slope[1:] - slope[:-1]
    curvature = dslope/new_dx
    new_new_x = new_x[:-1] + 0.5 * new_dx
    return curvature, new_new_x


