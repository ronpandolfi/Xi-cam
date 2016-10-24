import numpy as np

from slacxop import Operation


class MovingAverage(Operation):
    """Add two objects."""

    def __init__(self):
        input_names = ['augend', 'addend']
        output_names = ['sum']
        super(MovingAverage, self).__init__(input_names, output_names)
        self.input_doc['augend'] = 'array or number'
        self.input_doc['addend'] = 'array or number for which addition with augend is defined'
        self.output_doc['sum'] = 'augend plus addend'
        self.categories = ['1D DATA PROCESSING']

    def run(self):
        self.outputs['sum'] = self.inputs['augend'] + self.inputs['addend']


class SavitzkyGolay(Operation):
    """Add two objects."""

    def __init__(self):
        input_names = ['augend', 'addend']
        output_names = ['sum']
        super(SavitzkyGolay, self).__init__(input_names, output_names)
        self.input_doc['augend'] = 'array or number'
        self.input_doc['addend'] = 'array or number for which addition with augend is defined'
        self.output_doc['sum'] = 'augend plus addend'
        self.categories = ['1D DATA PROCESSING']

    def run(self):
        self.outputs['sum'] = self.inputs['augend'] + self.inputs['addend']


class RectangularUnweightedSmooth(Operation):
    """Applies rectangular (moving average) smoothing filter to 1d data.

    No error estimate used."""

    def __init__(self):
        input_names = ['data', 'm']
        output_names = ['smoothdata']
        super(RectangularUnweightedSmooth, self).__init__(input_names, output_names)
        self.input_doc['data'] = '1d ndarray'
        self.input_doc['m'] = 'integer number of data points to average locally'
        self.output_doc['smoothdata'] = 'smoothed 1d ndarray'
        self.categories = ['1D DATA PROCESSING']

    def run(self):

        self.outputs['sum'] = self.inputs['augend'] + self.inputs['addend']


def rectangular_unweighted_smooth(data, m):




def masked_mean_2d_axis_0(y2d, mask2d):
    '''
    Takes the mean of masked data along axis 0.

    :param y2d: 2d numpy float array
    :param mask2d: 2d numpy bool array
    :return mean: 1d numpy float array

    *y2d* is data; *mask2d* is its corresponding mask
    with values *True* for legitimate data, *False* otherwise.
    Assumes that each column of *y2d* has at least one valid element;
    otherwise the mean along axis 0 is not defined.
    Returns *mean*, the mean of *y2d* along axis 0.
    '''
    sum = (y2d * mask2d).sum(axis=0)
    num_elements = mask2d.sum(axis=0)
    mean = sum / num_elements
    return mean


def masked_variance_2d_axis_0(y2d, mask2d):
    '''
    Takes the variance of masked data along axis 0.

    :param y2d: 2d numpy float array
    :param mask2d: 2d numpy bool array
    :return variance: 1d numpy float array

    *y2d* is data; *mask2d* is its corresponding mask
    with values *True* for legitimate data, *False* otherwise.
    Assumes that each column of *y2d* has at least two valid elements;
    otherwise the variance along axis 0 is not defined.
    Returns *variance*, the variance of *y2d* along axis 0.
    '''
    mean = masked_mean_2d_axis_0(y2d, mask2d)
    difference = (y2d - mean) * mask2d
    num_elements = mask2d.sum(axis=0)
    variance = (difference ** 2).sum(axis=0) / (num_elements - 1)
    return variance

