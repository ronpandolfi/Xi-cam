import numpy as np

from ..slacxop import Operation
from .. import optools

class SavitzkyGolayWeighted(Operation):
    """
    Apply a Savitzky-Golay fit (polynomial fit approximation) to y(x) data.
    Requires error bars on y, which are used to weight the data during fitting.
    """

    def __init__(self):
        input_names = ['x', 'y', 'dy', 'order', 'base']
        output_names = ['y_smooth']
        super(SavitzkyGolayWeighted, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d array; independent variable'
        self.input_doc['y'] = '1d array; dependent variable, same shape as x'
        self.input_doc['dy'] = '1d array; error estimate in y, same shape as y'
        self.input_doc['order'] = 'integer order of polynomial approximation (zero to five)'
        self.input_doc['base'] = '-1, 0, or positive integer'
        self.output_doc['y_smooth'] = '1d array of smoothed (fit) y values'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.input_src['dy'] = optools.wf_input
        self.input_src['order'] = optools.user_input
        self.input_src['base'] = optools.user_input
        self.input_type['order'] = optools.int_type
        self.input_type['base'] = optools.int_type
        self.categories = ['PROCESSING']

    def run(self):
        self.outputs['y_smooth'] = savitzky_golay(self.inputs['x'], self.inputs['y'], 
        self.inputs['order'], self.inputs['base'], self.inputs['dy'])

class SavitzkyGolayUnweighted(Operation):
    """
    Apply a Savitzky-Golay fit (polynomial fit approximation) to y(x) data.
    Does not require error estimates for y: input data are equally weighted.
    """
    def __init__(self):
        input_names = ['x', 'y', 'order', 'base']
        output_names = ['y_smooth']
        super(SavitzkyGolayUnweighted, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d array; independent variable'
        self.input_doc['y'] = '1d array; dependent variable, same shape as x'
        self.input_doc['order'] = 'integer order of polynomial approximation (zero to five)'
        self.input_doc['base'] = '-1, 0, or positive integer'
        self.output_doc['y_smooth'] = '1d array of smoothed (fit) y values'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.input_src['order'] = optools.user_input
        self.input_src['base'] = optools.user_input
        self.input_type['order'] = optools.int_type
        self.input_type['base'] = optools.int_type
        self.categories = ['PROCESSING']

    def run(self):
        self.outputs['y_smooth'] = savitzky_golay(self.inputs['x'], self.inputs['y'], 
        self.inputs['order'], self.inputs['base'])

class MovingAverageUnweighted(Operation):
    """
    Applies rectangular moving average to 1d data.
    No error estimate used.
    """

    def __init__(self):
        input_names = ['x', 'window_size', 'window_shape']
        output_names = ['x_smooth']
        super(MovingAverageUnweighted, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d array'
        self.input_doc['window_size'] = 'odd integer: number of data points to include in averaging window'
        self.input_doc['window_shape'] = str('shape of the window to use in weighting the moving average. '
            + 'Must be either "square" or "triangle".')
        self.output_doc['x_smooth'] = '1d array: moving average of x'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['window_size'] = optools.user_input
        self.input_type['window_size'] = optools.int_type
        self.inputs['window_size'] = 3
        self.input_src['window_shape'] = optools.user_input
        self.input_type['window_shape'] = optools.str_type
        self.inputs['window_shape'] = 'square' 
        self.categories = ['PROCESSING']

    def run(self):
        self.outputs['x_smooth'] = moving_average(self.inputs['x'], self.inputs['window_size'], self.inputs['window_shape'])

class MovingAverageWeighted(Operation):
    """
    Applies rectangular moving average to 1d data.
    Data error estimate is used to weight the moving average.
    """

    def __init__(self):
        input_names = ['x', 'dx', 'window_size', 'window_shape']
        output_names = ['x_smooth']
        super(MovingAverageWeighted, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d array'
        self.input_doc['dx'] = '1d array, same shape as x'
        self.input_doc['window_size'] = 'odd integer: number of data points to include in averaging window'
        self.input_doc['window_shape'] = str('shape of the window to use in weighting the moving average. '
            + 'Must be either "square" or "triangle".')
        self.output_doc['x_smooth'] = '1d array: moving weighted average of x'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['dx'] = optools.wf_input
        self.input_src['window_size'] = optools.user_input
        self.input_type['window_size'] = optools.int_type
        self.inputs['window_size'] = 3
        self.input_src['window_shape'] = optools.user_input
        self.input_type['window_shape'] = optools.str_type
        self.inputs['window_shape'] = 'square' 
        self.categories = ['PROCESSING']

    def run(self):
        self.outputs['x_smooth'] = moving_average(self.inputs['x'], self.inputs['window_size'], self.inputs['window_shape'], self.inputs['dx'])

def moving_average(data, m, shape, error=np.zeros(1)):
    n = int(m)/2
    if (m != (2*n+1)):
        raise ValueError('Argument *m* should be an odd integer.')
    if error.any() and (data.shape != error.shape):
        raise ValueError('Arguments *data* and *error* should have the same shape.')
    if shape == 'square':
        shape_weight = square_weighting(n)
    elif shape == 'triangle':
        shape_weight = triangular_weighting(n)
    else:
        raise ValueError('Argument *shape* should be either "square" or "triangle".')
    if error.any():
        error_weight = specified_error_weights(error)
    else:
        error_weight = no_specified_error_weights(data)
    sum = data * shape_weight[0] * error_weight
    weights = shape_weight[0] * error_weight
    for ii in range(1, n+1):
        sum[ii:] += (data[:-ii] * shape_weight[ii] * error_weight[:-ii])
        sum[:-ii] += (data[ii:] * shape_weight[ii] * error_weight[ii:])
        weights[ii:] += (shape_weight[ii] * error_weight[:-ii])
        weights[:-ii] += (shape_weight[ii] * error_weight[ii:])
    mean = sum / weights
    return mean

def triangular_weighting(n):
    weights = (n + 1 - np.arange(0, int(n + 1), dtype=float))/float(n + 1)
    return weights

def square_weighting(n):
    weights = np.ones(n + 1, dtype=float)
    return weights

def no_specified_error_weights(data):
    weights = np.ones(data.shape, dtype=float)
    return weights

def specified_error_weights(error):
    weights = error**-2
    return weights

def savitzky_golay(x, y, order, base, dy=np.zeros(1)):
    if not dy.any():
        dy = np.ones(y.shape, dtype=float)
    m = choose_m(order, base) # Order, base checked here; number of fit points chosen
    order = int(order)
    size = x.size
    smoothed = np.zeros(size, dtype=float)
    for ii in range(size):
        # For each pixel, trim out a section of nearby pixels
        start, end = choose_start_and_end(m, base, ii, size)
        # Formulate the equation to be solved for polynomial coefficients
        matrix, vector = make_poly_matrices(x[start:end], y[start:end], dy[start:end], order)
        # Solve equation
        coefficients = (np.linalg.solve(matrix, vector)).flatten()
        # Find chosen approximation
        smoothed[ii] = polynomial(x[ii], coefficients)
    return smoothed

def choose_start_and_end(m, base, ii, size):
    n = (int(m)/2) + 1
    # Base controls edge conditions
    if base == -1:
        constant_base = True
    else:
        constant_base = False
    start = max(ii - n + 1, 0)
    end = min(ii + n - 1, size - 1)
    if constant_base:
        if (ii - n + 1 < 0):
            start = 0
            end = m
        elif (ii + n - 1 > size - 1):
            start = size - 1 - m
            end = size - 1
    return start, end

def choose_m(order, base):
    '''Choose a number of points to use for SG fit.

    :param order: integer order of polynomial fit
    :param base: an edge condition specification parameter
    :return m: integer number of data points to use

    Helper function for *savitzky_golay*.
    '''
    if ((order != int(order)) or (base != int(base))):
        raise ValueError('Arguments *order* and *base* must be integers.')
    if (order < 0):
        raise ValueError('Argument *order* must be a nonnegative integer.')
    if (base < -1):
        raise ValueError('Argument *base* must be -1, 0, or a positive integer.')
    if (order > 5):
        print "Warning: Using polynomials above order 5 not recommended.  There may be cases where this breaks."
    order = int(order)
    base = int(base)
    # "Minimal" point base case.  Chooses the minimum number of points
    # that is both sufficient to fit polynomial of order *order* and odd.
    if base == -1:
        is_order_odd = ((order/2)*2 != order)
        m = order + 1 + 1*is_order_odd
    # "Balanced" point base case.
    elif base == 0:
        m = 2*order + 1
    # "Additional" point base case.
    elif base > 0:
        m = 2*(order + base) + 1
    return m

def polynomial(x, coefficients):
    '''Evaluate a polynomial with given coefficients at location x.

    :param x: numeric value of coordinate
    :param coefficients: 1d ndarray of one or more elements; zeroth element is zero-order coefficient, 1st is 1st-order coefficient, etc.
    :return value: value of the specified polynomial at *x*

    In *coefficients*, zeroth element is zero-order coefficient (i.e. constant offset),
    1st is 1st-order coefficient (i.e. linear slope), etc.
    '''
    order = coefficients.size - 1
    powers = np.arange(0, order+1)
    value = ((x**powers) * coefficients).sum()
    return value

def vertical(array1d):
    '''Turn 1d array into 2d vertical vector.'''
    array1d = array1d.reshape((array1d.size, 1))
    return array1d

def horizontal(array1d):
    '''Turn 1d array into 2d horizontal vector.'''
    array1d = array1d.reshape((1, array1d.size))
    return array1d

def dummy(array1d):
    '''Turn 1d array into dummy-index vector for 2d matrix computation.

    Sum over the dummy index by taking *object.sum(axis=0)*.'''
    array1d = array1d.reshape((array1d.size, 1 , 1))
    return array1d

def make_poly_matrices(x, y, error, order):
    '''Make the matrices necessary to solve a polynomial fit of order *order*.

    :param x: 1d array representing independent variable
    :param y: 1d array representing dependent variable
    :param error: 1d array representing uncertainty in *y*
    :param order: integer order of polynomial fit
    :return matrix, vector: MC=V, where M is *matrix*, V is *vector*,
        and C is the polynomial coefficients to be solved for.
        *matrix* is an array of shape (order+1, order+1).
        *vector* is an array of shape (order+1, 1).
    '''
    if ((x.shape != y.shape) or (y.shape != error.shape)):
        raise ValueError('Arguments *x*, *y*, and *error* must all have the same shape.')
    index = np.arange(order+1)
    vector = (dummy(y) * dummy(x) ** vertical(index) * dummy(error)).sum(axis=0)
    index_block = horizontal(index) + vertical(index)
    matrix = (dummy(x) ** index_block * dummy(error)).sum(axis=0)
    return matrix, vector


