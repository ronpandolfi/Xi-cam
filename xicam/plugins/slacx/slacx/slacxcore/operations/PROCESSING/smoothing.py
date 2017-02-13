import numpy as np

from ..slacxop import Operation
from .. import optools



class SavitzkyGolay(Operation):
    """Applies a Savitzky-Golay (polynomial fit approximation) filter to 1d data.

    Uses error bars on intensity if available.  Set *dy* to *None* otherwise.
    """
    def __init__(self):
        input_names = ['x', 'y', 'dy', 'order', 'base']
        output_names = ['smoothdata']
        super(SavitzkyGolay, self).__init__(input_names, output_names)
        self.input_doc['x'] = '1d ndarray; independent variable'
        self.input_doc['y'] = '1d ndarray; dependent variable, same shape as *x*'
        self.input_doc['dy'] = '1d ndarray; error estimate in *y*, same shape as *y*; if unavailable, set to *None*'
        self.input_doc['order'] = 'integer order of polynomial approximation (zero to five)'
        self.input_doc['base'] = '-1, 0, or positive integer; see class docs'
        self.output_doc['smoothdata'] = 'smoothed 1d ndarray'
        # source & type
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.input_src['dy'] = optools.wf_input
        self.input_src['order'] = optools.user_input
        self.input_src['base'] = optools.user_input
        self.input_type['order'] = optools.int_type
        self.input_type['base'] = optools.int_type
        # defaults
        #self.inputs['dy'] = None
        self.categories = ['1D DATA PROCESSING.SMOOTHING']

    def run(self):
        self.outputs['smoothdata'] = savitzky_golay(self.inputs['x'], self.inputs['y'], self.inputs['order'], self.inputs['base'], self.inputs['dy'])



class RectangularSmooth(Operation):
    """Applies rectangular (moving average) smoothing filter to 1d data.

    User-specified error estimate used to weight points.  Set *dy* to *None* if unavailable."""

    def __init__(self):
        input_names = ['data', 'error', 'm']
        output_names = ['smoothdata']
        super(RectangularSmooth, self).__init__(input_names, output_names)
        self.input_doc['data'] = '1d ndarray'
        self.input_doc['error'] = '1d ndarray, same shape as data; if unavailable, set to *None*'
        self.input_doc['m'] = 'odd integer number of data points to average locally'
        self.output_doc['smoothdata'] = 'smoothed 1d ndarray'
        # source & type
        self.input_src['data'] = optools.wf_input
        self.input_src['error'] = optools.wf_input
        self.input_src['m'] = optools.user_input
        self.input_type['m'] = optools.int_type
        self.categories = ['1D DATA PROCESSING.SMOOTHING']

    def run(self):
        self.outputs['smoothdata'] = moving_average(self.inputs['data'], self.inputs['m'], 'square',
                                                    self.inputs['error'])


class TriangularSmooth(Operation):
    """Applies triangular-weighted (moving average) smoothing filter to 1d data.

    User-specified error estimate used to weight points.  Set *dy* to *None* if unavailable."""

    def __init__(self):
        input_names = ['data', 'error', 'm']
        output_names = ['smoothdata']
        super(TriangularSmooth, self).__init__(input_names, output_names)
        self.input_doc['data'] = '1d ndarray'
        self.input_doc['error'] = '1d ndarray, same shape as data; if unavailable, set to *None*'
        self.input_doc['m'] = 'odd integer number of data points to average locally'
        self.output_doc['smoothdata'] = 'smoothed 1d ndarray'
        # source & type
        self.input_src['data'] = optools.wf_input
        self.input_src['error'] = optools.wf_input
        self.input_src['m'] = optools.user_input
        self.input_type['m'] = optools.int_type
        self.categories = ['1D DATA PROCESSING.SMOOTHING']

    def run(self):
        self.outputs['smoothdata'] = moving_average(self.inputs['data'], self.inputs['m'], 'triangle',
                                                    self.inputs['error'])


def moving_average(data, m, shape, error=None):
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
    if error is not None:
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


def savitzky_golay(x, y, order, base, dy=None):
    if dy is None:
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




'''
Time testing code here

import time
n = 11
x = np.random.rand(n)
y = np.random.rand(n)
error = np.random.rand(n)
order = 5
matrix, vector = make_poly_matrices(x, y, error, order)




t0 = time.time()

t1 = time.time()
dt = t1 - t0
print "Elapsed time: %f seconds." % dt



#rectangular_unweighted_smooth_1(data, m) # data.size = 1024.  m = 3: 0.008655 s / 0.000416 s.  m = 5: 0.000369 s.  m = 7: 0.000319 s.
#rectangular_unweighted_smooth_2(data, m) # m = 3: 0.008419 s / 0.000274 s.  m = 5: 0.000292 s.  m = 7: 0.000324 s.

#solve_poly_1(matrix, vector) # (n, o) = (11, 5) 0.022688 s
#solve_poly_2(matrix, vector) # (n, o) = (11, 5) 0.010989 s
#solve_poly_3(matrix, vector) # (n, o) = (11, 5) 0.008370 s

'''

# Below this point: rejected & unused functions


def solve_poly_1(matrix, vector):
    '''
    Solve a set of linear equations MC = V for C.

    :param matrix: 2d array of dimensions (n, n), n some integer, M in equation
    :param vector: 2d array of dimensions (1, n), V in equation
    :return coefficients: 2d array of dimensions (1, n), C in equation
    '''
    matrix_inv = np.linalg.pinv(matrix)
    coefficients = np.matmul(matrix_inv, vector)
    return coefficients


def solve_poly_2(matrix, vector):
    '''
    Solve a set of linear equations MC = V for C.

    :param matrix: 2d array of dimensions (n, n), n some integer, M in equation
    :param vector: 2d array of dimensions (1, n), V in equation
    :return coefficients: 2d array of dimensions (1, n), C in equation
    '''
    matrix_inv = np.linalg.inv(matrix)
    coefficients = np.matmul(matrix_inv, vector)
    return coefficients


def solve_poly_3(matrix, vector):
    '''
    Solve a set of linear equations MC = V for C.

    :param matrix: 2d array of dimensions (n, n), n some integer, M in equation
    :param vector: 2d array of dimensions (1, n), V in equation
    :return coefficients: 2d array of dimensions (1, n), C in equation
    '''
    coefficients = np.linalg.solve(matrix, vector)
    return coefficients


def rectangular_unweighted_smooth_1(data, m):
    n = int(m)/2
    if (m != (2*n+1)):
        raise ValueError('Argument *m* should be an odd integer.')
    stacked_data, stacked_mask = shift_stack(data, n, n)
    smoothed = masked_mean_2d_axis_0(stacked_data, stacked_mask)
    return smoothed


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


def shift_stack(y, n1, n2):
    '''
    Creates a stack of index-shifted versions of y.

    :param y: 1d numpy float array
    :param n1: int
    :param n2: int
    :return local_neighborhood: 2d numpy float array
    :return element_exists: 2d numpy bool array

    Creates shifted versions of the input *y*,
    with shifts up to and including *n1* spaces downward in index
    and up to and including *n2* spaces upwards.
    The shifted versions are stacked together as *local_neighborhood*, like this
    (shown for a *y* of length 16 and *n1 = 4*, *n2 = 2*)
    [4 5 6 7 ... 15 __ __ __ __]
    [3 4 5 6 ... 14 15 __ __ __]
    [2 3 4 5 ... 13 14 15 __ __]
    [1 2 3 4 ... 12 13 14 15 __]
    [0 1 2 3 ... 11 12 13 14 15]
    [_ 0 1 2 ... 10 11 12 13 14]
    [_ _ 0 1 ...  9 10 11 12 13]
    with a corresponding mask array, *element_exists*,
    indicating whether an element holds information or not, like this
    [1 1 1 1 ...  1  0  0  0  0]
    [1 1 1 1 ...  1  1  0  0  0]
    [1 1 1 1 ...  1  1  1  0  0]
    [1 1 1 1 ...  1  1  1  1  0]
    [1 1 1 1 ...  1  1  1  1  1]
    [0 1 1 1 ...  1  1  1  1  1]
    [0 0 1 1 ...  1  1  1  1  1]
    '''
    local_neighborhood = np.zeros(((n1 + n2 + 1), y.size), dtype=float)
    element_exists = np.zeros(((n1 + n2 + 1), y.size), dtype=bool)
    for ii in range(n1 + n2 + 1):
        # ii ranges from 0 to n1 + n2; jj ranges from -n1 to n2
        jj = ii - n1
        if jj < 0:
            local_neighborhood[ii, :jj] = y[-jj:]
            element_exists[ii, :jj] = True
        elif jj == 0:
            local_neighborhood[ii, :] = y[:]
            element_exists[ii, :] = True
        else:
            local_neighborhood[ii, jj:] = y[:-jj]
            element_exists[ii, jj:] = True
    return local_neighborhood, element_exists

