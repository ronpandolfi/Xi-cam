import numpy as np
#from matplotlib import pyplot as plt
#from os.path import join
try:
    import cPickle as pickle
except ImportError:
    import pickle

from slacxop import Operation
import optools

reference_loc = 'slacx/slacxcore/operations/dmz/references/polydispersity_guess_references.pickle'


class GenerateSphericalDiffractionQ(Operation):
    """Generate a SAXS diffraction pattern for spherical nanoparticles.

    Uses r0 in real units and gives q in real units."""

    def __init__(self):
        input_names = ['r0', 'sigma_r_over_r0', 'intensity_at_zero_q', 'qmin', 'qmax', 'qstep']
        output_names = ['q', 'I']
        super(GenerateSphericalDiffractionQ, self).__init__(input_names, output_names)
        self.input_doc['r0'] = 'mean particle radius'
        self.input_doc['sigma_r_over_r0'] = 'width of distribution in r divided by r0'
        self.input_doc['intensity_at_zero_q'] = 'intensity at q = 0'
        self.input_doc['qmin'] = 'lower bound of q range of interest'
        self.input_doc['qmax'] = 'upper bound of q range of interest'
        self.input_doc['qstep'] = 'step size in q range of interest'
        self.input_src['r0'] = optools.op_input
        self.input_src['sigma_r_over_r0'] = optools.op_input
        self.input_src['intensity_at_zero_q'] = optools.op_input
        self.input_src['qmin'] = optools.text_input
        self.input_src['qmax'] = optools.text_input
        self.input_src['qstep'] = optools.text_input
        self.input_type['qmin'] = optools.float_type
        self.input_type['qmax'] = optools.float_type
        self.input_type['qstep'] = optools.float_type
        self.output_doc['q'] = '1d ndarray; wave vector values'
        self.output_doc['I'] = '1d ndarray; intensity values'
        self.categories = ['1D DATA PROCESSING']

    def run(self):
        self.outputs['q'] = gen_q_vector(self.inputs['qmin'], self.inputs['qmax'], self.inputs['qstep'])
        self.outputs['I'] = \
            generate_spherical_diffraction(self.inputs['r0'], self.inputs['sigma_r_over_r0'],
                                           self.inputs['intensity_at_zero_q'], self.outputs['q'])


class GenerateSphericalDiffraction(Operation):
    """Generate a SAXS diffraction pattern for spherical nanoparticles.

    Uses r0 in real units and gives q in real units."""

    def __init__(self):
        input_names = ['r0', 'sigma_r_over_r0', 'intensity_at_zero_q', 'q_vector']
        output_names = ['I']
        super(GenerateSphericalDiffraction, self).__init__(input_names, output_names)
        self.input_doc['r0'] = 'mean particle radius'
        self.input_doc['sigma_r_over_r0'] = 'width of distribution in r divided by r0'
        self.input_doc['intensity_at_zero_q'] = 'intensity at q = 0'
        self.input_doc['q_vector'] = 'q values of interest'
        self.input_src['r0'] = optools.op_input
        self.input_src['sigma_r_over_r0'] = optools.op_input
        self.input_src['intensity_at_zero_q'] = optools.op_input
        self.input_src['q_vector'] = optools.op_input
        self.output_doc['I'] = '1d ndarray; intensity values'
        self.categories = ['1D DATA PROCESSING']

    def run(self):
        self.outputs['I'] = \
            generate_spherical_diffraction(self.inputs['r0'], self.inputs['sigma_r_over_r0'],
                                           self.inputs['intensity_at_zero_q'], self.inputs['q_vector'])

'''
class GenerateSphericalDiffractionX(Operation):
    """Generate a SAXS diffraction pattern for spherical nanoparticles.

    Generates pattern in units x = r0 * q, with the assumption that x0 = r0 * q0 = 1."""

    def __init__(self):
        input_names = ['sigma_x_over_x0', 'intensity_at_zero_q', 'xmin', 'xmax', 'xstep']
        output_names = ['x', 'I']
        super(GenerateSphericalDiffractionX, self).__init__(input_names, output_names)
        self.input_doc['sigma_r_over_r0'] = 'width of distribution in r divided by r0'
        self.input_doc['intensity_at_zero_q'] = 'intensity at q = 0'
        self.input_doc['xmin'] = 'lower bound of x range of interest'
        self.input_doc['xmax'] = 'upper bound of x range of interest'
        self.input_doc['xstep'] = 'step size in x range of interest'
        self.input_src['sigma_x_over_x0'] = optools.text_input
        self.input_src['intensity_at_zero_q'] = optools.text_input
        self.input_src['xmin'] = optools.text_input
        self.input_src['xmax'] = optools.text_input
        self.input_src['xstep'] = optools.text_input
        self.input_type['sigma_x_over_x0'] = optools.float_type
        self.input_type['intensity_at_zero_q'] = optools.float_type
        self.input_type['xmin'] = optools.float_type
        self.input_type['xmax'] = optools.float_type
        self.input_type['xstep'] = optools.float_type
        self.output_doc['x'] = '1d ndarray; wave vector values'
        self.output_doc['I'] = '1d ndarray; intensity values'
        self.categories = ['1D DATA PROCESSING']

    def run(self):
        self.outputs['q'], self.outputs['I'] = \
            generate_spherical_diffraction(self.inputs['r0'], self.inputs['sigma_x_over_x0'],
                                           self.inputs['intensity_at_zero_q'], self.inputs['qmin'], self.inputs['qmax'],
                                           self.inputs['qstep'])
'''

def GuessPolydispersityUnweighted(Operation):
    """Guess the polydispersity of spherical diffraction pattern.

    Assumes the data have already been background subtracted, smoothed, and otherwise cleaned.

    """

    def __init__(self):
        input_names = ['q','I','dI']
        output_names = ['fractional_variation']
        super(GenerateSphericalDiffractionQ, self).__init__(input_names, output_names)
        # Documentation
        self.input_doc['q'] = '1d ndarray; wave vector values'
        self.input_doc['I'] = '1d ndarray; intensity values'
        self.output_doc['fractional_variation'] = 'normal distribution sigma divided by mean size'
        # Source and type
        self.input_src['q'] = optools.op_input
        self.input_src['I'] = optools.op_input
        self.categories = ['1D DATA PROCESSING']

    def run(self):
        q = self.inputs['q']
        I = self.inputs['I']
        dI = self.inputs['dI']

        x1, dipCoefficients = polydispersity_metric_xFirstDip(q, I)
        x2 = polydispersity_metric_x2(q, I)
        y1 = polydispersity_metric_firstDipHeight(dipCoefficients)
        y2 = polydispersity_metric_heightAtZero(x1, q, I)
        x0 = x1/x2
        y0 = y1/y2
        with open(reference_loc, 'rb') as handle:
            references = pickle.load(handle)
        x = references['xFirstDip']/references['']
        y = references['firstDipHeight']/references['heightAtZero']
        factor = references['factorVals']
        self.outputs['fractional_variation'] = guess_nearest_point_on_dual_trace(x0, y0, x, y, factor)



# xFirstDip, powerLaw, depthFirstDip, sigmaFirstDip, heightFirstDip, heightAtZero

def polydispersity_metric_xFirstDip(q, I, dI=np.zeros(1)):
    if not dI.any():
        dI = np.zeros(I.shape, dtype=float)
    scaled_I = (I - dI) * q ** 4
    dips = local_minima_detector(scaled_I)
    shoulders = local_maxima_detector(scaled_I)
    dips, shoulders = clean_extrema(dips, shoulders)
    xFirstDip, quadCoefficients = firstDipLoc(q, I, dips)
    return xFirstDip, quadCoefficients


def polydispersity_metric_heightAtZero(qFirstDip, q, I, dI=np.zeros(1)):
    if not dI.any():
        dI = np.ones(y.shape, dtype=float)
    low_q = (q < 0.75*qFirstDip)
    coefficients = arbitrary_order_fit(5, q[low_q], I[low_q])
    heightAtZero = coefficients[0]
    return heightAtZero

def firstDipLoc(x, y, dips):
    firstDipIndex = np.where(dips)[0][0]
    quadCoefficients = arbitrary_order_fit(2, x[firstDipIndex - 2:firstDipIndex + 2], y[firstDipIndex - 2:firstDipIndex + 2])
    xFirstDip = quadratic_extremum(quadCoefficients)
    return xFirstDip, quadCoefficients

def clean_extrema(dips, shoulders):
    # Forbid rapid oscillation
    dips_above = dips[1:].copy()
    dips_below = dips[:-1].copy()
    shoulders_above = shoulders[1:].copy()
    shoulders_below = shoulders[:-1].copy()
    dips[:-1] = dips[:-1] & (~shoulders_above)
    dips[1:] = dips[1:] & (~shoulders_below)
    shoulders[:-1] = shoulders[:-1] & (~dips_above)
    shoulders[1:] = shoulders[1:] & (~dips_below)
    # Mark endpoints False
    dips[0:10] = False
    dips[-1] = False
    shoulders[0] = False
    shoulders[-1] = False
    return dips, shoulders

def harsh_clean_extrema(dips, shoulders):
    # Forbid rapid oscillation
    dips_above = dips[1:].copy()
    dips_below = dips[:-1].copy()
    shoulders_above = shoulders[1:].copy()
    shoulders_below = shoulders[:-1].copy()
    dips[:-1] = dips[:-1] & (~shoulders_above)
    dips[1:] = dips[1:] & (~shoulders_below)
    shoulders[:-1] = shoulders[:-1] & (~dips_above)
    shoulders[1:] = shoulders[1:] & (~dips_below)
    # Play favorites
    size = dips_above.size
    onepercent = np.ceil(size/100.)
    dips_near_above = neighborhood(dips, onepercent)
    dips_near_below = dips[:-onepercent].copy()
    shoulders_near_above = shoulders[onepercent:].copy()
    shoulders_near_below = shoulders[:-onepercent].copy()

    dips[:-1] = dips[:-1] & (~shoulders_above)
    dips[1:] = dips[1:] & (~shoulders_below)
    shoulders[:-1] = shoulders[:-1] & (~dips_above)
    shoulders[1:] = shoulders[1:] & (~dips_below)
    # Mark endpoints False
    dips[0:10] = False
    dips[-1] = False
    shoulders[0] = False
    shoulders[-1] = False
    return dips, shoulders


def neighborhood_up(boolarray1d, n):
    size = boolarray1d.size
    neighbors = np.zeros(size, dtype=bool)
    for ii in range(1, n):
        neighbors = neighbors[ii:] | boolarray1d[:-ii]
    return neighbors

def neighborhood_down(boolarray1d, n):
    size = boolarray1d.size
    neighbors = np.zeros(size, dtype=bool)
    for ii in range(1, n):
        neighbors = neighbors[:-ii] | boolarray1d[ii:]
    return neighbors

def choose_extrema(q, I, dI):
    scaled_I = (I * q ** 4) / dI
    dips = local_minima_detector(scaled_I)
    shoulders = local_maxima_detector(scaled_I)
    dips, shoulders = clean_extrema(dips, shoulders)


def local_maxima_detector(y):
    '''
    Finds local maxima in ordered data y.

    :param y: 1d numpy float array
    :return maxima: 1d numpy bool array

    *maxima* is *True* at a local maximum, *False* otherwise.

    This function makes no attempt to reject spurious maxima of various sorts.
    That task is left to other functions.
    '''
    length = y.size
    greater_than_follower = np.zeros(length, dtype=bool)
    greater_than_leader = np.zeros(length, dtype=bool)
    greater_than_follower[:-1] = np.greater(y[:-1], y[1:])
    greater_than_leader[1:] = np.greater(y[1:], y[:-1])
    maxima = np.logical_and(greater_than_follower, greater_than_leader)
    # End points
    maxima[0] = greater_than_follower[0]
    maxima[-1] = greater_than_leader[-1]
    return maxima

def local_minima_detector(y):
    '''
    Finds local minima in ordered data *y*.

    :param y: 1d numpy float array
    :return minima: 1d numpy bool array

    *minima* is *True* at a local minimum, *False* otherwise.

    This function makes no attempt to reject spurious minima of various sorts.
    That task is left to other functions.
    '''
    minima = local_maxima_detector(-y)
    return minima

def arbitrary_order_fit(order, x, y, dy=np.zeros(1)):
    if not dy.any():
        dy = np.ones(y.shape, dtype=float)
    # Formulate the equation to be solved for polynomial coefficients
    matrix, vector = make_poly_matrices(x, y, dy, order)
    # Solve equation
    inverse = np.linalg.pinv(matrix)
    coefficients = np.matmul(inverse, vector)
    coefficients = coefficients.flatten()
    return coefficients

def quadratic_extremum(coefficients):
    extremum_location = -0.5*coefficients[1]/coefficients[2]
    return extremum_location

def polynomial_value(coefficients, x):
    powers = np.arange(coefficients.size)
    y = ((x ** powers) * coefficients).sum()
    return y

def polydispersity_metric_firstDipHeight(quadCoefficients, xdip):
    ydip = polynomial_value(quadCoefficients, xdip)
    return ydip

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
    if not error.any():
        error = np.ones(y.shape, dtype=float)
    if ((x.shape != y.shape) or (y.shape != error.shape)):
        raise ValueError('Arguments *x*, *y*, and *error* must all have the same shape.')
    index = np.arange(order+1)
    vector = (dummy(y) * dummy(x) ** vertical(index) * dummy(error)).sum(axis=0)
    index_block = horizontal(index) + vertical(index)
    matrix = (dummy(x) ** index_block * dummy(error)).sum(axis=0)
    return matrix, vector


def polydispersity_metric_x2(q, I):
    pass

def polydispersity_metric_y1(q, I):
    pass

def polydispersity_metric_y2(q, I):
    pass

def nearest_point_on_trace(x0, y0, x, y):
    pass

def guess_nearest_point_on_dual_trace(x0, y0, x, y, variable):
    vbestx = guess_nearest_point_on_single_trace(x0, x, variable)
    vbesty = guess_nearest_point_on_single_trace(y0, y, variable)
    vbest = 0.5*(vbestx + vbesty)
    return vbest

def guess_nearest_point_on_single_trace(x0, x, y):
    index2 = np.where(x > x0)[0][0]
    index1 = index2 - 1
    xdiff2 = x[index2] - x0
    xdiff1 = x0 - x[index1]
    ybest = (y[index1] * xdiff2 + y[index2] * xdiff1) / (xdiff1 + xdiff2)
    return ybest


def gen_q_vector(qmin, qmax, qstep):
    q = np.arange(qmin, qmax + qstep, qstep)
    return q


def generate_spherical_diffraction(r0, poly, i0, q):
    x = q * r0
    i = i0 * blur(x, poly)
    return i

def fullFunction(x):
    y = ((np.sin(x) - x * np.cos(x)) * x**-3)**2
    return y

def gauss(x, x0, sigma):
    y = ((sigma * (2 * np.pi)**0.5)**-1 )*np.exp(-0.5 * ((x - x0)/sigma)**2)
    return y

def generateRhoFactor(factor):
    '''

    :param factor: float
    :return:

    factor should be 0.1 for a sigma 10% of size
    '''
    factorCenter = 1
    factorMin = max(factorCenter-5*factor, 0.02)
    factorMax = factorCenter+5*factor
    factorVals = np.arange(factorMin, factorMax, factor*0.02)
    rhoVals = gauss(factorVals, factorCenter, factor)
    return factorVals, rhoVals

def factorStretched(x, factor):
    effective_x = x / factor
    y = fullFunction(effective_x)
    return y

def blur(x, factor):
    factorVals, rhoVals = generateRhoFactor(factor)
    deltaFactor = factorVals[1] - factorVals[0]
    ysum = np.zeros(x.shape)
    for ii in range(len(factorVals)):
        y = factorStretched(x, factorVals[ii])
        ysum += rhoVals[ii]*y*deltaFactor
    return ysum

# from pyFAI import AzimuthalIntegrator as ai
# help(ai.integrate1d)