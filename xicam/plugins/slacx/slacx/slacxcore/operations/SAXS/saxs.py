from os.path import join
from os import remove

import numpy as np
from scipy import interp

from scipy.optimize import curve_fit
try:
    import cPickle as pickle
except ImportError:
    import pickle

from ..slacxop import Operation
from .. import optools

reference_loc = join('slacx','slacxcore','operations','dmz','references','polydispersity_guess_references.pickle')
global references
references = {}

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
        self.input_src['r0'] = optools.wf_input
        self.input_src['sigma_r_over_r0'] = optools.wf_input
        self.input_src['intensity_at_zero_q'] = optools.wf_input
        self.input_src['qmin'] = optools.user_input
        self.input_src['qmax'] = optools.user_input
        self.input_src['qstep'] = optools.user_input
        self.input_type['qmin'] = optools.float_type
        self.input_type['qmax'] = optools.float_type
        self.input_type['qstep'] = optools.float_type
        self.output_doc['q'] = '1d ndarray; wave vector values'
        self.output_doc['I'] = '1d ndarray; intensity values'
        # source & type
        self.input_src['r0'] = optools.user_input
        self.input_src['sigma_r_over_r0'] = optools.user_input
        self.input_src['intensity_at_zero_q'] = optools.user_input
        self.input_src['qmin'] = optools.user_input
        self.input_src['qmax'] = optools.user_input
        self.input_src['qstep'] = optools.user_input
        self.input_type['r0'] = optools.float_type
        self.input_type['sigma_r_over_r0'] = optools.float_type
        self.input_type['intensity_at_zero_q'] = optools.float_type
        self.input_type['qmin'] = optools.float_type
        self.input_type['qmax'] = optools.float_type
        self.input_type['qstep'] = optools.float_type
        self.categories = ['SAXS']

    def run(self):
        self.outputs['I'] = \
            generate_spherical_diffraction(self.inputs['q_vector'], self.inputs['intensity_at_zero_q'],
                                           self.inputs['r0'], self.inputs['sigma_r_over_r0'])

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
        self.input_src['r0'] = optools.wf_input
        self.input_src['sigma_r_over_r0'] = optools.wf_input
        self.input_src['intensity_at_zero_q'] = optools.wf_input
        self.input_src['q_vector'] = optools.wf_input
        self.output_doc['I'] = '1d ndarray; intensity values'
        # source & type
        self.input_src['r0'] = optools.user_input
        self.input_src['sigma_r_over_r0'] = optools.user_input
        self.input_src['intensity_at_zero_q'] = optools.user_input
        self.input_src['q_vector'] = optools.wf_input
        self.input_type['r0'] = optools.float_type
        self.input_type['sigma_r_over_r0'] = optools.float_type
        self.input_type['intensity_at_zero_q'] = optools.float_type
        self.categories = ['SAXS']

    def run(self):
        q_vector, intensity_at_zero_q = self.inputs['q_vector'], self.inputs['intensity_at_zero_q']
        r0, sigma_r_over_r0 = self.inputs['r0'], self.inputs['sigma_r_over_r0']
#        print 'q_vector size, shape, sum', q_vector.size, q_vector.shape, q_vector.sum()
        I = generate_spherical_diffraction(q_vector, intensity_at_zero_q, r0, sigma_r_over_r0)
        self.outputs['I'] = I

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
        self.input_src['sigma_x_over_x0'] = optools.user_input
        self.input_src['intensity_at_zero_q'] = optools.user_input
        self.input_src['xmin'] = optools.user_input
        self.input_src['xmax'] = optools.user_input
        self.input_src['xstep'] = optools.user_input
        self.input_type['sigma_x_over_x0'] = optools.float_type
        self.input_type['intensity_at_zero_q'] = optools.float_type
        self.input_type['xmin'] = optools.float_type
        self.input_type['xmax'] = optools.float_type
        self.input_type['xstep'] = optools.float_type
        self.output_doc['x'] = '1d ndarray; wave vector values'
        self.output_doc['I'] = '1d ndarray; intensity values'
        self.categories = ['SAXS']

    def run(self):
        self.outputs['q'], self.outputs['I'] = \
            generate_spherical_diffraction(self.inputs['r0'], self.inputs['sigma_x_over_x0'],
                                           self.inputs['intensity_at_zero_q'], self.inputs['qmin'], self.inputs['qmax'],
                                           self.inputs['qstep'])
'''

class GenerateReferences(Operation):
    """Generate metrics used for guessing diffraction pattern properties.

    DOES NOT STORE.  IS NOT AUTOMATICALLY AVAILABLE TO OTHER OPERATIONS.

    Use OverwriteReferences to store the resulting dictionary, if desired."""

    def __init__(self):
        input_names = ['xmin', 'xmax', 'xstep', 'factormin', 'factormax', 'factorstep']
        output_names = ['references']
        super(GenerateReferences, self).__init__(input_names, output_names)
        # Documentation
        self.input_doc['xmin'] = 'lowest computed value of x; nuisance parameter'
        self.input_doc['xmax'] = 'highest computed value of x; nuisance parameter'
        self.input_doc['xstep'] = 'step size in x; nuisance parameter'
        self.input_doc['factormin'] = 'lowest computed value of factor'
        self.input_doc['factormax'] = 'highest computed value of factor'
        self.input_doc['factorstep'] = 'step size in factor'
        self.output_doc['references'] = 'dictionary of useful values'
        # Source and type
        self.input_src['xmin'] = optools.user_input
        self.input_src['xmax'] = optools.user_input
        self.input_src['xstep'] = optools.user_input
        self.input_src['factormin'] = optools.user_input
        self.input_src['factormax'] = optools.user_input
        self.input_src['factorstep'] = optools.user_input
        self.input_type['xmin'] = optools.float_type
        self.input_type['xmax'] = optools.float_type
        self.input_type['xstep'] = optools.float_type
        self.input_type['factormin'] = optools.float_type
        self.input_type['factormax'] = optools.float_type
        self.input_type['factorstep'] = optools.float_type
        self.categories = ['SAXS']

    def run(self):
        x = gen_q_vector(self.inputs['xmin'], self.inputs['xmax'], self.inputs['xstep'])
        factorVals = gen_q_vector(self.inputs['factormin'], self.inputs['factormax'], self.inputs['factorstep'])
        references = generate_references(x, factorVals)
        self.outputs['references'] = references

class OverwriteReferences(Operation):
    """Store previously generated metrics used for guessing diffraction pattern properties.

    USE WITH CAUTION as it will OVERWRITE any existing reference file."""

    def __init__(self):
        input_names = ['references']
        output_names = []
        super(OverwriteReferences, self).__init__(input_names, output_names)
        # Documentation
        self.input_doc['references'] = 'dictionary of useful values generated by GenerateReferences'
        # Source and type
        self.input_src['references'] = optools.wf_input
        self.categories = ['OUTPUT']

    def run(self):
        try:
            remove(reference_loc)
        except:
            pass
        dump_references(self.inputs['references'])


class FetchReferences(Operation):
    """Fetch previously generated and stored metrics used for guessing diffraction pattern properties.

    This function is for examination, and is not necessary for computation.
    The file in question is auto-fetched by operations that rely on it."""

    def __init__(self):
        input_names = []
        output_names = ['references']
        super(FetchReferences, self).__init__(input_names, output_names)
        # Documentation
        self.output_doc['references'] = 'dictionary of useful values generated by GenerateReferences'
        self.categories = ['SAXS']

    def run(self):
        try:
            self.outputs['references'] = load_references()
        except:
            print "No reference file was found at the appropriate location."

'''
class GuessPolydispersityWeighted(Operation):
    """Guess the polydispersity of spherical diffraction pattern.

    Assumes the data have already been background subtracted, smoothed, and otherwise cleaned."""

    def __init__(self):
        input_names = ['q','I','dI']
        output_names = ['fractional_variation','first_dip_q']
        super(GuessPolydispersityWeighted, self).__init__(input_names, output_names)
        # Documentation
        self.input_doc['q'] = '1d ndarray; wave vector values'
        self.input_doc['I'] = '1d ndarray; intensity values'
        self.input_doc['dI'] = '1d ndarray; error estimate of intensity values'
        self.output_doc['fractional_variation'] = 'normal distribution sigma divided by mean size'
        self.output_doc['qFirstDip'] = 'location in q of the first dip'
        # Source and type
        self.input_src['q'] = optools.wf_input
        self.input_src['I'] = optools.wf_input
        self.input_src['dI'] = optools.wf_input
        self.categories = ['SAXS']

    def run(self):
        q, I, dI = self.inputs['q'], self.inputs['I'], self.inputs['dI']
        fractional_variation, qFirstDip, _, _, _, _, _ = guess_polydispersity(q, I, dI)
        self.outputs['fractional_variation'], self.outputs['first_dip_q'] = fractional_variation, qFirstDip

class GuessSize(Operation):
        """Guess the mean size of spherical diffraction pattern.

        Assumes the data have already been background subtracted, smoothed, and otherwise cleaned.

        Requires foreknowledge of the fractional variation in size."""

        def __init__(self):
            input_names = ['fractional_variation', 'first_dip_q']
            output_names = ['mean_size']
            super(GuessSize, self).__init__(input_names, output_names)
            # Documentation
            self.input_doc['fractional_variation'] = 'normal distribution sigma divided by mean size'
            self.input_doc['first_dip_q'] = 'location in q of the first dip'
            self.output_doc['mean_size'] = 'mean size of particles'
            # Source and type
            self.input_src['fractional_variation'] = optools.wf_input
            self.input_src['first_dip_q'] = optools.wf_input
            self.categories = ['SAXS']

        def run(self):
            fractional_variation, first_dip_q = self.outputs['fractional_variation'], self.outputs['first_dip_q']
            mean_size = guess_size(fractional_variation, first_dip_q)
            self.outputs['mean_size'] = mean_size
'''

class GuessProperties(Operation):
    """Guess the polydispersity, mean size, and amplitude of spherical diffraction pattern.

    Assumes the data have already been background subtracted, smoothed, and otherwise cleaned."""

    def __init__(self):
        input_names = ['q', 'I', 'dI']
        output_names = ['fractional_variation', 'mean_size', 'amplitude_at_zero', 'qFirstDip', 'heightFirstDip', 'sigmaScaledFirstDip', 'heightAtZero', 'dips', 'shoulders']
        super(GuessProperties, self).__init__(input_names, output_names)
        # Documentation
        self.input_doc['q'] = '1d ndarray; wave vector values'
        self.input_doc['I'] = '1d ndarray; intensity values'
        self.input_doc['dI'] = '1d ndarray; error estimate of intensity values; use default value if none exists'
        self.output_doc['fractional_variation'] = 'normal distribution sigma divided by mean size'
        self.output_doc['mean_size'] = 'mean size of particles'
        self.output_doc['amplitude_at_zero'] = 'projected scattering amplitude at q=0'
        # Source and type
        self.input_src['q'] = optools.wf_input
        self.input_src['I'] = optools.wf_input
        self.input_src['dI'] = optools.wf_input
        # defaults
        self.inputs['dI'] = np.zeros(1, dtype=float)
        self.categories = ['SAXS']

    def run(self):
        q, I, dI = self.inputs['q'], self.inputs['I'], self.inputs['dI']
        fractional_variation, qFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero, dips, shoulders = guess_polydispersity(q, I, dI)
        self.outputs['qFirstDip'] = qFirstDip
        self.outputs['heightFirstDip'] = heightFirstDip
        self.outputs['sigmaScaledFirstDip'] = sigmaScaledFirstDip
        self.outputs['heightAtZero'] = heightAtZero
        self.outputs['dips'] = dips
        self.outputs['shoulders'] = shoulders
        mean_size = guess_size(fractional_variation, qFirstDip)
        amplitude_at_zero = polydispersity_metric_heightAtZero(qFirstDip, q, I, dI)
        #dips, shoulders = choose_dips_and_shoulders1(q, I, dI)
        self.outputs['fractional_variation'], self.outputs['mean_size'], self.outputs['amplitude_at_zero'] = \
            fractional_variation, mean_size, amplitude_at_zero


class OptimizeSphericalDiffractionFit(Operation):
    """From an initial guess, optimize r0, I0, and fractional_variation."""

    def __init__(self):
        input_names = ['q', 'I', 'amplitude_at_zero', 'mean_size', 'fractional_variation']
        output_names = ['amplitude_at_zero', 'mean_size', 'fractional_variation']
        super(OptimizeSphericalDiffractionFit, self).__init__(input_names, output_names)
        # Documentation
        self.input_doc['q'] = '1d ndarray; wave vector values'
        self.input_doc['I'] = '1d ndarray; intensity values'
        self.input_doc['amplitude_at_zero'] = 'estimate of intensity at q=0'
        self.input_doc['mean_size'] = 'estimate of mean particle size'
        self.input_doc['fractional_variation'] = 'estimate of normal distribution sigma divided by mean size'
        self.output_doc['fractional_variation'] = 'normal distribution sigma divided by mean size'
        self.output_doc['mean_size'] = 'mean particle size'
        self.output_doc['amplitude_at_zero'] = 'projected intensity at q=0'
        # Source and type
        self.input_src['q'] = optools.wf_input
        self.input_src['I'] = optools.wf_input
        self.input_src['amplitude_at_zero'] = optools.wf_input
        self.input_src['mean_size'] = optools.wf_input
        self.input_src['fractional_variation'] = optools.wf_input
        self.input_type['amplitude_at_zero'] = optools.float_type
        self.input_type['mean_size'] = optools.float_type
        self.input_type['fractional_variation'] = optools.float_type
        self.categories = ['SAXS']

    def run(self):
        q, I = self.inputs['q'], self.inputs['I']
        I0_in, r0_in, frac_in = self.inputs['amplitude_at_zero'], self.inputs['mean_size'], self.inputs['fractional_variation']
        popt, pcov = curve_fit(generate_spherical_diffraction, q, I, bounds=([I0_in*0.5, r0_in*0.5, frac_in*0.1], [I0_in/0.5, r0_in/0.5, frac_in/0.1]))
        self.outputs['amplitude_at_zero'] = popt[0]
        self.outputs['mean_size'] = popt[1]
        self.outputs['fractional_variation'] = popt[2]


class OptimizeSphericalDiffractionFit2(Operation):
    """From an initial guess, optimize r0, I0, and fractional_variation."""

    def __init__(self):
        input_names = ['q', 'I', 'amplitude_at_zero', 'mean_size', 'fractional_variation']
        output_names = ['amplitude_at_zero', 'mean_size', 'fractional_variation']
        super(OptimizeSphericalDiffractionFit2, self).__init__(input_names, output_names)
        # Documentation
        self.input_doc['q'] = '1d ndarray; wave vector values'
        self.input_doc['I'] = '1d ndarray; intensity values'
        self.input_doc['amplitude_at_zero'] = 'estimate of intensity at q=0'
        self.input_doc['mean_size'] = 'estimate of mean particle size'
        self.input_doc['fractional_variation'] = 'estimate of normal distribution sigma divided by mean size'
        self.output_doc['fractional_variation'] = 'normal distribution sigma divided by mean size'
        self.output_doc['mean_size'] = 'mean particle size'
        self.output_doc['amplitude_at_zero'] = 'projected intensity at q=0'
        # Source and type
        self.input_src['q'] = optools.wf_input
        self.input_src['I'] = optools.wf_input
        self.input_src['amplitude_at_zero'] = optools.wf_input
        self.input_src['mean_size'] = optools.wf_input
        self.input_src['fractional_variation'] = optools.wf_input
        self.input_type['amplitude_at_zero'] = optools.float_type
        self.input_type['mean_size'] = optools.float_type
        self.input_type['fractional_variation'] = optools.float_type
        self.categories = ['SAXS']

    def run(self):
        q, I = self.inputs['q'], self.inputs['I']
        I0_in, r0_in, frac_in = self.inputs['amplitude_at_zero'], self.inputs['mean_size'], self.inputs['fractional_variation']
        popt, pcov = curve_fit(generate_spherical_diffraction, q, I, bounds=([I0_in*0.5, r0_in*0.5, frac_in*0.1], [I0_in/0.5, r0_in/0.5, frac_in/0.1]))
        self.outputs['amplitude_at_zero'] = popt[0]
        self.outputs['mean_size'] = popt[1]
        self.outputs['fractional_variation'] = popt[2]

def generate_references(x, factorVals):
    #y0 = fullFunction(x)
    num_tests = len(factorVals)
    xFirstDip = np.zeros(num_tests)
    sigmaScaledFirstDip = np.zeros(num_tests)
    heightFirstDip = np.zeros(num_tests)
    heightAtZero = np.zeros(num_tests)
    #powerLawAll = np.zeros((num_tests, 2))
    #powerLawInitial = np.zeros((num_tests, 2))
    #depthFirstDip = np.zeros(num_tests)
    #y_x4_inf = np.zeros(num_tests)
    #lowQApprox = np.zeros((num_tests, 2))
    for ii in range(num_tests):
        factor = factorVals[ii]
        y = blur(x, factor)
        xFirstDip[ii], heightFirstDip[ii], sigmaScaledFirstDip[ii], heightAtZero[ii] = take_polydispersity_metrics(x, y)
    references = prep_for_pickle(factorVals, xFirstDip, sigmaScaledFirstDip, heightFirstDip, heightAtZero)
    return references

# xFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero # , powerLawAll, powerLawInitial

# I/O functions

def prep_for_pickle(factorVals, xFirstDip, sigmaScaledFirstDip, heightFirstDip, heightAtZero):
    references = {}
    references['factorVals'] = factorVals
    references['xFirstDip'] = xFirstDip
    references['heightFirstDip'] = heightFirstDip
    references['sigmaScaledFirstDip'] = sigmaScaledFirstDip
    references['heightAtZero'] = heightAtZero
    #references['powerLaw'] = powerLaw
    return references

def load_references():
    with open(reference_loc, 'rb') as handle:
        references = pickle.load(handle)
    return references

def dump_references(references):
    with open(reference_loc, 'wb') as handle:
        pickle.dump(references, handle)

# Functions about algebraic solutions

def arbitrary_order_solution(order, x, y, dy=np.zeros(1)):
    '''Solves for a polynomial "fit" of arbitrary order.'''
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
    '''Finds the location in independent coordinate of the extremum of a quadratic equation.'''
    extremum_location = -0.5*coefficients[1]/coefficients[2]
    return extremum_location

def polynomial_value(coefficients, x):
    '''Finds the value of a polynomial at a location.'''
    powers = np.arange(coefficients.size)
    try:
        x.size # distinguish sequence x from numeric x
        powers = horizontal(powers)
        coefficients = horizontal(coefficients)
        x = vertical(x)
        y = ((x ** powers) * coefficients).sum(axis=1)
        y = y.flatten()
    except AttributeError:
        y = ((x ** powers) * coefficients).sum()
    return y

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

def power_law_solution(x, y, dy=np.zeros(1)):
    '''Solves for a power law by solving for a linear fit in log-log space.'''
    # Note that if dy is zeros, logdy will also be zeros, triggering default behavior in arbitrary_order_solution also.
    if not dy.any():
        dy = np.ones(y.shape, dtype=float)
    logx = np.log(x)
    logy = np.log(y)
    logdy = np.log(dy)
    logLinearCoefficients = arbitrary_order_solution(1, logx, logy, logdy)
    prefactor = np.exp(logLinearCoefficients[0])
    exponent = logLinearCoefficients[1]
    powerCoefficients = np.array([prefactor, exponent])
    return powerCoefficients

# Functions about simulating SAXS patterns

def gen_q_vector(qmin, qmax, qstep):
    '''An inclusive-range version of np.arange.'''
    if np.mod((qmax - qmin), qstep) == 0:
        qmax = qmax + qstep
    q = np.arange(qmin, qmax, qstep)
    return q

def generate_spherical_diffraction(q, i0, r0, poly):
    x = q * r0
    i = i0 * blur(x, poly) * 9.
    return i

def fullFunction(x):
    y = ((np.sin(x) - x * np.cos(x)) * x**-3)**2
    return y

def gauss(x, x0, sigma):
    y = ((sigma * (2 * np.pi)**0.5)**-1 )*np.exp(-0.5 * ((x - x0)/sigma)**2)
    return y

def generateRhoFactor(factor):
    '''
    Generate a guassian distribution of number densities (rho).

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

# Funtions specifically about detecting SAXS properties

def guess_polydispersity(q, I, dI=np.zeros(1)):
#    global references
    if not dI.any():
        dI = np.ones(I.shape, dtype=float)
    dips, shoulders = choose_dips_and_shoulders(q, I, dI)
    qFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero = take_polydispersity_metrics(q, I, dI)
    x0 = qFirstDip / sigmaScaledFirstDip
    y0 = heightFirstDip / heightAtZero
    try:
        references = load_references()
    except:
        print no_reference_message
    x = references['xFirstDip'] / references['sigmaScaledFirstDip']
    y = references['heightFirstDip'] / references['heightAtZero']
    factor = references['factorVals']
    fractional_variation, _, best_xy = guess_nearest_point_on_nonmonotonic_trace_normalized([x0, y0], [x, y], factor)
    return fractional_variation, qFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero, dips, shoulders

def take_polydispersity_metrics(x, y, dy=np.zeros(1)):
    if not dy.any():
        dy = np.ones(y.shape)
    dips, shoulders = choose_dips_and_shoulders(x, y, dy)
    xFirstDip, heightFirstDip, scaledQuadCoefficients = first_dip(x, y, dips, dy)
    sigmaScaledFirstDip = polydispersity_metric_sigmaScaledFirstDip(x, y, dips, shoulders, xFirstDip, heightFirstDip)
    heightAtZero = polydispersity_metric_heightAtZero(xFirstDip, x, y, dy)
    return xFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero

def choose_dips_and_shoulders(q, I, dI=np.zeros(1)):
    '''Find the location of dips (low points) and shoulders (high points).'''
    if not dI.any():
        dI = np.ones(I.shape, dtype=float)
    dips = local_minima_detector(I)
    shoulders = local_maxima_detector(I)
    # Clean out end points and wussy maxima
    dips, shoulders = clean_extrema(dips, shoulders, I)
    return dips, shoulders

def clean_extrema(dips, shoulders, y):
    # Mark endpoints False
    dips[0:10] = False
    dips[-1] = False
    shoulders[0] = False
    shoulders[-1] = False
    # Reject weaksauce local maxima and minima
    extrema = dips | shoulders
    extrema_indices = np.where(extrema)[0]
    for ii in range(len(extrema_indices) - 3):
        four_indices = extrema_indices[ii:ii+4]
        is_dip = dips[extrema_indices[ii]]
        if is_dip:
            weak = upwards_weaksauce_identifier(four_indices, y)
        else:
            weak = downwards_weaksauce_identifier(four_indices, y)
        if weak:
            extrema[four_indices[1]] = False
            extrema[four_indices[2]] = False
    # Apply mask to shoulders & dips
    dips = dips * extrema
    shoulders = shoulders * extrema
    return dips, shoulders

def upwards_weaksauce_identifier(four_indices, y):
    a, b, c, d = four_indices
    if (y[a] < y[c]) & (y[b] < y[d]):
        weak = True
    else:
        weak = False
    return weak

def downwards_weaksauce_identifier(four_indices, y):
    weak = upwards_weaksauce_identifier(four_indices, -y)
    return weak

def guess_size(fractional_variation, first_dip_q):
#    global references
    try:
        references = load_references()
    except:
        print no_reference_message
    #xFirstDip, factorVals = references['xFirstDip'], references['factorVals']
    first_dip_x = interp(fractional_variation, references['factorVals'], references['xFirstDip'])
    mean_size = first_dip_x / first_dip_q
    return mean_size

def polydispersity_metric_heightFirstDip(scaledQuadCoefficients, xdip):
    ydip = polynomial_value(scaledQuadCoefficients, xdip) * (xdip**-4)
    return ydip

def polydispersity_metric_heightAtZero(qFirstDip, q, I, dI=np.zeros(1)):
    if not dI.any():
        dI = np.ones(I.shape, dtype=float)
    qlim = max(q[6], (qFirstDip - q[0])/2.)
    if qlim > 0.75*qFirstDip:
        print "Low-q sampling is poor and will likely affect estimate quality."
    low_q = (q < qlim)
    coefficients = arbitrary_order_solution(4, q[low_q], I[low_q], dI[low_q])
    heightAtZero = coefficients[0]
    return heightAtZero

def polydispersity_metric_qFirstDip(q, I, dips, dI=np.zeros(1)):
    '''Finds the location in *q* of the first dip.'''
    if not dI.any():
        dI = np.ones(I.shape, dtype=float)
    pad = 3
    firstDipIndex = np.where(dips)[0][0]
    lolim = firstDipIndex - pad
    hilim = firstDipIndex + pad
    scaled_I = I * q**4 / dI
    scaledQuadCoefficients = arbitrary_order_solution(2, q[lolim:hilim], scaled_I[lolim:hilim])
    xFirstDip = quadratic_extremum(scaledQuadCoefficients)
    return xFirstDip, scaledQuadCoefficients

def polydispersity_metric_sigmaScaledFirstDip(q, I, dips, shoulders, qFirstDip, heightFirstDip, dI=np.zeros(1)):
    if not dI.any():
        dI = np.ones(I.shape, dtype=float)
    scaled_I = I * q ** 4
    scaled_dI = dI * q ** 4
    powerCoefficients = power_law_solution(q[shoulders], I[shoulders], dI[shoulders])
    powerLawAtDip = powerCoefficients[0] * (qFirstDip ** powerCoefficients[1])
    scaledDipDepth = (powerLawAtDip - heightFirstDip) * (qFirstDip ** 4)
    pad = 3
    firstDipIndex = np.where(dips)[0][0]
    lolim = firstDipIndex - pad
    hilim = firstDipIndex + pad
    quadCoefficients = arbitrary_order_solution(2, q[lolim:hilim], scaled_I[lolim:hilim], scaled_dI[lolim:hilim])
    scaledDipCurvature = 2 * quadCoefficients[2]
    _, sigma = gauss_guess(scaledDipDepth, scaledDipCurvature)
    return sigma

# Other functions

def gauss_guess(signalMagnitude, signalCurvature):
    '''
    Guesses a gaussian intensity and width from signal magnitude and curvature.

    :param signalMagnitude: number-like with units of magnitude
    :param signalCurvature: number-like with units of magnitude per distance squared
    :return intensity, sigma:

    The solution given is not fitted; it is a first estimate to be used in fitting.
    '''
    signalMagnitude = np.fabs(signalMagnitude)
    signalCurvature = np.fabs(signalCurvature)
    sigma = (signalMagnitude / signalCurvature) ** 0.5
    intensity = signalMagnitude * sigma * (2 * np.pi) ** 0.5
    return intensity, sigma

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

def noiseless_curvature(x, y):
    '''
    Finds the curvature of y locally.  Does not account for noise.

    :param x: numpy float array, independent variable
    :param y: numpy float array, dependent variable
    :return curvature: numpy float array

    Compares subsequent pixels to find a local slope.
    Compares subsequent slopes to find a local curvature.
    The curvature is defined at a location 0.5*(x3 + x1) = 0.5*(x[2:] + x[:-2]).
    For evenly spaced data, the curvature is defined at x2 = x[1:-1].
    The curvature is not defined (np.nan) for the endpoints.
    '''
    curvature = np.zeros(x.size, dtype=float)
    y1 = y[:-2]
    y2 = y[1:-1]
    y3 = y[2:]
    x1 = x[:-2]
    x2 = x[1:-1]
    x3 = x[2:]
    # First derivatives
    yprime_one = (y2 - y1) / (x2 - x1)  # Defined at location 0.5*(x1 + x2)
    yprime_two = (y3 - y2) / (x3 - x2)  # Defined at location 0.5*(x2 + x3)
    # Second derivative
    # Defined at location 0.5*(x3 + x1).  For evenly spaced data, defined at x2.
    curvature[1:-1] = (yprime_two - yprime_one) / (0.5 * (x3 - x1))
    # Undefined at endpoints
    curvature[0] = np.nan
    curvature[-1] = np.nan
    return curvature

def point_to_point_variation(y):
    '''
    Calculate typical point-to-point variation as a weak metric of noisiness.

    :param y: A 1d ndarray of data values
    :return:
    '''
    length = y.size()
    y_low = y[:-1]
    y_high = y[1:]
    diff = y_high - y_low
    diff2mean = (diff**2).mean()
    variation = diff2mean**0.5
    return variation
    

def test():
    from os import listdir
    from os.path import join
    from matplotlib import pyplot as plt
    import numpy as np
    dir = "/Users/Amanda/Desktop/test"
    files = listdir(dir)
    csvfiles = [ii for ii in files if ii[-3:] == 'csv']
    fullfiles = [join(dir, ii) for ii in csvfiles]
    for ii in fullfiles:
        print "File: %s." % ii
        q = np.loadtxt(ii, dtype=float, delimiter=',', skiprows=1, usecols=(0,))
        I = np.loadtxt(ii, dtype=float, delimiter=',', skiprows=1, usecols=(1,))
        '''
        lo1, hi1, lo2, hi2 = choose_dips_and_shoulders3(q, I)
        q0, I0, dip_quadratic = first_dip(lo2, q, I)
        low_q, low_quadratic = polydispersity_metric_heightAtZero_2(q0, q, I)
        plot_one(q, I, lo1, hi1, lo2, hi2, q0, I0, dip_quadratic, low_quadratic)
        '''
        #xFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero = take_polydispersity_metrics(q, I)
        fractional_variation, qFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero, dips, shoulders = guess_polydispersity(q, I)
        mean_size = guess_size(fractional_variation, qFirstDip)
        plot_dos(q, I, qFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero, fractional_variation, mean_size)

        plt.title(ii)


def plot_one(x, y, lo1, hi1, lo2, hi2, x0, y0, coefficients1, coefficients2):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1)
    ax.plot(x, y, ls='-', marker='None', color='k', lw=2)
    # bad extrema
    ax.plot(x[lo1 & (~lo2)], y[lo1 & (~lo2)], ls='None', marker='x', color='b', lw=1)
    ax.plot(x[hi1 & (~hi2)], y[hi1 & (~hi2)], ls='None', marker='x', color='b', lw=1)
    # good extrema
    ax.plot(x[lo2], y[lo2], ls='None', marker='x', color='r', lw=1)
    ax.plot(x[hi2], y[hi2], ls='None', marker='x', color='r', lw=1)
    selection = ((x < x0*1.2) & (x > x0*0.8))
    ax.plot(x[selection], polynomial_value(coefficients1, x[selection]), ls='-', marker='None', color='b', lw=1)
    ax.plot(x0, y0, ls='None', marker='o', color='r', lw=1)
    selection = (x < x0*0.7)
    ax.plot(x[selection], polynomial_value(coefficients2, x[selection]), ls='-', marker='None', color='b', lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    return fig, ax


def plot_dos(q, I, q1, I1, sigmaScaledFirstDip, I0, poly, size):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1)
    ax.plot(q, I, ls='-', marker='None', color='k', lw=2)
    ax.plot(q1, I1, ls='None', marker='x', color='r', lw=1)
    selection = (q < q1*0.3)
    n = selection.sum()
    ax.plot(q[selection], np.ones(n)*I0, ls='-', marker='None', color='b', lw=1)
    modelI = generate_spherical_diffraction(q, I0, size, poly)
    ax.plot(q, modelI, ls='-', marker='None', color='r', lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    return fig, ax


def first_dip(q, I, dips, dI=np.zeros(1)):
    if not dI.any():
        dI = np.ones(I.shape, dtype=float)
    # if the first two "dips" are very close together, they are both marking the first dip
    # because I can't eliminate all spurious extrema with such a simple test
    dip_locs = np.where(dips)[0]
    q1, q2, q3 = q[dip_locs[:3]]
    mult = 4
    smult = 1.5
    # q1, q2 close compared to q2, q3 and compared to 0, q1
    if ((q2 - q1)*mult < (q3 - q2)) & ((q2 - q1)*mult < q1):
        case = 'two'
        scale = 0.5 * q3
    # (q2 - q1)*1.5 approximately equal to q1
    elif ((q2 - q1)*1.5*smult > q1) & ((q2 - q1)*1.5 < q1*smult):
        case = 'one'
        scale = 0.5 * q2
    else:
        case = 'mystery'
        scale = q1
#    print "Detected case: %s." % case
    if case == 'two':
        minq = q1 - scale*0.1
        maxq = q2 + scale*0.1
    else:
        minq = q1 - scale*0.1
        maxq = q1 + scale*0.1
    selection = ((q < maxq) & (q > minq))
    # make sure selection is large enough to get a useful sampling
    if selection.sum() < 9:
        print "Your sampling in q seems to be sparse and will likely affect the quality of the estimate."
    while selection.sum() < 9:
        selection[1:] = selection[1:] & selection[:-1]
        selection[:-1] = selection[1:] & selection[:-1]
    # fit local quadratic
    coefficients = arbitrary_order_solution(2, q[selection], I[selection], dI[selection])
    qbest = quadratic_extremum(coefficients)
    Ibest = polynomial_value(coefficients, qbest)
    return qbest, Ibest, coefficients


'''
def guess_nearest_point_on_single_monotonic_trace(x0, x, y):
    ybest = interp(x0, x, y)
    return ybest

def guess_nearest_point_on_dual_monotonic_trace(x0, y0, x, y, variable):
    vbestx = guess_nearest_point_on_single_monotonic_trace(x0, x, variable)
    vbesty = guess_nearest_point_on_single_monotonic_trace(y0, y, variable)
    vbest = 0.5*(vbestx + vbesty)
    return vbest
'''

def guess_nearest_point_on_nonmonotonic_trace_normalized(loclist, tracelist, coordinate):
    '''Finds the nearest point to location *loclist* on a trace *tracelist*.

    Uses normalized distances, i.e., the significance of each dimension is made equivalent.

    *loclist* and *tracelist* are lists of the same length and type.
    Elements of *loclist* are floats; elements of *tracelist* are arrays of the same shape as *coordinate*.
    *coordinate* is the independent variable along which *tracelist* is varying.

    *tracelist* must have decent sampling to start with or the algorithm will likely fail.'''
    tracesize = coordinate.size
    spacesize = len(loclist)
    distances = np.zeros(tracesize, dtype=float)
    for ii in range(spacesize):
        yii = tracelist[ii]
        meanii = yii.mean()
        varii = (((yii - meanii)**2).mean())**0.5
        normyii = (yii - meanii)/varii
        y0 = loclist[ii]
        normy0 = (y0 - meanii)/varii
        diffsquare = (normyii - normy0)**2
        distances = distances + diffsquare
    distances = distances**0.5
    best_indices = np.where(local_minima_detector(distances))[0]
    # At this point we've just found good neighborhoods to investigate
    # So we compare those neighborhoods
    n_candidates = best_indices.size
    best_coordinates = np.zeros(n_candidates)
    best_distances = np.zeros(n_candidates)
    for ii in range(n_candidates):
        best_index = best_indices[ii]
        # Assume distances is a fairly (but not perfectly) smooth function of coordinate
        lolim = best_index-2
        hilim = best_index+3
        if lolim < 0:
            lolim = None
        if hilim >= tracesize:
            hilim = None
        distance_coefficients = arbitrary_order_solution(2,coordinate[lolim:hilim],distances[lolim:hilim])
        best_coordinates[ii] = quadratic_extremum(distance_coefficients)
        best_distances[ii] = polynomial_value(distance_coefficients, best_coordinates[ii])
    # Identify the approximate/probable actual best point...
    best_distance = best_distances.min()
    best_coordinate = best_coordinates[np.where(best_distances == best_distance)[0][0]]
    # ...including the location of that point in space.
    best_location = np.zeros(spacesize)
    for jj in range(spacesize):
        coorddiff = np.fabs(best_coordinate - coordinate)
        best_index = np.where(coorddiff == coorddiff.min())[0][0]
        lolim = best_index-2
        hilim = best_index+3
        if lolim < 0:
            lolim = None
        if hilim >= tracesize:
            hilim = None
        xjj = tracelist[jj]
        xjj_coefficients = arbitrary_order_solution(2, coordinate[lolim:hilim], xjj[lolim:hilim])
        best_xjj = polynomial_value(xjj_coefficients, best_coordinate)
        best_location[jj] = best_xjj
    return best_coordinate, best_distance, best_location #, distances

no_reference_message = '''No reference file exists.  Use the GenerateReferences operation once to generate
an appropriate file; the file will be saved and need not be generated again.'''

# Hokay the real weak point here is determining where the dips and shoulders are in noisy data
# So we are addressing that today
# And specifically we are addressing it in real goddam data
# Okay, lesson learned: curvature, even log-log curvature, will not serve us here.
# We're gonna have to do something involving real minima.
