import numpy as np
from matplotlib import pyplot as plt
from os.path import join
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Values of x
xmin = 0.02
xmax = 50
xstep = 0.02
# Values of factor
minPercent = 1
maxPercent = 35
step = 1

outloc = 'slacx/slacxcore/operations/dmz/testplots'
reference_outloc = 'slacx/slacxcore/operations/dmz/references/polydispersity_guess_references.pickle'

def load_samples():
    fileloc = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/megaSAXSspreadsheet/megaSAXSspreadsheet.csv'
    data1 = np.loadtxt(fileloc, delimiter=',', skiprows=2, comments=',,,',
                   usecols=(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26))
    data2 = np.loadtxt(fileloc, delimiter=',', skiprows=2, usecols=(24, 25, 26))
    list_of_x_y_dy = []
    list_of_x_y_dy.append([data1[:, 0], data1[:, 1], data1[:, 2]])
    list_of_x_y_dy.append([data1[:, 3], data1[:, 4], data1[:, 5]])
    list_of_x_y_dy.append([data1[:, 6], data1[:, 7], data1[:, 8]])
    list_of_x_y_dy.append([data1[:, 9], data1[:, 10], data1[:, 11]])
    list_of_x_y_dy.append([data1[:, 12], data1[:, 13], data1[:, 14]])
    list_of_x_y_dy.append([data1[:, 15], data1[:, 16], data1[:, 17]])
    list_of_x_y_dy.append([data2[:, 0], data2[:, 1], data2[:, 2]])
    return list_of_x_y_dy

list_of_x_y_dy = load_samples()


def fullFunction(x):
    y = ((np.sin(x) - x * np.cos(x)) * x**-3)**2
    return y

def highQEnvelope(x):
    y = x**-4
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



def makeForPlots():
    x = np.arange(xmin, xmax, xstep)
    y1 = fullFunction(x)
    y2 = highQEnvelope(x)
    y02 = blur(x, 0.02)
    y05 = blur(x, 0.05)
    y10 = blur(x, 0.1)
    y20 = blur(x, 0.2)
    return x, y1, y2, y02, y05, y10, y20


def coPlots(x, y1, y2, y02, y05, y10, y20):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x, y1, c='k', lw=2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('q * const')
    ax1.set_ylabel('I * const')
    fig1.suptitle('Exact monodisperse spectrum')
    plt.close()

    fig2 = plt.figure()
    xsmall = (x > 0.1) & (x < 1)
    ax2 = fig2.add_subplot(111)
    ax2.plot(x[xsmall], y1[xsmall], c='k', lw=2)
    ysmall = (1 - 0.2*x[xsmall]**2)/9.
    ax2.plot(x[xsmall], ysmall, c='b', lw=1)
    ax2.set_xlabel('q * const')
    ax2.set_ylabel('I * const')
    fig2.suptitle('Low-q approximation')
    fig2.savefig(join(outloc, 'small_approx.pdf'))
    plt.close()

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    px, = ax4.plot(x, y1, c='k', lw=2)
    p02, = ax4.plot(x, y02, c='b', lw=1)
    p05, = ax4.plot(x, y05, c='g', lw=1)
    p10, = ax4.plot(x, y10, c='r', lw=1)
    p20, = ax4.plot(x, y20, c='purple', lw=1)
    ax4.legend((px, p02, p05, p10, p20), ('Monodisperse relation', '2% size variation', '5% size variation',
                                          '10% size variation', '20% size variation'), loc='lower left')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('q * const')
    ax4.set_ylabel('I * const')
    fig4.suptitle('Effects of size variation')
    fig4.savefig(join(outloc, 'polydispersity.pdf'))
    plt.close()

#x, y1, y2, y02, y05, y10, y20 = makeForPlots()
#coPlots(x, y1, y2, y02, y05, y10, y20)



def makeForCalcs():
    factorVals = np.arange(minPercent, maxPercent+step, step)*0.01
    x = np.arange(xmin, xmax, xstep)
    y0 = fullFunction(x)
    num_tests = len(factorVals)
    xFirstDip = np.zeros(num_tests)
    powerLaw = np.zeros((num_tests,2))
    depthFirstDip = np.zeros(num_tests)
    sigmaFirstDip = np.zeros(num_tests)
    y_x4_inf = np.zeros(num_tests)
    heightFirstDip = np.zeros(num_tests)
    lowQApprox = np.zeros((num_tests,2))
    for ii in range(num_tests):
        factor = factorVals[ii]
        y = blur(x, factor)
        x_high = np.arange(1e5, 1e5+1e3)
        y_x4_inf[ii] = (blur(x_high, factor) * x_high**4).mean()
        #logcurv = noiseless_curvature(np.log(x), np.log(y))
        scaled_y = y * x**4
        scaled_curv = noiseless_curvature(x, scaled_y)
        dips = local_minima_detector(scaled_y) & (scaled_curv > 0)
        shoulders = local_maxima_detector(scaled_y) & (scaled_curv < 0)
        #dips = local_minima_detector(y/(x**-4))
        #shoulders = local_maxima_detector(y/(x**-4))
        dips, shoulders = clean_extrema(dips, shoulders)
        plot_diagnostics(x, y, y0, y_x4_inf[ii], dips, shoulders, factor)
        xFirstDip[ii], quadCoefficients = firstDipLoc(x, scaled_y, dips)  ###
        print str(factorVals[ii]*100), quadCoefficients
        linCoefficients = linear_fit(np.log(x[shoulders]), np.log(y[shoulders]))
        powerLaw[ii,0] = np.e**(linCoefficients[0])
        powerLaw[ii,1] = linCoefficients[1]
        powerLawAtFirstDip = np.e**(linCoefficients[0] + np.log(xFirstDip[ii]) * linCoefficients[1])
        heightFirstDip[ii] = (quadCoefficients[0] + quadCoefficients[1]*xFirstDip[ii] +
                            (quadCoefficients[2]*(xFirstDip[ii]**2)))*(xFirstDip[ii]**-4)
        depthFirstDip[ii] = (powerLawAtFirstDip - heightFirstDip[ii])
        _, sigmaFirstDipLog = gauss_guess(depthFirstDip[ii], 2*quadCoefficients[2])
        sigmaFirstDip[ii] = np.e**sigmaFirstDipLog
        plot_factor(x, y, y0, dips, shoulders, linCoefficients, factor)
    return factorVals, xFirstDip, powerLaw, depthFirstDip, sigmaFirstDip

def firstDipLoc_1(x, dips):
    return x[dips][0]

def firstDipLoc(x, y, dips):
    firstDipIndex = np.where(dips)[0][0]
    quadCoefficients = quadratic_fit(x[firstDipIndex - 2:firstDipIndex + 2], y[firstDipIndex - 2:firstDipIndex + 2])
    xFirstDip = quadratic_extremum(quadCoefficients)
    return xFirstDip, quadCoefficients


def extremum_interpolated_location(x, y, index):
    quadCoefficients = quadratic_fit(x[index - 2:index + 2], y[index - 2:index + 2])
    xExtremum = quadratic_extremum(quadCoefficients)
    return xExtremum, quadCoefficients


def plot_diagnostics(x, y, y0, y_inf_scaled, dips, shoulders, factor):
    scaled_y = y * x**4
    scaled_y0 = y0 * x**4
    scaled_curv = noiseless_curvature(x, scaled_y)
    scaled_curv0 = noiseless_curvature(x, scaled_y0)
    #curv = noiseless_curvature(x, y)
    #curv0 = noiseless_curvature(x, y0)
    fig, axarray = plt.subplots(4, sharex=True)
    axarray[0].plot(x, y0, c='k', lw=2, marker=None)
    axarray[0].plot(x, y, c='b', lw=1, marker=None)
    axarray[0].plot(x[dips], y[dips], c='r', lw=0, marker='x')
    axarray[0].plot(x[shoulders], y[shoulders], c='purple', lw=0, marker='x')
    axarray[0].set_ylabel('$I$')
    axarray[0].set_yscale('log')
    axarray[1].plot(x, scaled_y0, c='k', lw=2, marker=None)
    axarray[1].plot(x, y_inf_scaled*np.ones(x.size), c='k', lw=0.5, marker=None)
    axarray[1].plot(x, scaled_y, c='b', lw=1, marker=None)
    axarray[1].plot(x[dips], scaled_y[dips], c='r', lw=0, marker='x')
    axarray[1].plot(x[shoulders], scaled_y[shoulders], c='purple', lw=0, marker='x')
    axarray[1].set_ylabel('$I \cdot q^4$')
    axarray[2].plot(x, scaled_curv0, c='k', lw=2, marker=None)
    axarray[2].plot(x, np.zeros(x.size), c='k', lw=0.5, marker=None)
    axarray[2].plot(x, scaled_curv, c='b', lw=1, marker=None)
    axarray[2].plot(x[dips], scaled_curv[dips], c='r', lw=0, marker='x')
    axarray[2].plot(x[shoulders], scaled_curv[shoulders], c='purple', lw=0, marker='x')
    axarray[2].set_ylabel('$\delta_q^2 (I \cdot q^4)$')
    axarray[3].plot(x[dips], (y_inf_scaled - scaled_y[dips]), c='r', lw=0, marker='x')
    axarray[3].plot(x[shoulders], (scaled_y[shoulders] - y_inf_scaled), c='purple', lw=0, marker='x')
    axarray[3].set_ylabel('Extrema vs $y_\inf$')
    axarray[3].set_yscale('log')
    fig.suptitle('$\Delta x / x_0 = $ %f' % factor)
    factorstring = str(factor)[2:]
    while len(factorstring) < 3:
        factorstring = factorstring + '0'
    if len(factorstring) > 3:
        factorstring = factorstring[:3]
    filename = 'diagnostic_factor%s.pdf' % factorstring
    fig.savefig(join(outloc, filename))
    plt.close()


def plot_factor(x, y, y0, dips, shoulders, coefficients, factor):
    not_too_low = (x > 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p0, = ax.plot(x[not_too_low], y0[not_too_low], c='k', lw=2, marker=None)
    pline, = ax.plot(x[not_too_low], y[not_too_low], c='b', lw=1, marker=None)
    plowex, = ax.plot(x[dips], y[dips], c='r', lw=0, marker='x')
    phighex, = ax.plot(x[shoulders], y[shoulders], c='purple', lw=0, marker='x')
    powerlaw = np.e**(coefficients[0] + np.log(x) * coefficients[1])
    pmodel, = ax.plot(x[not_too_low], powerlaw[not_too_low], c='g', lw=1, marker=None)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('q * const')
    ax.set_ylabel('I * const')
    fig.suptitle('$\Delta x / x_0 = $ %f' % factor)
    factorstring = str(factor)[2:]
    while len(factorstring) < 3:
        factorstring = factorstring + '0'
    if len(factorstring) > 3:
        factorstring = factorstring[:3]
    filename = 'factor%s.pdf' % factorstring
    fig.savefig(join(outloc, filename))
    plt.close()

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


def gauss_guess(signalMagnitude, signalCurvature):
    '''
    Guesses a gaussian + linear model for data segments.

    :param x: 1d numpy float array
    :param y: 1d numpy float array
    :param curvature: 1d numpy float array
    :param low_anchor_indices: 1d numpy int array
    :param high_anchor_indices: 1d numpy int array
    :param feature_indices: 1d numpy int array
    :return slope, offset, intensity, sigma: 1d numpy float arrays

    *x* and *y* are paired, ordered data,
    where *x* is the independent variable
    and *y* is the dependent variable.
    *curvature* is the second derivative of *y* with respect to *x*.
    *low_anchor_indices* indicate the start-points of a data segment about a feature
    and *high_anchor_indices* indicate the end-points of the same.
    At the anchor indices, *y* should be close to its background behavior,
    i.e. not dominated by the gaussian feature.
    *feature_indices* indicate the locations of the features themselves.
    The solution given is not fitted; it is a first estimate to be used in fitting.
    '''
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


def linear_fit(x, y):
    order = 1
    dy = np.ones(y.shape, dtype=float)
    # Formulate the equation to be solved for polynomial coefficients
    matrix, vector = make_poly_matrices(x, y, dy, order)
    # Solve equation
    inverse = np.linalg.pinv(matrix)
    coefficients = np.matmul(inverse, vector)
    coefficients = coefficients.flatten()
    return coefficients


def quadratic_fit(x, y):
    order = 2
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


factorVals, xFirstDip, powerLaw, depthFirstDip, sigmaFirstDip = makeForCalcs()

# factorVals: 0.01 - 0.35
# xFirstDip: 4.5 - 6.46
# powerLawMultiplier: 1.26 - 1.78
# depthFirstDip: 0.000277 - 0.00274
# sigmaFirstDip:

def plot_progression(factorVals, xFirstDip, powerLaw, depthFirstDip, sigmaFirstDip):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(factorVals*100, xFirstDip, c='k', lw=1, marker='.')
    ax1.set_xlabel('$\Delta r / r_0 \cdot 100$')
    ax1.set_ylabel('Location of first dip in units of $x = q \cdot r0$')
    fig1.savefig(join(outloc, 'dip_location.pdf'))
    plt.close()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    powerVal = powerLaw[:,0]*(xFirstDip**powerLaw[:,1])
    ax2.plot(factorVals*100, powerVal, c='k', lw=1, marker='.')
    ax2.set_xlabel('$\Delta r / r_0 \cdot 100$')
    ax2.set_ylabel('Height of power law at first dip, arbitrary units')
    fig2.savefig(join(outloc, 'power_law_height.pdf'))
    plt.close()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(factorVals*100, sigmaFirstDip, c='k', lw=1, marker='.')
    ax3.set_xlabel('$\Delta r / r_0 \cdot 100$')
    ax3.set_ylabel('Sigma "width" of first dip in units of $x = q \cdot r0$')
    fig3.savefig(join(outloc, 'dip_width.pdf'))
    plt.close()

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(factorVals*100, depthFirstDip, c='k', lw=1, marker='.')
    ax4.set_xlabel('$\Delta r / r_0 \cdot 100$')
    ax4.set_ylabel('Depth of first dip, arbitrary units')
    fig4.savefig(join(outloc, 'dip_depth.pdf'))
    plt.close()

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(factorVals*100, sigmaFirstDip/xFirstDip, c='k', lw=1, marker='.')
    ax5.set_xlabel('$\Delta r / r_0 \cdot 100$')
    ax5.set_ylabel('Width divided by location of first dip')
    fig5.savefig(join(outloc, 'dip_x_factor.pdf'))
    plt.close()

    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    ax6.plot(factorVals*100, depthFirstDip/powerVal, c='k', lw=1, marker='.')
    ax6.set_xlabel('$\Delta r / r_0 \cdot 100$')
    ax6.set_ylabel('Depth of first dip divided by power law')
    fig6.savefig(join(outloc, 'dip_y_factor.pdf'))
    plt.close()


plot_progression(factorVals, xFirstDip, powerLaw, depthFirstDip, sigmaFirstDip)

def prep_for_pickle():
    references = []
    references['factorVals'] = factorVals
    references['xFirstDip'] = xFirstDip
    references['powerLaw'] = powerLaw
    references['depthFirstDip'] = depthFirstDip
    references['sigmaFirstDip'] = sigmaFirstDip
    return references

def dump_references(references):
    with open(reference_outloc, 'wb') as handle:
        pickle.dump(references, handle)

references = prep_for_pickle()
dump_references(references)


def ReadMegaSAXS():
    fileloc = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/megaSAXSspreadsheet/megaSAXSspreadsheet.csv'
    data1 = np.loadtxt(fileloc, delimiter=',', skiprows=2, comments=',,,',
                       usecols=(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26))
    data2 = np.loadtxt(fileloc, delimiter=',', skiprows=2, usecols=(24, 25, 26))
    q1, I1, dI1 = data1[:, 0], data1[:, 1], data1[:, 2]
    q2, I2, dI2 = data1[:, 3], data1[:, 4], data1[:, 5]
    q3, I3, dI3 = data1[:, 6], data1[:, 7], data1[:, 8]
    q4, I4, dI4 = data1[:, 9], data1[:, 10], data1[:, 11]
    q5, I5, dI5 = data1[:, 12], data1[:, 13], data1[:, 14]
    q6, I6, dI6 = data1[:, 15], data1[:, 16], data1[:, 17]
    q7, I7, dI7 = data2[:, 0], data2[:, 1], data2[:, 2]
    return q1, I1, dI1 #, q2, I2, dI2, q3, I3, dI3, q4, I4, dI4, q5, I5, dI5, q6, I6, dI6, q7, I7, dI7
q1, I1, dI1 = ReadMegaSAXS()
def find_properties(q, I, dI):
    fractional_variation, qFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero, dips, shoulders = guess_polydispersity(q, I, dI)
    mean_size = guess_size(fractional_variation, qFirstDip)
    amplitude_at_zero = polydispersity_metric_heightAtZero(qFirstDip, q, I, dI)
    # dips, shoulders = choose_dips_and_shoulders(q, I, dI)
    return fractional_variation, mean_size, amplitude_at_zero, qFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero, dips, shoulders
fractional_variation, mean_size, amplitude_at_zero, qFirstDip, heightFirstDip, sigmaScaledFirstDip, heightAtZero, dips, shoulders = find_properties(q1, I1, dI1)
def plot_diagnostics(x, y, y0, y_inf_scaled, dips, shoulders, factor):
    scaled_y = y * x**4
    scaled_y0 = y0 * x**4
    scaled_curv = noiseless_curvature(x, scaled_y)
    scaled_curv0 = noiseless_curvature(x, scaled_y0)
    #curv = noiseless_curvature(x, y)
    #curv0 = noiseless_curvature(x, y0)
    fig, axarray = plt.subplots(4, sharex=True)
    axarray[0].plot(x, y0, c='k', lw=2, marker=None)
    axarray[0].plot(x, y, c='b', lw=1, marker=None)
    axarray[0].plot(x[dips], y[dips], c='r', lw=0, marker='x')
    axarray[0].plot(x[shoulders], y[shoulders], c='purple', lw=0, marker='x')
    axarray[0].set_ylabel('$I$')
    axarray[0].set_yscale('log')
    axarray[1].plot(x, scaled_y0, c='k', lw=2, marker=None)
    axarray[1].plot(x, y_inf_scaled*np.ones(x.size), c='k', lw=0.5, marker=None)
    axarray[1].plot(x, scaled_y, c='b', lw=1, marker=None)
    axarray[1].plot(x[dips], scaled_y[dips], c='r', lw=0, marker='x')
    axarray[1].plot(x[shoulders], scaled_y[shoulders], c='purple', lw=0, marker='x')
    axarray[1].set_ylabel('$I \cdot q^4$')
    axarray[2].plot(x, scaled_curv0, c='k', lw=2, marker=None)
    axarray[2].plot(x, np.zeros(x.size), c='k', lw=0.5, marker=None)
    axarray[2].plot(x, scaled_curv, c='b', lw=1, marker=None)
    axarray[2].plot(x[dips], scaled_curv[dips], c='r', lw=0, marker='x')
    axarray[2].plot(x[shoulders], scaled_curv[shoulders], c='purple', lw=0, marker='x')
    axarray[2].set_ylabel('$\delta_q^2 (I \cdot q^4)$')
    axarray[3].plot(x[dips], (y_inf_scaled - scaled_y[dips]), c='r', lw=0, marker='x')
    axarray[3].plot(x[shoulders], (scaled_y[shoulders] - y_inf_scaled), c='purple', lw=0, marker='x')
    axarray[3].set_ylabel('Extrema vs $y_\inf$')
    axarray[3].set_yscale('log')
    fig.suptitle('$\Delta x / x_0 = $ %f' % factor)
    factorstring = str(factor)[2:]
    while len(factorstring) < 3:
        factorstring = factorstring + '0'
    if len(factorstring) > 3:
        factorstring = factorstring[:3]
    filename = 'diagnostic_factor%s.pdf' % factorstring
    fig.savefig(join(outloc, filename))
    plt.close()
plot_diagnostics(q1, I1, y0, y_inf_scaled, dips, shoulders, factor)


import numpy as np
def spiral():
    phi = np.arange(0,10*np.pi, 0.1)
    r = phi/np.pi
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return r, phi, x, y
r, phi, x, y = spiral()
x0, y0 = 1.5, 5.1
best_r, best_distance, best_xy, distance = guess_nearest_point_on_nonmonotonic_trace_normalized([x0, y0], [x, y], r)
from matplotlib import pyplot as plt
def plot_spiral():
    fig, ax = plt.subplots(1)
    ax.plot(x,y,ls='-',marker='None')
    ax.plot(best_r*np.cos(phi),best_r*np.sin(phi),ls='-',marker='None')
    ax.plot(x0,y0, marker='x')
    ax.plot(best_xy[0],best_xy[1], marker='o')
    return fig, ax
def plot_distances():
    fig, ax = plt.subplots(1)
    ax.plot(r,distance,ls='-',marker='None')
    ax.plot(best_r,best_distance, marker='x')
    #ax.plot(best_xy[0],best_xy[1], marker='o')
    return fig, ax
plot_spiral()
plot_distances()
print 'Done!'
