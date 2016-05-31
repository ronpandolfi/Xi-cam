import numpy as np
from matplotlib import pyplot as plt
from os import listdir


# is it working now?

##### Functions that calculate things #####

def local_maxima_detector(y):
    '''
    Finds local maxima in ordered data y.

    :param y: 1d numpy array, ideally float dtype
    :return maxima: 1d numpy bool array

    maxima is True at a local maximum, False otherwise.

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
    Finds local minima in ordered data y.

    :param y: 1d numpy array, ideally float dtype
    :return minima: 1d numpy bool array

    minima is True at a local minimum, False otherwise.

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


def real_max(y):
    return y[~np.isnan(y)].max()


def real_min(y):
    return y[~np.isnan(y)].min()


def isolate_outliers(y, n):
    '''
    Isolates high and low outliers from normally distributed data.

    :param y: 1d numpy float array
    :param n: int
    :return normals, high_outliers, low_outliers: 1d numpy bool array x3

    y is assumed to be primarily normally distributed data.  Order of y is irrelevant.
    n is the number of standard deviations that is the cutoff for 'normal';
    in general this should be determined based on the size of y,
    but a value of 4 or 5 is good for most cases (<10,000 data points).
    Determination of an appropriate value of n is left to the user.
    '''
    normals = np.ones(y.size, bool)
    old_high_outliers = np.zeros(y.size, bool)
    old_low_outliers = np.zeros(y.size, bool)
    new_outliers = 1
    while new_outliers > 0:
        median = np.median(y[normals])
        std = np.std(y[normals])
        high_outliers = np.greater(y, median + n * std)
        low_outliers = np.less(y, median - n * std)
        new_high_outliers = high_outliers & (~old_high_outliers)
        new_low_outliers = low_outliers & (~old_low_outliers)
        new_outliers = new_high_outliers.sum() + new_low_outliers.sum()
        # Prep for next round, which may or may not trigger
        old_high_outliers = high_outliers[:]
        old_low_outliers = low_outliers[:]
        normals = (~high_outliers) & (~low_outliers)
    return normals, high_outliers, low_outliers


def find_zeros(y):
    '''
    Identifies the pixels just before zero-crossings.

    :param y: 1d numpy float array
    :return zeros: 1d numpy bool array

    y is ordered data.
    The discrete nature of arrays means that zero crossings generally happen between pixels,
    rather than on a specific unambiguous pixel.
    I arbitrarily chose to identify the pixel just before a zero crossing,
    i.e. with lower index number, rather than just after.
    zeros is a boolean array with value True for pixels
    just before y crosses from positive to negative
    and just before y crosses from negative to positive,
    False otherwise.
    '''
    positive = np.greater(y, 0)
    negative = np.less(y, 0)
    next_positive = np.zeros(y.size, dtype=bool)
    next_negative = np.zeros(y.size, dtype=bool)
    next_positive[:-1] = positive[1:]
    next_negative[:-1] = negative[1:]
    rising = positive & next_negative
    falling = negative & next_positive
    zeros = rising | falling
    return zeros


def calc_running_local_variance(y, n):
    '''
    Calculates the variance of pixel group, n to each side.

    :param y: 1d numpy float array
    :param n: int
    :return running_local_variance: 1d numpy float array

    y is ordered data.
    Creates shifted versions of the input y and stacks them together, like this
    (shown for a y of length 16 and n = 2)
    [2 3 4 5 ... 15 __ __]
    [1 2 3 4 ... 14 15 __]
    [0 1 2 3 ... 13 14 15]
    [_ 0 1 2 ... 12 13 14]
    [_ _ 0 1 ... 11 12 13]
    with a corresponding array indicating whether an element holds information or not, like this
    [1 1 1 1 ... 1  0  0]
    [1 1 1 1 ... 1  1  0]
    [1 1 1 1 ... 1  1  1]
    [0 1 1 1 ... 1  1  1]
    [0 0 1 1 ... 1  1  1]
    then takes the mean and variance of the elements of each column that exist.
    '''
    local_neighborhood = np.zeros(((2 * n + 1), y.size), dtype=float)
    element_exists = np.zeros(((2 * n + 1), y.size), dtype=bool)
    for ii in range(2 * n + 1):
        # ii ranges from 0 to 2n; jj ranges from -n to n
        jj = ii - n
        if jj < 0:
            local_neighborhood[ii, :jj] = y[-jj:]
            element_exists[ii, :jj] = True
        elif jj == 0:
            local_neighborhood[ii, :] = y[:]
            element_exists[ii, :] = True
        else:
            local_neighborhood[ii, jj:] = y[:-jj]
            element_exists[ii, jj:] = True
    running_local_sum = (local_neighborhood * element_exists).sum(axis=0)
    running_local_mean = running_local_sum / (element_exists.sum(axis=0))
    local_diffs = (local_neighborhood - running_local_mean) * element_exists
    running_local_variance = (local_diffs ** 2).sum(axis=0) / (element_exists.sum(axis=0) - 1)
    return running_local_variance


def linear_backgrounds(x, y, low_anchor_indices, high_anchor_indices):
    '''
    Finds parameters of a line connecting two points.

    :param x: 1d numpy float array
    :param y: 1d numpy float array
    :param low_anchor_indices: 1d numpy int array
    :param high_anchor_indices: 1d numpy int array
    :return slope, offset: 1d numpy float arrays

    x and y together are ordered data,
    where x is the independent variable
    and y is the dependent variable.
    low_anchor_indices indicate the start-points of a line segment
    and high_anchor_indices indicate the end-points of the same.
    The solution given is not fitted and does not account for data values between the end-points.
    It is simply a description of the line connecting those end-points.
    '''
    x1 = x[low_anchor_indices]
    x2 = x[high_anchor_indices]
    y1 = y[low_anchor_indices]
    y2 = y[high_anchor_indices]
    slope = (y2 - y1) / (x2 - x1)
    offset = y1 - slope * x1
    return slope, offset


def nested_boolean_indexing(truth_1, truth_2):
    '''
    Finds one array that slices like two input boolean arrays.

    :param truth_1: 1d numpy bool array
    :param truth_2: 1d numpy bool array
    :return ultimate_truth: 1d numpy bool array

    Finds a boolean array ultimate_truth such that
    (some_array[truth_1])[truth_2] = some_array[ultimate_truth]
    and corresponding integer array indices_ultimate such that
    (some_array[truth_1])[truth_2] = some_array[indices_ultimate]
    '''
    indices = np.arange(truth_1.size, dtype=int)
    indices_1 = indices[truth_1]
    indices_ultimate = indices_1[truth_2]
    ultimate_truth = np.zeros(truth_1.size, dtype=bool)
    ultimate_truth[indices_ultimate] = True
    return ultimate_truth, indices_ultimate


def integer_index_to_boolean(int_index, size):
    '''
    Converts an integer advanced indexing to a boolean advanced indexing.

    :param int_index: 1d numpy integer array
    :param size: integer
    :return bool_index: 1d numpy boolean array
    '''
    bool_index = np.zeros(size, dtype=bool)
    bool_index[int_index] = True
    return bool_index


def boolean_index_to_integer(bool_index):
    '''
    Converts a boolean advanced indexing to an integer advanced indexing.

    :param bool_index: 1d numpy boolean array
    :return int_index: 1d numpy integer array
    '''
    int_index = np.arange(bool_index.size, dtype=int)
    int_index = int_index[bool_index]
    return int_index


def find_low_variance(local_variance, noise_factor):
    '''
    Finds areas with low variance; smaller noise_factor is more strict.

    :param local_variance: 1d numpy float array
    :param noise_factor: float
    :return low_variance: 1d numpy bool array

    Permitted values of noise_factor are between 0 and 1, inclusive.
    For noise_factor = 0, the variance_cutoff is median_variance.
    For noise_factor = 1, the variance_cutoff is mean_variance.
    For values between 0 and 1, the variance_cutoff scales logarithmically between the two boundaries.
    Returns a boolean array with True for low-variance pixels, False otherwise.
    '''
    median_variance = np.median(local_variance)
    mean_variance = local_variance.mean()
    variance_cutoff = median_variance * (mean_variance / median_variance) ** noise_factor
    low_variance = np.less(local_variance, variance_cutoff)
    return low_variance


def sufficiently_separated_curv_zeros(feature_index, curv_zeros):
    '''
    Finds candidate items before and after, but not adjacent to, feature_index.

    :param feature_index: int
    :param curv_zeros: 1d numpy bool array
    :return curv_zeros_two_before, curv_zeros_two_after: 1d numpy bool arrays

    Returns candidate curv_zeros preceding feature_index, except the one immediately before, as curv_zeros_two_before.
    Returns candidate curv_zeros following feature_index, except the one immediately after, as curv_zeros_two_after.
    '''
    num_datapoints = curv_zeros.size
    indices = np.arange(num_datapoints, dtype=int)
    curv_zeros_indices = indices[curv_zeros]
    curv_zeros_two_before_indices = curv_zeros_indices[np.less_equal(curv_zeros_indices, feature_index)][:-1]
    curv_zeros_two_after_indices = curv_zeros_indices[np.greater_equal(curv_zeros_indices, feature_index)][1:]
    curv_zeros_two_before = integer_index_to_boolean(curv_zeros_two_before_indices, num_datapoints)
    curv_zeros_two_after = integer_index_to_boolean(curv_zeros_two_after_indices, num_datapoints)
    return curv_zeros_two_before, curv_zeros_two_after


def find_alternate_high_bound(curv_zeros_before_allowed, num_datapoints):
    '''
    Finds another point to serve as a false "high bound".

    :param curv_zeros_before_allowed: 1d numpy bool array
    :param num_datapoints: int
    :return high_bound: int

    This function is called when there are no permissible high_bound values
    that are actually higher than the feature you are trying to model.
    Instead of having one bound higher than the feature and one bound lower,
    and interpolating between them, two "bounds" both lower than the feature are found,
    and a linear relationship is extrapolated from their positions.
    Other than their x-value relative to the feature,
    they satisfy the same requirements as ordinary bounding points.
    '''
    indices = np.arange(num_datapoints, dtype=int)
    try:
        high_bound = (indices[curv_zeros_before_allowed])[-3]
    except IndexError:
        high_bound = (indices[curv_zeros_before_allowed])[-2]
    return high_bound


def find_alternate_low_bound(curv_zeros_after_allowed, num_datapoints):
    '''
    Finds another point to serve as a false "low bound".

    :param curv_zeros_after_allowed: 1d numpy bool array
    :param num_datapoints: int
    :return high_bound: int

    This function is called when there are no permissible low_bound values
    that are actually lower than the feature you are trying to model.
    Instead of having one bound higher than the feature and one bound lower,
    and interpolating between them, two "bounds" both higher than the feature are found,
    and a linear relationship is extrapolated from their positions.
    Other than their x-value relative to the feature,
    they satisfy the same requirements as ordinary bounding points.
    '''
    indices = np.arange(num_datapoints, dtype=int)
    try:
        low_bound = (indices[curv_zeros_after_allowed])[2]
    except IndexError:
        low_bound = (indices[curv_zeros_after_allowed])[1]
    return low_bound


def pick_slope_anchors(local_variance, gaussian_feature_indices, curv_zeros, noise_factor=0):
    '''
    Chooses anchor points for a linear background about features.

    :param local_variance: 1d numpy float array
    :param gaussian_feature_indices: 1d numpy int array
    :param noise_factor: float
    :return suggested_low_bound_indices, suggested_high_bound_indices: 1d numpy int arrays

    local_variance is a running local variance of some ordered data.
    gaussian_feature_indices are indices of centroids of additive gaussian features.
    noise_factor is a float between zero and one, inclusive.
    Lower values of noise_factor are more strict.  See find_low_variance() for further documentation.
    On my very nice (low-noise) test data a value of zero works well.
    Some data may need a larger value.
    suggested_low_bound_indices and suggested_high_bound_indices
    are the start and end points for a linear background fit for each feature.
    '''
    num_datapoints = local_variance.size
    num_features = gaussian_feature_indices.size
    low_variance = find_low_variance(local_variance, noise_factor)

    indices = np.arange(num_datapoints, dtype=int)
    suggested_low_bound_indices = np.zeros(gaussian_feature_indices.size, dtype=int)
    suggested_high_bound_indices = np.zeros(gaussian_feature_indices.size, dtype=int)
    no_good_background = np.zeros(num_features, dtype=bool)
    extrapolated_background = np.zeros(num_features, dtype=bool)
    for ii in range(num_features):
        jj = gaussian_feature_indices[ii]
        curv_zeros_two_before, curv_zeros_two_after = sufficiently_separated_curv_zeros(jj, curv_zeros)
        curv_zeros_before_allowed = curv_zeros_two_before & low_variance
        curv_zeros_after_allowed = curv_zeros_two_after & low_variance
        near_low_end = False
        near_high_end = False
        try:
            suggested_low_bound_indices[ii] = (indices[curv_zeros_before_allowed])[-1]
        except IndexError:
            near_low_end = True
            extrapolated_background[ii] = True
        try:
            suggested_high_bound_indices[ii] = (indices[curv_zeros_after_allowed])[0]
        except IndexError:
            near_high_end = True
            extrapolated_background[ii] = True
        if (near_low_end & near_low_end):
            no_good_background[ii] = True
        elif near_high_end:
            try:
                suggested_high_bound_indices[ii] = find_alternate_high_bound(curv_zeros_before_allowed, num_datapoints)
            except IndexError:
                no_good_background[ii] = True
        elif near_low_end:
            try:
                suggested_low_bound_indices[ii] = find_alternate_low_bound(curv_zeros_after_allowed, num_datapoints)
            except IndexError:
                no_good_background[ii] = True
    return suggested_low_bound_indices, suggested_high_bound_indices, no_good_background, extrapolated_background


def gauss_guess(x, y, curvature, low_anchor_indices, high_anchor_indices, feature_indices):
    '''
    Guesses a gaussian + linear model for data segments.

    :param x: 1d numpy float array
    :param y: 1d numpy float array
    :param curvature: 1d numpy float array
    :param low_anchor_indices: 1d numpy int array
    :param high_anchor_indices: 1d numpy int array
    :param feature_indices: 1d numpy int array
    :return slope, offset, intensity, sigma: 1d numpy float arrays

    x and y together are ordered data,
    where x is the independent variable
    and y is the dependent variable.
    curvature is the second derivative of y with respect to x.
    low_anchor_indices indicate the start-points of a data segment about a feature
    and high_anchor_indices indicate the end-points of the same.
    At the anchor indices, y should be close to its background behavior,
    i.e. not dominated by the gaussian feature.
    feature_indices indicate the locations of the features themselves.
    The solution given is not fitted; it is a first estimate to be used in fitting.
    '''
    number_features = low_anchor_indices.size
    slope, offset = linear_backgrounds(x, y, low_anchor_indices, high_anchor_indices)
    intensity = np.zeros(number_features, dtype=float)
    sigma = np.zeros(number_features, dtype=float)
    for ii in range(number_features):
        background_ii = slope[ii] * x + offset[ii]
        signal_ii = y - background_ii
        magnitude_ii = signal_ii[feature_indices[ii]]
        curvature_ii = curvature[feature_indices[ii]]
        sigma[ii] = (-magnitude_ii / curvature_ii) ** 0.5
        intensity[ii] = magnitude_ii * sigma[ii] * (2 * np.pi) ** 0.5
    return slope, offset, intensity, sigma


##### Figure-making functions #####

def figure_initial_maxima(x, y, maxima):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    ax.plot(x, y, ls='-', color='black', marker='None')
    ax.plot(x[maxima], y[maxima], ls='None', marker='+', color='red')
    ax.set_title('Naively detected local maxima')
    ax.set_xlabel('q')
    ax.set_ylabel('Intensity')
    #    plt.savefig('initial_maxima.pdf')
    return fig, ax


def figure_maxima_curvature(x, y, maxima, normed_curv, curvature_legit):
    # Two subplots, the axes array is 1-d
    fig, axarray = plt.subplots(2, sharex=True)
    axarray[0].set_title('Maxima of intensity, local curvature')
    axarray[0].plot(x, y, ls='-', color='black', marker='None')
    axarray[0].plot(x[maxima], y[maxima], ls='None', marker='+', color='red')
    axarray[0].set_ylabel('Intensity')
    axarray[1].plot(x[curvature_legit], normed_curv[curvature_legit], ls='-', marker='None', color='black')
    axarray[1].plot(x[maxima], normed_curv[maxima], ls='None', marker='+', color='red')
    axarray[1].set_ylabel('Curvature (scaled)')
    #    fig.savefig('maxima_curvature.pdf')
    return fig, axarray


def figure_curv_vs_max(x, y, exclusive_maxima, exclusive_curv_minima, max_and_curvmin, normed_curv, curvature_legit):
    fig, axarray = plt.subplots(2, sharex=True)
    axarray[0].set_title('Local maxima of intensity vs curvature minima')
    axarray[0].plot(x, y, ls='-', color='black', marker='None', lw=1)
    axarray[0].plot(x[exclusive_maxima], y[exclusive_maxima], ls='None', marker='+', color='red', ms=10)
    axarray[0].plot(x[exclusive_curv_minima], y[exclusive_curv_minima], ls='None', marker='+', color='blue', ms=10)
    axarray[0].plot(x[max_and_curvmin], y[max_and_curvmin], ls='None', marker='+', color='green', ms=10)
    axarray[0].set_ylabel('Intensity')
    axarray[1].plot(x[curvature_legit], normed_curv[curvature_legit], ls='-', marker='None', color='black', lw=1)
    axarray[1].plot(x[exclusive_maxima], normed_curv[exclusive_maxima], ls='None', marker='+', color='red', ms=10)
    axarray[1].plot(x[exclusive_curv_minima], normed_curv[exclusive_curv_minima], ls='None', marker='+', color='blue',
                    ms=10)
    axarray[1].plot(x[max_and_curvmin], normed_curv[max_and_curvmin], ls='None', marker='+', color='green', ms=10)
    axarray[1].set_ylabel('Curvature (scaled)')
    #    fig.savefig('curv_vs_max.pdf')
    return fig, axarray


def figure_curv_minima(x, y, curv_minima):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    ax.plot(x, y, ls='-', color='black', marker=',')
    ax.plot(x[curv_minima], y[curv_minima], ls='None', marker='+', color='red')
    ax.set_title('Curvature minima')
    ax.set_xlabel('q')
    ax.set_ylabel('Intensity')
    #    plt.savefig('curv_minima.pdf')
    return fig, ax


def figure_curv_minima_curvature(x, y, curv_minima, normed_curv, curvature_legit):
    fig, axarray = plt.subplots(2, sharex=True)
    axarray[0].set_title('Local minima of curvature')
    axarray[0].plot(x, y, ls='-', color='black', marker='None')
    axarray[0].plot(x[curv_minima], y[curv_minima], ls='None', marker='+', color='red')
    axarray[0].set_ylabel('Intensity')
    axarray[1].plot(x[curvature_legit], normed_curv[curvature_legit], ls='-', marker='None', color='black')
    axarray[1].plot(x[curv_minima], normed_curv[curv_minima], ls='None', marker='+', color='red')
    axarray[1].set_ylabel('Curvature (scaled)')
    #    fig.savefig('curv_minima_curvature.pdf')
    return fig, axarray


def figure_curv_minima_classified(x, y, curv_minima, high_outliers, normals, low_outliers, normed_curv):
    fig, axarray = plt.subplots(2, sharex=True)
    axarray[0].set_title('Curvature minima, classified as features, noise, and weirdness')
    axarray[0].plot(x, y, ls='-', color='black', marker='None', lw=1)
    axarray[0].plot(x[curv_minima][high_outliers], y[curv_minima][high_outliers], ls='None', marker='+', color='red',
                    ms=10)
    axarray[0].plot(x[curv_minima][normals], y[curv_minima][normals], ls='None', marker='+', color='blue', ms=10)
    axarray[0].plot(x[curv_minima][low_outliers], y[curv_minima][low_outliers], ls='None', marker='+', color='green',
                    ms=10)
    axarray[0].set_ylabel('Intensity')
    axarray[1].plot(x, normed_curv, ls='-', color='black', marker='None', lw=1)
    axarray[1].plot(x[curv_minima][high_outliers], normed_curv[curv_minima][high_outliers], ls='None', marker='+',
                    color='red', ms=10)
    axarray[1].plot(x[curv_minima][normals], normed_curv[curv_minima][normals], ls='None', marker='+', color='blue',
                    ms=10)
    axarray[1].plot(x[curv_minima][low_outliers], normed_curv[curv_minima][low_outliers], ls='None', marker='+',
                    color='green', ms=10)
    axarray[1].set_ylabel('Curvature (scaled)')
    #    fig.savefig('curv_minima_classified.pdf')
    return fig, axarray


def figure_curv_zeros(x, y, curv_zeros, normed_curv):
    fig, axarray = plt.subplots(2, sharex=True)
    axarray[0].set_title('Curvature zeros')
    axarray[0].plot(x, y, ls='-', color='black', marker='None')
    axarray[0].plot(x[curv_zeros], y[curv_zeros], ls='None', marker='+', color='red', ms=10)
    axarray[0].set_ylabel('Intensity')
    axarray[1].plot(x, normed_curv, ls='-', color='black', marker='None', lw=1)
    axarray[1].plot(x[curv_zeros], normed_curv[curv_zeros], ls='None', marker='+', color='red', ms=10)
    axarray[1].set_ylabel('Curvature (scaled)')
    #    fig.savefig('curv_zeros.pdf')
    return fig, axarray


def figure_running_variance(x, y, curv_zeros, running_local_variance):
    mean_variance = running_local_variance.mean()
    median_variance = np.median(running_local_variance)
    fig, axarray = plt.subplots(2, sharex=True)
    axarray[0].set_title('Running local variance & curvature zeros')
    axarray[0].plot(x, y, ls='-', color='black', marker='None')
    axarray[0].plot(x[curv_zeros], y[curv_zeros], ls='None', marker='+', color='red', ms=10)
    axarray[0].set_ylabel('Intensity')
    axarray[1].plot(x, np.log10(running_local_variance), ls='-', color='black', marker='None', lw=1)
    axarray[1].plot(x, np.log10(mean_variance) * np.ones(x.size), ls='-', color='blue', marker='None', lw=1)
    axarray[1].plot(x, np.log10(median_variance) * np.ones(x.size), ls='-', color='green', marker='None', lw=1)
    axarray[1].plot(x[curv_zeros], np.log10(running_local_variance)[curv_zeros], ls='None', marker='+', color='red',
                    ms=10)
    axarray[1].set_ylabel('Logarithm of running variance')
    #    fig.savefig('running_variance.pdf')
    return fig, axarray


def figure_slope_anchors_clipped(x, y, low_bound_indices, high_bound_indices, feature_indices_clipped):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    ax.plot(x, y, ls='-', color='black', marker='None')
    for ii in range(feature_indices_clipped.size):
        x1 = x[low_bound_indices[ii]]
        x2 = x[high_bound_indices[ii]]
        y1 = y[low_bound_indices[ii]]
        y2 = y[high_bound_indices[ii]]
        ax.plot([x1, x2], [y1, y2], ls='-', marker='.', color='blue')
    ax.set_title('Anchor points for feature fitting, excluding ends')
    ax.set_xlabel('q')
    ax.set_ylabel('Intensity')
    #    plt.savefig('slope_anchors_clipped.pdf')
    return fig, ax


def figure_naive_gauss_guess(x, y, low_bound_indices, high_bound_indices, feature_indices, slope,
                             offset, intensity, sigma):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    ax.plot(x, y, ls='-', color='black', marker='None')
    slope, offset = linear_backgrounds(x, y, low_bound_indices, high_bound_indices)
    number_features = feature_indices.size
    for ii in range(number_features):
        centroid = x[feature_indices[ii]]
        linear_model = slope[ii] * x + offset[ii]
        gauss_model = (intensity[ii] / (sigma[ii] * (2 * np.pi) ** 0.5)) \
                      * np.exp(-((x - centroid) ** 2) / (2 * sigma[ii] ** 2))
        # Fix plotting bpundaries if necessary
        if low_bound_indices[ii] < high_bound_indices[ii]:
            model_segment = (linear_model + gauss_model)[low_bound_indices[ii]: high_bound_indices[ii]]
            x_segment = x[low_bound_indices[ii]: high_bound_indices[ii]]
        else:
            high_x = centroid + 5 * sigma[ii]
            low_x = centroid - 5 * sigma[ii]
            segment = (np.greater(high_x, x) & np.less(low_x, x))
            print 'segment size, sum', segment.size, segment.sum()
            x_segment = x[segment]
            model_segment = (linear_model + gauss_model)[segment]
        # Mark as bad (red) if intensity < 0, good (cyan) otherwise
        if intensity[ii] > 0:
            ax.plot(x_segment, model_segment, ls=':', color='cyan')
        else:
            ax.plot(x_segment, model_segment, ls=':', color='red')
    ax.set_title('Naive gaussian parameters guess, %i features' % number_features)
    ax.set_xlabel('q')
    ax.set_ylabel('Intensity')
    #    plt.savefig('naive_gauss_guess.pdf')
    return fig, ax


##### Complex script functions #####


def batch_demo():
    data_folder = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Fang/spreadsheets1d/'
    file_list = listdir(data_folder)
    out_dir = 'batch_demo_figures/'

    # Quick 'n' dirty scrub of file_list by file name
    for ii in file_list:
        if (~ii.startswith('Sample') | ~ii.endswith('_1D.csv')):
            print 'Script wants to remove item %s from list of files to import...' % ii
            if ~ii.startswith('Sample'):
                print "...because it doesn't start with 'Sample' but instead starts with '%s'..." % ii[:6]
            if ~ii.endswith('_1D.csv'):
                print "...because it doesn't end with '_1D.csv' but instead ends with '%s'..." % ii[-7:]
            print '...but there is clearly something wrong with this picture, so it will not be removed.'
            #            file_list.remove(ii)

    plt.ioff()  # Don't show me batch plots!
    for ii in file_list:
        print "Reading file %s." % ii
        name_string = ii[6:-7]
        # Data intake
        data = np.genfromtxt(data_folder + ii, delimiter=',')
        (length, width) = data.shape  # (1096, 2)
        # The data is for some reason doubled.  Quick 2-line fix.
        length = length / 2
        data = data[0:length, :]
        x = data[:, 0]
        y = data[:, 1]
        name_location_string = out_dir + ii[:-7] + '_'

        # Find and classify curvature minima
        curvature = noiseless_curvature(x, y)
        normed_curv = curvature / (real_max(curvature) - real_min(curvature))
        curvature_legit = ~np.isnan(curvature)
        curv_minima = local_minima_detector(curvature)
        normals, high_outliers, low_outliers = isolate_outliers(curvature[curv_minima & curvature_legit], 4)
        fig, ax = figure_curv_minima_classified(x, y, curv_minima, high_outliers, normals, low_outliers, normed_curv)
        plt.savefig(name_location_string + 'curv_minima_classified.pdf')

        # Cleaning up some indexing mess, pointing to features
        features, feature_indices = nested_boolean_indexing(curv_minima, low_outliers)

        # Choose local-fit segments
        curv_zeros = find_zeros(curvature)
        running_variance = calc_running_local_variance(y, 2)
        fig, ax = figure_running_variance(x, y, curv_zeros, running_variance)
        plt.savefig(name_location_string + 'running_variance.pdf')
        low_bound_indices, high_bound_indices, no_good_background, extrapolated_background = \
            pick_slope_anchors(running_variance, feature_indices, curv_zeros, 0)
        slope, offset, intensity, sigma = gauss_guess(x, y, curvature, low_bound_indices, high_bound_indices,
                                                      feature_indices)
        fig, ax = figure_naive_gauss_guess(x, y, low_bound_indices, high_bound_indices, feature_indices, slope, offset,
                                           intensity, sigma)
        plt.savefig(name_location_string + 'naive_gauss_guess.pdf')
    plt.close('all')
    plt.ion()  # Interactive plotting back on


##### WIP functions #####


def process_demo_1():
    # Data intake
    data_folder = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Fang/spreadsheets1d/'
    file1 = 'Sample2_30x30_t60_0069_1D.csv'
    data1 = np.genfromtxt(data_folder + file1, delimiter=',')
    (length, width) = data1.shape  # (1096, 2)
    # The data is for some reason doubled.  Quick 2-line fix.
    length = length / 2
    data1 = data1[0:length, :]
    x = data1[:, 0]
    y = data1[:, 1]

    # Local maxima
    maxima = local_maxima_detector(y)
    print 'Initially detected %i local maxima.' % maxima.sum()
    figure_initial_maxima(x, y, maxima)

    # Curvature
    curvature = noiseless_curvature(x, y)
    normed_curv = curvature / (real_max(curvature) - real_min(curvature))
    curvature_legit = ~np.isnan(curvature)
    curv_minima = local_minima_detector(curvature)
    figure_maxima_curvature(x, y, maxima, normed_curv, curvature_legit)

    # Maxima vs curvature minima
    exclusive_curv_minima = curv_minima & (~maxima)
    exclusive_maxima = maxima & (~curv_minima)
    max_and_curvmin = maxima & curv_minima
    figure_curv_vs_max(x, y, exclusive_maxima, exclusive_curv_minima, max_and_curvmin, normed_curv, curvature_legit)
    figure_curv_minima(x, y, curv_minima)
    figure_curv_minima_curvature(x, y, curv_minima, normed_curv, curvature_legit)

    # Classifying curvature minima
    normals, high_outliers, low_outliers = isolate_outliers(curvature[curv_minima & curvature_legit], 4)
    print 'Found %i low outliers (features?), %i normals (noise), and %i high outliers (problems?).' % (
        low_outliers.sum(), normals.sum(), high_outliers.sum())
    figure_curv_minima_classified(x, y, curv_minima, high_outliers, normals, low_outliers, normed_curv)

    # Curvature zeros
    curv_zeros = find_zeros(curvature)
    figure_curv_zeros(x, y, curv_zeros, normed_curv)

    # Classifying curvature zeros
    running_local_variance = calc_running_local_variance(y, 2)
    mean_variance = running_local_variance.mean()
    median_variance = np.median(running_local_variance)
    print 'The median of the calculated running variance is %f, and the mean is %f.' % (median_variance, mean_variance)
    figure_running_variance(x, y, curv_zeros, running_local_variance)

    indices = np.arange(y.size, dtype=int)
    curv_minima_indices = indices[curv_minima]
    likely_gaussian_feature_indices = curv_minima_indices[low_outliers]
    likely_gaussian_features = np.zeros(y.size, dtype=bool)
    likely_gaussian_features[likely_gaussian_feature_indices] = True

    likely_gaussian_feature_indices_clipped = likely_gaussian_feature_indices[1:-1]
    likely_gaussian_feature_clipped = np.zeros(y.size, dtype=bool)
    likely_gaussian_feature_clipped[likely_gaussian_feature_indices_clipped] = True

    suggested_low_bound_indices, suggested_high_bound_indices \
        = pick_slope_anchors_mid_data(running_local_variance, likely_gaussian_feature_indices_clipped, curv_zeros, 0)
    figure_slope_anchors_clipped(x, y, suggested_low_bound_indices,
                                 suggested_high_bound_indices, likely_gaussian_feature_indices_clipped)

    slope, offset, intensity, sigma = gauss_guess(x, y, curvature, suggested_low_bound_indices,
                                                  suggested_high_bound_indices, likely_gaussian_feature_indices_clipped)
    figure_naive_gauss_guess(x, y, suggested_low_bound_indices, suggested_high_bound_indices,
                             likely_gaussian_feature_indices_clipped, slope, offset, intensity, sigma)


process_demo_1()


##### Run scripts, optional #####

# process_demo_1()
# batch_demo()

##### Not-yet-started and not-yet-used functions #####


def endpoint_curvature_cheat(curvature):
    '''
    Guesses the value of the curvature at the endpoints.

    :param curvature: numpy float array
    :return curvature: numpy float array
    '''
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def noisy_curvature(x, y, width):
    pass


def no_ends(x, clip):
    '''
    Returns a boolean array with pixels too close to the ends marked.

    :param x: numpy array
    :param clip: int
    :return clipped: numpy bool array

    x is a 1d array of the correct size.
    clip is an integer number of pixels that will be masked at each end.
    clipped is a boolean array with True for pixels with respectable locations
    and False for pixels with dodgy locations.
    '''
    clipped = np.ones(x.size, dtype=bool)
    clipped[clip:] = False
    clipped[:-clip] = False
    return clipped


def distance_from_nearest_neighbor(x, maxima):
    pass


def reject_pure_noise_maxima(x, y, maxima):
    pass


def cluster_maxima(x, y, maxima):
    pass
