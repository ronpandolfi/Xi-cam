import numpy as np
from matplotlib import pyplot as plt
import os


##### Functions that calculate things #####

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

    *y* is assumed to be primarily normally distributed data.  Order of *y* is irrelevant.
    *n* is the number of standard deviations that is the cutoff for 'normal';
    in general this should be determined based on the size of *y*,
    but a value of 4 or 5 is good for most cases (<10,000 data points).
    Determination of an appropriate value of *n* is left to the user.
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

    *y* is ordered data.
    The discrete nature of arrays means that zero crossings generally happen between pixels,
    rather than on a specific unambiguous pixel.
    I arbitrarily chose to identify the pixel just before a zero crossing,
    i.e. with lower index number, rather than just after.
    *zeros* is a boolean array with value *True* for pixels
    just before *y* crosses from positive to negative
    and just before *y* crosses from negative to positive,
    *False* otherwise.
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


def linear_backgrounds(x, y, low_anchor_indices, high_anchor_indices):
    '''
    Finds parameters of a line connecting two points.

    :param x: 1d numpy float array
    :param y: 1d numpy float array
    :param low_anchor_indices: 1d numpy int array
    :param high_anchor_indices: 1d numpy int array
    :return slope, offset: 1d numpy float arrays

    *x* and *y* are paired, ordered data,
    where *x* is the independent variable (coordinate)
    and *y* is the dependent variable (value).
    *low_anchor_indices* indicate the start-points of a line segment
    and *high_anchor_indices* indicate the end-points of the same.
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


def nested_boolean_indexing(boolean_slice_1, boolean_slice_2):
    '''
    Finds one array that slices like two input boolean arrays.

    :param boolean_slice_1: 1d numpy bool array
    :param boolean_slice_2: 1d numpy bool array
    :return ultimate_truth: 1d numpy bool array

    'Advanced indexing' in numpy returns, or sometimes returns,
    a copy instead of a view of the array being sliced.  I'm not 100% on the rules.
    This is my workaround for a case where a copy is definitely returned:
    when you slice like *(some_array[boolean_slice_1])[boolean_slice_2]*.
    Problematic when you are attempting to change specific elements of some_array.
    Finds a boolean array ultimate_truth such that
    *(some_array[boolean_slice_1])[boolean_slice_2] = some_array[ultimate_truth]*
    and corresponding integer array ultimate_indices such that
    *(some_array[boolean_slice_1])[boolean_slice_2] = some_array[ultimate_indices]*
    '''
    indices = np.arange(boolean_slice_1.size, dtype=int)
    indices_1 = indices[boolean_slice_1]
    ultimate_indices = indices_1[boolean_slice_2]
    ultimate_truth = np.zeros(boolean_slice_1.size, dtype=bool)
    ultimate_truth[ultimate_indices] = True
    return ultimate_truth, ultimate_indices


def integer_index_to_boolean(int_index, size):
    '''
    Converts an integer advanced indexing to a boolean advanced indexing.

    :param int_index: 1d numpy integer array
    :param size: integer
    :return bool_index: 1d numpy boolean array

    Finds an array *bool_index* such that
    *some_array[bool_index] = some_array[int_index]*
    for any some_array possessing *size* elements.
    '''
    bool_index = np.zeros(size, dtype=bool)
    bool_index[int_index] = True
    return bool_index


def boolean_index_to_integer(bool_index):
    '''
    Converts a boolean advanced indexing to an integer advanced indexing.

    :param bool_index: 1d numpy boolean array
    :return int_index: 1d numpy integer array

    Finds an array *int_index* such that
    *some_array[int_index] = some_array[bool_index]*.
    '''
    int_index = np.arange(bool_index.size, dtype=int)
    int_index = int_index[bool_index]
    return int_index


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


def calc_running_local_variance(y, n):
    '''
    Calculates the variance of pixel group, n to each side.

    :param y: 1d numpy float array
    :param n: int
    :return running_local_variance: 1d numpy float array

    *y* is ordered data.
    *n* is the number of pixels to each side included in the running variance.
    *n* should not be smaller than 1, and *n* usually should be much smaller than the size of *y*.
    *shift_stack()* creates shifted versions of the input y and stacks them together;
    *masked_variance_2d_axis_0()* finds the variance along axis 0, i.e. along each column.
    See *shift_stack()* and *masked_variance_2d_axis_0()* for further documentation.
    '''
    # local_neighborhood is shifted, concatenated data; element_exists is its mask
    local_neighborhood, element_exists = shift_stack(y, n, n)
    running_local_variance = masked_variance_2d_axis_0(local_neighborhood, element_exists)
    return running_local_variance


def find_low_variance(local_variance, noise_factor):
    '''
    Finds areas with low variance; smaller noise_factor is more strict.

    :param local_variance: 1d numpy float array
    :param noise_factor: float
    :return low_variance: 1d numpy bool array

    Permitted values of noise_factor are between 0 and 1, inclusive.
    For *noise_factor = 0*, the *variance_cutoff* is *median_variance*.
    For *noise_factor = 1*, the *variance_cutoff* is *mean_variance*.
    For values between 0 and 1, the *variance_cutoff* scales logarithmically between the two boundaries.
    Returns a boolean array with *True* for low-variance pixels, *False* otherwise.
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

    Returns candidate *curv_zeros* preceding *feature_index*,
    except the one immediately before, as *curv_zeros_two_before*.
    Returns candidate *curv_zeros* following *feature_index*,
    except the one immediately after, as *curv_zeros_two_after*.
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

    This function is called when there are no permissible *high_bound* values
    that are actually higher than the feature you are trying to model.
    Instead of having one bound higher than the feature and one bound lower,
    and interpolating between them, two "bounds" both lower than the feature are found,
    and a linear relationship is extrapolated from their positions.
    Other than their *x*-value relative to the feature,
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

    This function is called when there are no permissible *low_bound* values
    that are actually lower than the feature you are trying to model.
    Instead of having one bound higher than the feature and one bound lower,
    and interpolating between them, two "bounds" both higher than the feature are found,
    and a linear relationship is extrapolated from their positions.
    Other than their *x*-value relative to the feature,
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

    *local_variance* is a running local variance of some ordered data.
    *gaussian_feature_indices* are indices of centroids of additive gaussian features.
    *noise_factor* is a float between zero and one, inclusive.
    Lower values of *noise_factor* are more strict.  See *find_low_variance()* for further documentation.
    On my very nice (low-noise) test data a value of zero works well.
    Some data may need a larger value, or different provisions altogether.
    *suggested_low_bound_indices* and *suggested_high_bound_indices*
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


def read_mega_spreadsheet(filename, data_folder):
    '''
    Strictly for reading some data Liheng gave me.

    :param filename:
    :param data_folder:
    :return trimmed_data_list: A list of lists of 1d numpy float arrays
    '''
    data = np.genfromtxt(data_folder + filename, delimiter=',')
    # (length, width) = data.shape  # (429, 27)
    # Split data into components
    data1 = data[:, 0:3]
    data2 = data[:, 4:7]
    data3 = data[:, 8:11]
    data4 = data[:, 12:15]
    data5 = data[:, 16:19]
    data6 = data[:, 20:23]
    data7 = data[:, 24:27]
    data_list = [data1, data2, data3, data4, data5, data6, data7]
    #    print 'data 1 end', data1[-20:, :]
    #    print 'data 7 end', data7[-20:, :]
    trimmed_data_list = []
    for i in data_list:
        i_mask = ~np.isnan(i[:, 0])
        x = (i[:, 0])[i_mask]
        y = (i[:, 1])[i_mask]
        dy = (i[:, 2])[i_mask]
        trimmed_data_list.append([x, y, dy])
    return trimmed_data_list


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


def figure_smoothed_comparison(x_list, y_list, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    data, = ax.plot(x_list[0], y_list[0], ls='-', color='black', marker=',', lw=2)
    smoothed_1, = ax.plot(x_list[1], y_list[1], ls='-', color='blue', marker=',', lw=1)
    smoothed_2, = ax.plot(x_list[2], y_list[2], ls='-', color='green', marker=',', lw=1)
    smoothed_3, = ax.plot(x_list[3], y_list[3], ls='-', color='red', marker=',', lw=1)
    smoothed_4, = ax.plot(x_list[4], y_list[4], ls='-', color='magenta', marker=',', lw=1)
    handles = [data, smoothed_1, smoothed_2, smoothed_3, smoothed_4]
    # Shrink x axis by 20%; place legend to the right plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, prop={'size': 10}, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('q')
    ax.set_ylabel('Intensity')
    return fig, ax


def plot_peaks_flat_all(x, y_lists, masks, title, location, labels):
    # Feature region plot
    x_peak = x[masks[0]]
    x_peaks = []
    y_peaks = []
    for i in range(len(labels)):
        y_peak = (y_lists[i])[masks[0]]
        y_peaks.append(y_peak)
        x_peaks.append(x_peak)
    fig, ax = figure_smoothed_comparison(x_peaks, y_peaks, labels)
    ax.set_title(title)
    plt.savefig(location + 'peaks.pdf')
    # Continuum region plot
    x_flat = x[masks[1]]
    x_flats = []
    y_flats = []
    for i in range(len(labels)):
        y_flat = (y_lists[i])[masks[1]]
        y_flats.append(y_flat)
        x_flats.append(x_flat)
    fig, ax = figure_smoothed_comparison(x_flats, y_flats, labels)
    ax.set_title(title)
    plt.savefig(location + 'continuum.pdf')
    # Full range plot
    x_all = x[masks[2]]
    x_alls = []
    y_alls = []
    for i in range(len(labels)):
        y_all = (y_lists[i])[masks[2]]
        y_alls.append(y_all)
        x_alls.append(x_all)
    fig, ax = figure_smoothed_comparison(x_alls, y_alls, labels)
    ax.set_title(title)
    plt.savefig(location + 'all.pdf')


def figure_many_intensity_errors(data):
    num_data = len(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'indigo']
    for ii in range(num_data):
        x = data[ii][0]
        y = data[ii][1]
        dy = data[ii][2]
        ax.fill_between(x, y + dy, y - dy, facecolor=colors[ii], alpha=0.3)
        ax.plot(x, y, lw=3, ls='-', marker=',', color=colors[ii])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Intensity and error estimates, SAXS data')
    ax.set_xlabel('q')
    ax.set_ylabel('Intensity')
    return fig, ax


def figure_intensity_errors(data, out_folder):
    num_data = len(data)
    for ii in range(num_data):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = data[ii][0]
        y = data[ii][1]
        dy = data[ii][2]
        ax.fill_between(x, y + dy, y - dy, facecolor='k', alpha=0.3)
        ax.plot(x, y, lw=3, ls='-', marker=',', color='k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Intensity and error estimates, SAXS data')
        ax.set_xlabel('q')
        ax.set_ylabel('Intensity')
        plt.savefig(out_folder + 'intensity_errors_' + str(ii) + '.pdf')


##### Complex script functions #####


def process_demo_1():
    out_folder = 'process_demo_1_figures/'
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
    fig, ax = figure_initial_maxima(x, y, maxima)
    plt.savefig(out_folder + 'initial_maxima.pdf')
    #    plt.savefig(out_folder + '.pdf')

    # Curvature
    curvature = noiseless_curvature(x, y)
    normed_curv = curvature / (real_max(curvature) - real_min(curvature))
    curvature_legit = ~np.isnan(curvature)
    curv_minima = local_minima_detector(curvature)
    fig, ax = figure_maxima_curvature(x, y, maxima, normed_curv, curvature_legit)
    plt.savefig(out_folder + 'maxima_curvature.pdf')

    # Maxima vs curvature minima
    exclusive_curv_minima = curv_minima & (~maxima)
    exclusive_maxima = maxima & (~curv_minima)
    max_and_curvmin = maxima & curv_minima
    fig, ax = figure_curv_vs_max(x, y, exclusive_maxima, exclusive_curv_minima, max_and_curvmin, normed_curv,
                                 curvature_legit)
    plt.savefig(out_folder + 'curv_vs_max.pdf')
    fig, ax = figure_curv_minima(x, y, curv_minima)
    plt.savefig(out_folder + 'curv_minima.pdf')
    fig, ax = figure_curv_minima_curvature(x, y, curv_minima, normed_curv, curvature_legit)
    plt.savefig(out_folder + 'curv_minima_curvature.pdf')

    # Classifying curvature minima
    normals, high_outliers, low_outliers = isolate_outliers(curvature[curv_minima & curvature_legit], 4)
    print 'Found %i low outliers (features?), %i normals (noise), and %i high outliers (problems?).' % (
        low_outliers.sum(), normals.sum(), high_outliers.sum())
    fig, ax = figure_curv_minima_classified(x, y, curv_minima, high_outliers, normals, low_outliers, normed_curv)
    plt.savefig(out_folder + 'curv_minima_classified.pdf')

    # Curvature zeros
    curv_zeros = find_zeros(curvature)
    fig, ax = figure_curv_zeros(x, y, curv_zeros, normed_curv)
    plt.savefig(out_folder + 'curv_zeros.pdf')

    # Classifying curvature zeros
    running_local_variance = calc_running_local_variance(y, 2)
    mean_variance = running_local_variance.mean()
    median_variance = np.median(running_local_variance)
    print 'The median of the calculated running variance is %f, and the mean is %f.' % (median_variance, mean_variance)
    fig, ax = figure_running_variance(x, y, curv_zeros, running_local_variance)
    plt.savefig(out_folder + 'running_variance.pdf')

    indices = np.arange(y.size, dtype=int)
    curv_minima_indices = indices[curv_minima]
    likely_gaussian_feature_indices = curv_minima_indices[low_outliers]
    likely_gaussian_features = np.zeros(y.size, dtype=bool)
    likely_gaussian_features[likely_gaussian_feature_indices] = True

    likely_gaussian_feature_indices_clipped = likely_gaussian_feature_indices[1:-1]
    likely_gaussian_feature_clipped = np.zeros(y.size, dtype=bool)
    likely_gaussian_feature_clipped[likely_gaussian_feature_indices_clipped] = True

    suggested_low_bound_indices, suggested_high_bound_indices, no_good_background, extrapolated_background \
        = pick_slope_anchors(running_local_variance, likely_gaussian_feature_indices_clipped, curv_zeros, 0)
    fig, ax = figure_slope_anchors_clipped(x, y, suggested_low_bound_indices,
                                           suggested_high_bound_indices, likely_gaussian_feature_indices_clipped)
    plt.savefig(out_folder + 'slope_anchors_clipped.pdf')

    slope, offset, intensity, sigma = gauss_guess(x, y, curvature, suggested_low_bound_indices,
                                                  suggested_high_bound_indices, likely_gaussian_feature_indices_clipped)
    fig, ax = figure_naive_gauss_guess(x, y, suggested_low_bound_indices, suggested_high_bound_indices,
                                       likely_gaussian_feature_indices_clipped, slope, offset, intensity, sigma)
    plt.savefig(out_folder + 'naive_gauss_guess.pdf')


def batch_demo():
    data_folder = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Fang/spreadsheets1d/'
    file_list = os.listdir(data_folder)
    out_dir = 'batch_demo_figures/'

    # Quick 'n' dirty scrub of file_list by file name
    for ii in file_list:
        ii = str(ii)
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

def mean_smoother(y, n):
    '''
    Takes local running mean *n* pixels to each side.

    :param y: 1d numpy float array
    :param n: int
    :return mean_smoothed:  1d numpy float array
    '''
    y_stacked, mask = shift_stack(y, n, n)
    mean_smoothed = masked_mean_2d_axis_0(y_stacked, mask)
    return mean_smoothed


def masked_median_2d_axis_0(y2d, mask2d):
    '''
    Takes the median of masked data along axis 0.

    :param y2d: 2d numpy float array
    :param mask2d: 2d numpy bool array
    :return mean: 1d numpy float array

    *y2d* is data; *mask2d* is its corresponding mask
    with values *True* for legitimate data, *False* otherwise.
    Unlike mean and variance, median cannot negate elements by setting their weight to zero.
    This leads to unequally-sized regions, esp. near boundaries.
    In order to avoid unnecessary iteration over unevenly-sized boundary regions,
    those regions are split up by number of valid elements,
    e.g., all pixels with 3 valid elements are handled together.
    Returns *median*, the median of *y2d* along axis 0.
    '''
    median = np.zeros(y2d.size, dtype=float)
    mask = mask2d.any(axis=0)
    num_entries = mask2d.sum(axis=0)
    num_max_entries = num_entries.max()
    num_min_entries = num_entries.min()
    if num_min_entries == 0:
        num_min_entries = 1  # skip past empty columns; they're already masked zeros
    median = np.zeros(num_entries.size, dtype=float)
    for i in range(num_min_entries, num_max_entries + 1):
        i_entries = np.equal(num_entries, i)
        num_i = i_entries.sum()
        if num_i != 0:
            mask_i = i_entries & mask2d  # i_entries is broadcast to dimensions of mask2d
            # numpy aggregates items along rows first, then columns.
            # We want columns first for this application, so we get clever with transpose.
            medianees = y2d.T[mask_i.T]
            medianees = (medianees.reshape((num_i, i))).T
            median[i_entries] = np.median(medianees, axis=0)
    return median, mask


def median_smoother(y, n):
    '''
    Takes local running median *n* pixels to each side.

    :param y: 1d numpy float array
    :param n: int
    :return median_smoothed:  1d numpy float array
    :return median_mask:  1d numpy bool array

    If for whatever reason there are no valid elements at a pizel,
    the median_mask at that pixel is set to zero.
    This may need to be handled separately from overall image mask; not sure.
    Presently included for completeness, I guess.
    '''
    y_stacked, mask = shift_stack(y, n, n)
    median_smoothed, median_mask = masked_median_2d_axis_0(y_stacked, mask)
    return median_smoothed, median_mask


def curvature_from_quadratic_approximation(x, y, n):
    abc = local_quadratic_approximation(x, y, n)
    a = abc[0, :]
    curvature = 2 * a
    return curvature


def local_quadratic_approximation_no_errors(x, y, n):
    x_stacked, stack_mask = shift_stack(x, n, n)
    y_stacked, stack_mask = shift_stack(y, n, n)
    a, b, c = quadratic_approximation_no_errors(x_stacked, y_stacked, stack_mask)
    y_smoothed = a * x ** 2 + b * x + c
    curv_smoothed = 2 * a
    return y_smoothed, curv_smoothed, a, b, c


def local_quadratic_approximation_with_errors(x, y, dy, n):
    x_stacked, stack_mask = shift_stack(x, n, n)
    #    y_stacked, stack_mask = shift_stack(y, n, n)
    #    dy_stacked, stack_mask = shift_stack(dy, n, n)
    #    a, b, c = quadratic_approximation(x_stacked, y_stacked, dy_stacked, stack_mask)
    a, b, c = quadratic_approximation(x_stacked, y, dy, stack_mask)
    y_smoothed = a * x ** 2 + b * x + c
    curv_smoothed = 2 * a
    return y_smoothed, curv_smoothed, a, b, c


def quadratic_approximation(x, y, dy, mask):
    length = x.shape[1]
    # Notation in the form of
    # xe4_dyen2 = x exponent 4 (x**4) dy exponent negative 2 (dy**-2)
    xe4_dyen2 = (x ** 4 * dy ** -2 * mask).sum(axis=0)
    xe3_dyen2 = (x ** 3 * dy ** -2 * mask).sum(axis=0)
    xe2_dyen2 = (x ** 2 * dy ** -2 * mask).sum(axis=0)
    x_dyen2 = (x * dy ** -2 * mask).sum(axis=0)
    dyen2 = (dy ** -2 * mask).sum(axis=0)
    y_xe2_dyen2 = (y * x ** 2 * dy ** -2 * mask).sum(axis=0)
    y_x_dyen2 = (y * x * dy ** -2 * mask).sum(axis=0)
    y_dyen2 = (y * dy ** -2 * mask).sum(axis=0)
    abc = np.zeros((3, length), dtype=float)
    for i in range(length):
        fit_vector = np.array([[y_xe2_dyen2[i]],
                               [y_x_dyen2[i]],
                               [y_dyen2[i]]])
        fit_matrix = np.array([[xe4_dyen2[i], xe3_dyen2[i], xe2_dyen2[i]],
                               [xe3_dyen2[i], xe2_dyen2[i], x_dyen2[i]],
                               [xe2_dyen2[i], x_dyen2[i], dyen2[i]]])
        abc[:, i] = (np.dot(np.linalg.pinv(fit_matrix), fit_vector)).ravel()
    a = abc[0, :]
    b = abc[1, :]
    c = abc[2, :]
    return a, b, c


def quadratic_approximation_no_errors(x, y, mask):
    dy = np.ones(x.shape, dtype=float)
    return quadratic_approximation(x, y, dy, mask)


def quadratic_approximation_no_mask(x, y, dy):
    mask = np.ones(x.shape, dtype=bool)
    return quadratic_approximation(x, y, dy, mask)


def quadratic_approximation_no_mask_no_errors(x, y):
    dy = np.ones(x.shape, dtype=float)
    mask = np.ones(x.shape, dtype=bool)
    return quadratic_approximation(x, y, dy, mask)


# Finish later
def overplot_quadratic_approx(x, y, n, a, b, c):
    num_data = x.size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xn = shift_stack(x, n, n)
    y_smooth = a * x ** 2 + b * x + c
    yn = a * xn ** 2 + b * xn + c

    for ii in range(num_data):
        x = data[ii][0]
        y = data[ii][1]
        dy = data[ii][2]
        ax.fill_between(x, y + dy, y - dy, facecolor=colors[ii], alpha=0.3)
        ax.plot(x, y, lw=3, ls='-', marker=',', color=colors[ii])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Intensity and error estimates, SAXS data')
    ax.set_xlabel('q')
    ax.set_ylabel('Intensity')
    return fig, ax


# Ugh, finish later
def plot_peaks_flat_all_quadsanity(x, y, y_smooth, a, b, c, masks, title, location, labels):
    # Feature region plot
    x_peak = x[masks[0]]
    x_peaks = []
    y_peaks = []
    for i in range(len(labels)):
        y_peak = (y_lists[i])[masks[0]]
        y_peaks.append(y_peak)
        x_peaks.append(x_peak)
    fig, ax = figure_smoothed_comparison(x_peaks, y_peaks, labels)
    ax.set_title(title)
    plt.savefig(location + 'peaks.pdf')
    # Continuum region plot
    x_flat = x[masks[1]]
    x_flats = []
    y_flats = []
    for i in range(len(labels)):
        y_flat = (y_lists[i])[masks[1]]
        y_flats.append(y_flat)
        x_flats.append(x_flat)
    fig, ax = figure_smoothed_comparison(x_flats, y_flats, labels)
    ax.set_title(title)
    plt.savefig(location + 'continuum.pdf')
    # Full range plot
    x_all = x[masks[2]]
    x_alls = []
    y_alls = []
    for i in range(len(labels)):
        y_all = (y_lists[i])[masks[2]]
        y_alls.append(y_all)
        x_alls.append(x_all)
    fig, ax = figure_smoothed_comparison(x_alls, y_alls, labels)
    ax.set_title(title)
    plt.savefig(location + 'all.pdf')


def smoothing_demo_1():
    out_folder = '/Users/Amanda/PyCharmProjects/peak_detection/smoothing_demo_1_figures/'
    # Data intake
    data_folder = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Fang/spreadsheets1d/'
    file1 = 'Sample2_30x30_t60_0069_1D.csv'
    data = np.genfromtxt(data_folder + file1, delimiter=',')
    (length, width) = data.shape  # (1096, 2)
    # The data is for some reason doubled.  Quick 2-line fix.
    length = length / 2
    data = data[0:length, :]
    x = data[:, 0]
    y = data[:, 1]
    # masks specify interesting regions of the spectrum
    peaks_mask = (np.less(x, 5.5) & np.greater(x, 4.5))
    continuum_mask = (np.less(x, 2.0) & np.greater(x, 1.0))
    all_mask = np.ones(x.shape, dtype=bool)
    masks = [peaks_mask, continuum_mask, all_mask]

    plt.ioff()  # Don't show me batch plots!
    # mean smoothing
    mean_smoothed_3 = mean_smoother(y, 1)
    mean_smoothed_5 = mean_smoother(y, 2)
    mean_smoothed_7 = mean_smoother(y, 3)
    mean_smoothed_9 = mean_smoother(y, 4)
    y_list = [y, mean_smoothed_3, mean_smoothed_5, mean_smoothed_7, mean_smoothed_9]
    labels = ['Data', '3-pixel smoothed', '5-pixel smoothed', '7-pixel smoothed', '9-pixel smoothed']
    title = 'Mean smoothing comparison'
    location = out_folder + 'mean_smoothed_'
    plot_peaks_flat_all(x, y_list, masks, title, location, labels)

    # median smoothing
    median_smoothed_3, _ = median_smoother(y, 1)
    median_smoothed_5, _ = median_smoother(y, 2)
    median_smoothed_7, _ = median_smoother(y, 3)
    median_smoothed_9, _ = median_smoother(y, 4)
    y_list = [y, median_smoothed_3, median_smoothed_5, median_smoothed_7, median_smoothed_9]
    labels = ['Data', '3-pixel smoothed', '5-pixel smoothed', '7-pixel smoothed', '9-pixel smoothed']
    title = 'Median smoothing comparison'
    location = out_folder + 'median_smoothed_'
    plot_peaks_flat_all(x, y_list, masks, title, location, labels)

    # quadratic approximation smoothing
    quad_smoothed_5, _, a_5, b_5, c_5 = local_quadratic_approximation_no_errors(x, y, 2)
    quad_smoothed_7, _, a_7, b_7, c_7 = local_quadratic_approximation_no_errors(x, y, 3)
    quad_smoothed_9, _, a_9, b_9, c_9 = local_quadratic_approximation_no_errors(x, y, 4)
    quad_smoothed_11, _, a_11, b_11, c_11 = local_quadratic_approximation_no_errors(x, y, 5)
    a_list = [a_5, a_7, a_9, a_11]
    b_list = [b_5, b_7, b_9, b_11]
    c_list = [c_5, c_7, c_9, c_11]
    y_list = [y, quad_smoothed_5, quad_smoothed_7, quad_smoothed_9, quad_smoothed_11]
    labels = ['Data', '5-pixel smoothed', '7-pixel smoothed', '9-pixel smoothed', '11-pixel smoothed']
    title = 'Quadratic smoothing comparison'
    location = out_folder + 'quadratic_smoothed_'
    plot_peaks_flat_all_quadsanity(x, y_list, a_list, b_list, c_list, masks, title, location, labels)

    plt.close('all')
    plt.ion()  # Interactive plotting back on


def saxs_demo_do_one(x, y, suffix, out_folder):
    # Find and classify curvature maxima
    log_curv = noiseless_curvature(np.log(x), np.log(y))
    normed_curv = log_curv / (real_max(log_curv) - real_min(log_curv))
    curvature_legit = ~np.isnan(log_curv)
    log_curv_maxima = local_maxima_detector(log_curv)
    normals, high_outliers, low_outliers = isolate_outliers(log_curv[log_curv_maxima & curvature_legit], 4)
    fig, axarray = figure_curv_minima_classified(x, y, log_curv_maxima, low_outliers, normals, high_outliers,
                                                 normed_curv)
    axarray[0].set_title('Curvature maxima in log-log space, classified as features, noise, and weirdness')
    axarray[0].set_xscale('log')
    axarray[0].set_yscale('log')
    axarray[1].set_xscale('log')
    #        axarray[1].set_yscale('log')
    plt.savefig(out_folder + 'curv_maxima_classified_' + suffix + '.pdf')
    plt.close()

    # Cleaning up some indexing mess, pointing to features
    features, feature_indices = nested_boolean_indexing(log_curv_maxima, high_outliers)

    # Choose local-fit segments
    curv_zeros = find_zeros(log_curv)
    running_variance = calc_running_local_variance(np.log(y), 2)
    fig, axarray = figure_running_variance(x, y, curv_zeros, running_variance)
    axarray[0].set_xscale('log')
    axarray[0].set_yscale('log')
    axarray[1].set_xscale('log')
    plt.savefig(out_folder + 'running_variance_' + suffix + '.pdf')
    plt.close()
    low_bound_indices, high_bound_indices, no_good_background, extrapolated_background = \
        pick_slope_anchors(running_variance, feature_indices, curv_zeros, 0)
    slope, offset, intensity, sigma = gauss_guess(np.log(x), np.log(y), log_curv, low_bound_indices, high_bound_indices,
                                                  feature_indices)
    fig, ax = figure_naive_gauss_guess(np.log(x), np.log(y), low_bound_indices, high_bound_indices, feature_indices,
                                       slope, offset,
                                       intensity, sigma)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plt.savefig(out_folder + 'naive_gauss_guess_' + suffix + '.pdf')
    plt.close()


def saxs_demo_1():  # Pretty much just showing it reads in so far.  More to come.
    out_folder = '/Users/Amanda/PyCharmProjects/peak_detection/saxs_demo_1_figures/'
    # Data intake
    data_folder = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/megaSAXSspreadsheet/'
    file1 = 'megaSAXSspreadsheet.csv'
    data = read_mega_spreadsheet(file1, data_folder)
    [data1, data2, data3, data4, data5, data6, data7] = data
    [[x1, y1, dy1], [x2, y2, dy2], [x3, y3, dy3], [x4, y4, dy4], [x5, y5, dy5], [x6, y6, dy6], [x7, y7, dy7]] = data

    plt.ioff()  # Don't show me batch plots!
    fig, ax = figure_many_intensity_errors(data)
    plt.savefig(out_folder + 'many_intensity_errors.pdf')
    figure_intensity_errors(data, out_folder)

    for i in range(len(data)):
        [x, y, dy] = data[i]

        saxs_demo_do_one(x, y, str(i), out_folder)

        # Smooth
        mean_smoothed_19 = mean_smoother(y, 9)

        saxs_demo_do_one(x, mean_smoothed_19, 'SMOOTHED_' + str(i), out_folder)

    plt.close('all')
    plt.ion()  # Interactive plotting back on


def read_one_beam_csv(filename):
    # filename the full path name
    data = np.genfromtxt(filename, delimiter=',', skip_header=2, autostrip=True, usecols=(0, 4))
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def remove_non_csv(filenames):
    csv_only = []
    csv_count = 0
    for i in filenames:
        if i[-4:] == '.csv':
            csv_only.append(i)
            csv_count += 1
    # print '%i .csv files found.' % csv_count
    return csv_only


def saxs_demo_do_a_thing(x, y, suffix, out_folder):
    # Find and classify curvature maxima
    log_curv = noiseless_curvature(np.log(x), np.log(y))
    normed_curv = log_curv / (real_max(log_curv) - real_min(log_curv))
    curvature_legit = ~np.isnan(log_curv)
    log_curv_maxima = local_maxima_detector(log_curv)
    normals, high_outliers, low_outliers = isolate_outliers(log_curv[log_curv_maxima & curvature_legit], 4)
    fig, axarray = figure_curv_minima_classified(x, y, log_curv_maxima, low_outliers, normals, high_outliers,
                                                 normed_curv)
    a_title = '''Curvature maxima in log-log space, \n classified as features, noise, and weirdness'''
    axarray[0].set_xlim([2., 5.])
    axarray[1].set_xlim([2., 5.])
    fig.set_xlim([2., 5.])
    axarray[0].set_title(a_title)
    axarray[0].set_xscale('log')
    axarray[0].set_yscale('log')
    axarray[1].set_xscale('log')
    #        axarray[1].set_yscale('log')
    plt.savefig(out_folder + 'curv_maxima_classified_' + suffix + '.pdf')
    plt.close()

    # Cleaning up some indexing mess, pointing to features
    features, feature_indices = nested_boolean_indexing(log_curv_maxima, high_outliers)

    # Choose local-fit segments
    curv_zeros = find_zeros(log_curv)
    running_variance = calc_running_local_variance(np.log(y), 2)
    fig, axarray = figure_running_variance(x, y, curv_zeros, running_variance)
    axarray[0].set_xlim([2., 5.])
    axarray[1].set_xlim([2., 5.])
    fig.set_xlim([2., 5.])
    axarray[0].set_xscale('log')
    axarray[0].set_yscale('log')
    axarray[1].set_xscale('log')
    plt.savefig(out_folder + 'running_variance_' + suffix + '.pdf')
    plt.close()
    low_bound_indices, high_bound_indices, no_good_background, extrapolated_background = \
        pick_slope_anchors(running_variance, feature_indices, curv_zeros, 0)
    slope, offset, intensity, sigma = gauss_guess(np.log(x), np.log(y), log_curv, low_bound_indices, high_bound_indices,
                                                  feature_indices)
    fig, ax = figure_naive_gauss_guess(np.log(x), np.log(y), low_bound_indices, high_bound_indices, feature_indices,
                                       slope, offset,
                                       intensity, sigma)
    ax.set_xlim([2., 5.])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plt.savefig(out_folder + 'naive_gauss_guess_' + suffix + '.pdf')
    plt.close()


def saxs_demo_2():  # Pretty much just showing it reads in so far.  More to come.
    out_folder = '/Users/Amanda/PyCharmProjects/peak_detection/saxs_demo_2_figures/'
    # Data intake
    data_folder = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/beamtime/imagesR1/'
    files1 = os.listdir(data_folder)
    files1 = remove_non_csv(files1)
    ###
    files1 = files1[:5]
    ###

    plt.ioff()  # Don't show me batch plots!
    for i in range(len(files1)):
        x, y = read_one_beam_csv(data_folder + files1[i])
        tag = (files1[i])[10:-8] + str(i)
        saxs_demo_do_one(x, y, tag, out_folder)

        # Smooth
        mean_smoothed_19 = np.exp(mean_smoother(np.log(y), 9))
        tag = tag + '_logSMOOTHED'
        saxs_demo_do_one(x, mean_smoothed_19, tag, out_folder)

    plt.close('all')
    plt.ion()  # Interactive plotting back on
    print 'run done'


# Life is change


def monodisperse_model(q, R):
    pass


def smoother(y):
    pass


def binner(y, nbins, npix):
    pass


def convolution_smoother(y, kernel):
    pass


def gauss_smoother(x, y, x0, sigma):
    pass


def edge_preserving_smoothing(y, mysteries):
    pass


def pseudo_voigt(x, x0, gamma, sigma):
    fwhm_gauss = 2 * (2 * np.log(2)) ** 0.5 * sigma
    fwhm_lorentz = 2 * gamma
    # Approximation to the FWHM of the Voigt distribution, accurate to 0.02%, taken from Wikipedia
    fwhm_voigt = 0.5346 * fwhm_lorentz + (0.2166 * fwhm_lorentz ** 2 + fwhm_gauss ** 2) ** 0.5
    # Formula for a good pseudo-Voigt approximation, accurate to 1%, taken from Wikipedia
    # *f* and *eta* are constants used in that approximation
    f = (fwhm_gauss ** 5 + 2.69269 * fwhm_gauss ** 4 * fwhm_lorentz + 2.42843 * fwhm_gauss ** 3 * fwhm_lorentz ** 2
         + 4.47163 * fwhm_gauss ** 2 * fwhm_lorentz ** 3 + 0.07842 * fwhm_gauss * fwhm_lorentz ** 4 + fwhm_lorentz ** 5) ** 0.2
    eta = 1.36603 * (fwhm_lorentz / f) - 0.47719 * (fwhm_lorentz / f) ** 2 + 0.11116 * (fwhm_lorentz / f) ** 3
    gauss_profile = gaussian(x, x0, sigma)
    lorentz_profile = lorentzian(x, x0, gamma)
    pseudo_voigt_profile = eta * lorentz_profile + (1 - eta) * gauss_profile
    return pseudo_voigt_profile, fwhm_voigt


def lorentzian(x, x0, gamma):
    pass


def gaussian(x, x0, sigma):
    pass


def departure_from_linear(y, n):
    pass


def departure_from_model(y, model):
    pass


def collect_metrics(x, y):
    pass


def discriminator():
    data_folder = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Fang/spreadsheets1d/'
    file_list = os.listdir(data_folder)

    #    metrics = np.zeros(stuff, stuffy stuff)
    for ii in file_list:
        print "Reading file %s." % ii
        #        name_string = ii[6:-7]
        # Data intake
        data = np.genfromtxt(data_folder + ii, delimiter=',')
        (length, width) = data.shape  # (1096, 2)
        # The data is for some reason doubled.  Quick 2-line fix.
        length = length / 2
        data = data[0:length, :]
        x = data[:, 0]
        y = data[:, 1]
        #        name_location_string = out_dir + ii[:-7] + '_'
        metrics_ii = collect_metrics(x, y)


# amorphous_labels = np.array([])
#    broad_base_labels = np.array([])


def process_demo_2():
    out_folder = 'process_demo_2_figures/'
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
    fig, ax = figure_initial_maxima(x, y, maxima)
    plt.savefig(out_folder + 'initial_maxima.pdf')
    #    plt.savefig(out_folder + '.pdf')

    # Curvature
    curvature = noiseless_curvature(x, y)
    normed_curv = curvature / (real_max(curvature) - real_min(curvature))
    curvature_legit = ~np.isnan(curvature)
    curv_minima = local_minima_detector(curvature)
    fig, ax = figure_maxima_curvature(x, y, maxima, normed_curv, curvature_legit)
    plt.savefig(out_folder + 'maxima_curvature.pdf')

    # Maxima vs curvature minima
    exclusive_curv_minima = curv_minima & (~maxima)
    exclusive_maxima = maxima & (~curv_minima)
    max_and_curvmin = maxima & curv_minima
    fig, ax = figure_curv_vs_max(x, y, exclusive_maxima, exclusive_curv_minima, max_and_curvmin, normed_curv,
                                 curvature_legit)
    plt.savefig(out_folder + 'curv_vs_max.pdf')
    fig, ax = figure_curv_minima(x, y, curv_minima)
    plt.savefig(out_folder + 'curv_minima.pdf')
    fig, ax = figure_curv_minima_curvature(x, y, curv_minima, normed_curv, curvature_legit)
    plt.savefig(out_folder + 'curv_minima_curvature.pdf')

    # Classifying curvature minima
    normals, high_outliers, low_outliers = isolate_outliers(curvature[curv_minima & curvature_legit], 4)
    print 'Found %i low outliers (features?), %i normals (noise), and %i high outliers (problems?).' % (
        low_outliers.sum(), normals.sum(), high_outliers.sum())
    fig, ax = figure_curv_minima_classified(x, y, curv_minima, high_outliers, normals, low_outliers, normed_curv)
    plt.savefig(out_folder + 'curv_minima_classified.pdf')

    # Curvature zeros
    curv_zeros = find_zeros(curvature)
    fig, ax = figure_curv_zeros(x, y, curv_zeros, normed_curv)
    plt.savefig(out_folder + 'curv_zeros.pdf')

    # Classifying curvature zeros
    running_local_variance = calc_running_local_variance(y, 2)
    mean_variance = running_local_variance.mean()
    median_variance = np.median(running_local_variance)
    print 'The median of the calculated running variance is %f, and the mean is %f.' % (median_variance, mean_variance)
    fig, ax = figure_running_variance(x, y, curv_zeros, running_local_variance)
    plt.savefig(out_folder + 'running_variance.pdf')

    indices = np.arange(y.size, dtype=int)
    curv_minima_indices = indices[curv_minima]
    likely_gaussian_feature_indices = curv_minima_indices[low_outliers]
    likely_gaussian_features = np.zeros(y.size, dtype=bool)
    likely_gaussian_features[likely_gaussian_feature_indices] = True

    likely_gaussian_feature_indices_clipped = likely_gaussian_feature_indices[1:-1]
    likely_gaussian_feature_clipped = np.zeros(y.size, dtype=bool)
    likely_gaussian_feature_clipped[likely_gaussian_feature_indices_clipped] = True

    suggested_low_bound_indices, suggested_high_bound_indices, no_good_background, extrapolated_background \
        = pick_slope_anchors(running_local_variance, likely_gaussian_feature_indices_clipped, curv_zeros, 0)
    fig, ax = figure_slope_anchors_clipped(x, y, suggested_low_bound_indices,
                                           suggested_high_bound_indices, likely_gaussian_feature_indices_clipped)
    plt.savefig(out_folder + 'slope_anchors_clipped.pdf')

    slope, offset, intensity, sigma = gauss_guess(x, y, curvature, suggested_low_bound_indices,
                                                  suggested_high_bound_indices, likely_gaussian_feature_indices_clipped)
    fig, ax = figure_naive_gauss_guess(x, y, suggested_low_bound_indices, suggested_high_bound_indices,
                                       likely_gaussian_feature_indices_clipped, slope, offset, intensity, sigma)
    plt.savefig(out_folder + 'naive_gauss_guess.pdf')



##### Run scripts, optional #####


# process_demo_1()
# batch_demo()
# process_demo_2()
# smoothing_demo_1()
# saxs_demo_1()
#saxs_demo_2()


##### Not-yet-started and/or not-yet-used functions #####


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
