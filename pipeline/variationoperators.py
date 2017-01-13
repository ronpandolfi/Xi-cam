import collections
import numpy as np
import loader
from scipy import signal
import integration
import writer
#from skimage.measure import block_reduce  # Use this to subsample if you want

# README!
#
# To add a variation operation,
# 1. Your function must adhere to the signature below:
# (data, t, roi)
#    Where data is a list of ndarray thumbnails (5x), one for each frame,
#    t is the frame index to evaluate at, and roi is the region of interest mask.
# 2. Add your function to the 'operations' dictionary at the end of this module. The key is the display name.


# Operation signature:
# If calculating variation between only two consecutive frames, use current and previous
# ROI-mask is accept=1

def chisquared(data, t, roi):
    current = data[t].astype(float)
    previous = data[t - 1].astype(float)
    return np.sum(roi * np.square(current - previous))


def absdiff(data, t, roi):
    current = data[t].astype(float)
    previous = data[t - 1].astype(float)
    return np.sum(roi * np.abs(current - previous))


def normabsdiff(data, t, roi):
    current = data[t].astype(float)
    previous = data[t - 1].astype(float)
    return np.sum(roi * np.abs(current - previous) / previous)


def sumintensity(data, t, roi):
    current = data[t].astype(float)
    return np.sum(roi * current)


def normabsdiffderiv(data, t, roi):
    current = data[t].astype(float)
    previous = data[t - 1].astype(float)
    next = data[t + 1].astype(float)
    return -np.sum(roi * (np.abs(next - current) / current) + np.sum(np.abs(current - previous) / current))


def chisquaredwithfirst(data, t, roi):
    current = data[t].astype(float)
    first = data[0].astype(float)
    return np.sum(roi * np.square(current.astype(float) - first))


def radialintegration(data, t, roi):
    current = data[t]
    return integration.radialintegratepyFAI(current, cut=roi)[:2]


def angularcorrelationwithfirst(data, t, roi):
    # ROI is assumed to be in cake mode

    experiment.center = (experiment.center[0] / 5, experiment.center[1] / 5)

    currentcake, _, _ = integration.cake(data[t], experiment)
    firstcake, _, _ = integration.cake(data[0], experiment)
    # cakeroi, _, _ = integration.cake(np.ones_like(data), experiment)

    currentchi = np.sum(currentcake * roi, axis=0)
    firstchi = np.sum(firstcake * roi, axis=0)

    return signal.convolve(currentchi, firstchi)




operations = collections.OrderedDict([('Chi Squared', chisquared),
                                      ('Absolute Diff.', absdiff),
                                      ('Norm. Abs. Diff.', normabsdiff),
                                      ('Sum Intensity', sumintensity),
                                      ('Norm. Abs. Derivative', normabsdiffderiv),
                                      ('Chi Squared w/First Frame', chisquaredwithfirst),
                                      ('Angular autocorrelation w/First Frame', angularcorrelationwithfirst),
                                      ('Radial Integration', radialintegration)
])


experiment = None