import collections
import numpy as np

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
    current = data[t]
    previous = data[t - 1]
    return np.sum(roi * np.square(current.astype(float) - previous))


def absdiff(data, t, roi):
    current = data[t]
    previous = data[t - 1]
    return np.sum(roi * np.abs(current - previous))


def normabsdiff(data, t, roi):
    current = data[t]
    previous = data[t - 1]
    return np.sum(roi * np.abs(current - previous) / previous)


def sumintensity(data, t, roi):
    current = data[t]
    return np.sum(roi * current)


def normabsdiffderiv(data, t, roi):
    current = data[t]
    previous = data[t - 1]
    next = data[t + 1]
    return -np.sum(roi * (np.abs(next - current) / current) + np.sum(np.abs(current - previous) / current))


def chisquaredwithfirst(data, t, roi):
    current = data[t]
    first = data[0]
    return np.sum(roi * np.square(current.astype(float) - first))


operations = collections.OrderedDict([('Chi Squared', chisquared),
                                      ('Absolute Diff.', absdiff),
                                      ('Norm. Abs. Diff.', normabsdiff),
                                      ('Sum Intensity', sumintensity),
                                      ('Norm. Abs. Derivative', normabsdiffderiv),
                                      ('Chi Squared w/First Frame', chisquaredwithfirst)])