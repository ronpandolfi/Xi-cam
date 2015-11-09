import collections
import numpy as np

# README!
#
# To add a variation operation, define the function using the below signature, then add it the 'operations'
# dictionary at the end of this module. The key is the display name.


# Operation signature: previous, current, next, ROI-mask, first, last
# If calculating variation between only two consecutive frames, use current and previous
# ROI-mask is accept=1

def chisquared(p, c, n, r, f, l):
    return np.sum(r * np.square(c.astype(float) - p))


def absdiff(p, c, n, r, f, l):
    return np.sum(r * np.abs(c - p))


def normabsdiff(p, c, n, r, f, l):
    return np.sum(r * np.abs(c - p) / p)


def sumintensity(p, c, n, r, f, l):
    return np.sum(r * c)


def normabsdiffderiv(p, c, n, r, f, l):
    return -np.sum(r * (np.abs(n - c) / c) + np.sum(np.abs(c - p) / c))


def chisquaredwithfirst(p, c, n, r, f, l):
    return chisquared(f, c, n, r, f, l)


operations = collections.OrderedDict([('Chi Squared', chisquared),
                                      ('Absolute Diff.', absdiff),
                                      ('Norm. Abs. Diff.', normabsdiff),
                                      ('Sum Intensity', sumintensity),
                                      ('Norm. Abs. Derivative', normabsdiffderiv),
                                      ('Chi Squared w/First Frame', chisquaredwithfirst)])