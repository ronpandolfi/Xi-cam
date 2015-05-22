import numpy as np


def chisquared(p, c, n):
    return np.sum(np.square(c - p) / p)


def absdiff(p, c, n):
    return np.sum(np.abs(c - p))


def normabsdiff(p, c, n):
    return np.sum(np.abs(c - p) / p)


def sumintensity(p, c, n):
    return np.sub(c)


def normabsdiffderiv(p, c, n):
    return -np.sum(np.abs(n - c) / c) + np.sum(np.abs(c - p) / c)


operations = [chisquared, absdiff, normabsdiff, sumintensity, normabsdiffderiv]


def variation(operationindex, imga, imgb=None, imgc=None):
    try:

        return operations[operationindex](imga, imgb, imgc)
    except TypeError:
        print('Variation could not be determined for a frame.')