import numpy as np


def chisquared(c, p):
    return np.sum(np.square(c - p) / p)


def absdiff(c, p):
    return np.sum(np.abs(c - p))


def normabsdiff(c, p):
    return np.sum(np.abs(c - p) / p)


def sumintensity(c, p):
    return np.sub(c)


operations = [chisquared, absdiff, normabsdiff, sumintensity]


def variation(operation, imga, imgb=None):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return operation(imga, imgb)
    except TypeError:
        print('Variation could not be determined for a frame.')