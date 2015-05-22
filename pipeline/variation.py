import numpy as np
import loader
import scipy.ndimage


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


def filevariation(operationindex, filea, fileb, filec):
    p, _ = loader.loadpath(filea)
    c, _ = loader.loadpath(fileb)
    n, _ = loader.loadpath(filec)

    if p is not None and c is not None and n is not None:
        p = scipy.ndimage.zoom(p, 0.1, order=1)
        c = scipy.ndimage.zoom(c, 0.1, order=1)
        n = scipy.ndimage.zoom(n, 0.1, order=1)
        return variation(operationindex, p, c, n)
    else:
        print('Variation could not be determined for a frame.')
        return None

def variation(operationindex, imga, imgb=None, imgc=None):
    try:

        return operations[operationindex](imga, imgb, imgc)
    except TypeError:
        print('Variation could not be determined for a frame.')