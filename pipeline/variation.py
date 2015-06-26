import numpy as np
import loader
import scipy.ndimage
import warnings


def chisquared(p, c, n):
    return np.sum(np.square(c - p) / p)


def absdiff(p, c, n):
    return np.sum(np.abs(c - p))


def normabsdiff(p, c, n):
    return np.sum(np.abs(c - p) / p)


def sumintensity(p, c, n):
    return np.sum(c)


def normabsdiffderiv(p, c, n):
    return -np.sum(np.abs(n - c) / c) + np.sum(np.abs(c - p) / c)


operations = [chisquared, absdiff, normabsdiff, sumintensity, normabsdiffderiv]


def filevariation(operationindex, filea, c, filec):
    p, _ = loader.loadpath(filea)
    # c, _ = loader.loadpath(fileb)
    n, _ = loader.loadpath(filec)
    return variation(operationindex, p, c, n)


def variation(operationindex, imga, imgb=None, imgc=None):
    if imga is not None and imgb is not None and imgc is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = scipy.ndimage.zoom(imga, 0.1, order=1)
                c = scipy.ndimage.zoom(imgb, 0.1, order=1)
                n = scipy.ndimage.zoom(imgc, 0.1, order=1)
                p = scipy.ndimage.gaussian_filter(p, 3)
                c = scipy.ndimage.gaussian_filter(c, 3)
                n = scipy.ndimage.gaussian_filter(n, 3)
            with np.errstate(divide='ignore'):
                return operations[operationindex](p, c, n)
        except TypeError:
            print('Variation could not be determined for a frame.')
    else:
        print('Variation could not be determined for a frame.')
        return None