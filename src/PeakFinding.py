__author__ = 'remi'

from pylab import *
from scipy import signal
from scipy.ndimage import filters

maxfiltercoef = 5
cwtrange = np.arange(3, 100)
gaussiansigma = 2


def findpeaks(x, y):
    cwtdata = filters.gaussian_filter(signal.cwt(y, signal.ricker, cwtrange), gaussiansigma)
    maxima = (cwtdata == filters.maximum_filter(cwtdata, 5))
    maximaloc = np.where(maxima == 1)
    x = np.array(x)
    y = np.array(y)

    return list(np.array(np.vstack([x[maximaloc[1]], y[maximaloc[1]], maximaloc])))