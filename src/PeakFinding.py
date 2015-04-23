__author__ = 'remi'

from pylab import *
from scipy import signal

def PeakFinding(x,y):

    peakind = signal.find_peaks_cwt(y, np.arange(0.01,100))
    list_max_abscisse=(x[peakind])
    list_max_ordonnee=y[peakind]
    PeaksPosition=np.zeros((size(list_max_ordonnee),2))

    for i in range (0,size(list_max_ordonnee)):
        PeaksPosition[i,0]=list_max_abscisse[i]
        PeaksPosition[i,1]=list_max_ordonnee[i]


    print PeaksPosition

    return PeaksPosition