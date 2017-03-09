# -*- coding: utf-8 -*-
"""
Created on Nov 17 2016

@author: Fang Ren
"""

from scipy.signal import cwt, ricker, find_peaks_cwt
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from os.path import basename
from scipy import ndimage


path = 'C:\\Research_FangRen\\Data\\July2016\\Sample1\\Processed_old\\'
file = path + 'Sample1_24x24_t30_0001_1D.csv'

data = np.genfromtxt(file, delimiter = ',')
Qlist = data[:,0]
IntAve = data[:,1]

a1 = 1
a2 = 30
widths = np.arange(a1, a2)
cwt_coefficient = cwt(IntAve, ricker, widths)
peaks = find_peaks_cwt(IntAve, np.arange(a1, a2, 0.05))
peaks = peaks[1:-1]

h = 15 # number of points skipped in finite differences

peaks_accepted=[]
window = h

for peak in peaks:

    filter = np.nan_to_num(np.sqrt(-(IntAve[2*h:]-2*IntAve[h:-h]+IntAve[0:-2*h])))
    filterwindow = filter[max(peak-h - window, 0):min(peak-h + window, len(filter))]
    spectrawindow = IntAve[max(peak - window, h):min(peak + window, len(filter))]

    try:
        if np.any(filterwindow>spectrawindow/200): # np.percentile(filter,85) is also a good threshold
            peaks_accepted.append(peak)
    except ValueError:
        continue


plt.figure(1)
plt.subplot((311))
plt.pcolormesh(Qlist, widths, cwt_coefficient)
plt.plot(Qlist, [a1]* len(Qlist), 'r--')
plt.plot(Qlist, [a2]* len(Qlist), 'r--')
plt.xlim(0.65, 6.45)
plt.ylim(a1, a2)
# plt.clim(np.nanmin(np.log(cwt_coefficient)), np.nanmax(np.log(cwt_coefficient)))

plt.subplot((312))
plt.plot(Qlist[peaks_accepted], IntAve[peaks_accepted], linestyle = 'None', c = 'r', marker = 'o', markersize = 10)
plt.plot(Qlist[peaks], IntAve[peaks], linestyle = 'None', c = 'b', marker = 'o', markersize = 3)
plt.plot(Qlist, IntAve)
plt.xlim(0.65, 6.45)

plt.subplot((313))
plt.plot(Qlist[15:-15], filter)
plt.xlim(0.65, 6.45)
