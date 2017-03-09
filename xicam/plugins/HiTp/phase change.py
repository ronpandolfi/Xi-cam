"""
Created on Feb 7 2017

@author: Fang Ren
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from os.path import basename
from scipy.optimize import curve_fit
import imp


def func(x, *params):
    """
    create a Lorentzian fitted curve according to params
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        print i, params
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

Q = np.array(range(0, 590))
Q = Q * 0.01

params1 = [1, 100, 0.1, 2, 150, 0.08, 4, 120, 0.05]

params2 = [1.5, 100, 0.1, 3.5, 150, 0.1]

params3 = [1, 50, 0.1, 2, 75, 0.08, 4, 60, 0.05, 1.5, 50, 0.1, 3.5, 75, 0.1]


spectrum1 = func(Q, *params1)
spectrum2 = func(Q, *params2)
spectrum3 = func(Q, *params3)

plt.figure(1, (7, 5))
plt.subplot(311)
plt.plot(Q, spectrum1, label = 'phase A')
plt.legend()
plt.subplot(312)
plt.plot(Q, spectrum2, label = 'phase B')
plt.ylabel('Intensity')
plt.legend()
plt.subplot(313)
plt.plot(Q, spectrum3, label = 'A to B')
plt.xlabel('Q')
plt.legend()