"""
author: fangren
"""


from extract_peak_number import extract_peak_num
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os.path
from os.path import basename
from scipy.optimize import curve_fit


def func(x, *params):
    """
    create a Lorentzian fitted curve according to params
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        n = params[i+3]
        y = y + n * amp * np.exp(-4 * np.log(2) * ((x - ctr) / wid) ** 2) + (1 - n) * amp * wid ** 2 / 4 / (
        (x - ctr) ** 2 + wid ** 2 / 4)
    return y



def peak_fitting_GLS(imageFilename, processed_path, Qlist, IntAve, a1, a2):
    Qlist = Qlist[:647]
    IntAve = IntAve[:647]
    peak_num, peaks = extract_peak_num(Qlist, IntAve, a1, a2)
    guess = []
    low = []
    high = []
    #print peaks
    try:
        for peak in peaks:
            guess.append(Qlist[peak])
            low.append(Qlist[peak]-0.1)
            high.append(Qlist[peak]+0.1)

            guess.append(IntAve[peak])
            low.append(0)
            high.append(IntAve[peak]+10)

            guess.append(0.2)
            low.append(0)
            high.append(1)

            guess.append(0.5)
            low.append(0)
            high.append(1)
        popt, pcov = curve_fit(func, Qlist, IntAve, p0=guess, bounds = (low, high))
        #popt, pcov = curve_fit(func, Qlist, IntAve, p0=guess)
        fit = func(Qlist, *popt)
        plt.figure(1)
        plt.plot(Qlist, IntAve)
        plt.plot( Qlist, fit, 'r--')
        plt.plot(Qlist[peaks], IntAve[peaks], 'o')

        for i in range(0, len(popt), 4):
            ctr1 = popt[i]
            amp1 = popt[i+1]
            wid1 = popt[i+2]
            n1 = popt[i+3]
            curve1 = n1 * amp1 * np.exp( -4 * np.log(2) * ((Qlist - ctr1)/wid1)**2) + (1-n1) * amp1 * wid1**2 / 4 / ((Qlist-ctr1)**2 + wid1**2 / 4)
            plt.plot(Qlist, curve1)

        # generate a folder to put peak fitting files
        save_path = os.path.join(processed_path, 'peak_fitting')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(os.path.join(save_path, imageFilename[:-4] + '_peak_fitting_GLS'))
        plt.close()

        popt = np.reshape(popt, (popt.size/4, 4))
        np.savetxt(os.path.join(save_path, imageFilename[:-4] + '_peak_fitting_GLS.csv'), popt, delimiter=",")
    except RuntimeError:
        print "Failed to fit", imageFilename
        print "used the previous peak information"
        np.savetxt(os.path.join(save_path, imageFilename[:-4] + '_peak_fitting_GLS.csv'), popt, delimiter=",")
