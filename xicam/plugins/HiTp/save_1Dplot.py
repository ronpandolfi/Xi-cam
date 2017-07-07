"""
Created on Mar 2017

@author: fangren
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path


def save_1Dplot(Qlist, IntAve, peaks, imageFilename, save_path):
    # generate a column average image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Column average')
    ax.plot(Qlist, IntAve)
    ax.set_xlabel('Q')
    ax.set_ylabel('Intensity')
    ax.set_xlim((0.7, 6.4))
    fig.savefig(os.path.join(save_path, imageFilename[:-4]+'_1D'))