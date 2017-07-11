"""
Created on Mar 2017

@author: fangren
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os.path
import scipy.io

def save_Qchi(Q, chi, cake, imageFilename, save_path):
    scipy.io.savemat(os.path.join(save_path, imageFilename[:-4]+'_Qchi.mat'), {'Q':Q, 'chi':chi, 'cake':cake})
    Q, chi = np.meshgrid(Q, chi)
    plt.figure(1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Q-chi polarization corrected_log scale')
    meshplot = ax.pcolormesh(Q, chi, np.log(cake), cmap = 'viridis')
    ax.set_xlabel('Q')
    ax.set_ylabel('$\gamma$')
    ax.set_xlim((0.7, 6.8))
    ax.set_ylim((-56, 56))
    # the next two lines are contributed by S. Suram (JCAP)
    inds = np.nonzero(cake)
    meshplot.set_clim(scipy.stats.scoreatpercentile(np.log(cake[inds]), 5),
             scipy.stats.scoreatpercentile(np.log(cake[inds]), 95))
    fig.colorbar(meshplot)
    fig.savefig(os.path.join(save_path, imageFilename[:-4]+'_Qchi'))


