"""
Created on Mar 2017

@author: fangren
comtributor: S. Suram (JCAP)
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
    plt.title('Q-chi polarization corrected_log scale')
    plt.pcolormesh(Q, chi, np.log(cake), cmap = 'viridis')
    plt.xlabel('Q')
    plt.ylabel('$\gamma$')
    #plt.xlim((0.7, 6.8))
    #plt.ylim((-56, 56))
    plt.clim((0, np.log(np.nanmax(cake))))
    # the next two lines are contributed by S. Suram (JCAP)
    inds = np.nonzero(cake)
    plt.clim(scipy.stats.scoreatpercentile(np.log(cake[inds]), 5),
             scipy.stats.scoreatpercentile(np.log(cake[inds]), 95))
    plt.colorbar()
    plt.savefig(os.path.join(save_path, imageFilename[:-4]+'_Qchi'))

    
    plt.close()

