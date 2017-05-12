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
    plt.figure(2)
    plt.title('Column average')
    plt.plot(Qlist[peaks], IntAve[peaks], 'o')
    plt.plot(Qlist, IntAve)
    plt.xlabel('Q')
    plt.ylabel('Intensity')
    #plt.xlim((0.45, 5.87))

    plt.savefig(os.path.join(save_path, imageFilename[:-4]+'_1D'))
    
    plt.close()