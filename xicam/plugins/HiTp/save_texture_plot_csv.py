"""
Created on Mar 2017

@author: fangren
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path

def save_texture_plot_csv(Q, chi, cake, imageFilename, save_path):
    Q, chi = np.meshgrid(Q, chi)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('texture')
    
    keep = (cake != 0)
    chi = chi*np.pi/180
    
    cake *= keep.astype(np.int)
    #chi *= keep.astype(np.int)    
    IntSum = np.bincount((Q.ravel()*100).astype(int), cake.ravel().astype(int))
    count = np.bincount((Q.ravel()*100).astype(int), keep.ravel().astype(int))
    IntAve = list(np.array(IntSum)/np.array(count))
    
    textureSum = np.bincount((Q.ravel()*100).astype(int), (cake*np.cos(chi)).ravel())
    chiCount = np.bincount((Q.ravel()*100).astype(int), (keep*np.cos(chi)).ravel())
    
    texture = list(np.array(textureSum)/np.array(IntAve)/np.array(chiCount)-1)
    
    step = 0.01
    Qlen = int(np.max(Q)/step+1)
    Qlist_texture = [i*step for i in range(Qlen)]
    
    ax.plot(Qlist_texture, texture)
    ax.set_xlabel('Q')
    ax.set_ylabel('Texture')
    #plt.xlim((0.7, 6.4))
    fig.savefig(os.path.join(save_path, imageFilename[:-4] + '_texture'))
    plt.close()
    data = np.concatenate(([Qlist_texture], [texture]))
    np.savetxt(os.path.join(save_path, imageFilename[:-4]+'_texture.csv'), data.T, delimiter=',')
    del data
    return Qlist_texture, texture