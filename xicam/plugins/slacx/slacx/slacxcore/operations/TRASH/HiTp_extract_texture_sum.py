# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:18:45 2016

@author: Fang Ren
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

def save_texture_plot_csv(Q, chi, cake, imageFilename, save_path):
    Q, chi = np.meshgrid(Q, chi)
    plt.figure(3)
    plt.title('texture')
    
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
    
    plt.plot(Qlist_texture, texture)
    plt.xlabel('Q')
    plt.ylabel('Texture')
    plt.xlim((0.7, 6.4))
    plt.savefig(save_path + imageFilename[:-4] + '_texture')
    plt.close()
    
    rows = zip(Qlist_texture, texture)
    with open(save_path + imageFilename[:-4]+'_texture.csv', 'a') as csvoutput:
        writer = csv.writer(csvoutput, delimiter = ',', lineterminator = '\n')
        for row in rows:
            writer.writerow(row)
        csvoutput.close()

    return Qlist_texture, texture