# -*- coding: utf-8 -*-
"""
Created on Wed July 13 2016

@author: fangren
"""


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
from os.path import basename
import imp


path = 'C:\\Research_FangRen\\Publications\\on_the_fly_paper\\Sample_data\\'

def twoD_visualize(path):
    """
    create three lists for plotting: plate_x, plate_y, ROI1, ROI2, ROI3...
    """
    for filename in glob.glob(os.path.join(path, '*.csv')):
        if basename(filename)[0] == '1':
            print basename(filename)
            data = np.genfromtxt(filename, delimiter=',', skip_header = 1)
            plate_x = data[:,1]
            plate_y = data[:,2]
            ROI1 = data[:,15]
            ROI2 = data[:,16]
            ROI3 = data[:,17]
            ROI5 = data[:,19]
            crystallinity = data[:,51]
            texture = data[:,53]
            metal1 = data[:,54]
            metal2 = data[:,55]
            metal3 = data[:,56]
            peak_num = data[:,55]
            neighbor_distance = data[:,57]
            SNR = data[:,58]
#    return plate_x, plate_y, ROI1, ROI2, ROI3, ROI5, crystallinity, texture, metal1, metal2, metal3, peak_position, peak_width, peak_intensity
    return plate_x, plate_y, ROI1, ROI2, ROI3, ROI5, crystallinity, texture, metal1, metal2, metal3, peak_num, neighbor_distance, SNR
    # return plate_x, plate_y, ROI1, ROI2, ROI3, ROI5
           

plate_x, plate_y, ROI1, ROI2, ROI3, ROI5, crystallinity, texture, metal1, metal2, metal3, peak_num, neighbor_distance, SNR = twoD_visualize(path)
# plate_x, plate_y, ROI1, ROI2, ROI3, ROI5= twoD_visualize(path)

area = [125]

#
# plt.figure(5, figsize = (6, 4.5))
# # plt.title('crystallinity analysis')
# plt.scatter(plate_y, plate_x, c = np.log(crystallinity), s = area, marker = 's')
# plt.xlim((-36, 36))
# plt.ylim((-36, 36))
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.clim((0.2, 1.4))
# plt.savefig(path+'crystallinity analysis', dpi = 600)

# plt.figure(6, figsize = (6, 4.5))
# # plt.title('texture analysis')
# plt.scatter(plate_y, plate_x, c = np.log(texture), s = area, marker = 's')
# plt.xlim((-36, 36))
# plt.ylim((-36, 36))
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.clim((-11.1, -10.3))
# plt.savefig(path+'texture analysis', dpi = 600)


plt.figure(5, figsize = (6, 4.5))
plt.scatter(plate_y, plate_x, c = np.log(neighbor_distance), s = area, marker = 's')
plt.xlim((-36, 36))
plt.ylim((-36, 36))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.clim(-10.4, -5.6)
plt.savefig(path+'neighbor_distance', dpi = 600)


plt.figure(6, figsize = (6, 4.5))
plt.scatter(plate_y, plate_x, c = peak_num, s = area, marker = 's')
plt.xlim((-36, 36))
plt.ylim((-36, 36))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.clim(1, 8)
plt.savefig(path+'peak_num', dpi = 600)
#
#

plt.figure(7, figsize = (6, 4.5))
plt.scatter(plate_y, plate_x, c = 10*np.log(SNR), s = area, marker = 's')
plt.xlim((-36, 36))
plt.ylim((-36, 36))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.clim(150, 156)
plt.savefig(path+'SNR_2', dpi = 600)

