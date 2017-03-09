"""
Created on Feb 7 2017

@author: Fang Ren
"""
import numpy as np
import matplotlib.pyplot as plt

path = 'C:\\Research_FangRen\\publications\\on_the_fly_paper\\Sample_data\\'
file1 = path + 'amorphous.csv'
file2 = path + 'crystalline.csv'

data1 = np.genfromtxt(file1, delimiter = ',')
data2 = np.genfromtxt(file2, delimiter = ',')

plt.figure(1, (5,4))
plt.plot(data1[:,0], data1[:,1], label = 'amorphous')
plt.plot(data2[:,0], data2[:,1], label = 'crystalline')
plt.xlim(0.64, 5.9)
plt.ylim(500, 7000)
plt.legend()
plt.xlabel('Q')
plt.ylabel('Intensity')
plt.savefig(path+'amorphous and crystalline', dpi = 600)