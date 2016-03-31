from scipy.optimize import curve_fit
from scipy.optimize import basinhopping
import numpy as np
import matplotlib.pyplot as plt

import csv

f = open('data.txt')
csv_f = csv.reader(f)
p = 1

dx = np.empty([158])
dy = np.empty([158])
p = 0

for row in csv_f:
    dx[p] = float(row[0])
    dy[p] = float(row[1])
    p = p + 1

dx = dx[5:140]
dy = dy[5:140]
dy = dy - dy.min()
dy = dy / dy[-1]


def nexafs_f(args):
    ctr,amp,wid=args
    print ctr,amp,wid
    y = np.zeros_like(dx)
    #ctr = params[0]
    #amp = params[1]
    #wid = params[2]
    y = y + amp * (np.tanh(wid * dx - ctr) + 1.0)
    #for i in range(3, len(params), 3):
    #    ctr = params[i]
    #    amp = params[i + 1]
    #    wid = params[i + 2]
    y = y + amp * np.exp(-((dx - ctr) / wid) ** 2)
    return np.sum(y**2-dy**2)


def nexafs(params, *dxy):
    dx = dxy[0]
    dy = dxy[1]
    y = np.zeros_like(dx)
    ctr = params[0]
    amp = params[1]
    wid = params[2]
    y = y + amp * (np.tanh(wid * dx - ctr) + 1.0)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i + 1]
        wid = params[i + 2]
        y = y + amp * np.exp(-((dx - ctr) / wid) ** 2)
    di = np.sum((y - dy) ** 2)
    return di


dxy = np.array([dx, dy])
x0 = ((285.0, 0.5, 1.0))#, 290.0, 1.0, 10.0, 287.0, 0.5, 1.0, 320.0, 1.0, 10.0))
xl = ((280.0, 0.01, 0.1, 280.0, 0.01, 1.0, 280.0, 0.01, 0.1, 280.0, 0.01, 1.0))
xu = ((320.0, 10.0, 20.0, 320.0, 10.0, 20.0, 320.0, 10.0, 20.0, 320.0, 10.0, 20.0))
res = basinhopping(nexafs_f, x0, T=.2)
print(res)

#final = res[0]
# fit = nexafs_f(dx, *final)
# plt.close("all")
# plt.plot(dx, dy)
# plt.plot(dx, fit, 'r-')
# plt.show()
