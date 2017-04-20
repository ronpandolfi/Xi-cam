#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def trapezoid_form_factor(qy, qz, y1, y2, langle, rangle, h):
    m1 = np.tan(langle)
    m2 = np.tan(np.pi - rangle)
    t1 = qy + m1 * qz
    t2 = qy + m2 * qz
    with np.errstate(divide='ignore'):
        t3 = m1 * np.exp(-1j * qy * y1) * (1 - np.exp(-1j * h / m1 * t1)) / t1
        t4 = m2 * np.exp(-1j * qy * y2) * (1 - np.exp(-1j * h / m2 * t2)) / t2
        ff = (t4 - t3)/qy
    return ff 
    

def stacked_trapezoids(qy, qz, y1, y2, height, langle, rangle=None):
    if not isinstance(langle, np.ndarray):
        raise TypeError('anlges should be array')
    if rangle is not None:
        if not langle.size == rangle.size:
            raise ValueError('both angle array are not of same size')
    else:
        rangle = langle

    ff = np.zeros(qy.shape, dtype=np.complex)
    # loop over all the angles
    for i in range(langle.size):
        shift = height * i
        left, right = langle[i], rangle[i]
        ff += trapezoid_form_factor(qy, qz, y1, y2, left, right, height) * np.exp(-1j * shift * qz)
        m1 = np.tan(left)
        m2 = np.tan(np.pi - right)
        y1 += height / m1
        y2 += height / m2 
        
    return np.absolute(ff)**2

if __name__ == '__main__':
    qy = [np.linspace(-1, 1, 128), np.linspace(-1, 1, 100)]
    qz = [0., 0.5]
    #qy, qz = np.meshgrid(qy, qz)
    langle = np.deg2rad(np.repeat(80, 5))
    rangle = np.deg2rad(np.repeat(80, 5))
    ff = []
    for i in range(len(qz)):
        ff.append(stacked_trapezoids(qy[i], qz[i], -25, 25, 10, langle, rangle))
        plt.plot(ff[i])
    plt.show() 
