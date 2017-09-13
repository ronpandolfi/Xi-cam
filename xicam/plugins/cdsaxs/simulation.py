#! /usr/bin/env python

import numpy as np

def trapezoid_form_factor(qy, qz, y1, y2, langle, rangle, h):
    """ Simulation of the form factor of a trapezoid at qx, qz position

    Parameters
    ----------
    qy, qz (list of float): List of qx/qz at which the form factor is simulated
    y1, y2 (float): the value of the bottom right/left (y1/y2) position of the trapezoid => y2 - y1 = width of the bottom of the trapezoid
    langle, rangle (list of float):
    h (float): Height of the trapezoid

    Returns
    -------
    ff (list of float): list of the value of the form factor
    """
    m1 = np.tan(langle)
    m2 = np.tan(np.pi - rangle)
    t1 = qy + m1 * qz
    t2 = qy + m2 * qz
    with np.errstate(divide='ignore'):
        t3 = m1 * np.exp(-1j * qy * y1) * (1 - np.exp(-1j * h / m1 * t1)) / t1
        t4 = m2 * np.exp(-1j * qy * y2) * (1 - np.exp(-1j * h / m2 * t2)) / t2
        ff = (t4 - t3) / qy
    return ff

def stacked_trapezoids(qy, qz, y1, y2, height, langle, rangle=None):
    """ Simulation of the form factor of a trapezoid at qx, qz position

    Parameters
    ----------
    qy, qz (list of float): List of qx/qz at which the form factor is simulated
    y1, y2 (float): the value of the bottom right/left (y1/y2) position of the trapezoid => y2 - y1 = width of the bottom of the trapezoid
    height (float): Height of the trapezoid
    langle, rangle (list of float):

    Returns
    -------
    np.absolute(ff) ** 2 (list of float): Intensity of the form factor
    """
    if not isinstance(langle, np.ndarray):
        raise TypeError('anlges should be array')
    if rangle is not None:
        if not langle.size == rangle.size:
            raise ValueError('both angle array are not of same size')
    else:
        rangle = langle

    ff = np.zeros(qz.shape, dtype=np.complex)
    # loop over all the angles
    for i in range(langle.size):
        shift = height * i
        left, right = langle[i], rangle[i]
        ff += trapezoid_form_factor(qy, qz, y1, y2, left, right, height) * np.exp(-1j * shift * qz)
        m1 = np.tan(left)
        m2 = np.tan(np.pi - right)
        y1 += height / m1
        y2 += height / m2

    return np.absolute(ff) ** 2

#For RSOXS
'''
def stacked_trapezoids_shell(qy, qz, y1, y2, height, langle, rangle=None, y_off=None, h_off=None):
    if y_off is None and h_off in None:
        raise ValueError('say something meaningful')
    if rangle is not None:
        if not langle.size == rangle.size:
            raise ValueError('both angle array are not of same size')
    else:
        rangle = langle

    if not isinstance(h_off, float):
        raise TypeError('h_off mush be a scalar')

    if isinstance(y_off, np.ndarray):
        if len(y_off.shape == 1):
            if not y_off.shape[0] == langle.size:
                raise ValueError('y_off should be same size as number of side wall angles')
            dy = np.zeros((2, y_off.shape[0]))
            dy[:,:] = y_off
        elif len(y_off.shape) == 2:
            if y_off.shape[0] == 2:
                if not y_off.shape[1] == langle.size:
                    raise ValueError('y_off should be same size as number of side wall angles')
                dy = y_off
            elif y_off.shape[1] == 2:
                if not y_off.shape[0] == langle.size:
                    raise ValueError('y_off should be same size as number of side wall angles')
                dy = y_off.T
            else:
                raise ValueError('unsupported shape of y_off')
        else:
            raise ValueError('unsupported shape of y_off')
    elif isinstance(y_off, float):
        dy = np.empty((2, langle.shape[0]))
        dy.fill(y_off)
    else:
        raise TypeError('y_off can be either a numpy.ndarray or a float')

    ff_core = stacked_trapezoids(qy, qz, y1, y2, height, langle, rangle)
    ff = np.zeros(qz.shape, dtype=np.complex)
    for i in range(langle.size):
        shift = height * i
        left, right = langle[i], rangle[i]
        y1 -= dy[0, i]
        y2 += dy[1, i]
        if i == langle.size - 1: height += h_off
        ff += trapezoid_form_factor(qy, qz, y1, y2, left, right, height) * np.exp(-1j * shift * qz)
        m1 = np.tan(left)
        m2 = np.tan(np.pi - right)
        y1 += height / m1
        y2 += height / m2

    ff -= ff_core
    return ff

def multipyramid(h, w, a, nx, ny):
    if nx % 2 == 1:
        nx += 1

    n2 = nx / 2
    x0 = w / 2
    y0 = 0

    if not type(a) is np.ndarray:
        raise TypeError('Side-wall angle must be numpy array for multipyramid')

    # setup output array
    img = np.zeros((ny, n2))
    y, x = np.mgrid[0:ny, 0:n2]

    a = np.deg2rad(a)
    for i in range(a.size):
        A = np.sin(np.pi - a[i])
        B = -np.cos(np.pi - a[i])
        C = -(A * x0 + B * y0)
        d = A * x + B * y + C

        # update (x0, y0)
        y0 = (i + 1) * h
        x0 = -(B * y0 + C) / A

        # update image
        mask = np.logical_and(y >= i * h, y < (i + 1) * h)
        mask = np.logical_and(d < 0, mask)
        img[mask] = 1

    return np.hstack((np.fliplr(img), img))
'''