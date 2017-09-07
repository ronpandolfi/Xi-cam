#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class FitDWBA:
    def __init__ (self, qp, qz, wavelen, delta, beta):
        self.qp = qp
        self.qz = qz
        self.k0 = 2. * np.pi / wavelen
        self.alpha = None
        self.delta = delta
        self.beta = beta

    def calc_alpha(self, alphai):
        if self.alpha is None:
            self.alpha = np.arcsin(self.qz/self.k0 - np.sin(alphai))

    def distortQz(self, alphai):
        ki = -self.k0 * np.sin(alphai)
        kf = self.qz + ki
        return (kf-ki, kf+ki, -kf-ki, -kf+ki)

    def calcFresnelCoeffs(self, alpha, alphai):
        multilayer = MultiLayer()
        fc = multilayer.propagation_coeffs(alphai, alpha, self.k0, 0)
        return fc

    def calcIntensity(self, params, alphai):
        self.calc_alpha(alphai)
        qz = self.distortQz(alphai)
        fc = self.calcFresnelCoeffs(self.alpha, alphai)
        ff = np.zeros(self.qz.shape, dtype=np.complex_)

        # shape parameters
        y1 = params['left_coord']
        y2 = params['right_coord']
        dh = params['delta_h']
        la = params['langle']
        for i in range(4):
            ff +=  fc[:,i] * stacked_trapezoids(self.qp, qz[i], y1, y2, dh, la)
        return ff

if __name__ == '__main__':

    # parameters
    params = { 'left_coord':-25, 'right_coord':25, 'delta_h':10 }
    params['langle'] = np.deg2rad(np.repeat(80, 5))
    alphai = np.deg2rad(0.3)
    delta = 4.88E-06
    beta = 7.37E-08
    qp = 0.125
    qz = np.linspace(0, 2, 600)
    wavelen = 0.123984

    fit = FitDWBA(qp, qz, wavelen, delta, beta)
    intensity = fit.calcIntensity(params, alphai)
    plt.semilogy(qz, np.abs(intensity)**2)
    plt.show()


#Formfacto.py
def trapezoid_form_factor(qy, qz, y1, y2, langle, rangle, h):
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
    return ff


#Reflectivity.py
class Layer:
    def __init__(self, delta, beta, order, thickness):
        self.one_n2 = 2 * np.complex(delta, beta)
        self.order = order
        self.thickness = thickness
        self.zval = 0

class MultiLayer:
    def __init__(self):
        self.layers = [Layer(0, 0, 0, 0)]
        self.substrate = Layer(4.88E-6, 7.37E-08, -1, 0)
        self._setup_ = False

    def insert(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError('only Layer types can be inserted into multilayered object')
            exit(101)
        if not layer.order > 0:
            raise ValueError('the order of layer must be greater than 0');
            exit(102)
        self.layer.insert(layer.order, layer)

    def setup_multilayer(self):
        if self._setup_ : return

        # put substrate at the end
        self.layers.append(self.substrate)

        # calc z of every interface
        nlayer = len(self.layers)
        for i in range(nlayer-2, -1, -1):
            self.layers[i].zval = self.layers[i+1].thickness

        # run only once
        self._setup_ = True

    def parratt_recursion(self, alpha, k0, order):
        self.setup_multilayer()
        nlayer = len(self.layers)

        # sin(alpha)
        sin_a = np.sin(alpha)

        # initialize
        R = np.zeros(nlayer, np.complex_)
        T = np.zeros(nlayer, np.complex_)
        T[-1] = np.complex(1, 0)

        # cacl k-value
        kz = np.zeros(nlayer, np.complex_)
        for i in range(nlayer):
            kz[i] =  -k0 * np.sqrt(sin_a**2 - self.layers[i].one_n2)

        # iterate from bottom to top
        for i in range(nlayer-2, -1, -1):
            pij = (kz[i] + kz[i+1])/(2.* kz[i])
            mij = (kz[i] - kz[i+1])/(2.* kz[i])
            z = self.layers[i].zval
            exp_p = np.exp(-1j*(kz[i+1]+kz[i])*z)
            exp_m = np.exp(-1j*(kz[i+1]-kz[i])*z)
            a00 = pij * exp_m
            a01 = mij * np.conjugate(exp_p)
            a10 = mij * exp_p
            a11 = pij * np.conjugate(exp_m)
            T[i] = a00 * T[i+1] + a01 * R[i+1]
            R[i] = a10 * T[i+1] + a11 * R[i+1]
        # normalize
        t0 = T[0]
        for i in range(nlayer):
            T[i] /= t0
            R[i] /= t0
        return T[order], R[order]

    def propagation_coeffs(self, alphai, alpha, k0, order):
        Ti, Ri = self.parratt_recursion(alphai, k0, order)
        Tf = np.zeros(alpha.shape, np.complex_)
        Rf = np.zeros(alpha.shape, np.complex_)
        fc = np.zeros((alpha.shape[0], 4), np.complex_)

        nz = alpha.size
        for i in range(nz):
            Tf[i], Rf[i] = self.parratt_recursion(alpha[i], k0, order)
            fc[i,0] = Ti * Tf[i].conjugate()
            fc[i,1] = Ri * Tf[i].conjugate()
            fc[i,2] = Ti * Rf[i].conjugate()
            fc[i,3] = Ri * Rf[i].conjugate()
            fc[alpha < 0, :] = 0
        return fc

'''
def reflectivity(alpha, delta, beta):
    dns2 = 2 * complex(delta, beta)
    kz = np.sin(alpha)
    kt = np.sqrt(np.sin(alpha)**2 - dns2)
    Rf = (kz - kt)/(kz + kt)
    Tf = 2 * kz / (kz + kt)
    return Rf, Tf

'''