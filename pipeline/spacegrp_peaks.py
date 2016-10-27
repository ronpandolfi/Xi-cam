#! /usr/bin/env python
# --coding: utf-8 --
import itertools
import numpy as np
from numpy.linalg import norm
from scipy.optimize import root
import sgexclusions
from xicam import config

def volume(a, b, c, alpha=None, beta=None, gamma=None):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and \
            isinstance(c, np.ndarray):
        return abs(np.dot(a, np.cross(b, c)))
    elif isinstance(a, float) and isinstance(b, float) and isinstance(c, float):
        if alpha is None:
            alpha = np.pi / 2
        if beta is None:
            beta = np.pi / 2
        if gamma is None:
            gamma = np.pi / 2
        abc2 = (a * b * c ) ** 2
        t1 = np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        t2 = np.cos(alpha) ** 2 + np.cos(beta) ** 2 + np.cos(gamma) ** 2
        return np.sqrt(abc2 * (1 - t2 + 2 * t1))
    else:
        raise ValueError('vectors must be either numpy.ndarray or real types')


def crystal2sample(a, b, c, alpha=None, beta=None, gamma=None):
    R = np.zeros((3, 3), dtype=np.float)

    # don't need to do all this if a,b,c are vectors
    # we should convert them to vectors if they are not
    if alpha is None:
        alpha = np.pi / 2
    if beta is None:
        beta = np.pi / 2
    if gamma is None:
        gamma = np.pi / 2
    t1 = np.cos(gamma) * np.cos(beta) - np.cos(alpha)
    vol = volume(a, b, c, alpha, beta, gamma)
    R[0, 0] = a
    R[0, 1] = b * np.cos(gamma)
    R[0, 2] = c * np.cos(beta)
    R[1, 1] = b * np.sin(gamma)
    R[1, 2] = -c * t1 / np.sin(gamma)
    R[2, 2] = vol / (a * b * np.sin(gamma))
    return R


def orientation_uvw(uvw, R):
    e = R.dot(uvw)
    return e / norm(e)


def orientation_hkl(hkl, R):
    h = hkl[0]
    k = hkl[1]
    l = hkl[2]
    if h == 0 and k == 0 and l == 0:
        raise ValueError("one of [h k l] must be non-zero")

    A_c = np.zeros(3, dtype=np.float)
    B_c = np.zeros(3, dtype=np.float)
    C_c = np.zeros(3, dtype=np.float)

    # vector Ac
    if not h == 0:
        A_c[0] = 1. / hkl[0]
    else:
        A_c[0] = 1.;
        if not k == 0:
            A_c[1] = 1
        else:
            A_c[2] = 1

    # vector Bc
    if not k == 0:
        B_c[1] = 1. / hkl[1]
    else:
        B_c[1] = 1.;
        if not l == 0:
            B_c[2] = 1
        else:
            B_c[0] = 1

    # vector Cc
    if not l == 0:
        C_c[2] = 1. / hkl[2]
    else:
        C_c[2] = 1.
        if not h == 0:
            C_c[0] = 1.;
        else:
            C_c[1] = 1.;
    AsBs = np.inner(R, (B_c - A_c))
    BsCs = np.inner(R, (C_c - B_c))
    t1 = np.cross(AsBs, BsCs)
    return t1 / norm(t1)


def RotationMatrix(u, theta):
    R = np.zeros((3, 3), dtype=np.float)
    cost = np.cos(theta)
    sint = np.sin(theta)
    R[0, 0] = cost + u[0] ** 2 * ( 1 - cost )
    R[0, 1] = u[0] * u[1] * ( 1 - cost ) - u[2] * sint
    R[0, 2] = u[0] * u[2] * ( 1 - cost ) + u[1] * sint
    R[1, 0] = u[0] * u[1] * ( 1 - cost ) - u[2] * sint
    R[1, 1] = cost + u[1] ** 2 * ( 1 - cost )
    R[1, 2] = u[1] * u[2] * ( 1 - cost ) - u[0] * sint
    R[2, 0] = u[0] * u[2] * ( 1 - cost ) - u[1] * sint
    R[2, 1] = u[1] * u[2] * ( 1 - cost ) + u[0] * sint
    R[2, 2] = cost + u[2] ** 2 * ( 1 - cost )
    return R;


def reciprocalvectors(a, b, c):
    V = volume(a, b, c)
    mra = 2 * np.pi / V * np.cross(b, c)
    mrb = 2 * np.pi / V * np.cross(c, a)
    mrc = 2 * np.pi / V * np.cross(a, b)
    vecs = np.vstack([mra, mrb, mrc])
    return vecs


def equations(x, G, alphai, k):
    ci = np.cos(alphai)
    si = np.sin(alphai)
    y = np.zeros(4, dtype=float)
    y[0] = x[0] ** 2 + x[2] ** 2 - G[0] ** 2 - G[1] ** 2
    y[1] = x[0]  - x[1] * ci - x[3] * si
    y[2] = x[1] * si - x[3] * ci + G[2]
    y[3] = (x[1] - k) ** 2 + x[2] ** 2 + x[3] ** 2 - k ** 2
    return y

def jacobian(x, G, alphai, k):
        ci = np.cos(alphai)
        si = np.sin(alphai)
        jac = np.zeros((4,4),dtype=float)
        jac[0,0] = 2 * x[0]
        jac[0,1] = 1
        jac[1,1] = -ci
        jac[1,2] = si
        jac[1,3] = 2 * (x[1] - k)
        jac[2, 0] = 2 * x[2]
        jac[2,3] = 2 * x[2]
        jac[3,1] = -si
        jac[3,2] = -ci
        jac[3,3] = 2 * x[3]
        return jac

def reflection_condtion(hkl, unitcell, space_grp):
    if unitcell is None:
        return True
    tot = 0.
    c = -2 * np.pi * np.complex(0, 1)
    for pos in unitcell:
        tot += np.real(np.exp(c * np.dot(pos, hkl)))
    if abs(np.real(tot)) < 1.0E-06:
        return False
    else:
        return True


def alpha_exit(qz, alphai, k):
    alf = np.arcsin(qz / k - np.sin(alphai))
    return alf

def theta_exit(q, alpha, alphai, k):
    t1 = np.cos(alpha)**2 + np.cos(alphai)**2 - (q[0]**2 + q[1]**2)/(k*k)
    t2 = 2 * np.cos(alpha) * np.cos(alphai)
    return np.sign(q[1]) * np.arccos(t1/t2)

def angles_to_pixels(angles, center, sdd, pixel_size=None):
    if pixel_size is None:
        pixel_size = [172e-6, 172e-6]
    if np.NaN in angles:
        return np.NAN, np.NAN
    tan_2t = np.tan(angles[:, 0])
    tan_al = np.tan(angles[:, 1])
    x = tan_2t * sdd
    px = sdd * tan_2t / pixel_size[0] #+ center[0]
    py = np.sqrt(sdd ** 2 + x ** 2) * tan_al  / pixel_size[1] #+ center[1]
    pixels = np.zeros((px.size, 2))
    pixels[:,0] = px
    pixels[:,1] = py
    return pixels.astype(int)

class peak(object):
    def __init__(self, mode):
        self.mode = mode # either 'Transmission' or 'Reflection'
        self.hkl = None
        self.x = None
        self.y = None
        self.twotheta = None
        self.alphaf = None
        self.qpar = None
        self.qz = None

    def pos(self):
        return self.x, self.y

    # qz, alphaf and y-poition are going to be different
    def copy(self):
        val = peak(self.mode)
        val.twotheta = self.twotheta
        val.qpar = self.qpar
        val.hkl = self.hkl
        val.x = self.x
        return val

    def position(self, center, sdd, pixels):
        tan_2t = np.tan(self.twotheta)
        tan_al = np.tan(self.alphaf)
        x= tan_2t * sdd
        self.x = sdd * tan_2t / pixels# + config.activeExperiment.center[0]
        self.y = np.sqrt(sdd ** 2 + x ** 2) * tan_al / pixels# + config.activeExperiment.center[1]

    def isAt(self, pos):
        if np.isnan(self.twotheta): return False
        return int(self.x) == int(pos.x()) and int(self.y) == int(pos.y())

    def __str__(self):
        s = u"Peak type: {}\n".format(self.mode)
        s += u"Lattice vector (h,k,l): {}\n".format(self.hkl)
        if self.twotheta is not None: s += u"2\u03B8: {}\n".format(self.twotheta)
        if self.alphaf is not None: s += u"\u03B1f: {}\n".format(self.alphaf)
        if self.qpar is not None: s += u"q\u2225: {}\n".format(self.qpar/1e10)
        if self.qz is not None: s += u"q\u27c2: {}\n".format(self.qz/1e10)
        return s

def qvalues(twotheta, alphaf, alphai, wavelen):
    k = 2 * np.pi / wavelen
    qx = k * (np.cos(alphaf) * np.cos(twotheta) - np.cos(alphai))
    qy = k * np.cos(alphaf) * np.sin(twotheta)
    qz = k * np.sin(alphaf) + np.sin(alphai)
    return np.sqrt(qx**2 + qy**2), qz

def dwba_componets(Gz, k, nu, alphai):
    alphaf = np.zeros(4, dtype=np.float)
    gprime = Gz/k
    t1 = nu**2 - np.cos(alphai)**2
    if np.real(t1) < 0:
        print 'Error: incoming angle is below the critical angle'
        return None
    cprime = np.sqrt(t1)
    cosa1 = nu**2 - (gprime + cprime)**2 
    cosa2 = nu**2 - (gprime - cprime)**2 
    a1 = np.arccos(np.sqrt(cosa1))
    a2 = np.arccos(np.sqrt(cosa2))
    #TODO: this is a hack a1 should always be transmission
    if abs(a2) > abs(a1):
        alphaf[0] = a1
        alphaf[1] = a2
    else:
        alphaf[0] = a2
        alphaf[1] = a1
    
    return alphaf

def find_peaks(a, b, c, alpha=None, beta=None, gamma=None, normal=None,
               norm_type="uvw", wavelen=0.123984e-9, refdelta=2.236E-06, refbeta=-1.8790E-09, order=5, unitcell=None, space_grp=None):
    # rotation matrix from crystal coordinates for sample coordinates
    if alpha is not None: alpha = np.deg2rad(alpha)
    if beta is not None: beta = np.deg2rad(beta)
    if gamma is not None: gamma = np.deg2rad(gamma)

    A = crystal2sample(a, b, c, alpha, beta, gamma)
    # unit vector normal to sample plane
    normal = np.array(normal)
    if norm_type == "xyz":
        e_norm = normal / norm(normal)
    elif norm_type == "hkl":
        if not normal.dtype.type is np.int_:
            raise TypeError("hkl type must be integer datatype")
        e_norm = orientation_hkl(normal, A)
    elif norm_type == "uvw":
        n1 = np.array(normal, dtype=float)
        e_norm = orientation_uvw(n1, A)
    else:
        raise ValueError("error: unable to process normal direction")

    # Rotation axis
    e_z = np.array([0, 0, 1], dtype=float)
    e_r = np.cross(e_norm, e_z)
    e_r /= (norm(e_r) if norm(e_r) != 0 else 1)

    # Angle of rotation
    theta = np.arccos(np.dot(e_z, e_norm))
    # Rotation Matrix
    R = RotationMatrix(e_r, theta)

    # Lattice vectors in sample frame
    V = np.dot(R, A)
    a = V[:, 0]
    b = V[:, 1]
    c = V[:, 2]
    RV = reciprocalvectors(a, b, c)

    nu = 1 - np.complex(refdelta, refbeta)
    HKL = itertools.product(range(-order, order + 1), repeat=3)
    alphai = np.deg2rad(config.activeExperiment.getvalue('Incidence Angle (GIXS)'))
    k = 2 * np.pi / wavelen
    peaks = list()
    for hkl in HKL:
        if not sgexclusions.check(hkl, space_grp): continue
        if (reflection_condtion(hkl, unitcell, space_grp)):
            G = RV[0, :] * hkl[0] + RV[1, :] * hkl[1] + RV[2, :] * hkl[2]
            qpar = np.sqrt(G[0]**2 + G[1]**2)
            alpha = alpha_exit(G[2], alphai, k)
            theta = theta_exit(G, alpha, alphai, k)

            # compute DWBA components
            alphaf = dwba_componets(G[2], k, nu, alphai)
            if alphaf is None:
                continue

            # get the transmission peaks
            transmission = peak('Transmission')
            transmission.hkl = hkl
            transmission.twotheta = theta
            transmission.alphaf = alphaf[0]
            transmission.qpar = qpar
            transmission.qz = G[2]
            peaks.append(transmission)

            # calulate reflection peaks
            reflection = peak('Reflection')
            reflection.hkl = hkl
            reflection.twotheta = theta
            reflection.alphaf = alphaf[1]
            reflection.qpar = qpar
            reflection.qz = G[2]
            peaks.append(reflection)

    return peaks


if __name__ == '__main__':
    ang1 = np.deg2rad(90.)
    ang2 = np.deg2rad(90.)
    ang3 = np.deg2rad(90.)
    n = np.array([0, 0, 1], dtype=float)
    peaks = find_peaks(10.E-09, 10.E-09, 10.E-09, alpha=ang1, beta=ang2, gamma=ang3, normal=n, norm_type='uvw', wavelen=0.123984E-09,
                       order=2)
    trans_peaks = []
    refle_peaks = []
    for key in peaks:
        trans_peaks.append(peaks[key][0])
        refle_peaks.append(peaks[key][1])

    center = np.array([672,224], dtype=int)
    tr = np.array(trans_peaks, dtype=float)
    p1 = angles_to_pixels(tr,center, sdd=1.84)
    rf = np.array(refle_peaks, dtype=float)
    p2 = angles_to_pixels(rf,center, sdd=1.84)
    print str(p1)
    print str(p2)
