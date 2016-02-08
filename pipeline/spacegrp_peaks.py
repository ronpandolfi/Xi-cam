#! /usr/bin/env python

import itertools
import numpy as np
from numpy.linalg import norm
from scipy.optimize import fsolve


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
            Ac[1] = 1
        else:
            Ac[2] = 1

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
        C_c[2] = 1. / hkl[1]
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
    cosa = np.cos(alphai)
    sina = np.sin(alphai)
    y = np.zeros(4, dtype=float)
    y[0] = x[0] ** 2 + x[2] ** 2 - G[0] ** 2 - G[1] ** 2
    y[1] = x[1] * cosa + x[3] * sina - x[0]
    y[2] = -x[1] * sina + x[3] * cosa - G[2]
    y[3] = (x[1] - k) ** 2 + x[2] ** 2 + x[3] ** 2 - k ** 2
    return y


def reflection_condtion(hkl, unitcell, space_grp):
    if space_grp is None and unitcell is None:
        return True
    tot = 0.
    c = -2 * np.pi * np.complex(0, 1)
    for pos in unitcell:
        tot += np.real(np.exp(c * np.dot(pos, hkl)))
    if abs(np.real(tot)) < 1.0E-06:
        return False
    else:
        return True


def alpha_exit(q, alphai, nu, k):
    t1 = (q[2] / k) ** 2 + np.sin(alphai) ** 2
    t2 = 2 * q[2] / k * np.sqrt(nu ** 2 - 1 + np.sin(alphai) ** 2)
    af1 = np.arcsin(np.real(np.sqrt(t1 - t2)))
    af2 = np.arcsin(np.real(np.sqrt(t1 + t2)))
    return af1, af2


def theta_exit(q, alphai, alphaf, k):
    t1 = np.cos(alphaf) ** 2 + np.cos(alphai) ** 2 - (q[0] ** 2 + q[1] ** 2) / k ** 2
    t2 = 2 * np.cos(alphaf) * np.cos(alphai)
    return np.arccos(t1 / t2)


def find_peaks(a, b, c, alpha=None, beta=None, gamma=None, normal=None,
               norm_type="uvw", wavelen=0.123984, order=4, unitcell=None, space_grp=None):
    # rotation matrix from crystal coordinates for sample coordinates
    A = crystal2sample(a, b, c, alpha, beta, gamma)
    # unit vector normal to sample plane
    if norm_type == "xyz":
        if isinstance(normal, np.ndarray):
            e_norm = normal / norm(normal)
        else:
            n1 = np.array(normal, dtype=float)
            e_norm = n1 / norm(n1)
    elif norm_type == "hkl":
        if not n.dtype.dtype is np.int_:
            raise TypeError("hkl type must be integer datatype")
        e_norm = orientation_hkl(normal, A)
    elif norm_type == "uvw":
        if isinstance(normal, np.ndarray):
            n1 = np.array(normal, dtype=float)
            e_norm = orientation_uvw(n1, A)
        else:
            n1 = np.array(normal, dtype=float)
            e_norm = orientation_uvw(n1, A)
    else:
        raise ValueError("error: unable to process normal direction")

    # Rotation axis
    e_z = np.array([0, 0, 1], dtype=float)
    e_r = np.cross(e_norm, e_z)
    e_r /= norm(e_r)

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

    nu = 1 - np.complex(2.236E-06, -1.8790E-09)
    HKL = itertools.product(range(-order, order + 1), repeat=3)
    alphai = np.deg2rad(0.2)
    k = 2 * np.pi / wavelen
    peaks = dict()
    for hkl in HKL:
        if (reflection_condtion(hkl, space_grp, unitcell)):
            G = RV[0, :] * hkl[0] + RV[1, :] * hkl[1] + RV[2, :] * hkl[2]
            skip = True
            for it in range(10):
                x = np.random.rand(4)
                y,info,ier,msg = fsolve(equations, x, args=(G, alphai, k), full_output=True)
                if ier == 1:
                    skip = False
                    break
            if skip:
                continue
            q = [y[0], y[2], G[2]]
            al_t, al_r = alpha_exit(q, alphai, nu, k)
            th_t = theta_exit(q, alphai, al_t, k)
            th_r = theta_exit(q, alphai, al_r, k)
            transmission = [th_t, al_t]
            reflection = [th_r, al_r]
            key = '{0}{1}{2}'.format(hkl[0], hkl[1], hkl[2])
            peaks[key] = (transmission, reflection)
    return peaks


def angles_to_pixels(angles, center, sdd, pixel_size=[172E-6, 172E-06]):
    tan_2t = np.tan(angles[:, 0])
    tan_al = np.tan(angles[:, 1])
    x = tan_2t * sdd
    px = (tan_2t * sdd) / pixel_size[0] + center[0]
    py = np.sqrt(sdd ** 2 * tan_al ** 2 - x ** 2) / pixel_size[1] + center[1]
    return np.vstack([px, py]).astype(int)


if __name__ == '__main__':
    ang1 = np.deg2rad(90.)
    ang2 = np.deg2rad(90.)
    ang3 = np.deg2rad(90.)
    n = np.array([0, 0, 1], dtype=float)
    peaks = find_peaks(10., 10., 10., alpha=ang1, beta=ang2, gamma=ang3, normal=n, norm_type='uvw', wavelen=0.123984,
                       order=1)
    for key in peaks:
        print key + " -> " + str(peaks[key])

