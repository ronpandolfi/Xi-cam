import numpy as np
import itertools

latvecmaxr=4
latvecmaxz=2

def reciprocalvectors(a, b, c, order=2):
    V = abs(np.dot(a, np.cross(b, c)))
    mra = 2 * np.pi / V * np.cross(b, c)
    mrb = 2 * np.pi / V * np.cross(c, a)
    mrc = 2 * np.pi / V * np.cross(a, b)
    mi = np.vstack([mra, mrb, mrc])
    combs = itertools.product(range(-order, order + 1), repeat=3)
    vecs = [np.sum((mi.T * np.array(comb)).T, axis=0) for comb in combs]
    return vecs


def latticevectors(a, b, c, zoffset, order=2):
    mi = np.vstack([a, b, c])
    combs = itertools.product(range(-order, order + 1), repeat=3)
    vecs = [np.sum((mi.T * np.array(comb)).T, axis=0) for comb in combs]
    vecs = [zoffsetvec(vec,zoffset) for vec in vecs if vec[2] >= 0 and np.sqrt(vec[0]**2+vec[1]**2)<=latvecmaxr and vec[2]<=latvecmaxz]
    return vecs

def zoffsetvec(vec,zoffset=0):
    vec[2]=vec[2]+zoffset
    return vec


def latticelines(a, b, c, zoffset, order=2):
    vecs = latticevectors(a, b, c, zoffset, order)

    lines = []
    for vec in vecs:
        lines.extend([(vec, vec + a), (vec, vec - a), (vec, vec + b), (vec, vec - b), (vec, vec + c), (vec, vec - c)])

    return lines


def vecs2angles(v):
    v = np.array(v)
    vproj = v.copy()
    vproj[:, 2] = 0
    return np.arctan2(vproj[:, 1], vproj[:, 0]) / np.pi * 180.


if __name__ == '__main__':
    a = [1, 0, 0]
    b = [.5, np.sqrt(3) / 2, 0]
    c = [0, 0, 1]
    recipvecs = np.array(reciprocalvectors(a, b, c))

    print(list(set(vecs2angles(recipvecs))))

    exit(0)
