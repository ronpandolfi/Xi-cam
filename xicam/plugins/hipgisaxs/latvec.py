import numpy as np
import itertools


def reciprocalvectors(a, b, c, order=2):
    V = abs(np.dot(a, np.cross(b, c)))
    mra = 2 * np.pi / V * np.cross(b, c)
    mrb = 2 * np.pi / V * np.cross(c, a)
    mrc = 2 * np.pi / V * np.cross(a, b)
    mi = np.vstack([mra, mrb, mrc])
    combs = itertools.product(range(-order, order + 1), repeat=3)
    vecs = [np.sum((mi.T * np.array(comb)).T, axis=0) for comb in combs]
    return vecs

def combs_generator(x,y,z):
    for i in range(-x/2,x/2+1):
        for j in range(-y/2,y/2+1):
            for k in range(0,z+1):
                yield (i,j,k)

def latticevectors(a, b, c, zoffset, maxreps=100, repetitions=None,scaling=1.):
    if repetitions is None: repetitions = [0,0,0]
    mi = np.vstack([a, b, c])
    combs = combs_generator(*repetitions) #itertools.product(range(-order, order + 1), repeat=3)
    # vecs = [np.sum((mi.T * np.array(comb)).T, axis=0) for comb in combs]
    # vecs = [zoffsetvec(vec,zoffset) for vec in vecs if vec[2] >= 0 and np.sqrt(vec[0]**2+vec[1]**2)<=maxr and vec[2]<=maxz]
    vecs = [zoffsetvec(np.sum((mi.T * np.array(comb)*scaling).T, axis=0),zoffset) for comb in combs]
    vecs = sorted(vecs,key=lambda v: np.vdot(v,v))[:maxreps]
    return vecs

def zoffsetvec(vec,zoffset=0):
    vec[2]=vec[2]+zoffset
    return vec


def latticelines(a, b, c, zoffset, maxreps=100, repetitions=None, scaling=1.):
    vecs = latticevectors(a, b, c, zoffset, maxreps, repetitions, scaling)

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
