#!/usr/bin/env python

import sys
import time

import numpy as np
import pylab
import scipy.sparse
import scipy.optimize

def tic(label=None):
    global TIC_START
    if label is not None:
        print label,
    sys.stdout.flush()
    TIC_START = time.time()

def toc(label=None):
    t = time.time() - TIC_START
    if label is not None:
        print label,
    print "%fs" % t
    sys.stdout.flush()
    return t

def D(dim):
    """Discrete first derivative operator with no boundary."""
    operator = np.zeros((3, dim))
    operator[0,2:]  =  0.5
    operator[2,:-2] = -0.5
    return scipy.sparse.dia_matrix((operator, (1,0,-1)), shape=(dim,dim))

def D2(dim):
    """Discrete second derivative operator with no boundary."""
    operator = np.zeros((3, dim))
    operator[0,2:]  =  1
    operator[1,1:-1]  = -2
    operator[2,:-2] =  1
    return scipy.sparse.dia_matrix((operator, (1,0,-1)), shape=(dim,dim))


def center_diff(domain, n=1, axis=-1):
    """Like numpy.diff, but centered instead of forward."""
    xs = domain.copy()
    dx = np.zeros_like(xs,dtype=float)
    if axis == 0:
        for i in xrange(n):
            dx[:-1]  += np.diff(xs, axis=axis)
            dx[1:]   += np.diff(xs[::-1], axis=axis)[::-1]*-1
            dx[1:-1] *= 0.5
            t = xs; xs = dx; dx = t
    if axis == 1:
        for i in xrange(n):
            dx[:,:-1]  += np.diff(xs, axis=axis)
            dx[:,1:]   += np.diff(xs[::-1], axis=axis)[::-1]*-1
            dx[:,1:-1] *= 0.5
            t = xs; xs = dx; dx = t
    return xs

def sinh_space(exact, high, density, size):
    """Sigmoidal space with high density around 'exact'. Use ~ 0.93."""
    # def g(x, K, c, p): return K + c/p * np.sinh(p*x + np.arcsinh(-p*K/c))
    # c = float(density)
    density = float(density)
    K = float(exact)
    Smax = float(high)
    # p = scipy.optimize.root(lambda p: g(1.0, K, c, p)-1.0, 1)
    # print p
    # print p.success, p.r, g(1.0, K, c, p.r)-1
    # p = p.r
    deps = 1./size * (np.arcsinh((Smax - K)*(1/density)) - np.arcsinh(-K/density))
    eps = np.arcsinh(-K/density) + np.arange(size)*deps
    space = K + density * np.sinh(eps)
    return space

def exponential_space(low, exact, high, ex, n):
    """Ex is the exponent used to map to the new space."""
    v = np.zeros(n)
    l = pow(low,1./ex)
    h = pow(high,1./ex)
    x = pow(exact,1./ex)
    dv = (h - l) / (n-1)

    j = 0
    d = 1e100
    for i in range(n):
        if (l + i*dv > x):
        # if abs(i*dv - x) < d:
            # d = abs(i*dv - x)
            j = i-1
            break
    if (j == 0):
        print "Did not find thingy."
        assert(j != 0)
    dx = x - (l + j*dv)
    h += (n-1) * dx/j
    dv = (h - l) / (n-1)
    for i in range(n):
        v[i] = l + pow(i*dv, ex)
    return v

def cubic_sigmoid_space(exact, high, density, n):
    """Cheap and bad sigmoid curve. Use sinh instead."""
    if density == 0:
        return linspace(exact - (high - exact), high, n)

    y = np.zeros(n)
    dx = 1.0/(n-1)
    scale = (float(high)-exact)/(density**3 + density)
    for i in range(n):
        x = (2*(i*dx)-1)*density
        y[i] = exact + (x**3+x)*scale

    return y



def nonuniform_backward_coefficients(deltas):
    """The coefficients for tridiagonal matrices operating on a non-uniform
    grid.

    L = spl.dia_matrix((fst, (0,-1,-2)), shape=(fst.shape[1], fst.shape[1]))
    """
    d = deltas.copy()
    fst = np.zeros((3,len(d)))
    snd = fst.copy()
    for i in range(2,len(d)-1):
        fst[0,i]   = (d[i-1]+2*d[i])  / (d[i]*(d[i-1]+d[i]));
        fst[1,i-1] = (-d[i-1] - d[i]) / (d[i-1]*d[i]);
        fst[2,i-2] = d[i]             / (d[i-1]*(d[i-1]+d[i]));

        denom = (0.5*(d[i]+d[i-1])*d[i]*d[i-1]);
        snd[0,i]   = d[i-1] / denom;
        snd[1,i-1] = -(d[i]+d[i-1]) / denom;
        snd[2,i-2] = d[i] / denom;


    L1 = scipy.sparse.dia_matrix((fst.copy(), (0, -1, -2)), shape=(len(d),len(d)))
    L2 = scipy.sparse.dia_matrix((snd.copy(), (0, -1, -2)), shape=(len(d),len(d)))
    return L1,L2
    return fst, snd


def nonuniform_forward_coefficients(deltas):
    """The coefficients for tridiagonal matrices operating on a non-uniform
    grid.

    L = spl.dia_matrix((fst, (2,1,0)), shape=(fst.shape[1], fst.shape[1]))
    """
    d = deltas.copy()
    fst = np.zeros((3,len(d)))
    snd = fst.copy()
    for i in range(1,len(d)-2):
        fst[0,i+2] = -d[i+1]           / (d[i+2]*(d[i+1]+d[i+2]))
        fst[1,i+1] = (d[i+1] + d[i+2])  /         (d[i+1]*d[i+2])
        fst[2,i]   = (-2*d[i+1]-d[i+2]) / (d[i+1]*(d[i+1]+d[i+2]))

        denom = (0.5*(d[i+2]+d[i+1])*d[i+2]*d[i+1]);
        snd[0,i+2] =   d[i+1]         / denom
        snd[1,i+1] = -(d[i+2]+d[i+1]) / denom
        snd[2,i]   =   d[i+2]         / denom

    L1 = scipy.sparse.dia_matrix((fst.copy(), (2, 1, 0)), shape=(len(d),len(d)))
    L2 = scipy.sparse.dia_matrix((snd.copy(), (2, 1, 0)), shape=(len(d),len(d)))
    return L1,L2


def nonuniform_center_coefficients(deltas):
    """
    The coefficients for tridiagonal matrices operating on a non-uniform grid.

    L = spl.dia_matrix((fst, (1,0,-1)), shape=(fst.shape[1], fst.shape[1]))
    """
    d = deltas.copy()
    fst = np.zeros((3,len(d)))
    snd = fst.copy()
    for i in range(1,len(d)-1):
        fst[0,i+1] =            d[i]  / (d[i+1]*(d[i]+d[i+1]))
        fst[1,i]   = (-d[i] + d[i+1]) /         (d[i]*d[i+1])
        fst[2,i-1] =         -d[i+1]  / (d[i  ]*(d[i]+d[i+1]))

        snd[0,i+1] = 2  / (d[i+1]*(d[i]+d[i+1]))
        snd[1,i]   = -2 /       (d[i]*d[i+1])
        snd[2,i-1] = 2  / (d[i  ]*(d[i]+d[i+1]))
    L1 = scipy.sparse.dia_matrix((fst.copy(), (1, 0, -1)), shape=(len(d),len(d)))
    L2 = scipy.sparse.dia_matrix((snd.copy(), (1, 0, -1)), shape=(len(d),len(d)))
    return L1, L2
    return fst, snd

def nonuniform_center_forward_coefficients(deltas, upwind_from=None):
    d = deltas.copy()
    if upwind_from is None:
        return nonuniform_center_coefficients(d)
    u = upwind_from

    C1, C2 = nonuniform_center_coefficients(d)
    F1, F2 = nonuniform_forward_coefficients(d)
    U1 = np.zeros((4, len(d)))
    U1[0,:u+2] = 0
    U1[0,u+2:] = F1.data[0,u+2:]

    U1[1,:u+1] = C1.data[0,:u+1]
    U1[1,u+1:] = F1.data[1,u+1:]

    U1[2,:u] = C1.data[1,:u]
    U1[2,u:] = F1.data[2,u:]

    U1[3,:u-1] = C1.data[0,:u-1]
    U1[3,u-1:] = 0.0

    U1 = scipy.sparse.dia_matrix((U1, (2, 1, 0, -1)), shape=(len(d),len(d)))
    return U1


def cs(deltas):
    B1,B2 = nonuniform_backward_coefficients(deltas)
    C1,C2 = nonuniform_center_coefficients(deltas)
    F1,F2 = nonuniform_forward_coefficients(deltas)
    return F1,F2,C1,C2,B1,B2
