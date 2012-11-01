#!/usr/bin/env python

import numpy as np
import pylab
import scipy.sparse
import scipy.optimize
import time

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


def center_diff(domain, n=1, axis=0):
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
    THIS IS WRONG. CHECK THE ALIGNMENT.

    L = spl.dia_matrix((fst, (1,0,-1)), shape=(fst.shape[1], fst.shape[1]))
    """
    d = deltas.copy()
    fst = np.zeros((3,len(d)))
    snd = fst.copy()
    for i in range(1,len(d)-1):
        fst[0,i] =            d[i]  / (d[i+1]*(d[i]+d[i+1]))
        fst[1,i-1] = (-d[i] + d[i+1]) /         (d[i]*d[i+1])
        fst[2,i-2] =         -d[i+1]  / (d[i  ]*(d[i]+d[i+1]))

        snd[0,i+1] = 2  / (d[i+1]*(d[i]+d[i+1]))
        snd[1,i]   = -2 /       (d[i]*d[i+1])
        snd[2,i-1] = 2  / (d[i  ]*(d[i]+d[i+1]))
    return fst, snd


def nonuniform_forward_coefficients(deltas):
    """The coefficients for tridiagonal matrices operating on a non-uniform
    grid.
    THIS IS WRONG. CHECK THE ALIGNMENT.

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
    return fst, snd


def nonuniform_center_coefficients(deltas):
    """The coefficients for tridiagonal matrices operating on a non-uniform
    grid.

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
    return fst, snd
