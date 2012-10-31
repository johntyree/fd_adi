#!/usr/bin/env python
"""Demonstration of 1D Black Scholes using FTCS BTCS CTCS and Smoothed CTCS."""


import numpy as np
import scipy.stats
import scipy.linalg as spl
import scipy.sparse as sps
from pylab import *
from utils import fp

x = np.log(100)
k = 100.
r = 0.06
t = 1.
v = 2.**2
dt = 1/30.
I = 15

def D(dim):
    '''Discrete first derivative operator with no boundary.'''
    operator = np.zeros((3, dim))
    operator[0,2:]  =  0.5
    operator[2,:-2] = -0.5
    return sps.dia_matrix((operator, (1,0,-1)), shape=(dim,dim))

def D2(dim):
    '''Discrete second derivative operator with no boundary.'''
    operator = np.zeros((3, dim))
    operator[0,2:]  =  1
    operator[1,1:-1]  = -2
    operator[2,:-2] =  1
    return sps.dia_matrix((operator, (1,0,-1)), shape=(dim,dim))

def init(xs, vs, k):
    u = np.ones((len(xs),len(vs))).T * np.exp(xs)
    u = u.T
    return np.maximum(0, u-k)

def bs_call(s, k, r, sig, t):
    N = scipy.stats.distributions.norm.cdf
    d1 = (np.log(s/k) + (r+0.5*sig**2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig*np.sqrt(t)
    return (N(d1)*s - N(d2)*k*np.exp(-r * t), N(d1))

def center_diff(xs):
    dx = np.zeros_like(xs,dtype=float)
    dx[:-1]  += np.diff(xs)
    dx[1:]   += np.diff(xs[::-1])[::-1]*-1
    dx[1:-1] *= 0.5
    return dx

I += not (I%2)
xs = np.linspace(-1, 1, I)
xs = 3*np.sqrt(v)*xs*x + x
dxs = center_diff(xs)
dss = center_diff(np.exp(xs))
ids = (80 < np.exp(xs)) & (np.exp(xs) < 120)
idx = find(xs[ids] == x)
dx = dxs[idx][0]
ds = dss[ids]


bs, delta = [x[ids] for x in bs_call(np.exp(xs), k, r, np.sqrt(v), t)]

Vi = init(xs, [1], k)
V = np.copy(Vi)


def impl(V,dt,n):
    global As, Ass
    As = D(I)/dx * (r-0.5*v)
    print "dx", dx
    fp(As,3,'e')
    Rs = np.zeros_like(V)
    Rs[-1] = 1
    Rs *= (r-0.5*v)

    Ass = D2(I)/(dx**2) * (0.5*v)
    Ass.data[1, -1] = -2/dx**2
    Ass.data[2, -2] =  2/dx**2
    Rss = np.zeros_like(V)
    Rss[-1] = 2*dx/dx**2
    Rss *= (0.5*v)

    # L  = (As + Ass - r*np.eye(I))*-dt + np.eye(I)
    L = As.copy()
    L.data += Ass.data
    L.data[1,:] -= r
    L.data *= -dt
    L.data[1,:] += 1
    R  = (Rs + Rss)*dt
    for i in xrange(n):
        V = spl.solve_banded((1,1), L.data, V + R)
    return V

def crank(V,dt,n):
    global As, Ass
    dt *= 0.5

    As = D(I)/dx * (r-0.5*v)
    Rs = np.zeros_like(V)
    Rs[-1] = 1
    Rs *= (r-0.5*v)

    Ass = D2(I)/(dx**2) * (0.5*v)
    Ass.data[1, -1] = -2/dx**2
    Ass.data[2, -2] =  2/dx**2
    Rss = np.zeros_like(V)
    Rss[-1] = 2*dx/dx**2
    Rss *= (0.5*v)

    L = As.copy()
    L.data += Ass.data
    L.data[1,:] -= r
    L.data *= dt
    L.data[1,:] += 1
    R  = (Rs + Rss)*dt
    Li = As.copy()
    Li.data += Ass.data
    Li.data[1,:] -= r
    Li.data *= -dt
    Li.data[1,:] += 1

    for i in xrange(n):
        V = L.dot(V) + R
        V = spl.solve_banded((1,1), Li.data, V + R)
    return V


# Trim for plotting
front = 45
back = 35

# V = impl(Vi,dt, int(t/dt))
# V = V[ids][:,0]
# print V[idx] - bs[idx]
# dVds = center_diff(V)/(ds)
# plot((np.exp(xs)/k*100)[ids][front:-back],
     # (V-bs)[front:-back],
     # '*', label="impl")


# ## Rannacher smoothing to damp oscilations at the discontinuity
# V = impl(Vi,0.5*dt,4)
# V = crank(V,dt, int(t/dt)-2)
# V = V[ids][:,0]
# print V[idx] - bs[idx]
# dVds = center_diff(V)/(ds)
# plot((np.exp(xs)/k*100)[ids][front:-back],
    # (V-bs)[front:-back],
    # 'x', label="smooth")


# title("Error in Price")
# xlabel("% of strike")
# ylabel("Error")
# legend(loc=0)
# show()

