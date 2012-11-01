#!/usr/bin/env python
"""Demonstration of 1D Black Scholes using FTCS BTCS CTCS and Smoothed CTCS."""


import numpy as np
import scipy.stats
import scipy.linalg as spl
import scipy.sparse as sps
from pylab import *
from visualize import fp
from time import time
from utils import center_diff

x = np.log(100)
k = 100.
r = 0.06
t = 1
v = 0.2**2
dt = 1/100.
I = 20000
I += not (I%2)
xs = np.linspace(-1, 1, I)
xs = 4*np.sqrt(v)*xs*x + x
dx = center_diff(xs)
ds = center_diff(np.exp(xs))
ids = (0 < np.exp(xs)) & (np.exp(xs) < 1400)
idx = isclose(xs[ids], x)
dx = dx[idx][0]
ds = ds[ids]

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

bs, delta = [x[ids] for x in bs_call(np.exp(xs), k, r, np.sqrt(v), t)]

Vi = init(xs, [1], k)
V = np.copy(Vi)


def expl(V,dt,n):
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

    # Le  = (As + Ass - r*np.eye(I))*dt + np.eye(I)
    Le = As.copy()
    Le.data += Ass.data
    Le.data[1,:] -= r
    Le.data *= dt
    Le.data[1,:] += 1

    R  = (Rs + Rss)*dt

    print "GO!"
    for i in xrange(n):
        V = Le.dot(V) + R
    return V

def impl(V,dt,n):
    V = V.copy()
    mu_s = r-0.5*v
    gamma2_s = 0.5*v
    As = D(I)/dx * mu_s
    Rs = np.zeros_like(V[:,0])
    Rs[-1] = 1
    Rs *= mu_s
    # fp(As.data,2,'e')
    # fp(Rs[newaxis,:],2,'e')
    # raw_input()

    Ass = D2(I)/(dx**2) * gamma2_s
    Ass.data[1, -1] = -2/dx**2
    Ass.data[2, -2] =  2/dx**2
    Rss = np.zeros_like(V[:,0])
    Rss[-1] = 2*dx/dx**2
    Rss *= gamma2_s
    # fp(Ass.data,2,'e')
    # fp(Rss[newaxis,:],2,'e')
    # raw_input()

    # L  = (As + Ass - r*np.eye(I))*-dt + np.eye(I)
    L = As.copy()
    L.data += Ass.data
    L.data[1,:] -= r
    L.data *= -dt
    L.data[1,:] += 1

    R  = (Rs + Rss)*dt
    # V = V[:,0].copy()
    # R = R[:,0].copy()
    print V.shape
    print R.shape
    # sys.exit()
    start = time()
    for i in xrange(n):
        if not i % 100:
            print int(i*100.0/float(n)),
        V[:,0] = spl.solve_banded((1,1), L.data, V[:,0] + R, overwrite_b=True)
    print
    print "Impl time:", time() - start
    return V

def crank(V,dt,n):
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

    start = time()
    for i in xrange(n):
        if not i % 100:
            print int(i*100.0/float(n)),
        V = L.dot(V) + R
        V = spl.solve_banded((1,1), Li.data, V + R)
    print
    print "Crank time:", time() - start
    return V


# Trim for plotting
front = 1
back = 1
line_width = 2
          # exp  imp   cr   smo
markers = ['--', '--', ':', ':']

# V = expl(Vi,dt, int(t/dt))
# V = V[ids][:,0]
# print V[idx] - bs[idx]
# dVds = center_diff(V)/(ds)
# plot((np.exp(xs)/k*100)[ids][front:-back],
    # (V-bs)[front:-back],
    # markers[0], lw=line_width, label="exp")


V = impl(Vi,dt, int(t/dt))
V = V[ids][:,0]
print V[idx] - bs[idx]
dVds = center_diff(V)/(ds)
plot((np.exp(xs)/k*100)[ids][front:-back],
     (V-bs)[front:-back],
     markers[1], lw=line_width, label="impl")


V = crank(Vi,dt, int(t/dt))
V = V[ids][:,0]
print V[idx] - bs[idx]
dVds = center_diff(V)/(ds)
plot((np.exp(xs)/k*100)[ids][front:-back],
     (dVds - delta)[front:-back],
     markers[2], lw=line_width, label="crank")


## Rannacher smoothing to damp oscilations at the discontinuity
V = impl(Vi,0.5*dt,4)
V = crank(V,dt, int(t/dt)-2)
V = V[ids][:,0]
print V[idx] - bs[idx]
dVds = center_diff(V)/(ds)
plot((np.exp(xs)/k*100)[ids][front:-back],
    (V - bs)[front:-back],
    markers[3], lw=line_width, label="smooth")


# shift = 0.5
# Vi2 = init(xs+shift*dx, [1], k)
# bs, delta = [x[ids] for x in bs_call(np.exp(xs+shift*dx), k, r, np.sqrt(v), t)]
# V = crank(Vi2, dt, int(t/dt))
# V = V[ids][:,0]
# print V[idx] - bs[idx]
# dVds = center_diff(V)/(ds)
# plot((np.exp(xs)/k*100)[ids][front:-back],
     # (dVds - delta)[front:-back],
     # '--', label="cr-shft+")

# shift = -0.5
# Vi2 = init(xs+shift*dx, [1], k)
# bs, delta = [x[ids] for x in bs_call(np.exp(xs+shift*dx), k, r, np.sqrt(v), t)]
# V = crank(Vi2, dt, int(t/dt))
# V = V[ids][:,0]
# print V[idx] - bs[idx]
# dVds = center_diff(V)/(ds)
# plot((np.exp(xs)/k*100)[ids][front:-back],
     # (dVds - delta)[front:-back],
     # '--', label="cr-shft-")



title("Error in Price")
xlabel("% of strike")
ylabel("Error")
legend(loc=0)
show()
