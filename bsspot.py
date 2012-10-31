#!/usr/bin/env python
"""Demonstration of 1D Black Scholes using FTCS BTCS CTCS and Smoothed CTCS."""


import numpy as np
import scipy.stats
import scipy.linalg as spl
import scipy.sparse as sps
from pylab import *
from visualize import fp
from utils import sinh_space,center_diff,D,D2,nonuniform_center_coefficients
from time import time

# x = np.log(100)
spot = 100
k = 100.
r = 0.06
t = 1
v = 0.2**2
dt = 1/10.
I = 2000
I += not (I%2)
# spots = linspace(0,1400,I)
spots = sinh_space(k, 4000, 100, I)
plot(spots)
title("Spots")
show()
ds = center_diff(spots)
trim = (0 <= spots) & (spots <= 400)
trim = slice(None)
ids = isclose(spots[trim], spot)
ds = ds[I//2]
dss = np.hstack((np.nan, np.diff(spots)))

def init(spots, vs, k):
    u = np.ones((len(spots),len(vs))).T * spots
    u = u.T
    return np.maximum(0, u-k)

def bs_call(s, k, r, sig, t):
    N = scipy.stats.distributions.norm.cdf
    d1 = (np.log(s/k) + (r+0.5*sig**2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig*np.sqrt(t)
    return (N(d1)*s - N(d2)*k*np.exp(-r * t), N(d1))

bs, delta = [x[trim] for x in bs_call(spots, k, r, np.sqrt(v), t)]

Vi = init(spots, [1], k)
V = np.copy(Vi)

mu_s = r*spots
gamma2_s = 0.5*v*spots**2
fst, snd = nonuniform_center_coefficients(dss)


# As = D(I)/ds
As = sps.dia_matrix((fst.copy(), (1, 0, -1)), shape=(I,I))

Rs = np.zeros_like(V[:,0])
Rs[-1] = 1

As.data[0,1:]  *= mu_s[:-1]
As.data[1,:]   *= mu_s
As.data[2,:-1] *= mu_s[1:]
Rs *= mu_s

# Ass = D2(I)/(ds**2)
Ass = sps.dia_matrix((snd.copy(), (1, 0, -1)), shape=(I,I))
Ass.data[1, -1] = -2/dss[-1]**2
Ass.data[2, -2] =  2/dss[-1]**2

Rss = np.zeros_like(V[:,0])
Rss[-1] = 2*dss[-1]/dss[-1]**2

Ass.data[0,1:]  *= gamma2_s[:-1]
Ass.data[1,:]   *= gamma2_s
Ass.data[2,:-1] *= gamma2_s[1:]
Rss *= gamma2_s

def expl(V,dt,n):
    factor = 1
    dt /= factor
    n  *= factor


    V = V.copy()[:,newaxis]

    # Le  = (As + Ass - r*np.eye(I))*dt + np.eye(I)
    Le = As.copy()
    Le.data += Ass.data
    Le.data[1,:] -= r
    Le.data *= dt
    Le.data[1,:] += 1

    R  = (Rs + Rss)*dt

    print "GO!"
    for i in xrange(n):
        if not i % (100*factor):
            if isnan(V).any():
                break
            print int(i*100.0/float(n)),
        V[:,0] = Le.dot(V[:,0]) + R
    return V[:,0]

def impl(V,dt,n):
    V = V.copy()

    # L  = (As + Ass - r*np.eye(I))*-dt + np.eye(I)
    Li = As.copy()
    Li.data += Ass.data
    Li.data[1,:] -= r
    Li.data *= -dt
    Li.data[1,:] += 1
    R  = (Rs + Rss)*dt
    start = time()
    for i in xrange(n):
        if not i % 100:
            print int(i*100.0/float(n)),
        V[:,0] = spl.solve_banded((1,1), Li.data, V[:,0] + R, overwrite_b=True)
    print
    print "Impl time:", time() - start
    return V

def crank(V,dt,n):
    global CLe, CLi, CAs, CAss, CRs, CRss, Cdt
    V = V.copy()
    dt *= 0.5
    # Le  = (As + Ass - r*np.eye(I))*dt + np.eye(I)
    Le = As.copy()
    Le.data += Ass.data
    Le.data[1,:] -= r
    Le.data *= dt
    Le.data[1,:] += 1

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
        V[:,0] = Le.dot(V[:,0]) + R
        V[:,0] = spl.solve_banded((1,1), Li.data, V[:,0] + R, overwrite_b=True)
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
# if not isnan(V).any():
    # V = V[trim][:,0]
    # print V[ids] - bs[ids]
    # dVds = center_diff(V)/(ds)
    # plot((spots/k*100)[trim][front:-back],
        # (V-bs)[front:-back], label="exp")


V = impl(Vi,dt, int(t/dt))
V = V[trim][:,0]
print V[ids] - bs[ids]
dVds = center_diff(V)/(ds)
plot((spots/k*100)[trim][front:-back],
     (V-bs)[front:-back],
     markers[1], lw=line_width, label="impl")


V = crank(Vi,dt, int(t/dt))
V = V[trim][:,0]
print V[ids] - bs[ids]
dVds = center_diff(V)/(ds)
plot((spots/k*100)[trim][front:-back],
     (V-bs)[front:-back],
     markers[2], lw=line_width, label="crank")


## Rannacher smoothing to damp oscilations at the discontinuity
V = impl(Vi, 0.5*dt, 4)
V = crank(V, dt, int(t/dt)-2)
V = V[trim][:,0]
print V[ids] - bs[ids]
dVds = center_diff(V,)/(ds)
plot((spots/k*100)[trim][front:-back],
    (V - bs)[front:-back].T,
    markers[3], lw=line_width, label="smooth")


# shift = 0.5
# Vi2 = init(xs+shift*ds, [1], k)
# bs, delta = [x[trim] for x in bs_call(np.exp(xs+shift*ds), k, r, np.sqrt(v), t)]
# V = crank(Vi2, dt, int(t/dt))
# V = V[trim][:,0]
# print V[ids] - bs[ids]
# dVds = center_diff(V)/(ds)
# plot((spots/k*100)[trim][front:-back],
     # (dVds - delta)[front:-back],
     # '--', label="cr-shft+")

# shift = -0.5
# Vi2 = init(xs+shift*ds, [1], k)
# bs, delta = [x[trim] for x in bs_call(np.exp(xs+shift*ds), k, r, np.sqrt(v), t)]
# V = crank(Vi2, dt, int(t/dt))
# V = V[trim][:,0]
# print V[ids] - bs[ids]
# dVds = center_diff(V)/(ds)
# plot((spots/k*100)[trim][front:-back],
     # (dVds - delta)[front:-back],
     # '--', label="cr-shft-")



title("Error in Price")
xlabel("% of strike")
ylabel("Error")
legend(loc=0)
show()
