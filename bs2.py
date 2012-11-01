#!/usr/bin/env python
"""Demonstration of 1D Black Scholes using FTCS BTCS CTCS and Smoothed CTCS."""

import scipy.stats
import scipy.linalg as spl
import scipy.sparse as sps
from pylab import *
from utils import *
from visualize import *
from itertools2 import *
from heston import bs_call_delta

x = np.log(100)
spot = 100.0
k = 100.0
r = 0.06
t = 1.0
v0 = 0.2**2
dt = 1/100.0
nspots = 20000
nspots += not (nspots%2)

vars = linspace(0.2,10,3)
nvols = len(vars)
idv = isclose(vars, v0) # The actual vol we care about

xs = linspace(-1, 1, nspots)
xs = 3*sqrt(max(vars))*xs*x + x
# xs = sinh_space(x, 3*np.sqrt(v0)*x + x, 0.93, nspots)
# xs = sinh_space(x, log(200), 1., nspots)
spots = np.exp(xs)
dxs = np.hstack((nan, np.diff(xs)))
ids = (0 < np.exp(xs)) & (np.exp(xs) < 1200)
idx = isclose(xs, x)
# dx = dxs[1]

def init(spots, vars, k):
    u = np.ones((len(spots),len(vars))).T * spots
    u = u.T
    return np.maximum(0, u-k)

Vi = init(spots, vars, k)
V = Vi.copy()
bs, delta = bs_call_delta(spots[:,newaxis], k, r, np.sqrt(vars)[newaxis,:], t)


L1 = [0]*nvols
R1 = [0]*nvols
fst, snd = nonuniform_center_coefficients(dxs.copy())
for j, v in enumerate(vars):
    mu_s = (r-0.5*v)*ones(nspots)
    gamma2_s = 0.5*v*ones(nspots)



    As = sps.dia_matrix((fst.copy(), (1, 0, -1)), shape=(nspots,nspots))
    As.data[0,1:]  *= mu_s[:-1]
    As.data[1,:]   *= mu_s
    As.data[2,:-1] *= mu_s[1:]

    Rs = np.zeros_like(V[:,j])
    Rs[-1] = 1
    Rs *= mu_s

    Ass = sps.dia_matrix((snd.copy(), (1, 0, -1)), shape=(nspots,nspots))
    Ass.data[1, -1] = -2/dxs[-1]**2  # Last elem of center diag
    Ass.data[2, -2] =  2/dxs[-1]**2  # last elem of sub diag
    Ass.data[0,1:]  *= gamma2_s[:-1]
    Ass.data[1,:]   *= gamma2_s
    Ass.data[2,:-1] *= gamma2_s[1:]

    Rss = np.zeros_like(V[:,j])
    Rss[-1] = 2*dxs[-1]/dxs[-1]**2
    Rss *= gamma2_s

    # L1 = (As + Ass - 1/2*r*np.eye(nspots))*dt + np.eye(nspots)
    # We have to save the dt though, to generalize our time stepping functions
    L1[j] = As.copy()
    L1[j].data += Ass.data
    L1[j].data[1,:] -= r

    R1[j] = (Rs + Rss)

def impl(V, L1, R1, L2, R2, dt, n, crumbs=None, callback=None):
    V = V.copy()
    if len(V.shape) == 1:
        V = V[:,newaxis]
    L1i = [x.copy() for x in L1]
    R1  = [x.copy() for x in R1]
    for j in xrange(len(L1)):
        L1i[j].data *= -dt
        L1i[j].data[1,:] += 1
        R1[j] *= dt
    for k in xrange(n):
        for j in xrange(V.shape[1]):
            V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j])
    return V

def crank(V, L1, R1, L2, R2, dt, n, crumbs=None, callback=None):
    dt *= 0.5
    V = V.copy()
    if len(V.shape) == 1:
        V = V[:,newaxis]
    L1  = [x.copy() for x in L1]
    L1i = [x.copy() for x in L1]
    R1  = [x.copy() for x in R1]

    for j in xrange(len(L1)):
        L1[j].data *= dt
        L1[j].data[1,:] += 1
        L1i[j].data *= -dt
        L1i[j].data[1,:] += 1
        R1[j] *= dt

    for k in xrange(n):
        for j in xrange(V.shape[1]):
            V[:,j] = L1[j].dot(V[:,j]) + R1[j]
            V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j])
    return V



# Trim for plotting
front = 1
back = 1

V = impl(Vi, L1, R1, None, None, dt, int(t/dt))
print V[idx,idv] - bs[idx,idv]
# dVds = center_diff(V)/(ds)
if 0:
    for j,v in enumerate(vars):
        Vc = V[:,j]
        bsc = bs[:,j]
        plot((np.exp(xs)/k*100)[ids][front:-back],
             (Vc-bsc)[ids][front:-back],
             label="impl")
else:
    wireframe((V-bs)[ids,:][front:-back,:], xs[ids][front:-back], sqrt(vars))
    wireframe((V-bs)[ids,:][front:-back,:], spots[ids][front:-back], sqrt(vars))
title("Error in Price")
ylabel("% of strike")
xlabel("Vol")
legend(loc=0)
show()



## Rannacher smoothing to damp oscilations at the discontinuity
# V = impl(Vi, L1, R1, None, None, 0.5*dt, 4)
# V = crank(V, L1, R1, None, None, dt, int(t/dt)-2)
V = crank(V, L1, R1, None, None, dt, int(t/dt))
print V[idx] - bs[idx]
if 0:
    for j,v in enumerate(vars):
        Vc = V[:,j]
        bsc = bs[:,j]
        plot((np.exp(xs)/k*100)[ids][front:-back],
             (Vc-bsc)[ids][front:-back], label=("Vol %.2f" % v))
    title("Error in Price")
    xlabel("% of strike")
else:
    wireframe((V-bs)[ids,:][front:-back,:], xs[ids][front:-back], sqrt(vars))
    wireframe((V-bs)[ids,:][front:-back,:], spots[ids][front:-back], sqrt(vars))
    title("Error in Price")
    ylabel("% of strike")
    xlabel("Vol")
legend(loc=0)
show()

