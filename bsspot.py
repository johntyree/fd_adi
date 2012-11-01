#!/usr/bin/env python
"""Demonstration of 1D Black Scholes using FTCS BTCS CTCS and Smoothed CTCS."""


import sys
import numpy as np
import scipy.stats
import scipy.linalg as spl
import scipy.sparse as sps
from pylab import *
from utils import sinh_space,center_diff,D,D2,nonuniform_center_coefficients
from visualize import fp,wireframe
from heston import bs_call_delta
from time import time


# TEMPORARY PARAMETERS
rate_Spot_Var = 0.0


spot = 100.0
k = 100.0
r = 0.06
t = 1
# v0 = 3
dt = 1/30.0
nspots = 20000
nspots += not (nspots%2)
# spots = linspace(0,1400,nspots)
spots = sinh_space(k, 20000, 1, nspots)
# plot(spots)
# title("Spots")
# show()
ds = center_diff(spots)
trims = (0 <= spots) & (spots <= 400)
trims = slice(None)
ids = isclose(spots[trims], spot)
ds = ds[nspots//2]
dss = np.hstack((np.nan, np.diff(spots)))

# kappa = 1e-4
# theta = v0
# sigma = 1e-4
# rho = 0
nvols = 3

spotdensity = 0.  # 0 is linear
varexp = 1        # 1 is linear
vars = linspace(1,2,nvols)
vars = array((1.,))
nvols = len(vars)
dvs = np.hstack((nan, np.diff(vars)))

def init(spots, vs, k):
    u = np.ones((len(spots),len(vs))).T * spots
    u = u.T
    return np.maximum(0, u-k)


Vi = init(spots, vars, k)
V = np.copy(Vi)
bs, delta = [x[trims] for x in bs_call_delta(spots[:,newaxis], k, r,
                                            np.sqrt(vars)[newaxis,:], t)]

L1_ = [0]*nvols
R1_ = [0]*nvols
start = time()
print "Building As(s)",
sys.stdout.flush()
fst, snd = nonuniform_center_coefficients(dss)
for j, v in enumerate(vars):
    mu_s = r*spots
    gamma2_s = 0.5*v*spots**2


    As = sps.dia_matrix((fst.copy(), (1, 0, -1)), shape=(nspots,nspots))

    Rs = np.zeros(nspots)
    Rs[-1] = 1

    As.data[0,1:]  *= mu_s[:-1]
    As.data[1,:]   *= mu_s
    As.data[2,:-1] *= mu_s[1:]
    Rs *= mu_s

    Ass = sps.dia_matrix((snd.copy(), (1, 0, -1)), shape=(nspots,nspots))
    Ass.data[1, -1] = -2/dss[-1]**2
    Ass.data[2, -2] =  2/dss[-1]**2

    Rss = np.zeros(nspots)
    Rss[-1] = 2*dss[-1]/dss[-1]**2

    Ass.data[0,1:]  *= gamma2_s[:-1]
    Ass.data[1,:]   *= gamma2_s
    Ass.data[2,:-1] *= gamma2_s[1:]
    Rss *= gamma2_s

    L1_[j] = As.copy()
    L1_[j].data += Ass.data
    L1_[j].data[1,:] -= (1 - rate_Spot_Var)*r

    R1_[j] = (Rs + Rss)
print time() - start

# mu_v = kappa*(theta - vars)
# gamma2_v = 0.5*sigma**2*vars

# L2_ = [0]*nspots
# R2_ = [0]*nspots
# fst, snd = nonuniform_center_coefficients(dvs)
# start = time()
# print "Building Av(v)",
# sys.stdout.flush()
# for i, s in enumerate(spots):
    # Av = sps.dia_matrix((fst.copy(), (1, 0, -1)), shape=(nvols,nvols))
    # Av *= 0
    # # Av.data[0, 1] = -1 / dvs[1]
    # # Av.data[1, 0] =  1 / dvs[1]
    # # # Av.data[0, 1:]  =  1 / dvs[1:]
    # # # Av.data[1,:-1]  = -1 / dvs[1:]
    # # # Av.data[2,:]   *= 0
    # # # Av.data[2,:-1] *= mu_v[1:]

    # # Av.data[1:-1] = -1  # This is to cancel out the previous value so we can
                        # # # set the dirichlet boundary condition in R.
                        # # # Then we have U_i + -U_i + R

    # # Av.data[0,1:]  *= mu_v[:-1]
    # # Av.data[1,:]   *= mu_v
    # # Av.data[2,:-1] *= mu_v[1:]

    # Rv = np.zeros(nvols)
    # # Rv[-1] = s - k
    # # Rv *= mu_v

    # Avv = sps.dia_matrix((snd.copy(), (1, 0, -1)), shape=(nvols,nvols))
    # Avv *= 0
    # # Avv.data[0, 1] =  2/dvs[1]**2
    # # Avv.data[1, 0] = -2/dvs[1]**2

    # # Avv.data[0, 1:]  *= gamma2_v[:-1]
    # # Avv.data[1, :]   *= gamma2_v
    # # Avv.data[2, :-1] *= gamma2_v[1:]


    # Rvv = np.zeros(nvols)
    # # Rvv[0] = 2*dvs[1]/dvs[1]**2
    # # Rvv *= gamma2_v

    # L2_[i] = Av.copy()
    # L2_[i].data += Avv.data
    # L2_[i].data[1,:] -= rate_Spot_Var*r

    # R2_[i] = (Rv + Rvv)
# print time() - start

def impl(V,dt,n):
    V = V.copy()
    L1i = [x.copy() for x in L1_]
    R1  = [x.copy() for x in R1_]
    # L2i = [x.copy() for x in L2_]
    # R2  = [x.copy() for x in R2_]

    # L  = (As + Ass - r*np.eye(nspots))*-dt + np.eye(nspots)
    for j in xrange(nvols):
        L1i[j].data *= -dt
        L1i[j].data[1,:] += 1
        R1[j] *= dt
    # for i in xrange(nspots):
        # L2i[i].data *= -dt
        # L2i[i].data[1,:] += 1
        # R2[i] *= dt

    start = time()
    print_step = max(1, int(n / 10))
    to_percent = 100.0/n
    print "Impl:",
    for k in xrange(n):
        if not k % print_step:
            print int(k*to_percent),
        for j in xrange(nvols):
            V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j], overwrite_b=True)
        # for i in xrange(len(R2)):
            # V[i,:] = spl.solve_banded((1,1), L2i[i].data, V[i,:] + R2[i], overwrite_b=True)
    print "  (%fs)" % (time() - start)
    return V

def crank(V,dt,n):
    V = V.copy()
    dt *= 0.5

    L1e = [x.copy() for x in L1_]
    L1i = [x.copy() for x in L1_]
    R1  = [x.copy() for x in R1_]
    # L2e = [x.copy() for x in L2_]
    # L2i = [x.copy() for x in L2_]
    # R2  = [x.copy() for x in R2_]

    for j in xrange(nvols):
        L1e[j].data *= dt
        L1e[j].data[1,:] += 1
        R1[j]  = (Rs + Rss)*dt
        L1i[j].data *= -dt
        L1i[j].data[1,:] += 1

    # for i in xrange(nspots):
        # L2e[i].data *= dt
        # L2e[i].data[1,:] += 1
        # R2[i] *= dt
        # L2i[i].data *= -dt
        # L2i[i].data[1,:] += 1


    start = time()
    print_step = max(1, int(n / 10))
    to_percent = 100.0/n
    print "Crank:",
    for k in xrange(n):
        if not k % print_step:
            print int(k*to_percent),
        for j in xrange(nvols):
            V[:,j] = L1e[j].dot(V[:,j]) + R1[j]
        # for i in xrange(len(R2)):
            # V[i,:] = spl.solve_banded((1,1), L2i[i].data, V[i,:] + R2[i], overwrite_b=True)
            # V[i,:] = L2e[i].dot(V[i,:]) + R2[i]
        for j in xrange(nvols):
            V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j], overwrite_b=True)
    print "  (%fs)" % (time() - start)
    return V


# Trim for plotting
front = 1
back = 1
line_width = 2
          # exp  imp   cr   smo
markers = ['--', '--', ':', ':']

def p1(V, spots, vars, marker_idx, label):
    plot((spots/k*100)[trims][front:-back],
         (V-bs)[front:-back],
         markers[marker_idx], lw=line_width, label=label)
    title("Error in Price")
    xlabel("% of strike")
    ylabel("Error")
    legend(loc=0)

def p2(V, spots, vars, marker_idx, label):
    wireframe(V-bs, spots, vars)
    title("Error in Price")
    xlabel("Var")
    ylabel("% of strike")
    show()


p = p1

# V = impl(Vi,dt, int(t/dt))
# V = V[trims,:]
# print V[ids] - bs[ids]
# dVds = center_diff(V)/(ds)
# p(V, spots, vars, 1, "impl")

V = crank(Vi,dt, int(t/dt))
V = V[trims,:]
print V[ids] - bs[ids]
dVds = center_diff(V)/(ds)
p(V, spots, vars, 2, "crank")

## Rannacher smoothing to damp oscilations at the discontinuity
V = impl(Vi, 0.5*dt, 4)
V = crank(V, dt, int(t/dt)-2)
V = V[trims,:]
print V[ids] - bs[ids]
p(V, spots, vars, 3, "smooth")

if p is p1:
    show()
