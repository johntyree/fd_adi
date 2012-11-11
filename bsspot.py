#!/usr/bin/env python
"""Demonstration of 2D Heston using BTCS CTCS and Smoothed CTCS."""

# Defaults: spot = k = 100
#           v0 = 0.2
#           dt = 1/100.0
#           kappa = 1
#           theta = v0
#           sigma = 0.2
#           rho = 0
#           nspots = 200
#           nvols = 200

# Pretty much everything goes to infinity at high vol boundary. Trimming shows
# the interesting area looks reasonable.

# High theta oscillates wildly at (vol < v0)
# theta = 3.2

# For low theta, Analytical has problems at low vol and spot (FD ok)
# theta = 0.01

# At high sigma, FD explodes around strike at low vol. Analytical jagged but
# better.
# sigma = 3
# Still well behaved at sigma = 1

# Crank explodes around strike at low vol if nvols is low. Implicit ok.
# nvols = 40
# sigma = 1
# With sigma at 0.2 again, both explode.

# Well behaved when nspots is low
# nspots = 40

# Crank has small oscillations when dt is exactly 1/10.0. Implicit is way way too large. O(5000)
# dt = 1/10.0
# At other large dt, everything is ok again.
# dt = 1/2.0 ... 1/8.0 .. 1/11.0.. etc

# At small dt, impl and crank are both ok, but with big dip at strike and low
# vol. Err O(0.05)

import sys
import numpy as np
import scipy.stats
import scipy.linalg as spl
import scipy.sparse as sps
from pylab import *
import utils
from visualize import fp, wireframe, surface, anim
from heston import bs_call_delta, hs_call
from time import time

# ion()


# TEMPORARY PARAMETERS
rate_Spot_Var = 0.5


spot = 80.0
k = 100.0
r = 0.06
t = 1
v0 = 0.2
dt = 1/10.0

kappa = 1
theta = 0.02
sigma = 0.2
rho = 0

nspots = 100
nvols = 400

spotdensity = 100.0  # infinity is linear?
varexp = 4
spots = utils.sinh_space(k, 2000, spotdensity, nspots)
spot = spots[min(abs(spots - spot)) == abs(spots - spot)][0]
k = spots[min(abs(spots - k)) == abs(spots - k)][0]
vars = utils.exponential_space(0.00, v0, 10., varexp, nvols)
# plot(spots); title("Spots"); show()
# plot(vars); title("Vars"); show()

trims = (0 < spots) & (spots < k*2.0)
trimv = (v0*0.1 < vars) & (vars < v0*2.0)
# trims = slice(None)
# trimv = slice(None)

tr = lambda x: x[trims,:][:,trimv]

ids = isclose(spots[trims], spot)
idv = isclose(vars[trimv], v0)
dss = np.hstack((np.nan, np.diff(spots)))
dvs = np.hstack((nan, np.diff(vars)))
# flip_idx_var = None

BADANALYTICAL = False

def init(spots, nvols, k):
    return tile(np.maximum(0,spots-k), (nvols,1)).T


Vi = init(spots, nvols, k)
V = np.copy(Vi)
bs, delta = [x for x in bs_call_delta(spots[:,newaxis], k, r,
                                            np.sqrt(vars)[newaxis,:], t)]
utils.tic("Heston Analytical:")
# hss = array([hs_call(spots, k, r, np.sqrt(vars),
             # dt*i, kappa, theta, sigma, rho) for i in range(int(t/dt)+1)])
# hs = hss[-1]
hs = hs_call(spots, k, r, np.sqrt(vars),
             t, kappa, theta, sigma, rho)
utils.toc()
hs[isnan(hs)] = 0.0
if max(hs.flat) > spots[-1]*2:
    BADANALYTICAL = True
    print "Warning: Analytical solution looks like trash."

if len(sys.argv) > 1:
    if sys.argv[1] == '0':
        print "Bail out with arg 0."
        sys.exit()

L1_ = []
R1_ = []
utils.tic("Building As(s):")
sys.stdout.flush()
# As_, Ass_ = utils.nonuniform_center_coefficients(dss)
As_, Ass_ = utils.nonuniform_complete_coefficients(dss)
assert(not isnan(As_.data).any())
assert(not isnan(Ass_.data).any())
for j, v in enumerate(vars):
    # Be careful not to overwrite our operators
    As, Ass = As_.copy(), Ass_.copy()
    m = 2

    mu_s = r*spots
    gamma2_s = 0.5*v*spots**2

    Rs = np.zeros(nspots)
    Rs[-1] = 1


    As.data[m-2, 2:]  *= mu_s[:-2]
    As.data[m-1, 1:]  *= mu_s[:-1]
    As.data[m, :]     *= mu_s
    As.data[m+1, :-1] *= mu_s[1:]
    As.data[m+2, :-2] *= mu_s[2:]

    Rs *= mu_s

    Rss = np.zeros(nspots)
    Rss[-1] = 2*dss[-1]/dss[-1]**2


    Ass.data[m,   -1] = -2/dss[-1]**2
    Ass.data[m+1, -2] =  2/dss[-1]**2

    Ass.data[m-2, 2:]  *= gamma2_s[:-2]
    Ass.data[m-1, 1:]  *= gamma2_s[:-1]
    Ass.data[m, :]     *= gamma2_s
    Ass.data[m+1, :-1] *= gamma2_s[1:]
    Ass.data[m+2, :-2] *= gamma2_s[2:]

    Rss *= gamma2_s

    L1_.append(As.copy())
    L1_[j].data += Ass.data
    L1_[j].data[m,:] -= (1 - rate_Spot_Var)*r

    R1_.append((Rs + Rss).copy())
utils.toc()

mu_v = kappa*(theta - vars)
gamma2_v = 0.5*sigma**2*vars

L2_ = []
R2_ = []
downwind_from = min(find(vars > theta))
# downwind_from = None
print "Downwind from:", downwind_from
utils.tic("Building Av(v):")
# Avc_, Avvc_ = utils.nonuniform_center_coefficients(dvs)
Av_, Avv_ = utils.nonuniform_complete_coefficients(dvs, downwind_from=downwind_from)
assert(not isnan(Av_.data).any())
assert(not isnan(Avv_.data).any())
for i, s in enumerate(spots):
    # Be careful not to overwrite our operators
    Av, Avv = Av_.copy(), Avv_.copy()

    m = 2

    Av.data[m, 0]   = -1 / dvs[1]
    Av.data[m-1, 1] =  1 / dvs[1]

    Av.data[m-2, 2:]  *= mu_v[:-2]
    Av.data[m-1, 1:]  *= mu_v[:-1]
    Av.data[m, :]     *= mu_v
    Av.data[m+1, :-1] *= mu_v[1:]
    Av.data[m+2, :-2] *= mu_v[2:]

    Av.data[m, -1] = -1  # This is to cancel out the previous value so we can
                          # set the dirichlet boundary condition using R.
                          # Then we have U_i + -U_i + R

    Rv = np.zeros(nvols)
    Rv *= mu_v
    Rv[-1] = s-k

    Avv.data[m, 0]   = -2/dvs[1]**2
    Avv.data[m-1, 1] =  2/dvs[1]**2

    Avv.data[m-2, 2:]  *= gamma2_v[:-2]
    Avv.data[m-1, 1:]  *= gamma2_v[:-1]
    Avv.data[m, :]     *= gamma2_v
    Avv.data[m+1, :-1] *= gamma2_v[1:]
    Avv.data[m+2, :-2] *= gamma2_v[2:]


    Rvv = np.zeros(nvols)
    Rvv[0] = 2*dvs[1]/dvs[1]**2
    Rvv *= gamma2_v

    L2_.append(Av.copy())
    L2_[i].data += Avv.data
    L2_[i].data[m,:] -= rate_Spot_Var*r

    R2_.append(Rv + Rvv)
utils.toc()

def impl(V,L1,R1x,L2,R2x,dt,n):
    V = V.copy()
    L1i = [x.copy() for x in L1]
    R1  = [x.copy() for x in R1x]
    L2i = [x.copy() for x in L2]
    R2  = [x.copy() for x in R2x]

    m = 2

    # L  = (As + Ass - r*np.eye(nspots))*-dt + np.eye(nspots)
    for j in xrange(nvols):
        L1i[j].data *= -dt
        L1i[j].data[m,:] += 1
        R1[j] *= dt
    for i in xrange(nspots):
        L2i[i].data *= -dt
        L2i[i].data[m,:] += 1
        R2[i] *= dt

    print_step = max(1, int(n / 10))
    to_percent = 100.0/n
    utils.tic("Impl:")
    for k in xrange(n):
        if not k % print_step:
            if isnan(V).any():
                print "Impl fail @ t = %f (%i steps)" % (dt*k, k)
                return zeros_like(V)
            print int(k*to_percent),
        for j in xrange(nvols):
            # V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j], overwrite_b=True)
            V[:,j] = spl.solve_banded((abs(min(L1i[j].offsets)),abs(max(L1i[j].offsets))),
                                      L1i[j].data, V[:,j] + R1[j], overwrite_b=True)
        for i in xrange(nspots):
            # V[i,:] = spl.solve_banded((1,1),
                                      # L2i[i].data, V[i,:] + R2[i], overwrite_b=True)
            # V[i,:] = spl.solve_banded((1,1),
                                      # L2i[i].data[1:4,:], V[i,:] + R2[i], overwrite_b=True)
            V[i,:] = spl.solve_banded((abs(min(L2i[i].offsets)),abs(max(L2i[i].offsets))),
                                      L2i[i].data, V[i,:] + R2[i], overwrite_b=True)
    utils.toc()
    return V

def crank(V,L1,R1x,L2,R2x,dt,n):
    V = V.copy()
    dt *= 0.5

    L1e = [x.copy() for x in L1]
    L1i = [x.copy() for x in L1]
    R1  = [x.copy() for x in R1x]
    L2e = [x.copy() for x in L2]
    L2i = [x.copy() for x in L2]
    R2  = [x.copy() for x in R2x]

    for j in xrange(nvols):
        L1e[j].data *= dt
        L1e[j].data[1,:] += 1
        R1[j]  *= dt
        L1i[j].data *= -dt
        L1i[j].data[1,:] += 1

    for i in xrange(nspots):
        L2e[i].data *= dt
        L2e[i].data[1,:] += 1
        R2[i] *= dt
        L2i[i].data *= -dt
        L2i[i].data[1,:] += 1


    print_step = max(1, int(n / 10))
    to_percent = 100.0/n
    utils.tic("Crank:")
    for k in xrange(n):
        if not k % print_step:
            if isnan(V).any():
                print "Crank fail @ t = %f (%i steps)" % (dt*k, k)
                return zeros_like(V)
            print int(k*to_percent),
        for j in xrange(nvols):
            V[:,j] = L1e[j].dot(V[:,j]) + R1[j]
        for i in xrange(len(R2)):
            V[i,:] = spl.solve_banded((2,2), L2i[i].data, V[i,:] + R2[i], overwrite_b=True)
            V[i,:] = L2e[i].dot(V[i,:]) + R2[i]
        for j in xrange(nvols):
            V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j], overwrite_b=True)
    utils.toc()
    return V


# Trim for plotting
front = 1
back = 1
line_width = 2
          # exp  imp   cr   smo
markers = ['--', '--', ':', '--']

def p1(V, analytical, spots, vars, marker_idx, label):
    if BADANALYTICAL:
        label += " - bad analytical!"
    plot((spots/k*100)[front:-back],
         (V-analytical)[front:-back],
         markers[marker_idx], lw=line_width, label=label)
    title("Error in Price")
    xlabel("% of strike")
    ylabel("Error")
    legend(loc=0)

def p2(V, analytical, spots, vars, marker_idx=0, label=""):
    surface(V-analytical, spots, vars)
    if BADANALYTICAL:
        label += " - bad analytical!"
    title("Error in Price (%s)" % label)
    xlabel("Var")
    ylabel("% of strike")
    show()

def p3(V, analytical, spots, vars, marker_idx=0, label=""):
    surface((V-analytical)/analytical, spots, vars)
    if BADANALYTICAL:
        label += " - bad analytical!"
    title("Relative Error in Price (%s)" % label)
    xlabel("Var")
    ylabel("% of strike")
    show()



p = p3
vis = lambda V=V, p=p2: p(V, hs, spots, vars, 0, "")
vis2 = lambda V=V, p=p2: p(tr(V), tr(hs), spots[trims], vars[trimv], 0, "")


# Vs = impl(Vi,L1_, R1_, L2_, R2_,
           # dt, int(t/dt)
          # , crumbs=[Vi]
          # # , callback=lambda v, t: force_boundary(v, hs)
         # )
# Vi = Vs[0]
# print tr(Vi)[ids,idv] - tr(hs)[ids,idv]

Vs = crank(Vi, L1_, R1_, L2_, R2_,
           dt, int(t/dt)
           , crumbs=[Vi]
           # , callback=lambda v, t: force_boundary(v, hs)
          )
Vc = Vs[-1]
print tr(Vc)[ids,idv] - tr(hs)[ids,idv]

## Rannacher smoothing to damp oscilations at the discontinuity
Vs = impl(Vi,L1_, R1_, L2_, R2_,
         dt, 4
         , crumbs=[Vi]
         # , callback=lambda v, t: force_boundary(v, hs)
        )
Vs.extend(crank(Vs[-1], L1_, R1_, L2_, R2_,
          dt, int(t/dt)-4
          , crumbs=[]
          # , callback=lambda v, t: force_boundary(v, hs)
         )
        )
Vr = Vs[-1]
print tr(Vr)[ids,idv] - tr(hs)[ids,idv]


# p(tr(Vi), tr(hs), spots[trims], vars[trimv], 1, "impl")
p(tr(Vc), tr(hs), spots[trims], vars[trimv], 2, "crank")
p(tr(Vr), tr(hs), spots[trims], vars[trimv], 3, "smooth")
# p(Vi, hs, spots, vars, 1, "impl")
p(Vc, hs, spots, vars, 1, "crank")
p(Vr, hs, spots, vars, 1, "smooth")



if p is p1:
    show()
