#!/usr/bin/env python
"""Demonstration of 1D Black Scholes using FTCS BTCS CTCS and Smoothed CTCS."""

# Boundary to S-k not S
# look at mismatch for very high vol
# consider the case of high d1 and d2


import sys
import numpy as np
import scipy.stats
import scipy.linalg as spl
import scipy.sparse as sps
from pylab import *
from utils import *
from visualize import *
from itertools2 import *
import heston
# def main():
    # global ids, idx, idv, spots, dss, vars, dvs, fst, snd, Vi, L1, L2, R1, R2
    # global Ass, mu_s, gamma2_s, G, V, spot, k, r, t, v, dt, nspots, nvols, Rs, Rss
    # global Rv, Rvv, Av, Avv, As, hs, bs, COSboundary


x = np.log(100)
spot = 100.0
k = 100.0
r = 0.06
t = 1.0
v0 = 0.2**2
dt = 1/30.0
nspots = 20000
nspots += not (nspots%2)

kappa = 1e-4
theta = v0
sigma = 1e-4
nvols = 40
rho = 0

spotdensity = 0.  # 0 is linear
varexp = 1        # 1 is linear

if len(sys.argv) > 1:
    spotdensity = float(sys.argv[1])
    print spotdensity

vars = linspace(0.2,10,3)
nvols = len(vars)
idv = isclose(vars, v0) # The actual vol we care about
dvs = np.hstack((nan, np.diff(vars)))

xs = np.linspace(-1, 1, nspots)
xs = 3*np.sqrt(max(vars))*xs*x + x
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
bs, delta = heston.bs_call_delta(spots[:,newaxis], k, r, np.sqrt(vars)[newaxis,:], t)
hs = heston.hs_call(spots, k, r, np.sqrt(vars), t, kappa, theta, sigma, rho, HFUNC=heston.HestonCos)

L1 = [0]*nvols
R1 = [0]*nvols
fst, snd = nonuniform_center_coefficients(dxs.copy())
print "Building As(s)"
sys.stdout.flush()
for j, v in enumerate(vars):
    mu_s = (r-0.5*v)*ones(nspots)
    gamma2_s = 0.5*v*ones(nspots)
    # mu_s = r*spots
    # gamma2_s = 0.5*v*spots**2

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
    L1[j].data[1,:] -= r #TODO Should be 0.5*r
    # L1[j].data[1,:] -= 0.5*r

    R1[j] = (Rs + Rss)

L2 = [0]*nspots
R2 = [0]*nspots
fst, snd = nonuniform_center_coefficients(dvs.copy())
print "Building Av(v)\n"
sys.stdout.flush()
for i, s in enumerate(spots):
    mu_v = kappa*(theta - vars)
    gamma2_v = 0.5*sigma**2*vars

    Av = sps.dia_matrix((fst.copy(), (1, 0, -1)), shape=(nvols,nvols))
    Av.data[0, 1] = -1 / dvs[1]
    Av.data[1, 0] =  1 / dvs[1]
    # Av.data[0, 1:]  =  1 / dvs[1:]
    # Av.data[1,:-1]  = -1 / dvs[1:]
    # Av.data[2,:]   *= 0
    # Av.data[2,:-1] *= mu_v[1:]

    Av.data[1:-1] = -1  # This is to cancel out the previous value so we can
                        # set the dirichlet boundary condition in R.
                        # Then we have U_i + -U_i + R

    Av.data[0,1:]  *= mu_v[:-1]
    Av.data[1,:]   *= mu_v
    Av.data[2,:-1] *= mu_v[1:]

    Rv = np.zeros_like(V[i,:])
    Rv[-1] = s - k
    Rv *= mu_v

    Avv = sps.dia_matrix((snd.copy(), (1, 0, -1)), shape=(nvols,nvols))
    Avv.data[0, 1] =  2/dvs[1]**2
    Avv.data[1, 0] = -2/dvs[1]**2

    Avv.data[0, 1:]  *= gamma2_v[:-1]
    Avv.data[1, :]   *= gamma2_v
    Avv.data[2, :-1] *= gamma2_v[1:]


    Rvv = np.zeros_like(V[i,:])
    Rvv[0] = 2*dvs[1]/dvs[1]**2
    Rvv *= gamma2_v

    # L2 = (Av + Avv - 1/2*r*np.eye(nvols))*dt + np.eye(nvols)
    # We have to save the dt though, to generalize our time stepping functions
    L2[i] = Av.copy()
    L2[i].data += Avv.data
    L2[i].data[1,:] -= 0.5*r

    R2[i] = (Rv + Rvv)


def impl(V, L1, R1, L2, R2, dt, n, crumbs=None, callback=None):
    V = V.copy()
    if len(V.shape) == 1:
        V = V[:,newaxis]
    L1i = [x.copy() for x in L1]
    R1  = [x.copy() for x in R1]
    # L2i = [x.copy() for x in L2]
    # R2  = [x.copy() for x in R2]
    for j in xrange(len(L1)):
        L1i[j].data *= -dt
        L1i[j].data[1,:] += 1
        R1[j] *= dt
    # for i in xrange(len(L2)):
        # L2i[i].data *= -dt
        # L2i[i].data[1,:] += 1
        # R2[i] *= dt
    for k in xrange(n):
        # if callback is not None:
            # callback(V, k*dt)
        for j in xrange(V.shape[1]):
            V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j])
        # for i in xrange(V.shape[0]):
            # V[i,:] = spl.solve_banded((1,1), L2i[i].data, V[i,:] + R2[i])
        if crumbs is not None:
            crumbs.append(V.copy())
    if crumbs is not None:
        return crumbs
    return V

def crank(V, L1, R1, L2, R2, dt, n, crumbs=None, callback=None):
    dt *= 0.5
    V = V.copy()
    if len(V.shape) == 1:
        V = V[:,newaxis]
    L1  = [x.copy() for x in L1]
    L1i = [x.copy() for x in L1]
    R1  = [x.copy() for x in R1]
    L2i = [x.copy() for x in L2]
    R2  = [x.copy() for x in R2]

    for j in xrange(len(L1)):
        L1[j].data *= dt
        L1[j].data[1,:] += 1
        L1i[j].data *= -dt
        L1i[j].data[1,:] += 1
        R1[j] *= dt
    for i in xrange(len(L2)):
        L2[j].data *= dt
        L2[j].data[1,:] += 1
        L2i[i].data *= -dt
        L2i[i].data[1,:] += 1
        R2[i] *= dt

    for k in xrange(n):
        # if callback is not None:
            # callback(V, k*dt)
        for j in xrange(V.shape[1]):
            V[:,j] = L1[j].dot(V[:,j]) + R1[j]
        # for i in xrange(V.shape[0]):
            # V[i,:] = spl.solve_banded((1,1), L2i[i].data, V[i,:] + R2[i])
        # for i in xrange(V.shape[0]):
            # V[i,:] = L2[i].dot(V[i,:]) + R2[i]
        for j in xrange(V.shape[1]):
            V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j])
        if crumbs is not None:
            crumbs.append(V.copy())
    if crumbs is not None:
        return crumbs
    return V



# Trim for plotting
front = 1
back = 1

V = impl(Vi, L1, R1, L2, R2, dt, int(t/dt))
# V = impl(Vi, L1, R1, L2, R2, dt, int(t/dt), [], callback=COSboundary)
# G = impl(Vi, L1, R1, L2, R2, dt, int(t/dt))
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
# plot_it(G, "impl", bs)
# plot_it(V[-1], "impl - forced")
# plot_price_err(hs, "Heston - Black", spots/k*100, bs, delta, ids, vars)
# print "Max analytical err vs BS: ", max(abs(hs.flatten() - bs.flatten()))
# print "Max FD err (forced): ", max(abs(V[-1].flatten() - hs.flatten()))
print "Max FD err: ", max(abs(V.flatten() - bs.flatten()))


# ## Rannacher smoothing to damp oscilations at the discontinuity
V = impl(Vi, L1, R1, None, None, 0.5*dt, 4)
V = crank(V, L1, R1, L2, R2, dt, int(t/dt)-2)
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

# def BSboundary(V,t):
    # m = heston.bs_call_delta(maximum(spots,0.0001), k, r, maximum(sqrt(vars),0.0001), t)[0]
    # V[0,:] = m[0,:] # top
    # V[:,0] = m[:,0] # left
    # V[-1,:] = m[-1,:] # bottom
    # V[:,-1] = m[:,-1] # right


# def COSboundary(V, t):
    # m = heston.hs_call(spots, k, r, sqrt(vars), t, kappa, theta, sigma, rho,
                       # HFUNC=heston.HestonCos)
    # V[0,:] = m[0,:] # top
    # V[:,0] = m[:,0] # left
    # V[-1,:] = m[-1,:] # bottom
    # V[:,-1] = m[:,-1] # right

# def d(s, v, t=1,sigma=0.00001):
    # h = heston.hs_call(s, 99, .06, v, float(t), 1, None, sigma, 0)
    # b = heston.bs_call_delta(s, 99, .06, v, float(t))[0]
    # return (h,b,h-b)

# def bs_stream(s, k, r, vol, dt):
    # for i in it.count():
        # yield heston.bs_call_delta(s, k, r, vol, i*dt)[0]



# def expl(V, L1, R1, L2, R2, dt, n, crumbs=None):
    # # dt /= 2
    # V = V.copy()
    # L1 = [x.copy() for x in L1]
    # R1 = [x.copy() for x in R1]
    # L2 = [x.copy() for x in L2]
    # R2 = [x.copy() for x in R2]
    # for j, l in enumerate(L1):
        # L1[j].data *= dt
        # L1[j].data[1,:] += 1
        # R1[j] *= dt
    # for i, l in enumerate(L2):
        # L2[i].data *= dt
        # L2[i].data[1,:] += 1
        # R2[i] *= dt
    # if crumbs is not None:
        # crumbs.append(V.copy())
    # for k in xrange(n):
        # for j in xrange(V.shape[1]):
                # V[:,j] = L1[j].dot(V[:,j]) + R1[j]
        # for i in xrange(V.shape[0]):
                # V[i,:] = L2[i].dot(V[i,:]) + R2[i]
        # if crumbs is not None:
            # crumbs.append(V.copy())
    # if crumbs is not None:
        # return crumbs
    # return V

# spots = linspace(0, 200, nspots)
# # spot = spots[8]
# dss = np.hstack((nan, np.diff(spots)))
# ids = (spot*0.8 < spots) & (spots < spot*1.2) # Just the range we care about
# ids = (0 < spots) & (spots < 400) # Just the range we care about
# idvs = (0 < vars) & (vars < 1) # Just the range we care about
# idx = isclose(spots, spot) # The actual spot we care about


# def plot_it(domain, label, analytical=None):
    # plot_price(domain, spots, k, vars, label)
    # if analytical is not None:
        # print domain[idx,idv] - analytical[idx,idv]
    # # plot_delta_err(domain, label, spots/k*100, analytical, delta, ids)
        # plot_price_err(domain, spots, k, vars, analytical, label=label)
    # # plot_price(domain[:,idvs], label, spots/k*100, ids, vars[idvs])
    # # plot_price(analytical, "Analytical Price", spots/k*100, None, vars)

# V = expl(Vi, L1, R1, L2, R2, dt, int(t/dt))
# if not isnan(V).any():
    # plot_it(V,"exp")
