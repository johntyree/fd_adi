#!/usr/bin/env python
"""Demonstration of 1D Black Scholes using FTCS BTCS CTCS and Smoothed CTCS."""


import sys
import numpy as np
import scipy.stats
import scipy.linalg as spl
import scipy.sparse as sps
import itertools as it
from pylab import *
from mpl_toolkits.mplot3d import axes3d
import heston
import time

def exponential_space(low, exact, high, ex, n):
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
    print dx
    h += (n-1) * dx/j
    dv = (h - l) / (n-1)
    for i in range(n):
        v[i] = l + pow(i*dv, ex)
    return v

def cubic_sigmoid_space(exact, high, density, n):
    if density == 0:
        return linspace(exact - (high - exact), high, n)

    y = zeros(n)
    dx = 1.0/(n-1)
    scale = (float(high)-exact)/(density**3 + density)
    for i in range(n):
        x = (2*(i*dx)-1)*density
        y[i] = exact + (x**3+x)*scale

    return y


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

def init(spots, vars, k):
    u = np.ones((len(spots),len(vars))).T * spots
    u = u.T
    return np.maximum(0, u-k)

def hs_call(ss, k, r, vs, t, kappa, theta, sigma, rho, density=None, varexp=None):
    fname = \
    'heston_grids/hs_%i_density_%f_%i_exp_%f_kappa_%f_theta_%f_sigma_%f_rho_%f.npy' % (
        len(ss), density, len(vs), varexp, kappa, theta, sigma, rho)
    try:
        ret = np.load(fname)
    except IOError:
        ret = empty((len(ss), len(vs)))
        tot = len(ss)*len(vs)
        for i in xrange(len(ss)):
            print float(i*len(vs)) / tot
            for j in xrange(len(vs)):
                s = ss[i]
                if isclose(vs[j], 0):
                    v = 0.0001
                else:
                    v = vs[j]
                ret[i,j] = heston.call(s, k, r, sqrt(v), t,
                                       kappa, theta, sigma, rho)
        np.save(fname, ret)
    return ret


def bs_call(s, k, r, v, t):
    N = scipy.stats.distributions.norm.cdf
    s = s[:,newaxis]
    v = v[newaxis,:]
    d1 = (np.log(s/k) + (r+0.5*v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v*np.sqrt(t)
    return (N(d1)*s - N(d2)*k*np.exp(-r * t), N(d1))

def center_diff(xs):
    dx = np.zeros_like(xs,dtype=float)
    dx[:-1]  += np.diff(xs)
    dx[1:]   += np.diff(xs[::-1])[::-1]*-1
    dx[1:-1] *= 0.5
    return dx

def str_coeff():
    fst = zeros((3,7))*nan
    fst[:,0]  = nan
    fst[:,-1] = nan
    for i in range(1,6):
        fst[0,i+1] = i
        fst[1,i] = i
        fst[2,i-1] = i
    return fst


def nonuniform_backward_coefficients(deltas):
    d = deltas.copy()
    fst = zeros((3,len(d)))
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
    d = deltas.copy()
    fst = zeros((3,len(d)))
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
    d = deltas.copy()
    fst = zeros((3,len(d)))
    snd = fst.copy()
    for i in range(1,len(d)-1):
        fst[0,i+1] =            d[i]  / (d[i+1]*(d[i]+d[i+1]))
        fst[1,i]   = (-d[i] + d[i+1]) /         (d[i]*d[i+1])
        fst[2,i-1] =         -d[i+1]  / (d[i  ]*(d[i]+d[i+1]))

        snd[0,i+1] = 2  / (d[i+1]*(d[i]+d[i+1]))
        snd[1,i]   = -2 /       (d[i]*d[i+1])
        snd[2,i-1] = 2  / (d[i  ]*(d[i]+d[i+1]))
    return fst, snd

def filterprint(A, prec=1, fmt="f", predicate=lambda x: x == 0, blank='- '):
    '''
    Pretty print a NumPy array, hiding values which match a predicate
    (default: x == 0). predicate must be callable.
    '''
    if hasattr(A, "todense"):
        A = A.todense()
    if A.ndim == 1: # Print 1-D vectors as columns
        A = A[:,newaxis]
    tmp = "% .{0}{1}".format(prec, fmt)
    xdim, ydim = np.shape(A)
    pad = max(len(tmp % x) for x in A.flat)
    fmt = "% {pad}.{prec}{fmt}".format(pad=pad, prec=prec, fmt=fmt)
    bstr = "{:>{pad}}".format(blank, pad=pad)
    for i in range(xdim):
        for j in range(ydim):
            if not predicate(A[i,j]):
                print fmt % A[i,j],
            else:
                print bstr,
        print
    return
fp = filterprint

def expl(V, L1, R1, L2, R2, dt, n, crumbs=None):
    # dt /= 2
    V = V.copy()
    L1 = [x.copy() for x in L1]
    R1 = [x.copy() for x in R1]
    L2 = [x.copy() for x in L2]
    R2 = [x.copy() for x in R2]
    for j, l in enumerate(L1):
        L1[j].data *= dt
        L1[j].data[1,:] += 1
        R1[j] *= dt
    for i, l in enumerate(L2):
        L2[i].data *= dt
        L2[i].data[1,:] += 1
        R2[i] *= dt
    for k in xrange(n):
        for j in xrange(V.shape[1]):
                V[:,j] = L1[j].dot(V[:,j]) + R1[j]
        for i in xrange(V.shape[0]):
                V[i,:] = L2[i].dot(V[i,:]) + R2[i]
        if crumbs is not None:
            crumbs.append(V.copy())
    if crumbs is not None:
        return crumbs
    return V

def impl(V, L1, R1, L2, R2, dt, n, crumbs=None):
    V = V.copy()
    L1i = [x.copy() for x in L1]
    R1  = [x.copy() for x in R1]
    L2i = [x.copy() for x in L2]
    R2  = [x.copy() for x in R2]
    for j, l in enumerate(L1):
        L1i[j].data *= -dt
        L1i[j].data[1,:] += 1
        R1[j] *= dt
    for i, l in enumerate(L2):
        L2i[i].data *= -dt
        L2i[i].data[1,:] += 1
        R2[i] *= dt
    for k in xrange(n):
        for j in xrange(V.shape[1]):
            V[:,j] = spl.solve_banded((1,1), L1i[j].data, V[:,j] + R1[j])
        for i in xrange(V.shape[0]):
            V[i,:] = spl.solve_banded((1,1), L2i[i].data, V[i,:] + R2[i])
        if crumbs is not None:
            crumbs.append(V.copy())
    if crumbs is not None:
        return crumbs
    return V

def crank(V, L, R, dt, n):
    V = V.copy()
    L  = [x.copy() for x in L]
    Li = [x.copy() for x in L]
    R  = [x.copy() for x in R]

    dt *= 0.5
    for j, l in enumerate(L):
        L[j].data *= dt
        L[j].data[1,:] += 1
        Li[j].data *= -dt
        Li[j].data[1,:] += 1
        R[j] *= dt
    for i in xrange(n):
        for j in xrange(V.shape[1]):
            V[:,j] = L[j].dot(V[:,j]) + R[j]
            V[:,j] = spl.solve_banded((1,1), Li[j].data, V[:,j] + R[j])
    return V


def plot_price_err(V, label, spots, bs, _, ids=slice(None), vars=None):
    # Trim for plotting
    front = 2
    back = 2
    assert(0 < V.ndim < 3)
    if V.ndim == 1 or V.shape[1] == 1:
        plot((spots/k*100)[ids][front:-back],
             (V - bs)[ids][front:-back], label=label)
        xlabel("% of strike")
        ylabel("Error")
        title("Error in Price")
    if V.ndim == 2 and V.shape[1] > 1:
        assert(vars is not None)
        wireframe((V-bs)[ids,:] , (spots/k*100)[ids],(vars))
        xlabel("Var")
        ylabel("% of strike")
        title("Error in Price: {0}".format(label))
    legend(loc=0)

def sinh_space(high, exact, density, size):
    c = float(density)
    K = float(exact)
    Smax = float(high)
    def g(x, K, c, p): return K + c/p * sinh(p*x + arcsinh(-p*K/c))
    p = scipy.optimize.root(lambda p: g(1, K, c, p)-1, 1)
    print p.success, p.r, g(1, K, c, p.r)
    p = p.r
    deps = 1./size * (arcsinh((Smax - K)*p/c) - arcsinh(-p*K/c))
    eps = arcsinh(-p*K/c) + arange(size)*deps
    space = K + c/p * sinh(eps)
    return space



def plot_price(V, label, spots, ids=slice(None), vars=None):
    # Trim for plotting
    front = 2
    back = 2
    assert(0 < V.ndim < 3)
    if V.ndim == 1 or V.shape[1] == 1:
        plot((spots/k*100)[ids][front:-back],
             V[ids][front:-back], label=label)
        xlabel("% of strike")
        ylabel("Price")
        title("Price")
    if V.ndim == 2 and V.shape[1] > 1:
        assert(vars is not None)
        wireframe(V[ids,:] , (spots/k*100)[ids],(vars))
        xlabel("Var")
        ylabel("% of strike")
        title("Price: {0}".format(label))
    legend(loc=0)


def plot_delta_err(V, label, spots, bs, delta, ids=slice(None)):
    # Trim for plotting
    front = 2
    back = 2
    dVds = center_diff(V)/center_diff(spots)
    plot((spots)[ids][front:-back],
         (dVds - delta)[ids][front:-back], label=label)
    title("Error in $\Delta$")
    xlabel("% of strike")
    ylabel("Error")
    legend(loc=0)


def main():
    global ids, idx, idv, spots, dss, vars, dvs, fst, snd, Vi, L1, L2, R1, R2
    global Ass, mu_s, gamma2_s, V, spot, k, r, t, v, dt, nspots, nvols, Rs, Rss
    global Rv, Rvv, Av, Avv, As, hs, bs
    spot = 100.
    k = 99.
    r = 0.06
    t = 1.
    v = 0.2**2
    dt = 1/1000.
    kappa = 0.01
    theta = 0.04
    sigma = 0.001
    nspots = 80
    nspots += not (nspots%2)
    nvols = 40
    rho = 0

    spotdensity = 0.  # 0 is linear
    varexp = 1        # 1 is linear

    if len(sys.argv) > 1:
        spotdensity = float(sys.argv[1])
        print spotdensity
    # spots = cubic_sigmoid_space(spot, spot*2-1,
                                    # spotdensity, nspots)
    # spots = linspace(0, 980, nspots)
    # spot = spots[8]
    dss = np.hstack((nan, np.diff(spots)))
    ids = (spot*0.8 < spots) & (spots < spot*1.2) # Just the range we care about
    ids = (0 < spots) & (spots < 400) # Just the range we care about
    idvs = (0 < vars) & (vars < 1) # Just the range we care about
    idx = isclose(spots, spot) # The actual spot we care about

    # vars = exponential_space(0.001, v, 10, varexp, nvols)
    vars = linspace(0, 10, nvols)
    nvols = len(vars)
    dvs = np.hstack((nan, np.diff(vars)))
    idv = isclose(vars, v) # The actual vol we care about



    Vi = init(spots, vars, k)
    V = Vi.copy()


    def plot_it(domain, label):
        global bs, hs
        if sigma < 0.01:
            print "Low sigma, using black scholes solution."
            bs, delta = bs_call(spots, k, r, np.sqrt(vars), t)
            analytical = bs
        else:
            hs = hs_call(spots, k, t, np.sqrt(vars), t, kappa, theta, sigma,
                         rho, spotdensity, varexp)
            analytical = hs
        print domain[idx,idv] - analytical[idx,idv]
        # plot_delta_err(domain, label, spots/k*100, analytical, delta, ids)
        plot_price_err(domain, label, spots/k*100, analytical, delta, ids, vars)
        # plot_price(domain[:,idvs], label, spots/k*100, ids, vars[idvs])
        # plot_price(domain, label, spots/k*100, None, vars)

    L1 = [0]*nvols
    R1 = [0]*nvols
    for j, v in enumerate(vars):
        mu_s = r*spots
        gamma2_s = 0.5*v*spots**2

        fst, snd = nonuniform_center_coefficients(dss)
        As = sps.dia_matrix((fst, (1, 0, -1)), shape=(nspots,nspots))
        As.data[0,1:]  *= mu_s[:-1]
        As.data[1,:]   *= mu_s
        As.data[2,:-1] *= mu_s[1:]

        Rs = np.zeros_like(V[:,j])
        Rs[-1] = 1
        Rs *= mu_s

        Ass = sps.dia_matrix((snd, (1, 0, -1)), shape=(nspots,nspots))
        Ass.data[1, -1] = -2/dss[-1]**2
        Ass.data[2, -2] =  2/dss[-1]**2
        Ass.data[0,1:]  *= gamma2_s[:-1]
        Ass.data[1,:]   *= gamma2_s
        Ass.data[2,:-1] *= gamma2_s[1:]


        Rss = np.zeros_like(V[:,j])
        Rss[-1] = 2*dss[-1]/dss[-1]**2
        Rss *= gamma2_s

        # L1 = (As + Ass - 1/2*r*np.eye(nspots))*dt + np.eye(nspots)
        # We have to save the dt though, to generalize our time stepping functions
        L1[j] = As.copy()
        L1[j].data += Ass.data
        L1[j].data[1,:] -= 0.5*r

        R1[j] = (Rs + Rss)

    L2 = [0]*nspots
    R2 = [0]*nspots
    for i, s in enumerate(spots):
        mu_v = kappa*(theta - vars)
        gamma2_v = 0.5*sigma**2*vars

        fst, snd = nonuniform_center_coefficients(dvs)
        Av = sps.dia_matrix((fst, (1, 0, -1)), shape=(nvols,nvols))
        Av.data[0, 1] = -1 / dvs[1]
        Av.data[1, 0] =  1 / dvs[1]
        # Av.data[0, 1:]  =  1 / dvs[1:]
        # Av.data[1,:-1]  = -1 / dvs[1:]
        # Av.data[2,:] *= 0
        # Av.data[2,:-1] *= mu_v[1:]
        Av.data[0,1:]  *= mu_v[:-1]
        Av.data[1,:]   *= mu_v
        Av.data[2,:-1] *= mu_v[1:]

        Rv = np.zeros_like(V[i,:])
        Rv[-1] = s
        # Rv *= mu_v

        Avv = sps.dia_matrix((snd, (1, 0, -1)), shape=(nvols,nvols))
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

    # V = expl(Vi, L1, R1, L2, R2, dt, int(t/dt))
    # if not isnan(V).any():
        # plot_it(V,"exp")


    V = impl(Vi, L1, R1, L2, R2, dt, int(t/dt), [])
    plot_it(V[-1], "impl")


    # V = crank(Vi, L, R, dt, int(t/dt))
    # plot_it(V, "crank")


    ## Rannacher smoothing to damp oscilations at the discontinuity
    # V = impl(Vi, L1, R1, 0.5*dt, 4)
    # V = crank(V, L1, R1, dt, int(t/dt)-2)
    # plot_it(V, "smooth")


    show()

def iterate(f, x0):
    while 1:
        yield x0
        x0 = f(x0)

def take(n, seq, step=1):
    for i, x in enumerate(seq):
        if i/step == n:
            raise StopIteration
        if not i % step:
            yield x

def bs_stream(s, k, r, v, dt):
    for i in it.count():
        yield bs_call(s, k, r, v, i*dt)[0]


def wireframe(domain, xs, ys, ax=None):
    if ax is None:
        fig = figure()
        ax = fig.add_subplot(111, projection='3d')
    X, Y = meshgrid(ys, xs)
    wframe = ax.plot_surface(X, Y, domain,
                               rstride=(max(int(np.shape(X)[0]/25.0), 1)),
                               cstride=(max(int(np.shape(X)[1]/25.0), 1)),
                              cmap=cm.jet)
    return wframe, ax

def lineplot(domain, xs, ys=None, ax=None):
    if ax is None:
        fig = figure()
        ax = fig.add_subplot(111)
    lines = ax.plot(xs, domain)
    return lines, ax


def anim(plotter, domains, xs, ys):
    """
    A very simple 'animation' of a 3D plot
    """
    ion()
    tstart = time.time()
    oldcol = None
    ax = None
    try:
        for i, Z in enumerate(domains):
            # Remove old line collection before drawing
            if oldcol is not None:
                ax.collections.remove(oldcol)

            oldcol, ax = plotter(Z, xs, ys, ax)
            title("#%02i" % (i,))

            draw()
    except StopIteration:
        pass

    print 'FPS: %f' % (100 / (time.time() - tstart))

if __name__ == "__main__":
    pass
    # main()
