#!/usr/bin/env python
"""Demonstration of 1D Black Scholes using FTCS BTCS CTCS and Smoothed CTCS."""

import numpy as np
from pylab import *
from utils import fp, wireframe
def w(m): wireframe(m,spots,sqrt(vars)); show()

def center_diff(xs):
    dx = np.zeros_like(xs,dtype=float)
    dx[:-1]  += np.diff(xs)
    dx[1:]   += np.diff(xs[::-1])[::-1]*-1
    dx[1:-1] *= 0.5
    return dx

def g(x, K, c, p): return K + c/p * sinh(p*x + arcsinh(-p*K/c))
def sinh_space(exact, high, density, size):
    c = float(density)
    K = float(exact)
    Smax = float(high)
    p = scipy.optimize.root(lambda p: g(1.0, K, c, p)-1.0, -1)
    print p.success, p.r, g(1.0, K, c, p.r)-1
    p = p.r
    deps = 1./size * (arcsinh((Smax - K)*p/c) - arcsinh(-p*K/c))
    eps = arcsinh(-p*K/c) + arange(size)*deps
    space = K + c/p * sinh(eps)
    return space

# x = np.log(100)
# spot = 100
# k = 100.
# r = 0.06
# t = 1.
# v0 = 0.2**2
# dt = 1/30.
# nspots = 200

# vars = array(linspace(0.01,10,100))
# # vars = array([2., 2.])**2
# nvols = len(vars)
# idv = find(vars == v0)

# nspots += not (nspots%2)
# xs = np.linspace(-1, 1, nspots)
# xs = 3*np.sqrt(5)*xs*x + x
# # xs = sinh_space(x, 3*np.sqrt(v0)*x + x, 0.93, nspots)
# # xs = sinh_space(x, log(200), 1., nspots)
# spots = np.exp(xs)
# dxs = np.hstack((nan, np.diff(xs)))
# ids = (0 < np.exp(xs)) & (np.exp(xs) < 1200)
# idx = find(xs == x)
# # dx = dxs[1]

def init(spots, vars, k):
    u = np.ones((len(spots),len(vars))).T * spots
    u = u.T
    return np.maximum(0, u-k)


def exponential_space(low, high, exact, ex, n):
    v = np.zeros(n);
    l = pow(low,1./ex);
    h = pow(high,1./ex);
    x = pow(exact,1./ex);
    dv = (h - l) / (n-1);
    j = 0

    d = 1e100
    for i in range(n):
        if (i*dv > x):
        # if abs(i*dv - x) < d:
            # d = abs(i*dv - x)
            j = i-1
            break

    dx = x - j*dv;
    print dx
    h += (n-1) * dx/j;
    dv = (h - l) / (n-1);
    for i in range(n):
        v[i] = pow(i*dv, ex)
    return v;

def cubic_sigmoid_space(exact, high, density, n):
    y = zeros(n)

    dx = 1.0/(n-1)
    scale = (float(high)-exact)/(density**3 + density)
    for i in range(n):
        x = (2*(i*dx)-1)*density
        y[i] = exact + (x**3+x)*scale

    return y


def centered_linspace_low_center_high(low, center, high, n):
    l = np.empty(n);
    dx = (high - low) / float(n);
    for i in xrange(n):
        l[i] = center + dx * np.ceil(i - (center-low)/(high-low)*n);
    return l

def centered_linspace_dx(center, dx, n):
    l = np.empty(n);
    for i in xrange(n):
        l[i] = center + dx * (i - n/2);
    return l;


def spot_boundary(domain, logspots):
    domain[0,:] = 0
    # ds = 0.5 * (-3*np.exp(logspots[-1])
                # + 4*np.exp(logspots[-2])
                # - 1*np.exp(logspots[-3]))
    # domain[-1,:] = (0.5 * (4*domain[-2,:] - 1*domain[-3,:]) - ds) / -1.5
    domain[-1,:] = domain[-2,:] + (logspots[-1] - logspots[-2])
    return

def vol_boundary(domain, logspots, k):
    # Left column except for top and bottom row
    # Vol = 0, just enfore the PDE itself.
    # for i in xrange(1,len(logspots)-1):
        # dx = logspots[1] - logspots[0]
        # dv = vars[1] - vars[0]

        # # One-sided difference in v due to boundary
        # du_dv = (domain[i  ,1] - domain[i  ,0]) / dv;
        # du_dx = (domain[i+1,0] - domain[i-1,0]) / (2*dx)

        # domain[i,0] = (kappa*theta*du_dv + r*du_dx - r*domain[i,0]) * dt + previous[i,0];

    # Right column
    # v -> \infty so V = S
    domain[:,-1] = np.exp(logspots[-1]) - k
    return


def enforce_boundary(domain, logspots, k):
    # Top and bottom rows
    # domain = domain.copy()
    spot_boundary(domain, logspots)
    vol_boundary(domain, logspots, k)
    return domain



# def bs_call(s, k, r, sig, t):
    # def N(x): return sp.distributions.norm.cdf(x)
    # d1 = (np.log(s/k) + (r+0.5*sig**2) * t) / (sig * np.sqrt(t))
    # d2 = d1 - sig*np.sqrt(t)
    # return N(d1)*s - N(d2)*k*np.exp(-r * t)


def matrix_stream(fin):
    """An iterator of matrices."""
    ret = []
    dom = False
    with open(fin) as f:
        m = []
        for l in f:
            l = l.strip()
            if l:
                m.append(map(float, l.split()))
            else:
                if dom is False:
                    s = np.shape(m)
                    yield np.meshgrid(np.arange(s[0]), np.arange(s[1]))
                    dom = True
                _ = yield np.array(m)
                m = []


def run(domain, dt=1e-2, t=10, step=1):
    (X,_) = domain.next()
    Z = domain

    tstart = time.time()
    Z.send(0)
    for i in xrange(0, int(t/dt)):
        if i % step == 0:
            s = (Z.send(dt))
            # print s.dump("data/%i.np" % i)
            for i in range(np.shape(X)[0]):
                for j in range(np.shape(X)[1]):
                    print s[i, j],
                print
            print
        else:
            Z.send(dt)

    # print 'FPS: %f' % (100 / (time.time() - tstart))

# if __name__ == "__main__":
    # run(diffusion_process(), dt=1e-3, step=100)

# def heatmap_anim(domain, dt=1e-4, t=10, step=1):
    # """
    # A very simple 'animation' of a 3D plot
    # """

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # (X,Y) = domain.next()
    # Z = domain

    # wframe = None
    # tstart = time.time()
    # Z.send(0)
    # for i in xrange(0, int(t/dt)):
        # if i % step == 0:
            # ax.imshow(Z.send(dt))
            # plt.draw()
        # else:
            # Z.send(dt)

    # print 'FPS: %f' % (100 / (time.time() - tstart))

# def wireframe_anim(domain, dt=1e-3, t=10, step=1):
    # """
    # A very simple 'animation' of a 3D plot
    # """

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # (X,Y) = domain.next()
    # Z = domain

    # wframe = None
    # tstart = time.time()
    # Z.send(0)
    # try:
        # for _ in np.linspace(0, t, t/dt):
            # for i in xrange(0, int(t/dt)):
                # if i % step == 0:
                    # oldcol = wframe
                    # wframe = ax.plot_wireframe(X, Y, Z.send(dt),
                                               # rstride=(int(np.shape(X)[0]/50.0)),
                                               # cstride=(int(np.shape(X)[1]/50.0)))

                    # # Remove old line collection before drawing
                    # if oldcol is not None:
                        # ax.collections.remove(oldcol)

                    # plt.draw()
                # else:
                    # Z.send(dt)
    # except StopIteration:
        # pass

    # print 'FPS: %f' % (100 / (time.time() - tstart))

