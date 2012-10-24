#!/usr/bin/env python
"""Description of file."""


# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt

from pylab import *
import numpy as np
# import scipy.stats as sp
# import scipy as sp
from math import pow


import time

def init(xs, vs, k):
    u = np.ones((len(xs),len(vs))).T * np.exp(xs)
    u = u.T
    return np.maximum(0, u-k)

def D(dim, boundary=False):
    domain = np.arange(dim)
    return discrete_first(domain, boundary)

def D2(dim, boundary=False):
    domain = np.arange(dim)
    return discrete_second(domain, boundary)

def discrete_first(domain, boundary=False):
    '''Discrete first derivative operator with no boundary.'''
    operator = np.zeros((len(domain), len(domain)))
    (xdim, ydim) = np.shape(operator)
    if boundary:
        operator[0,1] = 0.5
    else:
        xstart, xend = 1, xdim-1
        ystart, yend = 1, ydim-1
    for i in xrange(xstart, xend):
        for j in xrange(1, ydim-1):
            if i == j:
                operator[i][j-1] = -0.5
                operator[i][j+1] =  0.5
    # operator[-1][-2] = -1
    # operator[-1][-1] =  1
    return operator

def discrete_second(domain, boundary=False):
    '''Discrete second derivative operator with no boundary.'''
    operator = np.zeros((len(domain), len(domain)))
    (xdim, ydim) = np.shape(operator)
    if boundary:
        xstart, xend = 0, xdim
        ystart, yend = 0, ydim
    else:
        xstart, xend = 1, xdim-1
        ystart, yend = 1, ydim-1
    for i in xrange(xstart, xend):
        for j in xrange(ystart, yend):
            if i == j:
                operator[i][j-1] =  1
                operator[i][j  ] = -2
                operator[i][j+1] =  1
    return operator





# def ddx(mat, n):
    # '''Return nth discrete derivative w.r.t. x.'''
    # return ddy(mat.T, n).T

# def ddy(mat, n):
    # '''Return nth discrete derivative w.r.t. y.'''
    # ret = mat
    # for i in range(n):
        # ret = discrete_first(ret).dot(ret)
    # return ret

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

