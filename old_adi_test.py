#!/usr/bin/env python
"""Description of file."""


import platform

pypy = False
if platform.python_implementation() == 'PyPy':
    pypy = True

if pypy:
    import numpypy
else:
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt

import numpy as np
# import scipy as sp

import time



N = 4
A = np.zeros((N, N))
B = np.arange(N*N).reshape((N, N))
source = np.zeros((N, N))


def linspace(min, max, n):
    if pypy:
        return np.arange(0, n) * (max-min) + min
    else:
        return np.linspace(min, max, n)

def meshgrid(x, y):
    if pypy:
        xx = x.repeat(len(x)).reshape((len(x), len(x)))
        yy = y.repeat(len(y)).reshape((len(y), len(y)))
        return (xx, yy)
    else:
        return np.meshgrid(x, y)

def zeros_like(x):
    return np.zeros(np.shape(x))


def centered_ddx(mat):
    '''Discrete first derivative operator.'''
    for i in xrange(1, len(mat)):
        for j in xrange(1, len(mat)):
            mat[i][j-1] = -1
            mat[i][j+1] =  1

def ddx(mat, n):
    '''Return discrete derivatives w.r.t. x.'''
    return ddy(mat.T, n).T

def ddy(mat, n):
    '''Return discrete derivatives w.r.t. y.'''
    ret = mat
    for i in range(n):
        ret = discrete_first(ret).dot(ret)
    return ret



def discrete_first(domain):
    '''Discrete first derivative operator with one-sided boundaries.'''
    operator = zeros_like(domain.T)
    operator[0][0] += -1.0
    operator[0][1] +=  1.0
    (xdim, ydim) = np.shape(domain)  # dot product is rows * cols
    for i in xrange(1, xdim-1):
        for j in xrange(1, ydim-1):
            if i == j:
                operator[i][j-1] += -0.5
                operator[i][j+1] +=  0.5
    operator[xdim-1][ydim-2] += -1.0
    operator[xdim-1][ydim-1] +=  1.0
    return operator

def diffusion_process(n=50, xmin=0, xmax=1, ymin=0, ymax=1):
    xs = linspace(xmin, ymax, n)
    ys = linspace(ymin, ymax, n)
    X, Y = meshgrid(xs, ys)
    yield (X,Y)
    u = np.zeros((n,n))
    while 1:
        u[:,0]   = 0.0
        u[0,:]   = 0.0
        u[:,n-1] = 0.0
        u[n-1,:] = 0.0
        u[n//2, n//2] = 1e3
        u[n//2, n//2-1] = 1e3
        u[n//2, n//2+1] = 1e3
        u[n//2-1, n//2] = 1e3
        u[n//2+1, n//2] = 1e3
        u[n//2+1, n//2+1] = 1e3
        u[n//2+1, n//2-1] = 1e3
        u[n//2-1, n//2+1] = 1e3
        u[n//2-1, n//2-1] = 1e3
        u = (ddy(u,2) + ddx(u,2))*dt + u
        dt = (yield u)

def wavy_process():
    xs = linspace(-1, 1, 50)
    ys = linspace(-1, 1, 50)
    X, Y = meshgrid(xs, ys)
    R = 1 - np.sqrt(X**2 + Y**2)
    yield (X,Y)
    phi = (yield)
    while 1:
        v = np.cos(2 * np.pi * X + phi) * R
        phi = (yield v)


def matrix_stream(fin):
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
                    yield meshgrid(np.arange(s[0]), np.arange(s[1]))
                    dom = True
                _ = yield np.array(m)
                m = []


def run(domain, dt=1e-2, t=10, step=1):
    (X,_) = domain.next()
    Z = domain

    tstart = time.time()
    Z.send(0)
    for i in xrange(0, int(t//dt)):
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

if not pypy:
    def heatmap_anim(domain, dt=1e-4, t=10, step=1):
        """
        A very simple 'animation' of a 3D plot
        """

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        (X,Y) = domain.next()
        Z = domain

        wframe = None
        tstart = time.time()
        Z.send(0)
        for i in xrange(0, int(t//dt)):
            if i % step == 0:
                ax.imshow(Z.send(dt))
                plt.draw()
            else:
                Z.send(dt)

        print 'FPS: %f' % (100 / (time.time() - tstart))

    def wireframe_anim(domain, dt=1e-3, t=10, step=1):
        """
        A very simple 'animation' of a 3D plot
        """

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        (X,Y) = domain.next()
        Z = domain

        wframe = None
        tstart = time.time()
        Z.send(0)
        try:
            for _ in linspace(0, t, t/dt):
                for i in xrange(0, int(t//dt)):
                    if i % step == 0:
                        oldcol = wframe
                        wframe = ax.plot_wireframe(X, Y, Z.send(dt),
                                                   rstride=(int(np.shape(X)[0]/50.0)),
                                                   cstride=(int(np.shape(X)[1]/50.0)))

                        # Remove old line collection before drawing
                        if oldcol is not None:
                            ax.collections.remove(oldcol)

                        plt.draw()
                    else:
                        Z.send(dt)
        except StopIteration:
            pass

        print 'FPS: %f' % (100 / (time.time() - tstart))
