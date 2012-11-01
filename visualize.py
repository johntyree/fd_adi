#!/usr/bin/env python

import numpy as np
import pylab
from mpl_toolkits.mplot3d import axes3d
from utils import center_diff

def filterprint(A, prec=1, fmt="f", predicate=lambda x: x == 0, blank='- '):
    """
    Pretty print a NumPy array, hiding values which match a predicate
    (default: x == 0). predicate must be callable.
    """
    if hasattr(A, "todense"):
        A = A.todense()
    if A.ndim == 1: # Print 1-D vectors as columns
        A = A[:,np.newaxis]
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

def lineplot(domain, xs, ys=None, ax=None):
    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
    lines = ax.plot(xs, domain)
    return lines, ax


def wireframe(domain, xs, ys, ax=None):
    """Convenience func for plotting a wireframe."""
    if domain.shape != (len(xs), len(ys)):
        print "Domain shape %s does match axes %s!." % (domain.shape, (len(xs),
                                                                    len(ys)))
    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(ys, xs)
    wframe = ax.plot_surface(X, Y, domain,
                             rstride=(max(int(np.shape(X)[0]/15.0), 1)),
                             cstride=(max(int(np.shape(X)[1]/15.0), 1)),
                             alpha=0.3)
    xmin = np.min(ax.xaxis.get_tick_positions()[1])
    ymax = np.max(ax.yaxis.get_tick_positions()[1])
    zmin = np.min(ax.zaxis.get_tick_positions()[1])
    xmin = np.min(ys)
    ymax = np.max(xs)
    zmin = np.min(domain)
    print xmin, ymax, zmin
    cset = ax.contour(X, Y, domain, zdir='x', offset=xmin)
    cset = ax.contour(X, Y, domain, zdir='y', offset=ymax)
    cset = ax.contour(X, Y, domain, zdir='z', offset=zmin)
    return wframe, ax


def surface(domain, xs, ys, ax=None):
    """Convenience func for plotting a surface."""
    if domain.shape != (len(xs), len(ys)):
        print "Domain shape %s does match axes %s!." % (domain.shape, (len(xs),
                                                                    len(ys)))
    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(ys, xs)
    wframe = ax.plot_surface(X, Y, domain,
                               rstride=(max(int(np.shape(X)[0]/25.0), 1)),
                               cstride=(max(int(np.shape(X)[1]/25.0), 1)),
                              cmap=pylab.cm.jet)
    return wframe, ax

def anim(plotter, domains, xs, ys, FPS=2):
    """
    A very simple 'animation' of a 3D plot. This works best with QT backend for
    some reason and not at all with inline :(.

        %pylab qt
        anim(wireframe, [V1, V2....], spots, vols, FPS=1)
    """
    SPF = 1.0 / FPS
    pylab.ion()
    tstart = time.time()
    oldcol = None
    ax = None
    try:
        for i, Z in enumerate(domains):
            # Remove old line collection before drawing
            if oldcol is not None:
                ax.collections.remove(oldcol)

            frame_time = time.time()
            oldcol, ax = plotter(Z, xs, ys, ax)
            pylab.xlabel("#%02i" % (i,))

            time.sleep(max(j, SPF - frame_time))
            draw()
    except StopIteration:
        pass
    pylab.ioff()

    print 'FPS: %f' % (100 / (time.time() - tstart))



def plot_price_err(V, spots, k, vars, analytical, label=None, ids=slice(None)):
    # Trim for plotting
    front = 2
    back = 2
    assert(0 < V.ndim < 3)
    if V.ndim == 1 or V.shape[1] == 1:
        pylab.plot((spots/k*100)[ids][front:-back],
             (V - analytical)[ids][front:-back], label=label)
        pylab.xlabel("% of strike")
        pylab.ylabel("Error")
        pylab.title("Error in Price")
    if V.ndim == 2 and V.shape[1] > 1:
        assert(vars is not None)
        # if ids is None:
            # ids = slice(len(spots))
        wireframe((V-analytical)[ids,:] , (spots/k*100)[ids],(vars))
        pylab.xlabel("Var")
        pylab.ylabel("% of strike")
        pylab.title("Error in Price: {0}".format(label))
    pylab.legend(loc=0)


def plot_price(V, spots, k, vars, label=None, ids=slice(None)):
    # Trim for plotting
    front = 2
    back = 2
    assert(0 < V.ndim < 3)
    if V.ndim == 1 or V.shape[1] == 1:
        pylab.plot((spots/k*100)[ids][front:-back],
             V[ids,front:-back], label=label)
        pylab.xlabel("% of strike")
        pylab.ylabel("Price")
        pylab.title("Price")
    if V.ndim == 2 and V.shape[1] > 1:
        assert(vars is not None)
        if ids is None:
            ids = slice(len(spots))
        wireframe(V[ids,:] , (spots/k*100)[ids],(vars))
        pylab.xlabel("Var")
        pylab.ylabel("% of strike")
        pylab.title("Price: {0}".format(label))
    pylab.legend(loc=0)


def plot_delta_err(V, spots, k, vars, label=None, ids=slice(None)):
    # Trim for plotting
    front = 2
    back = 2
    dVds = center_diff(V)/center_diff(spots)
    pylab.plot((spots)[ids][front:-back],
         (dVds - delta)[ids][front:-back], label=label)
    pylab.title("Error in $\Delta$")
    pylab.xlabel("% of strike")
    pylab.ylabel("Error")
    pylab.legend(loc=0)

def plot_dUdv(vars, V, bs, i=16):
    print i
    print bs.shape
    print V.shape
    dUdv = center_diff(V[i,:])/center_diff(vars)
    dUdbs = center_diff(bs[i,:])/center_diff(vars)
    # pylab.plot(vars, center_diff(bs[i,:]), label="BS")
    pylab.plot(vars, dUdbs, label="BS")
    pylab.plot(vars, dUdv, label="ADI")
    pylab.title("First deriv w.r.t. var")
    pylab.ylabel(r"$\frac{\partial U}{\partial v}$")
    pylab.legend(loc=0)
    pylab.show()

def plot_d2Udv2(vars, V, bs, i=16):
    pylab.plot(vars, center_diff(center_diff(bs[i,:])), label="BS")
    pylab.plot(vars, center_diff(center_diff(V[i,:])), label="ADI")
    pylab.title("Second deriv w.r.t. var")
    pylab.ylabel(r"$\frac{\partial^2 U}{\partial v^2}$")
    pylab.legend(loc=0)
    pylab.show()

