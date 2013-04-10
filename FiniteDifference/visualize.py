#!/usr/bin/env python
# coding: utf8

import numpy as np
import pylab
from mpl_toolkits.mplot3d import axes3d
from utils import center_diff
import time

def filterprint(domain, prec=1, fmt="f", predicate=lambda x: x == 0, blank='- '):
    """
    Pretty print a NumPy array, hiding values which match a predicate
    (default: x == 0). predicate must be callable.

    Aliased to fp.

    Print an array with three decimal places in scientific notation when the
    value is larger than 2, and '-' otherwise.

        fp(domain, 3, 'e', lambda x: x > 2)

    """
    if hasattr(domain, "todense"):
        domain = domain.todense()
    # if domain.ndim == 1: # Print 1-D vectors as columns
        # domain = domain[:,np.newaxis]
    tmp = "% .{0}{1}".format(prec, fmt)
    domain = np.atleast_2d(domain)
    xdim, ydim = np.shape(domain)
    pad = max(len(tmp % x) for x in domain.flat)
    fmt = "% {pad}.{prec}{fmt}".format(pad=pad, prec=prec, fmt=fmt)
    bstr = "{:>{pad}}".format(blank, pad=pad)
    for i in range(xdim):
        for j in range(ydim):
            if not predicate(domain[i,j]):
                print fmt % domain[i,j],
            else:
                print bstr,
        print
    return
fp = filterprint


def lineplot(domain, xs=None, dummy=None, ax=None, style='b'):
    """Regular plot of 1D line, à la pylab.plot(), but with a uniform interface
    for use with anim().
    """
    if xs is None:
        xs = np.arange(domain.shape[0])
    if domain.shape != xs.shape:
        print "Domain shape %s does match axes %s." % (domain.shape, xs.shape)
    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
    lines = ax.plot(xs, domain, style)
    return lines, ax


def wireframe(domain, xs=None, ys=None, ax=None):
    """
    Plot a wireframe with countour plots projected on the "walls".

    domain is a 2D array, xs and ys are both 1D vectors. The shapes must align
    such that domain.shape == (length(xs), length(ys)).

    The ax parameter is for reusing the plot for animation. Ignore it.
    """
    if xs is None:
        xs = np.arange(domain.shape[0])
    if ys is None:
        ys = np.arange(domain.shape[1])
    if domain.shape != (len(xs), len(ys)):
        print "Domain shape %s does match axes %s." % (domain.shape, (len(xs),
                                                                       len(ys)))
        return
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
    # print xmin, ymax, zmin
    cset = ax.contour(X, Y, domain, zdir='x', offset=xmin)
    cset = ax.contour(X, Y, domain, zdir='y', offset=ymax)
    cset = ax.contour(X, Y, domain, zdir='z', offset=zmin)
    return wframe, ax


def surface(domain, xs=None, ys=None, ax=None):
    """
    Plot the usual colored 3D surface.

    domain is a 2D array, xs and ys are both 1D vectors. The shapes must align
    such that domain.shape == (length(xs), length(ys)).

    The ax parameter is for reusing the plot for animation. Ignore it.
    """
    if xs is None:
        xs = np.arange(domain.shape[0])
    if ys is None:
        ys = np.arange(domain.shape[1])
    if domain.shape != (len(xs), len(ys)):
        print "Domain shape %s does match axes %s." % (domain.shape, (len(xs),
                                                                    len(ys)))
        return
    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(ys, xs)
    wframe = ax.plot_surface(X, Y, domain,
                               rstride=(max(int(np.shape(X)[0]/25.0), 1)),
                               cstride=(max(int(np.shape(X)[1]/25.0), 1)),
                              cmap=pylab.cm.jet)
    return wframe, ax

def anim(plotter, domains, xs=None, ys=None, FPS=2):
    """
    A very simple 'animation' of a 3D plot. This does not work with the inline
    backend of IPython, so use TkAgg GTKAgg or something.

    plotter is one of @surface@, @wireframe@, or @lineplot@.
    domains is an iterable of domain snapshots. See the docs for the plotter for
    clarification.

        %pylab TkAgg
        anim(surface, [domain1, domain2, ...], xs, ys, FPS=1)
    """
    SPF = 1.0 / FPS
    start = time.time()
    oldcol = None
    ax = None
    pylab.ion()
    try:
        for i, Z in enumerate(domains):
            frame_start = time.time()
            # Remove old line collection before drawing
            if oldcol is not None:
                if ax.collections:
                    ax.collections.remove(oldcol)
                elif ax.lines:
                    for l in oldcol:
                        ax.lines.remove(l)

            oldcol, ax = plotter(Z, xs, ys, ax)
            pylab.xlabel("#%02i" % (i,))

            frame_time = time.time() - frame_start
            if i > 0:
                time.sleep(max(0, SPF - frame_time))
            pylab.draw()
    except KeyboardInterrupt:
        pass
    except StopIteration:
        pass
    finally:
        stop = time.time()
        pylab.ioff()
        pylab.draw()
        pylab.show()

    # print 'FPS: %f' % ((i+1) / (stop - start))



def trim1d(F, V=None):
    if V is None:
        V = F.grid.domain[-1]
    xs = F.grid.mesh[0]
    trimx = (0.0 * F.option.spot <= xs) & (xs <= 2.0*F.option.spot)
    tr = lambda x: x[trimx]
    res = tr(V)
    try:
        a = tr(F.grid_analytical)
    except NotImplementedError:
        a = None
    return res, a, xs[trimx]

def trim2d(V, (xs, xmin, xmax), (ys, ymin, ymax)):
    trimx = (xmin <= xs) & (xs <= xmax)
    trimy = (ymin <= ys) & (ys <= ymax)
    tr = lambda x: x[trimx, :][:, trimy]
    res = tr(V)
    return res, xs[trimx], ys[trimy]

def error1d(F):
    V = F.grid.domain[-1]
    res, a, xs = trim1d(F)
    inf_norm = max(abs(res-a).flat)
    norm2 = pylab.sqrt(sum(((res-a)**2).flat))
    mae = pylab.mean((res-a).flat)
    return mae

def error2d(F):
    V = F.grid.domain[-1]
    res, a, xs, ys = trim2d(F)
    inf_norm = max(abs(res-a).flat)
    norm2 = pylab.sqrt(sum(((res-a)**2).flat))
    mae = pylab.mean((res-a).flat)
    return mae

def price_surface1d(F, trim=True):
    res = F.grid.domain[-1]
    xs = F.grid.mesh
    if trim:
        res, a, xs = trim1d(F)
    pylab.plot(xs, res)

def price_surface2d(F, trim=True):
    res = F.grid.domain[-1]
    xs, ys = F.grid.mesh
    if trim:
        res, a, xs, ys = trim2d(F)
    surface(res, xs, ys)

def error_surface2d(F, label="", trim=True):
    a = F.grid_analytical
    res = F.grid.domain[-1]
    xs, ys = F.grid.mesh
    if trim:
        res, a, xs, ys = trim2d(F)
    p_absolute_error(res, a, xs, ys, label=label)

def p_absolute_error(V, analytical, spots, vars, marker_idx=0, label="", bad=False):
    surface(V - analytical, spots, vars)
    # wireframe(V - analytical, spots, vars)
    if bad:
        label += " - bad analytical!"
    else:
        # label += " - $||V - V*||^\infty = %.2e$" % max(abs(V-analytical).flat)
        label += " - MAE = %.2e$" % np.mean(abs(V-analytical).flat)
    pylab.title("Error in Price (%s)" % label)
    pylab.xlabel("Var")
    pylab.ylabel("% of strike")
    pylab.show()

def p_price(V, analytical, spots, vars, marker_idx=0, label="", bad=False):
    surface(V, spots, vars)
    # wireframe(V - analytical, spots, vars)
    if bad:
        label += " - bad analytical!"
    pylab.title("Price (%s)" % label)
    pylab.xlabel("Var")
    pylab.ylabel("% of strike")
    pylab.show()


p = p_absolute_error
pp = p_price



def plot_dUds_err(domain, analytical, spots, vars, plotter=wireframe):
    """
    Plot the error in the derivative down each columns.
    """
    dUds = center_diff(domain,axis=0)/center_diff(vars)
    danalyticalds = center_diff(analytical,axis=0)/center_diff(vars)
    plotter(dUds - danalyticalds, spots, vars)
    pylab.title("Error in $\Delta$")
    pylab.xlabel("Var")
    pylab.ylabel("% of strike")

def plot_dUdv(domain, spots, vars, plotter=wireframe):
    """
    Plot the error in the derivative across each row.
    """
    dUdv = center_diff(domain,axis=1)/center_diff(vars)
    plotter(dUdv, spots, vars)
    pylab.title("First deriv w.r.t. var")
    pylab.xlabel("Var")
    pylab.ylabel("% of strike")
    pylab.show()

def plot_d2Udv2(domain, spots, vars, plotter=wireframe):
    """
    Plot the error in the second derivative across each row.
    """
    d2Udv2 = center_diff(domain,n=2,axis=1)/center_diff(vars, 2)
    plotter(d2Udv2, spots, vars)
    pylab.title("Second deriv w.r.t. var")
    pylab.xlabel("Var")
    pylab.ylabel("% of strike")
    pylab.show()
