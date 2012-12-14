#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

# import sys
# import os
# import itertools as it

from __future__ import division

import numpy as np

from Option import Option, BarrierOption
from Grid import Grid
from FiniteDifferenceEngine import FiniteDifferenceEngineADI
import utils
import scipy.stats

class BlackScholesOption(Option):

    def __init__(self
                , spot=100
                , strike=99
                , interest_rate=0.06
                , volatility=0.2
                , variance=None
                , tenor=1.0
                ):
        Option.__init__(self, spot, strike, interest_rate, volatility,
                variance, tenor)


        def mu_s(t, *dim):     return r * dim[0]

        def gamma2_s(t, *dim): return 0.5 * v * dim[0]**2
        self.coefficients = {()   : lambda t: -self.interest_rate.value,
                             (0,) : mu_s,
                             (0,0): gamma2_s}

        self.boundaries = {
                            # D: U = 0              Von Neumann: dU/dS = 1
                (0,)  : ((0, lambda *args: 0.0), (1, lambda t, x: 1.0)),
                            # D: U = 0              Free boundary
                (0,0) : ((0, lambda *args: 0.0), (None, lambda *x: None))}

        self.schemes = {}


    def compute_analytical(self):
        return self._call_delta()[0]

    @property
    def delta(self):
        return self._call_delta()[1]

    def _call_delta(self):
        N = scipy.stats.distributions.norm.cdf
        s = self.spot
        k = self.strike
        r = self.interest_rate.value
        t = self.tenor
        vol = np.maximum(1e-10, np.sqrt(self.variance.value))

        if self.tenor == 0.0:
            d1 = np.infty
        else:
            d1 = ((np.log(s/float(k)) + (r + 0.5*vol**2) * t)
                / (vol * np.sqrt(t)))
            d2 = d1 - vol*np.sqrt(t)
        return (N(d1)*s - N(d2)*k*np.exp(-r * t), N(d1))

class BlackScholesBarrierOption(BarrierOption, BlackScholesOption):
    def __init__(self
                , spot=100
                , strike=80
                , interest_rate=0.06
                , volatility=0.2
                , variance=None
                , tenor=1.0
                , top=None
                , bottom=None
                ):
        """Default is up and out."""
        assert not (top and bottom)
        BarrierOption.__init__(self, spot=spot, strike=strike,
                               interest_rate=interest_rate,
                               volatility=volatility, variance=variance,
                               tenor=tenor, top=top, bottom=bottom)
        BlackScholesOption.__init__(self, spot=spot, strike=strike,
                               interest_rate=interest_rate,
                               volatility=volatility, variance=variance,
                               tenor=tenor)

    def monte_carlo_paths(self, dt, n, callback=lambda *x: None):
        #TODO: make this work with callback, etc
        raise NotImplementedError
        dW = np.randn(t/dt + 1, npaths)
        dW[0,:] = 0
        W = np.cumsum(dW, axis=0)
        del dW
        rate_process = np.arange(t/dt + 1) * (r - 0.5*sig**2)*dt
        paths = spot * np.exp(rate_process[:,None] + sig*W*sqrt(dt))
        del rate_process, W
        barrier_paths = callback(paths, state)
        sT = barrier_paths[-1,:]


    def compute_analytical(self):
        assert not (self.top and self.bottom)
        exp = np.exp
        log = np.log
        sqrt = np.sqrt
        N = scipy.stats.distributions.norm.cdf
        s = self.spot
        k = self.strike
        r = self.interest_rate.value
        t = self.tenor
        # if not self.top:
            # raise NotImplementedError("Only doing up and * options.")
        c = BlackScholesOption.compute_analytical(self)
        if self.top:
            barrier = self.top
        elif self.bottom:
            barrier = self.bottom
        else:
            return c

        s = np.atleast_1d(s)
        s[s==0] = 1e-10
        knockin, h = barrier
        sig = sqrt(self.variance.value)
        lam = (r + sig*sig / 2) / (sig*sig)
        x1 = log(s/h)/(sig*sqrt(t)) + lam*sig*sqrt(t)
        y = log(h*h / (s*k)) / (sig*sqrt(t)) + lam*sig*sqrt(t)
        y1 = log(h/s)/(sig*sqrt(t)) + lam*sig*sqrt(t)

        if self.top:
            ret = (s*N(x1) - k*exp(-r*t) * N(x1 - sig*sqrt(t))
                - s*(h/s)**(2*lam)*(N(-y) - N(-y1))
                + k*exp(-r*t)*(h/s)**(2*lam-2)*(N(-y + sig*sqrt(t)) - N(-y1 + sig*sqrt(t))))
        elif self.bottom:
            ret = s*(h/s)**(2*lam)*N(y) - k*exp(-r*t)*(h/s)**(2*lam-2)*N(y-sig*sqrt(t))

        if not knockin:
            ret = c - ret

        return ret

    def features(self):
        d = BarrierOption.features(self)
        d[0] = "BlackScholesBarrierOption <%s>" % hex(id(self))
        return d





class BlackScholesFiniteDifferenceEngine(FiniteDifferenceEngineADI):

    def __init__(self, option,
            grid=None,
            spot_max=1500.0,
            nspots=100,
            spotdensity=7.0,
            force_exact=True,
            flip_idx_spot=False,
            schemes={},
            cache=True,
            coefficients=None,
            boundaries=None,
            force_bandwidth=False,
            verbose=True
            ):
        """@option@ is a BlackScholesOption"""
        self.cache = cache
        assert isinstance(option, Option)
        self.option = option

        if grid:
            self.grid = grid
            spots = self.grid.mesh[0]
        else:
            spots = utils.sinh_space(option.spot, spot_max, spotdensity, nspots, force_exact=force_exact)
            grid = Grid([spots], initializer=lambda *x: np.maximum(x[0]-option.strike,0))

        # self.spot_idx = np.argmin(np.abs(spots - np.log(spot)))
        # self.spot = np.exp(spots[self.spot_idx])
        self.spots = spots
        spot = spots[self.idx]
        self.spots = spots

        # self.option = BlackScholesOption(spot=np.exp(spot), strike=k, interest_rate=r,
                                     # variance=v, tenor=t)
        # G = Grid([spots], initializer=lambda *x: np.maximum(np.exp(x[0])-option.strike,0))

        def mu_s(t, *dim):
            # return np.zeros_like(dim[0], dtype=float) + (r - 0.5 * v)
            return option.interest_rate.value * dim[0]

        def gamma2_s(t, *dim):
            # return 0.5 * v + np.zeros_like(dim[0], dtype=float)
            return 0.5 * option.variance.value * dim[0]**2

        if not coefficients:
            coefficients = {()   : lambda t: -option.interest_rate.value,
                    (0,) : mu_s,
                    (0,0): gamma2_s}

        if not boundaries:
            boundaries = {          # D: U = 0              VN: dU/dS = 1
                    (0,)  : ((0, lambda *args: 0.0), (1, lambda t, x: 1.0)),
                    # (0,)  : ((0, lambda *args: 0.0), (1, lambda t, *x: np.exp(x[0]))),
                            # D: U = 0              Free boundary
                    (0,0) : ((0, lambda *args: 0.0), (None, lambda *x: 1.0))}

        self.grid = grid
        self.coefficients = coefficients
        self.boundaries = boundaries
        self.schemes = schemes
        self.force_bandwidth = force_bandwidth
        self._initialized = False


    @property
    def idx(self):
        return np.argmin(np.abs(self.spots - self.option.spot))

    @property
    def price(self):
        return self.grid[self.idx]

    @property
    def grid_analytical(self):
        BS = self.option
        orig = BS.spot
        BS.spot = self.grid.mesh[0]
        ret = BS.analytical
        BS.spot = orig
        return ret






def main():
    """Run main."""

    return 0

if __name__ == '__main__':
    main()
