#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

# import sys
# import os
# import itertools as it

import numpy as np

from Option import Option
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
                , dt = None
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
        vol = np.maximum(1e-10, self.volatility)

        if self.tenor == 0.0:
            d1 = np.infty
        else:
            d1 = ((np.log(s/k) + (r+0.5*vol**2) * t)
                / (vol * np.sqrt(t)))
            d2 = d1 - vol*np.sqrt(t)
        return (N(d1)*s - N(d2)*k*np.exp(-r * t), N(d1))


class BlackScholesFiniteDifferenceEngine(FiniteDifferenceEngineADI):

    def __init__(self, option,
            spot_max=1500.0,
            nspots=100,
            spotdensity=7.0,
            force_exact=True,
            flip_idx_spot=False,
            schemes=None,
            cache=True,
            verbose=True
            ):
        """@option@ is a BlackScholesOption"""
        self.cache = cache
        self.option = option

        spots = utils.sinh_space(option.spot, spot_max, spotdensity, nspots, force_exact=force_exact)

        # self.spot_idx = np.argmin(np.abs(spots - np.log(spot)))
        # self.spot = np.exp(spots[self.spot_idx])
        self.idx = np.argmin(np.abs(spots - option.spot))
        spot = spots[self.idx]

        # self.option = BlackScholesOption(spot=np.exp(spot), strike=k, interest_rate=r,
                                     # variance=v, tenor=t)
        # G = Grid([spots], initializer=lambda *x: np.maximum(np.exp(x[0])-option.strike,0))
        G = Grid([spots], initializer=lambda *x: np.maximum(x[0]-option.strike,0))

        def mu_s(t, *dim):
            # return np.zeros_like(dim[0], dtype=float) + (r - 0.5 * v)
            return option.interest_rate.value * dim[0]

        def gamma2_s(t, *dim):
            # return 0.5 * v + np.zeros_like(dim[0], dtype=float)
            return 0.5 * option.variance.value * dim[0]**2

        self.coefficients = {()   : lambda t: -option.interest_rate.value,
                  (0,) : mu_s,
                  (0,0): gamma2_s}

        self.boundaries = {          # D: U = 0              VN: dU/dS = 1
                (0,)  : ((0, lambda *args: 0.0), (1, lambda t, x: 1.0)),
                # (0,)  : ((0, lambda *args: 0.0), (1, lambda t, *x: np.exp(x[0]))),
                        # D: U = 0              Free boundary
                (0,0) : ((0, lambda *args: 0.0), (None, lambda *x: 1.0))}

        FiniteDifferenceEngineADI.__init__(self, G, coefficients=self.coefficients,
                boundaries=self.boundaries, schemes={})

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
