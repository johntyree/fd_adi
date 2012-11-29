#!/usr/bin/env python
# coding: utf8
"""Options"""

# import sys
# import os
# import itertools as it

import numpy as np
import scipy.stats


class Option(object):
    """Base class for vanilla option contracts."""
    def __init__(self
                 , spot=100
                 , strike=99
                 , interest_rate=0.06
                 , volatility=0.2
                 , variance=None
                 , tenor=1.0
                 , dt = None
                 ):
        self.spot = spot
        self.strike = strike
        # Constant rate
        self.interest_rate = MeanRevertingProcess(mean=interest_rate, volatility=0)

        if variance is not None:
            volatility = np.sqrt(variance)
        else:
            variance = volatility**2
            # Constant rate
        self._variance = MeanRevertingProcess(mean=variance, volatility=0)

        self.tenor = tenor
        self.dt = dt

    # We can fake const attributes by using properties wihtout setters.
    @property
    def variance(self):
        return self._variance

    @property
    def volatility(self):
        "The volatility property, based on variance."
        return np.sqrt(self.variance.value)

    # def __repr__(self):
        # return "\n\t".join(self.features())

    def __str__(self):
        return [ "Option <%s>" % hex(id(self))
            , "Spot: %s" % self.spot
            , "Strike: %s" % self.strike
            , "Interest: %s" % self.interest_rate()
            , "Volatility: %s" % self.volatility
            , "Variance: %s" % self.variance
            , "Tenor: %s" % self.tenor
            , "dt: %s" % self.dt
            ]


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
                variance, tenor, dt)

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

    @property
    def analytical(self):
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

class MeanRevertingProcess(object):
    def __init__(  self
            , mean=0
            , volatility=1
            , value=None
             ):
        self.value = value if value is not None else mean
        self.mean = mean
        self.volatility = volatility

    def __call__(self, t=None):
        return self.value

    def __repr__(self):
        return "%s (%s, %s)" % (self.value, self.mean, self.volatility)

    # def __add__(self, val):
        # self.add(val, inplace=False)
    # def __iadd__(self, val):
        # self.add(val, inplace=True)
    # def __mul__(self, val):
        # self.mul(val, inplace=False)
    # def __imul__(self, val):
        # self.mul(val, inplace=True)

    # def add(self, other, inplace=False):
        # MeanRevertingProcess(self.mean,
                             # self.volatility,
                             # self.value+other)



# class AsianOption(Option):
    # """Asian style options."""

def main():
    """Run main."""
    import time
    H = HestonOption( spot=100
                    , strike=99
                    , interest_rate=0.06
                    , volatility = 0.2
                    , tenor=1.0
                    , mean_reversion=1
                    , mean_variance=None # default to current var
                    , vol_of_variance = 0.4
                    , correlation = 0.0
               )
    print H
    print time.time()
    return 0

if __name__ == '__main__':
    main()
