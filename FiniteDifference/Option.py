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

    def __init__( self
                , spot=100
                , strike=99
                , interest_rate=0.06
                , volatility=0.2
                , variance=None
                , tenor=1.0
                ):
        self.spot = float(spot)
        self.strike = float(strike)
        # Constant rate
        self.interest_rate = MeanRevertingProcess(mean=interest_rate, volatility=0)

        if variance is not None:
            volatility = np.sqrt(variance)
        else:
            variance = volatility**2.0
            # Constant rate
        self._variance = MeanRevertingProcess(mean=variance, volatility=0)

        self.tenor = float(tenor)
        self._analytical = None

    # We can fake const attributes by using properties wihtout setters.

    @property
    def analytical(self):
        return self.compute_analytical()
        # if self._analytical is None:
            # self._analytical = self.compute_analytical()
        # return self._analytical


    @property
    def variance(self):
        return self._variance


    @property
    def volatility(self):
        "The volatility property, based on variance."
        return np.sqrt(self.variance.value)

    # def __repr__(self):
        # return "\n\t".join(self.features())


    def _desc(self):
        return [ "Option <%s>" % hex(id(self))
            , "Spot: %s" % self.spot
            , "Strike: %s" % self.strike
            , "Interest: %s" % self.interest_rate()
            , "Volatility: %s" % self.volatility
            , "Variance: %s" % self.variance
            , "Tenor: %s" % self.tenor
            ]

    def __str__(self):
        return "\n".join(self._desc())


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

    def __str__(self):
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


class BarrierOption(Option):
    """Base class for barrier option contracts."""

    def __init__( self
                , spot=100
                , strike=99
                , interest_rate=0.06
                , volatility=0.2
                , variance=None
                , tenor=1.0
                , top=None
                , bottom=None
                ):
        Option.__init__(self, spot=spot, strike=strike,
                        interest_rate=interest_rate,
                        volatility=volatility,
                        variance=variance,
                        tenor=tenor)
        self.top = top
        self.bottom = bottom

    def _desc(self):
        d = Option._desc(self)
        d[0] = "BarrierOption <%s>" % hex(id(self))
        d.extend([ "Upper Barrier: %s" % (self.top,)
                 , "Lower Barrier: %s" % (self.bottom,)])
        return d



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
