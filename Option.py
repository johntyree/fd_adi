#!/usr/bin/env python
# coding: utf8
"""Options"""

# import sys
# import os
# import itertools as it

import numpy as np


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
        self.interest_rate = MeanRevertingProcess(mean=interest_rate, volatility=0)

        if variance is not None and volatility is not None:
            assert(np.close(volatility**2, variance))
        if volatility:
            variance = volatility**2
        else:
            volatility = np.sqrt(variance)
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


class HestonOption(Option):
    def __init__(self
                , spot=100
                , strike=99
                , interest_rate=0.06
                , volatility=0.2
                , tenor=1.0
                , dt=None
                , mean_reversion=1
                , mean_variance=None
                , vol_of_variance=0.4
                , correlation=0):
        Option.__init__(self
                , spot=spot
                , strike=strike
                , interest_rate=interest_rate
                , volatility=volatility
                , tenor=tenor
                , dt=dt)
        self.mean_reversion = mean_reversion
        self.mean_variance = mean_variance if mean_variance is not None else volatility**2
        self.vol_of_variance = vol_of_variance
        self.correlation = correlation

    def __repr__(self):
        return "\n\t".join(self.features())

    def features(self):
        s = Option.features(self)
        s[0] = "HestonOption <%s>" % hex(id(self))
        s.extend(
                [ "Mean Reversion: %s" % self.mean_reversion
                , "Mean Variance: %s" % self.mean_variance
                , "Vol of Variance: %s" % self.vol_of_variance
                , "Correlation %s" % self.correlation
                ])
        return s


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
