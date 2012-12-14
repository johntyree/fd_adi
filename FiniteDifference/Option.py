#!/usr/bin/env python
# coding: utf8
"""Options"""


from __future__ import division

import time
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
        self.monte_carlo_callback = lambda *x: None


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


    def features(self):
        return [ "Option <%s>" % hex(id(self))
            , "Spot: %s" % self.spot
            , "Strike: %s" % self.strike
            , "Interest: %s" % self.interest_rate()
            , "Volatility: %s" % self.volatility
            , "Variance: %s" % self.variance
            , "Tenor: %s" % self.tenor
            ]

    def __str__(self):
        return "\n".join(self.features())

    def monte_carlo(self, dt=0.01, npaths=100000,
                    callback=None):
        if not callback:
            callback = self.monte_carlo_callback
        start = time.time()
        s = self.monte_carlo_paths(dt, npaths, callback)
        duration = time.time() - start
        payoff = np.maximum(s - self.strike, 0)
        p = np.exp(-self.interest_rate.value * self.tenor) * payoff
        stdp = np.std(p)
        return { "expected": np.mean(p)
               , "error": stdp / np.sqrt(npaths)
               , "duration": duration
               , "n": npaths
               , "std": stdp
               }


    def monte_carlo_paths(self, dt=None, npaths=None, callback=None):
        raise NotImplementedError



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
        self._top = top
        self._bottom = bottom
        self.monte_carlo_callback = self._callback_from_boundary((self.bottom, self.top))

    def top():
        doc = "The top boundary."
        def fget(self):
            return self._top
        def fset(self, value):
            self._top = value
            self.monte_carlo_callback = self._callback_from_boundary((self.bottom, self.top))
        return locals()
    top = property(**top())

    def bottom():
        doc = "The bottom boundary."
        def fget(self):
            return self._bottom
        def fset(self, value):
            self._bottom = value
            self.monte_carlo_callback = self._callback_from_boundary((self.bottom, self.top))
        return locals()
    bottom = property(**bottom())


    def features(self):
        d = Option.features(self)
        d[0] = "BarrierOption <%s>" % hex(id(self))
        d.extend([ "Upper Barrier: %s" % (self.top,)
                 , "Lower Barrier: %s" % (self.bottom,)])
        return d



    def _callback_from_boundary(self, b):
        def knockin_top(bound):
            def f(s, state):
                state |= s >= bound
            return f
        def knockout_top(bound):
            def f(s, state):
                state &= s < bound
            return f
        def knockin_bot(bound):
            def f(s, state):
                state |= s <= bound
            return f
        def knockout_bot(bound):
            def f(s, state):
                state &= s > bound
            return f
        def apply_both(f1, f2):
            def f(*args, **kwargs):
                f2(*args, **kwargs)
                f1(*args, **kwargs)
            return f
        f = lambda *x: None
        bot, top = b
        if bot:
            if bot[0]: # knockin
                f = apply_both(f, knockin_bot(bot[1]))
            else: #knockout
                f = apply_both(f, knockout_bot(bot[1]))
        if top:
            if top[0]: # knockin
                f = apply_both(f, knockin_top(top[1]))
            else: #knockout
                f = apply_both(f, knockout_top(top[1]))
        return f



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
