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
                , tenor=1.0):
        self.attrs = ['spot', 'strike', 'interest_rate', 'variance', 'tenor']
        self.Type = "Option"
        self.spot = float(spot)
        self.strike = float(strike)
        # Constant rate
        if np.isscalar(interest_rate):
            # Constant rate
            self.interest_rate = MeanRevertingProcess(mean=interest_rate, volatility=0)
        else:
            self.interest_rate =  interest_rate

        if np.isscalar(variance):
            # Constant rate
            self._variance = MeanRevertingProcess(mean=variance, volatility=0)
        elif variance is not None:
            self._variance = variance
        else:
            # Constant rate
            self._variance = MeanRevertingProcess(mean=volatility**2.0, volatility=0)

        self.tenor = float(tenor)
        self._analytical = None
        # Only set monte_carlo_callback if it's empty
        try:
            if self.monte_carlo_callback is None:
                self.monte_carlo_callback = lambda *x: None
        except AttributeError:
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
        "The volatility property, computed from variance."
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


    def __repr__(self):
        l = ["{attr}={val}".format(attr=attr, val=repr(getattr(self, attr))) for attr in self.attrs]
        s = self.Type + "(" + "\n, ".join(l) + ')'
        return s


    def __str__(self):
        return "\n".join(self.features())


    def monte_carlo(self, dt=0.001, npaths=10000, with_payoff=False,
                    callback=None):
        if not callback:
            callback = self.monte_carlo_callback
        start = time.time()
        s = self.monte_carlo_paths(dt, npaths, callback)
        duration = time.time() - start
        payoff = np.maximum(s - self.strike, 0)
        p = np.exp(-self.interest_rate.value * self.tenor) * payoff
        stdp = np.std(p)
        ret = { "expected": np.mean(p)
               , "error": stdp / np.sqrt(npaths)
               , "duration": duration
               , "n": npaths
               , "std": stdp
               , "dt": dt
               }
        if with_payoff:
            ret['payoff'] = payoff
        return ret


    def __eq__(self, other):
        for attr in self.attrs:
            if not getattr(self, attr) == getattr(other, attr):
                # print self.Type, attr
                return False
        return True


    def monte_carlo_paths(self, dt=None, npaths=None, callback=lambda *x: None):
        raise NotImplementedError



class MeanRevertingProcess(object):

    attrs = ['mean', 'volatility', 'value', 'reversion']

    def __init__(  self
            , mean=0
            , volatility=1
            , value=None
            , reversion=0
             ):
        self.Type = 'MeanRevertingProcess'
        self.value = value if value is not None else mean
        self.reversion = reversion
        self.mean = mean
        self.volatility = volatility

    def __call__(self, t=None):
        return self.value

    def __str__(self):
        return "%s (mean: %s, vol: %s)" % (self.value, self.mean, self.volatility)

    def __eq__(self, other):
        for attr in self.attrs:
            if not getattr(self, attr) == getattr(other, attr):
                # print 'MRP', attr
                return False
        return True

    def __repr__(self):
        l = ["{attr}={val}".format(attr=attr, val=repr(getattr(self, attr))) for attr in self.attrs]
        s = self.Type + "(" + ", ".join(l) + ')'
        return s

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
        self.Type = 'BarrierOption'
        self.attrs += ['top', 'bottom']
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


    def __repr__(self):
        l = ["{attr}={val}".format(attr=attr, val=repr(getattr(self, attr))) for attr in self.attrs]
        s = self.Type + "(" + "\n, ".join(l) + ')'
        return s


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
    o = Option()
    p = eval(repr(o))
    print o == p
    q = eval(repr(p))
    print p == q

    b = BarrierOption()
    print repr(b)
    print
    p = eval(repr(b))
    print b == p
    print
    print repr(p)
    print
    q = eval(repr(p))
    print p == q
    print
    print repr(q)
    return 0

if __name__ == '__main__':
    main()
