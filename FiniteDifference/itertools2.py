#!/usr/bin/env python

from itertools import *

def iterate(f, x0):
    """Return a stream of applications of f:
        [x, f(x), f(f(x))...]
    """
    while True:
        yield x0
        x0 = f(x0)

def itake(n, seq, step=1):
    """Return an iterator over seq that stops after n results.

        anim(wireframe, itake(10, domainlist), spots, vars)
    """
    for i, x in enumerate(seq):
        if i/step == n:
            raise StopIteration
        elif not i % step:
            yield x
