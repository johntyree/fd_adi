#!/usr/bin/env python
# coding: utf8
"""<+Module Description.+>"""

# import sys
# import os
# import itertools as it

import utils
from utils import sinh_space, exponential_space
import numpy as np

class Grid(object):
    """#:class documentation"""
    def __init__(self
            , mesh=(np.linspace(0, 1, 5), np.linspace(0, 2, 5))
            , initializer=lambda x, y: x+y
        ):
        """Create an n-dimensional grid from an iterable of monotonic spaces."""

        mesh = tuple(mesh)
        for m in mesh:
            if len(m) > 1:
                s = np.sign(np.diff(m))
                assert(all(s == s[0]))
        self._mesh = mesh
        self.domain = initializer(*(x.T for x in np.meshgrid(*mesh)))
        self._shape = tuple(map(len, mesh))
        self.dx = [np.hstack((np.nan, np.diff(m))) for m in mesh]

        self.initializer = initializer

    def reset(self):
        self.__init__(mesh=self.mesh, initializer=self.initializer)

    def copy(self):
        return Grid(mesh=self._mesh, initializer=self.initializer)

    def __str__(self):
        keys = utils.attr_dict(self)
        keys['hexid'] = hex(id(self))
        keys['ndim'] = self.ndim
        return ("""
Grid object <%(hexid)s>
    mesh: (%(_mesh)s
    ndim: %(ndim)i
    dx  : %(dx)s
    shape: %(shape)s
    domain:\n%(domain)s
        """) % keys

    @property
    def mesh(self):
        return self._mesh

    def max(self, dim):
        return self.mesh[dim][-1]
    def min(self, dim):
        return self.mesh[dim][0]

    @property
    def shape(self):
        return self._shape

    def __getattr__(self, name):
        return self.domain.__getattribute__(name)

    def __getitem__(self, indices):
        return self.domain[indices]



def test_copy():
    """Run main."""
    k = 0.5
    g = Grid(initializer=lambda x,y: np.maximum(x-k,0))
    h = g.copy()
    g.domain[1,:] = 4
    print g.mesh
    print h.mesh
    print
    print g.domain
    print h.domain
    print g.shape, h.shape
    assert(g.domain != h.domain)
    g.reset()
    assert(g.domain != h.domain)
    g.reset()
    print g.mesh
    print g.domain
    print g.shape
    return 0

if __name__ == '__main__':
    main()
