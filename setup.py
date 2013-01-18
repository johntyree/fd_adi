#!/usr/bin/env python
# coding: utf8

from setuptools import setup
from Cython.Distutils import build_ext, Extension
import Cython.Compiler.Options
from Cython.Build import cythonize


import numpy


cython_modules = cythonize('FiniteDifference/*.pyx',
                           cython_include_dirs=["FiniteDifference", numpy.get_include()],
                           nthreads=4
                           )
for m in cython_modules:
    m.include_dirs += ["FiniteDifference", numpy.get_include()]
    m.extra_compile_args += ["-std=c++0x", "-fpermissive"]

setup(
    author = "John Tyree",
    name = "FiniteDifference",
    version = 0,
    cmdclass = {'build_ext': build_ext},
    packages = ['FiniteDifference'],
    scripts = ['benchmark.py', 'convergence.py', 'heat_eq.py', 'HestonExample.py'],
    ext_modules = cython_modules
)
