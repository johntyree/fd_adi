
import glob
import os
import sys

# from distutils.core import setup
from setuptools import setup
# from distutils.extension import Extension
from Cython.Distutils import build_ext, Extension
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

import numpy

cython_sources = glob.glob(os.path.join("FiniteDifference", "*.pyx"))
cython_modules = []

for fn in cython_sources:
    cython_modules.append(
        Extension(os.path.splitext(os.path.basename(fn))[0],
            sources=[fn],
            extra_compile_args=["-O3"],
            include_dirs=[numpy.get_include()],
            cython_include_dirs=[numpy.get_include()],
            cython_c_in_temp=True,
            # cython_directives={'annotate': True},
            cython_cplus=True,
            # language="c++",
            # libraries=["-lfoo"]
            # extra_objects=["libsomelib.so"],
            # runtime_library_dirs=["."]
        )
    )

setup(
    cmdclass = {'build_ext': build_ext},
    packages = ['FiniteDifference'],
    scripts = ['benchmark.py', 'convergence.py', 'heat_eq.py', 'HestonExample.py'],
    ext_package = 'FiniteDifference',
    ext_modules = cython_modules
)
