from distutils.core import setup
# from distutils.extension import Extension
from Cython.Distutils import build_ext,Extension
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    packages = ['FiniteDifference'],
    scripts = ['benchmark.py', 'convergence.py', 'heat_eq.py', 'HestonExample.py'],
    ext_package = 'FiniteDifference',
    ext_modules = [
        Extension("FiniteDifferenceEngine",
            sources=["FiniteDifference/FiniteDifferenceEngine.pyx"],
            extra_compile_args=["-O3"],
            include_dirs=[numpy.get_include(), "FiniteDifference"],
            cython_include_dirs=["FiniteDifference/"],
            cython_c_in_temp=True,
            # cython_cplus=False,
            # language="c++",
            # libraries=["-lfoo"]
            # extra_objects=["libsomelib.so"],
            # runtime_library_dirs=["."]
        )
    ]
)
