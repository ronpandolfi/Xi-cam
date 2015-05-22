#! /usr/bin/env python
from __future__ import division, absolute_import, print_function
import os
from numpy.distutils.core import Extension
from Cython.Build import cythonize

os.environ["CC"] = "cc.exe"
os.environ["CXX"] = "c++.exe"

boost_inc = '-IC:\\boost_1_58_0\\boost1580'

ext = Extension(name='cWarpImage',
                sources=['cWarpImage.cc', 'warpimage.h', 'remesh.h', 'remesh.cc', 'kdtree2.hpp', 'kdtree2.cpp'],
                extra_compile_args=['-O3', '-ffast-math', boost_inc],
                # extra_compile_args = ['-O0 -g', boost_inc]
)

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='cWarpImage',
          description="Transform image to new coordinate system, faster",
          version="1.0.0",
          author="Dinesh Kumar",
          author_email="dkumar@lbl.gov",
          ext_modules=cythonize(ext)
    )
# End of setup_example.py
