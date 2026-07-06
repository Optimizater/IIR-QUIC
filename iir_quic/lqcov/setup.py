from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sysconfig


eigen_path = "D:/Program/eigen-3.4.0"  # Windows path
if not os.path.exists(eigen_path):  # If the Windows path does not exist, try the Linux default path
    eigen_path = "/usr/include/eigen3"

ext_modules = [
    Extension(
        "lqcov_cython",
        sources=["lqcov.pyx", "lqcov_impl.cpp"],
        language="c++",
        include_dirs=[
            eigen_path,
            np.get_include(),
            os.path.dirname(os.path.abspath(__file__)),  # numpy_eigen.hpp
            sysconfig.get_path('include')  # Python.h
        ],
        # extra_compile_args=["-std=c++11"],
    )
]

setup(
    name="lqcov_cython",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",  # Use Python 3
            "embedsignature": True,
        },
    ),
)