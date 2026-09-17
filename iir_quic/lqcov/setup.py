from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import sysconfig


eigen_path = "D:/Program/eigen-3.4.0"  # Windows路径
if not os.path.exists(eigen_path):  # 如果Windows路径不存在，尝试Linux默认路径
    eigen_path = "/usr/include/eigen3"

# MinGW has no MS_WIN64 defined by default, so the CPython headers leave
# SIZEOF_VOID_P undefined and Cython's generated size check fails to compile.
# The MinGW runtime is linked statically so the .pyd loads without PATH setup.
if sys.platform == "win32":
    extra_compile_args = ["-DMS_WIN64"]
    extra_link_args = ["-static", "-static-libgcc", "-static-libstdc++"]
else:
    extra_compile_args = []
    extra_link_args = []

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
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="lqcov_cython",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",  # 使用 Python 3
            "embedsignature": True,  # 保留函数签名
        },
    ),
)