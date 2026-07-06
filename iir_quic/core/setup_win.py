import os
from pathlib import Path

import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize


BASE_DIR = Path(__file__).resolve().parent


def _existing_dir(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_dir() else None


def _find_lapack_root_from_path():
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        bin_dir = _existing_dir(entry)
        if bin_dir is None:
            continue
        parent = bin_dir.parent
        include_dir = parent / "include"
        lib_dir = parent / "lib"
        if include_dir.is_dir() and lib_dir.is_dir() and any(
            (lib_dir / name).exists() for name in ("liblapack.dll.a", "liblapack.lib")
        ):
            return parent
    return None


def _detect_lapack_root():
    for env_name in ("LAPACK_ROOT", "LAPACK_DIR"):
        root = _existing_dir(os.environ.get(env_name))
        if root is not None:
            return root

    path_root = _find_lapack_root_from_path()
    if path_root is not None:
        return path_root

    default_root = Path(r"D:\Program\lapack")
    if default_root.is_dir():
        return default_root

    raise RuntimeError(
        "Cannot find LAPACK. Set LAPACK_ROOT/LAPACK_DIR, or add the LAPACK bin directory to PATH."
    )


LAPACK_ROOT = _detect_lapack_root()
lapack_include_path = _existing_dir(os.environ.get("LAPACK_INCLUDE_DIR")) or (LAPACK_ROOT / "include")
lapack_lib_path = _existing_dir(os.environ.get("LAPACK_LIB_DIR")) or (LAPACK_ROOT / "lib")

ext_modules = [
    Extension(
        name="IRL_core",
        sources=[str(BASE_DIR / "QUIC_IRL.C"), str(BASE_DIR / "IRL_core.pyx")],
        include_dirs=[numpy.get_include(), str(lapack_include_path)],
        library_dirs=[str(lapack_lib_path)],
        libraries=["lapack", "blas"],
        extra_compile_args=[
            "-O3",
            "-mtune=native",
            "-march=native",
            "-w",
            "-DMS_WIN64",
        ],
        language="c++",
    )
]

setup(
    name="IRL_core",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": 3}),
)