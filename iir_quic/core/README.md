# Introduction

This module is modified based on the [QUIC code](https://github.com/osdf/pyquic) and provides the code for solving sub-problems in IIR-QUIC.

# How to build

To build the module, you need to have `lapack` installed.

For Ubuntu, you can install it using the following command:

```bash
sudo apt-get install libblas-dev liblapack-dev
```

Then build the module with the interpreter of the virtual environment set up in the
[top-level README](../../README.md) — a system `python` normally has no Cython installed. Run the build from inside
this directory: `--inplace` drops the extension into the current working directory.

```bash
cd iir_quic/core
../../.venv/bin/python setup.py build_ext --inplace
```

# Windows

LAPACK is located through the `LAPACK_ROOT`/`LAPACK_DIR` environment variables (fallback: `D:\Program\lapack`),
and the build uses the MinGW-w64 `g++` toolchain, so it has to be selected explicitly:

```powershell
cd iir_quic\core
..\..\.venv\Scripts\python.exe setup_win.py build_ext --inplace --compiler=mingw32
```

The `-DMS_WIN64` flag is added automatically; `liblapack.dll` and the MinGW runtime DLLs are located at import
time by [`__init__.py`](__init__.py), which searches `LAPACK_ROOT`/`LAPACK_DIR`/`MINGW_DIR`, the `PATH` entries and
the `D:\Program\lapack\bin` / `D:\Program\mingw64\bin` fallbacks.
