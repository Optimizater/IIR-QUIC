# Introduction

IIR-QUIC[^ref] is a novel Inexact Iteratively Reweighted algorithm based on the QUadratic approximation for sparse Inverse Covariance (QUIC) method.

# Installation

Python 3.11 is required. Create the environment with [requirements.txt](requirements.txt).

For Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For Linux and macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The environment interpreter is `.venv\Scripts\python.exe` on Windows and `.venv/bin/python` elsewhere.

The build products are not tracked in this repository (Cython translations `IRL_core.cpp`/`lqcov.cpp`, compiled
extensions `*.pyd`/`*.so`, `build/`), so the core module has to be built after cloning, following the instructions
in [iir_quic/core/README.md](iir_quic/core/README.md).

You can use [test.py](test.py) as an example to understand how to use IIR-QUIC to estimate the precision matrix of a tridiagonal matrix.

[^ref]: _Efficient QUIC-Based Inexact Iterative Reweighting for Sparse Inverse Covariance Estimation with Nonconvex Partly Smooth Regularization_
