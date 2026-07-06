'''
IRL core module

modified quic function
'''

import os
from pathlib import Path


def _existing_dir(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_dir() else None


def _path_entries():
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        path = _existing_dir(entry)
        if path is not None:
            yield path


if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _dll_dirs = []

    for env_name in ("LAPACK_ROOT", "LAPACK_DIR"):
        root = _existing_dir(os.environ.get(env_name))
        if root is not None:
            _dll_dirs.append(root / "bin")

    mingw_root = _existing_dir(os.environ.get("MINGW_DIR"))
    if mingw_root is not None:
        _dll_dirs.append(mingw_root / "bin")

    for path_entry in _path_entries():
        if path_entry not in _dll_dirs:
            if (path_entry / "liblapack.dll").exists() or (path_entry / "libblas.dll").exists():
                _dll_dirs.append(path_entry)
            elif (path_entry / "libgfortran-5.dll").exists() or (path_entry / "libstdc++-6.dll").exists():
                _dll_dirs.append(path_entry)

    for fallback in (Path(r"D:\Program\lapack\bin"), Path(r"D:\Program\mingw64\bin")):
        if fallback.is_dir() and fallback not in _dll_dirs:
            _dll_dirs.append(fallback)

    for _dll_dir in _dll_dirs:
        if _dll_dir.is_dir():
            os.add_dll_directory(str(_dll_dir))

from . import IRL_core
import numpy as np

def quic(
    S, L, mode="default", tol=1e-6, max_iter=1000, X0=None, W0=None, path=None, msg=0
):
    """
    Args:
        S:The empirical nxn covariance matrix.

        L:Regularization parameters per element of the inverse
          covariance matrix. Can be a scalar or nxn matrix.

        mode:Computation mode: one of "default", "path", "trace".

        tol: **(useless para)** Convergence threshold.

        max_iter:Maximum number of Newton iterations.

        X0:Initial guess for the inverse covariance matrix. If
                    not provided, the diagonal identity matrix is used.

        W0:Initial guess for the covariance matrix. If not provided
           the diagonal identity matrix is used.

        path:In "path" mode, an array of float values for scaling L.

        msg:Verbosity level.
    """

    assert mode in ["default", "path", "trace"], (
        "QUIC:arguments\n" + "Invalid mode, use: 'default', 'path' or 'trace'."
    )

    # Empircal covariance matrix S
    Sn, Sm = S.shape
    # assert S.dtype is np.float64, "QUIC:type\n" +\
    #        "Expected a double covariance matrix S."
    assert Sn == Sm, (
        "QUIC:dimensions\n" + "Expected a square empircal covariance matrix S."
    )

    # Regularization parameter matrix L
    if (type(L) == float) or (type(L) == np.float64):
        _L = np.empty((Sn, Sm))
        _L[:] = L
    else:
        Ln, Lm = L.shape
        assert (Ln == Sn) and (Lm == Sn), (
            "QUIC:dimensions\n"
            + "The regularization parameter L is not a scalar or a matching matrix."
        )
        assert L.dtype == np.float64, (
            "QUIC:type\n" + "Expected a double regularization parameter matrix L."
        )
        _L = L

    # Path
    if mode == "path":
        assert path is not None, (
            "QUIC:dimensions\n" + "Please specify the path scaling values."
        )
        # assert (type(path) is np.ndarray) and (path.dtype is np.float64), "QUIC:type\n" +\
        #        "Expected a double array for path."
        pathLen = path.shape[0]
    else:
        path = np.empty(1)
        pathLen = 1

    if X0 is None:
        assert W0 is None, (
            "QUIC:initializations\n"
            + "You specified an initial value for W0 but not for X0."
        )
        if mode == "path":
            # Note here: memory layout is important:
            # a row of X/W holds a flattened Sn x Sn matrix,
            # one row for every element in _path_.
            X = np.empty((pathLen, Sn * Sn))
            X[0, :] = np.eye(Sn).ravel()
            W = np.empty((pathLen, Sn * Sn))
            W[0, :] = np.eye(Sn).ravel()
        else:
            X = np.eye(Sn)
            W = np.eye(Sn)
    else:
        assert W0 is not None, (
            "QUIC:initializations\n"
            + "You specified an initial value for X0 but not for W0."
        )

        assert X0.dtype == np.float64, (
            "QUIC:type\n" + "Expected a double initial inverse covariance matrix X0."
        )

        assert W0.dtype == np.float64, (
            "QUIC:type\n" + "Expected a double initial covariance matrix W0."
        )

        X0n, X0m = X0.shape
        assert (X0n == Sn) and (X0m == Sn), (
            "QUIC:dimensions\n"
            + "Matrix dimensions should match for initial inverse covariance matrix X0."
        )

        W0n, W0m = W0.shape
        assert (W0n == Sn) and (W0m == Sn), (
            "QUIC:dimensions\n"
            + "Matrix dimensions should match for initial covariance matrix W0"
        )

        if mode == "path":
            # See note above wrt memory layout
            X = np.empty((pathLen, Sn * Sn))
            X[0, :] = X0.ravel()
            W = np.empty((pathLen, Sn * Sn))
            W[0, :] = W0.ravel()
        else:
            X = np.empty(X0.shape)
            X[:] = X0
            W = np.empty(W0.shape)
            W[:] = W0

    if mode == "path":
        optSize = pathLen
        iterSize = pathLen
    elif mode == "trace":
        optSize = max_iter
        iterSize = 1
    else:
        optSize = 1
        iterSize = 1

    opt = np.zeros(optSize)
    cputime = np.zeros(optSize)
    dGap = np.zeros(optSize)
    iters = np.zeros(iterSize, dtype=np.uint32)

    info_list = np.zeros(4)

    IRL_core.quic(
        mode.encode("utf-8"),
        Sn,
        S,
        _L,
        pathLen,
        path,
        tol,
        msg,
        max_iter,
        X,
        W,
        opt,
        cputime,
        iters,
        dGap,
        info_list
    )

    if optSize == 1:
        opt = opt[0]
        cputime = cputime[0]
        dGap = dGap[0]

    if iterSize == 1:
        iters = iters[0]
    
    logdet = info_list[0]
    trSX = info_list[1]
    numActive = info_list[2]
    stepsize = info_list[3]

    return X, W, opt, cputime, iters, dGap, logdet, trSX, numActive, stepsize
