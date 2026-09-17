"""
Private implementation module for the public API functions.
Import via iir_quic/__init__.py
"""

import numpy as np

from .IRLQUIC import iir_quic as _iir_quic_impl
from .config import RAND_SEED


def _initial_point(S: np.ndarray, rho: float, X0, eps0) -> tuple[np.ndarray, np.ndarray]:
    """Validate the optional starting point/perturbation, filling in the defaults."""
    _dim = S.shape[0]

    if type(X0) == np.ndarray:
        assert X0.shape[0] == X0.shape[1] and X0.shape[0] == _dim, "Input X0 must be a square matrix with the same dimension as S."
    elif X0 == None:
        X0 = np.diag(1 / (np.diag(S)  + rho))
    else:
        raise ValueError("Input X0 must be ndarray or None format.")

    if type(eps0) == np.ndarray:
        assert eps0.shape[0] == eps0.shape[1] and eps0.shape[0] == _dim, "Input eps0 must be a square matrix with the same dimension as S."
    elif eps0 == None:
        np.random.seed(RAND_SEED)
        mat = np.random.randn(_dim, _dim) * 0.5
        mat = (mat + mat.T) / 2
        eps0 = np.abs(mat)
    else:
        raise ValueError("Input eps0 must be ndarray or None format.")

    return X0, eps0


def _check_common(S: np.ndarray, rho: float, mu: float, MaxIter: int, tolerance: float, msg: bool):
    assert type(S) == np.ndarray and S.shape[0] == S.shape[1], "Input S must be a square matrix."
    assert type(mu) == float and 0 < mu and mu < 1, "Parameter mu must be in the range (0, 1)."
    assert type(rho) == float and rho > 0, "Parameter rho must be positive."
    assert type(MaxIter) == int and MaxIter > 0, "MaxIter must be a positive integer."
    assert type(tolerance) == float and tolerance > 0, "Tolerance must be a positive number."
    assert type(msg) == bool, "msg must be a boolean value."


def iir_quic(
    S: np.ndarray,
    rho: float,
    X0: np.ndarray | None = None,
    eps0: np.ndarray | None = None,
    mu: float = 0.1,
    p: float = 0.5,
    penalty: str = "lp",
    MaxIter: int = 3000,
    tolerance: float = 1e-5,
    msg=False,
    vartheta: float = 0.1,
    inner_max_iter: int = 2000,
    zero_diag: bool | None = None,
):
    """
    Parameters:
        S(np.ndarray):      The empirical nxn covariance matrix.
        rho (float):        Regularization parameter.
        X0 (np.ndarray):    Initial point for the iterates.
        eps0 (np.ndarray):  Initial epsilon values.
        mu (float):         Epsilon decay factor, which belongs to (0,1).
        p (float):          Parameter of the non-convex lp regularization term
                            (0 < p < 1).  Unused for the "scad" and "mcp"
                            penalties, whose weights are driven by `rho`.
        penalty (str):      Non-convex regularizer: "lp", "scad" or "mcp".
        MaxIter (int):      Maximum number of iterations.
        tolerance (float):  Tolerance for the stopping criterion.
        msg(bool):          print message if `msg=True`
        vartheta (float):   Inexact inner-solver acceptance parameter in (0, 1/2).
        inner_max_iter(int):Maximum number of inner QUIC Newton iterations.
        zero_diag(bool):    Zero the diagonal penalty weights; None follows config.OFF_DIAG.

    Returns
    ------
    X : np.ndarray
        Estimated sparse precision matrix

    f_val_list : np.ndarray
        Objective values

    KKT_list : list
        Stationarity residuals

    time_list : list
        Time taken for each iteration
    """

    _check_common(S, rho, mu, MaxIter, tolerance, msg)
    if penalty == "lp":
        assert type(p) == float and 0 < p and p < 1, "Parameter p must be in the range (0, 1)."

    X0, eps0 = _initial_point(S, rho, X0, eps0)

    result = _iir_quic_impl(
        S, mu, eps0, X0, rho, p, penalty=penalty, MaxIter=MaxIter, tolerance=tolerance,
        vartheta=vartheta, inner_max_iter=inner_max_iter, zero_diag=zero_diag, msg=msg,
    )

    return result["X"], result["f_val_list"], result["KKT_list"], result["time_list"]
