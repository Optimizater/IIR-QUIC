"""
Private implementation module for the public API functions.
Import via iir_quic/__init__.py
"""

from .IRLQUIC import iRL1_simplified2, iRL1_scad, iRL1_mcp
from .config import RAND_SEED
import numpy as np


def iir_quic(
    S: np.ndarray,
    rho: float,
    X0: np.ndarray | None = None,
    eps0: np.ndarray | None = None,
    mu: float = 0.1,
    p: float = 0.5,
    MaxIter: int = 3000,
    tolerance: float = 1e-5,
    msg=False,
):
    """
    Parameters:
        S(np.ndarray):      The empirical nxn covariance matrix.
        rho (float):        Regularization parameter.
        X0 (np.ndarray):    Initial point for the iterates.
        eps0 (np.ndarray):  Initial epsilon values.
        mu (float):         Epsilon decay factor, which belongs to (0,1).
        p (float):          Parameter for the non-convex regularization term (0 < p < 1).
        MaxIter (int):      Maximum number of iterations.
        tolerance (float):  Tolerance for the stopping criterion.
        msg(bool):          print message if `msg=True`

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

    assert type(S) == np.ndarray and S.shape[0] == S.shape[1], "Input S must be a square matrix."
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
    
    assert type(mu) == float and 0 < mu and mu < 1, "Parameter mu must be in the range (0, 1)."
    assert type(p) == float and 0 < p and p < 1, "Parameter p must be in the range (0, 1)."
    assert type(rho) == float and rho > 0, "Parameter rho must be positive."

    assert type(MaxIter) == int and MaxIter > 0, "MaxIter must be a positive integer."
    assert type(tolerance) == float and tolerance > 0, "Tolerance must be a positive number."
    assert type(msg) == bool, "msg must be a boolean value."

    X, _, f_val_list, KKT_list, time_list = iRL1_simplified2(S, mu, eps0, X0, rho, p, MaxIter, tolerance, 1, None, msg)

    return X, f_val_list, KKT_list, time_list


def iir_quic_scad(
    S: np.ndarray,
    rho: float,
    X0: np.ndarray | None = None,
    eps0: np.ndarray | None = None,
    mu: float = 0.1,
    MaxIter: int = 3000,
    tolerance: float = 1e-5,
    msg=False,
):
    """
    IIR-QUIC with SCAD.

    Parameters:
        S(np.ndarray):      The empirical nxn covariance matrix.
        rho (float):        Regularization parameter.
        X0 (np.ndarray):    Initial point for the iterates.
        eps0 (np.ndarray):  Initial epsilon values.
        mu (float):         Epsilon decay factor, which belongs to (0,1).
        MaxIter (int):      Maximum number of iterations.
        tolerance (float):  Tolerance for the stopping criterion.
        msg(bool):          print message if `msg=True`

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

    assert type(S) == np.ndarray and S.shape[0] == S.shape[1], "Input S must be a square matrix."
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
    
    assert type(mu) == float and 0 < mu and mu < 1, "Parameter mu must be in the range (0, 1)."
    assert type(rho) == float and rho > 0, "Parameter rho must be positive."

    assert type(MaxIter) == int and MaxIter > 0, "MaxIter must be a positive integer."
    assert type(tolerance) == float and tolerance > 0, "Tolerance must be a positive number."
    assert type(msg) == bool, "msg must be a boolean value."

    X, _, KKT_list, time_list, f_val_list = iRL1_scad(S, mu, eps0, X0, rho, 0.5, MaxIter, tolerance, 1, None, msg)

    return X, f_val_list, KKT_list, time_list


def iir_quic_mcp(
    S: np.ndarray,
    rho: float,
    X0: np.ndarray | None = None,
    eps0: np.ndarray | None = None,
    mu: float = 0.1,
    MaxIter: int = 3000,
    tolerance: float = 1e-5,
    msg=False,
):
    """
    IIR-QUIC with MCP.

    Parameters:
        S(np.ndarray):      The empirical nxn covariance matrix.
        rho (float):        Regularization parameter.
        X0 (np.ndarray):    Initial point for the iterates.
        eps0 (np.ndarray):  Initial epsilon values.
        mu (float):         Epsilon decay factor, which belongs to (0,1).
        MaxIter (int):      Maximum number of iterations.
        tolerance (float):  Tolerance for the stopping criterion.
        msg(bool):          print message if `msg=True`

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

    assert type(S) == np.ndarray and S.shape[0] == S.shape[1], "Input S must be a square matrix."
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
    
    assert type(mu) == float and 0 < mu and mu < 1, "Parameter mu must be in the range (0, 1)."
    assert type(rho) == float and rho > 0, "Parameter rho must be positive."

    assert type(MaxIter) == int and MaxIter > 0, "MaxIter must be a positive integer."
    assert type(tolerance) == float and tolerance > 0, "Tolerance must be a positive number."
    assert type(msg) == bool, "msg must be a boolean value."

    X, _, KKT_list, time_list, f_val_list = iRL1_mcp(S, mu, eps0, X0, rho, 0.5, MaxIter, tolerance, 1, None, msg)

    return X, f_val_list, KKT_list, time_list
