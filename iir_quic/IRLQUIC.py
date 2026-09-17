"""
IIR-QUIC: Inexact Iteratively Reweighted QUIC.

Solves min_X {-logdet(X) + tr(SX) + rho * Phi(X)} with nonconvex regularizers
(lp quasi-norm, SCAD, MCP) by alternating weight updates and inexact weighted
l1-regularized QUIC subproblem solves (Algorithm 1 of the manuscript).

All subproblems are solved by iir_quic.core (the inexact inner rule with the
acceptance parameter vartheta).  When config.OFF_DIAG is True the diagonal
penalty weights are zeroed (W_ii = 0), matching the off-diagonal problem
solved by l_pCOV; the objectives and KKT residuals are evaluated with the
corresponding off-diagonal penalty.
"""

import time

import numpy as np

from . import alg
from . import core as IRL_core
from .config import NONZERO, OFF_DIAG


def _use_off_diag(zero_diag):
    return OFF_DIAG if zero_diag is None else zero_diag


# FIX: weight functions for SCAD/MCP are parameterised by the regularization
# parameter rho (their knot / slope), while the lp weights use the exponent p.
def _weight_scale(penalty: str, modelPara: float, regularizationPara: float) -> float:
    return modelPara if penalty == "lp" else regularizationPara


# --------------------------------------------------------------------------
# Penalty weights
# --------------------------------------------------------------------------
def update_weights(
    iterate: np.ndarray, perturbation: np.ndarray, modelPara: float,
    zero_diag: bool | None = None,
) -> np.ndarray:
    """W_ij = p * (|X_ij| + E_ij)^(p-1) for the lp quasi-norm."""
    zero_diag = _use_off_diag(zero_diag)
    adjustment = np.maximum(np.abs(iterate) + perturbation, 1e-13)
    weight = modelPara * np.power(adjustment, modelPara - 1)
    if zero_diag:
        np.fill_diagonal(weight, 0.0)
    return weight


def update_weights_scad(
    iterate: np.ndarray, perturbation: np.ndarray, lam: float,
    zero_diag: bool | None = None,
) -> np.ndarray:
    """W_ij = phi'(|X_ij| + E_ij) for the SCAD penalty."""
    zero_diag = _use_off_diag(zero_diag)
    adjustment = np.maximum(np.abs(iterate) + perturbation, 1e-13)
    weight = alg.scad_derivative(adjustment, lam)
    if zero_diag:
        np.fill_diagonal(weight, 0.0)
    return weight


def update_weights_mcp(
    iterate: np.ndarray, perturbation: np.ndarray, lam: float,
    zero_diag: bool | None = None,
) -> np.ndarray:
    """W_ij = phi'(|X_ij| + E_ij) for the MCP penalty."""
    zero_diag = _use_off_diag(zero_diag)
    adjustment = np.maximum(np.abs(iterate) + perturbation, 1e-13)
    weight = alg.mcp_derivative(adjustment, lam)
    if zero_diag:
        np.fill_diagonal(weight, 0.0)
    return weight


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------
def _penalty_matrix(X: np.ndarray, zero_diag: bool) -> np.ndarray:
    pen = np.abs(X)
    if zero_diag:
        np.fill_diagonal(pen, 0.0)
    return pen


def objective_function(
    S: np.ndarray, X: np.ndarray, lam: float, p: float,
    perturbation: np.ndarray, use_perturbation: bool = True,
    zero_diag: bool | None = None,
) -> float:
    """-logdet(X) + tr(SX) + lam * sum(|X|^p) (with optional smoothing)."""
    zero_diag = _use_off_diag(zero_diag)
    log_det = np.linalg.slogdet(X)[1]
    trace_term = np.trace(S @ X)
    pen = _penalty_matrix(X, zero_diag)
    if use_perturbation:
        pen = pen + perturbation
    return -log_det + trace_term + lam * np.sum(np.power(pen, p))


def objective_function_scad(
    S: np.ndarray, X: np.ndarray, lam: float, modelPara: float,
    perturbation: np.ndarray, use_perturbation: bool = True,
    zero_diag: bool | None = None,
) -> float:
    """-logdet(X) + tr(SX) + lam * sum phi_SCAD(|X|)."""
    zero_diag = _use_off_diag(zero_diag)
    log_det = np.linalg.slogdet(X)[1]
    trace_term = np.trace(S @ X)
    pen = _penalty_matrix(X, zero_diag)
    if use_perturbation:
        pen = pen + perturbation
    return -log_det + trace_term + lam * alg.matrix_scad(pen, lam)


def objective_function_mcp(
    S: np.ndarray, X: np.ndarray, lam: float, modelPara: float,
    perturbation: np.ndarray, use_perturbation: bool = True,
    zero_diag: bool | None = None,
) -> float:
    """-logdet(X) + tr(SX) + lam * sum phi_MCP(|X|)."""
    zero_diag = _use_off_diag(zero_diag)
    log_det = np.linalg.slogdet(X)[1]
    trace_term = np.trace(S @ X)
    pen = _penalty_matrix(X, zero_diag)
    if use_perturbation:
        pen = pen + perturbation
    return -log_det + trace_term + lam * alg.matrix_mcp(pen, lam)


# --------------------------------------------------------------------------
# Stationarity residuals (KKT)
# --------------------------------------------------------------------------
def KKT_condition(
    S: np.ndarray, X: np.ndarray, X_inv: np.ndarray, lam: float, p: float,
    zero_diag: bool | None = None,
) -> float:
    """Stationarity residual of the lp problem, scaled by the dimension."""
    zero_diag = _use_off_diag(zero_diag)
    _dim = X.shape[0]

    nz = np.abs(X) > NONZERO
    pen_grad = np.zeros_like(X)
    pen_grad[nz] = p * np.power(np.abs(X[nz]), p - 1) * np.sign(X[nz])
    if zero_diag:
        np.fill_diagonal(pen_grad, 0.0)

    res = np.abs(S - X_inv + lam * pen_grad)
    optRes = np.max(res[nz]) * _dim
    return optRes


def KKT_condition_scad(
    S: np.ndarray, X: np.ndarray, X_inv: np.ndarray, lam: float,
    modelPara: float, zero_diag: bool | None = None,
) -> float:
    """Stationarity residual of the SCAD problem, scaled by the dimension."""
    zero_diag = _use_off_diag(zero_diag)
    _dim = X.shape[0]

    grad_f: np.ndarray = S - X_inv

    # (3a) nonzero entries
    nz = np.abs(X) > NONZERO
    pen_grad = np.zeros_like(X)
    pen_grad[nz] = alg.scad_derivative(np.abs(X[nz]), lam) * np.sign(X[nz])
    if zero_diag:
        np.fill_diagonal(pen_grad, 0.0)

    optRes_unscaled_1 = np.max(
        np.abs(grad_f[nz] + lam * pen_grad[nz])
    )

    # (3b) zero entries: |grad_f| must stay below the flat-tail threshold
    zr = np.abs(X) <= NONZERO
    lambda_sq = lam ** 2
    res = np.zeros_like(X[zr])

    mask1 = grad_f[zr] < -lambda_sq
    res[mask1] = -(grad_f[zr][mask1] + lambda_sq)

    mask2 = grad_f[zr] > lambda_sq
    res[mask2] = grad_f[zr][mask2] - lambda_sq

    optRes_unscaled = max(optRes_unscaled_1, np.max(np.abs(res)))
    return optRes_unscaled * _dim


def KKT_condition_mcp(
    S: np.ndarray, X: np.ndarray, X_inv: np.ndarray, lam: float,
    modelPara: float, zero_diag: bool | None = None,
) -> float:
    """Stationarity residual of the MCP problem, scaled by the dimension."""
    zero_diag = _use_off_diag(zero_diag)
    _dim = X.shape[0]

    grad_f: np.ndarray = S - X_inv

    # (3a) nonzero entries
    nz = np.abs(X) > NONZERO
    pen_grad = np.zeros_like(X)
    pen_grad[nz] = alg.mcp_derivative(np.abs(X[nz]), lam) * np.sign(X[nz])
    if zero_diag:
        np.fill_diagonal(pen_grad, 0.0)

    # (3a) nonzero entries
    nz = np.abs(X) > NONZERO
    optRes_unscaled_1 = np.max(
        np.abs(grad_f[nz] + lam * pen_grad[nz])
    )

    # (3b) zero entries
    zr = np.abs(X) <= NONZERO
    lambda_sq = lam ** 2
    res = np.zeros_like(X[zr])

    mask1 = grad_f[zr] < -lambda_sq
    res[mask1] = -(grad_f[zr][mask1] + lambda_sq)

    mask2 = grad_f[zr] > lambda_sq
    res[mask2] = grad_f[zr][mask2] - lambda_sq

    optRes_unscaled = max(optRes_unscaled_1, np.max(np.abs(res)))
    return optRes_unscaled * _dim


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
_WEIGHT_FN = {
    "lp": update_weights,
    "scad": update_weights_scad,
    "mcp": update_weights_mcp,
}
_OBJECTIVE_FN = {
    "lp": objective_function,
    "scad": objective_function_scad,
    "mcp": objective_function_mcp,
}
_KKT_FN = {
    "lp": KKT_condition,
    "scad": KKT_condition_scad,
    "mcp": KKT_condition_mcp,
}


def iir_quic(
    SampleCov: np.ndarray,
    reduce_para: float,
    perturbationInit: np.ndarray,
    IteratesInit: np.ndarray,
    regularizationPara: float,
    modelPara: float,
    penalty: str = "lp",
    MaxIter: int = 3000,
    tolerance: float = 1e-5,
    vartheta: float = 0.1,
    inner_max_iter: int = 2000,
    zero_diag: bool | None = None,
    msg: bool = False,
) -> dict:
    """
    Run IIR-QUIC (Algorithm 1 of the manuscript).

    Args:
        SampleCov (np.ndarray):      Empirical n x n covariance matrix.
        reduce_para (float):         Perturbation decay factor mu in (0,1).
        perturbationInit (ndarray):  Initial perturbation matrix E^0.
        IteratesInit (ndarray):      Initial point X^0 (positive definite).
        regularizationPara (float):  Regularization parameter rho.
        modelPara (float):           Model parameter p in (0,1] (unused for
                                     "scad"/"mcp", whose weights use rho).
        penalty (str):               "lp", "scad" or "mcp".
        MaxIter (int):               Maximum number of outer iterations.
        tolerance (float):           KKT stopping tolerance.
        vartheta (float):            Inexact inner acceptance parameter in (0, 0.5).
        inner_max_iter (int):        Max inner QUIC Newton iterations.
        zero_diag (bool | None):     Zero the diagonal penalty weights; None
                                     follows the global config.OFF_DIAG.
        msg (bool):                  Print per-iteration information if True.

    Returns:
        dict with keys X, f_val_list, f_val_list2, KKT_list, time_list,
        nnz_list, nAct_list, fix_norm_list, eps_norm_list, iterations.
    """
    if penalty not in _WEIGHT_FN:
        raise ValueError(f"penalty must be one of {list(_WEIGHT_FN)}")
    if not 0.0 < vartheta < 0.5:
        raise ValueError("vartheta must be in (0, 0.5)")
    zero_diag = _use_off_diag(zero_diag)
    weight_fn = _WEIGHT_FN[penalty]
    objective_fn = _OBJECTIVE_FN[penalty]
    kkt_fn = _KKT_FN[penalty]

    SampleCov = np.ascontiguousarray(SampleCov, dtype=np.float64)
    iterate = np.ascontiguousarray(IteratesInit, dtype=np.float64).copy()
    iterate_inv = np.linalg.inv(iterate)
    perturbation = np.asarray(perturbationInit, dtype=np.float64).copy()
    _dim = iterate.shape[0]

    f_val_list = [
        objective_fn(SampleCov, iterate, regularizationPara, modelPara,
                     perturbation, zero_diag=zero_diag)
    ]
    f_val_list2 = [
        objective_fn(SampleCov, iterate, regularizationPara, modelPara,
                     perturbation, use_perturbation=False, zero_diag=zero_diag)
    ]
    KKT_list = [
        kkt_fn(SampleCov, iterate, iterate_inv, regularizationPara, modelPara,
               zero_diag=zero_diag)
    ]
    nnz_list = [np.sum(np.abs(iterate) > NONZERO)]
    nAct_list = []
    fix_norm_list = [np.nan]
    eps_norm_list = [np.max(np.abs(perturbation)) * _dim]
    time_list = [0.0]

    start_time = time.perf_counter()
    for outer_iter in range(MaxIter):
        if msg:
            print(f"============ iter: {outer_iter} ============")

        weight = np.ascontiguousarray(
            weight_fn(iterate, perturbation, _weight_scale(penalty, modelPara, regularizationPara),
                      zero_diag=zero_diag),
            dtype=np.float64,
        )
        lam_matrix = np.ascontiguousarray(
            regularizationPara * weight, dtype=np.float64
        )
        iterate_next, iterate_next_inv, _, _, _, _, _, _, numActive, stepsize = (
            IRL_core.quic(
                S=SampleCov,
                L=lam_matrix,
                mode="default",
                max_iter=inner_max_iter,
                X0=iterate,
                W0=iterate_inv,
                msg=2 if msg else 0,
                vartheta=vartheta,
            )
        )

        iterate_next = np.asarray(iterate_next, dtype=np.float64)
        iterate_next_inv = np.asarray(iterate_next_inv, dtype=np.float64)
        nAct_list.append(numActive)
        fix_norm_list.append(np.max(np.abs(iterate_next - iterate)) * _dim)

        iterate = iterate_next.copy()
        iterate_inv = iterate_next_inv.copy()

        perturbation = reduce_para * perturbation

        kkt_value = kkt_fn(SampleCov, iterate, iterate_inv, regularizationPara,
                           modelPara, zero_diag=zero_diag)
        KKT_list.append(kkt_value)
        f_val_list.append(
            objective_fn(SampleCov, iterate, regularizationPara, modelPara,
                         perturbation, zero_diag=zero_diag)
        )
        f_val_list2.append(
            objective_fn(SampleCov, iterate, regularizationPara, modelPara,
                         perturbation, use_perturbation=False,
                         zero_diag=zero_diag)
        )
        nnz_list.append(np.sum(np.abs(iterate) > NONZERO))
        eps_norm_list.append(np.max(np.abs(perturbation)) * _dim)
        time_list.append(time.perf_counter() - start_time)

        if msg:
            print(f"===== KKT: {kkt_value} =====")

        if kkt_value < tolerance:
            break

    return {
        "X": iterate,
        "f_val_list": f_val_list,
        "f_val_list2": f_val_list2,
        "KKT_list": KKT_list,
        "time_list": time_list,
        "nnz_list": nnz_list,
        "nAct_list": nAct_list,
        "fix_norm_list": fix_norm_list,
        "eps_norm_list": eps_norm_list,
        "iterations": len(KKT_list) - 1,
    }