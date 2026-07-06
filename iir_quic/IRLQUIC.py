'''
DIIR-QUIC main program
'''

import numpy as np
import time

from . import core as IRL_core

from .config import NONZERO
from . import alg


def update_weights(
    iterate: np.ndarray, perturbation: np.ndarray, modelPara: float
) -> np.ndarray:
    """
    Compute the new weights based on the current iterate and perturbation.

    Args:
      iterate (np.ndarray): Current iterate.
      perturbation (np.ndarray): Current perturbation.
      modelPara (float): Parameter for the non-convex regularization term.

    Returns:
      np.ndarray: Updated weights.
    """
    # Ensure that the term (|x_k| + epsilon_k) is not too small by using a lower bound

    adjustment = np.maximum(np.abs(iterate) + perturbation, 1e-13)
    return modelPara * np.power(adjustment, modelPara - 1)


def update_weights_scad(
    iterate: np.ndarray, perturbation: np.ndarray, lam: float
) -> np.ndarray:
    # Ensure that the term (|x_k| + epsilon_k) is not too small by using a lower bound

    adjustment = np.maximum(np.abs(iterate) + perturbation, 1e-13)
    return alg.scad_derivative(adjustment, lam)


def update_weights_mcp(
    iterate: np.ndarray, perturbation: np.ndarray, lam: float
) -> np.ndarray:
    # Ensure that the term (|x_k| + epsilon_k) is not too small by using a lower bound

    adjustment = np.maximum(np.abs(iterate) + perturbation, 1e-13)
    return alg.mcp_derivative(adjustment, lam)


def objective_function(
    S: np.ndarray, X: np.ndarray, lam: float, p: float, perturbation: np.ndarray
) -> float:
    """-logdet(X) + tr(SX) + lam * sum(|X|^p)"""
    log_det = np.linalg.slogdet(X)[1]
    trace_term = np.trace(S @ X)
    lp_term = lam * np.sum(np.power(np.abs(X) + perturbation, p))
    f_val = -log_det + trace_term + lp_term

    return f_val


def objective_function_scad(
    S: np.ndarray, X: np.ndarray, lam: float, perturbation: np.ndarray
) -> float:
    """-logdet(X) + tr(SX) + lam * sum(|X|^p)"""
    log_det = np.linalg.slogdet(X)[1]
    trace_term = np.trace(S @ X)
    scad_term = lam * alg.matrix_scad(np.abs(X) + perturbation, lam)

    f_val = -log_det + trace_term + scad_term

    return f_val


def objective_function_mcp(
    S: np.ndarray, X: np.ndarray, lam: float, perturbation: np.ndarray
) -> float:
    """-logdet(X) + tr(SX) + lam * sum(|X|^p)"""
    log_det = np.linalg.slogdet(X)[1]
    trace_term = np.trace(S @ X)
    scad_term = lam * alg.matrix_mcp(np.abs(X) + perturbation, lam)

    f_val = -log_det + trace_term + scad_term

    return f_val


def KKT_condition(
    S: np.ndarray, X: np.ndarray, X_inv: np.ndarray, lam: float, p: float
):
    """Stationarity resdual"""
    _dim = X.shape[0]

    non_zero_indices = np.where(np.abs(X) > NONZERO)
    elementwise_product = np.power(np.abs(X[non_zero_indices]), p - 1) * np.sign(
        X[non_zero_indices]
    )
    optRes_unscaled = np.max(
        np.abs(
            S[non_zero_indices]
            - X_inv[non_zero_indices]
            + lam * p * elementwise_product
        )
    )

    optRes = optRes_unscaled * _dim
    return optRes


def KKT_condition_scad(S: np.ndarray, X: np.ndarray, X_inv: np.ndarray, lam: float):
    """Stationarity resdual"""
    _dim = X.shape[0]

    grad_f: np.ndarray = S - X_inv

    # (3a)
    non_zero_indices = np.where(np.abs(X) > NONZERO)
    optRes_unscaled_1 = np.max(
        np.abs(
            grad_f[non_zero_indices]
            + lam
            * alg.scad_derivative(np.abs(X[non_zero_indices]), lam)
            * np.sign(X[non_zero_indices])
        )
    )

    # (3b)
    zero_indices = np.where(np.abs(X) <= NONZERO)
    lambda_sq = lam**2
    res = np.zeros_like(X[zero_indices])

    mask1 = grad_f[zero_indices] < -lambda_sq
    res[mask1] = -(grad_f[zero_indices][mask1] + lambda_sq)

    mask2 = grad_f[zero_indices] > lambda_sq
    res[mask2] = grad_f[zero_indices][mask2] - lambda_sq

    optRes_unscaled_2 = np.max(np.abs(res))

    optRes_unscaled = max(optRes_unscaled_1, optRes_unscaled_2)

    optRes = optRes_unscaled * _dim
    return optRes


def KKT_condition_mcp(S: np.ndarray, X: np.ndarray, X_inv: np.ndarray, lam: float):
    """Stationarity resdual"""
    _dim = X.shape[0]

    grad_f: np.ndarray = S - X_inv

    # (3a)
    non_zero_indices = np.where(np.abs(X) > NONZERO)
    optRes_unscaled_1 = np.max(
        np.abs(
            grad_f[non_zero_indices]
            + lam
            * alg.mcp_derivative(np.abs(X[non_zero_indices]), lam)
            * np.sign(X[non_zero_indices])
        )
    )

    # (3b)
    zero_indices = np.where(np.abs(X) <= NONZERO)
    lambda_sq = lam**2
    res = np.zeros_like(X[zero_indices])

    mask1 = grad_f[zero_indices] < -lambda_sq
    res[mask1] = -(grad_f[zero_indices][mask1] + lambda_sq)

    mask2 = grad_f[zero_indices] > lambda_sq
    res[mask2] = grad_f[zero_indices][mask2] - lambda_sq

    optRes_unscaled_2 = np.max(np.abs(res))

    optRes_unscaled = max(optRes_unscaled_1, optRes_unscaled_2)

    optRes = optRes_unscaled * _dim
    return optRes



def iRL1_simplified2(
    SampleCov: np.ndarray,
    reduce_para: float,
    perturbationInit: np.ndarray,
    IteratesInit: np.ndarray,
    regularizationPara: float,
    modelPara: float,
    MaxIter: int = 3000,
    tolerance: float = 1e-5,
    alpha=0.5,
    Sigma=None,
    msg=True,
) -> tuple[np.ndarray, int, list[float], list[float], list[float]]:
    """
    Main loop for the Extrapolated Proximal Iteratively Reweighted L1 algorithm.

    Args:
        SampleCov(np.ndarray):          The empirical nxn covariance matrix.
        reducePara (float):             Epsilon decay factor, which belongs to (0,1).
        perturbationInit (np.ndarray):  Initial epsilon values.
        IteratesInit (np.ndarray):      Initial point for the iterates.
        regularizationPara (float):     Regularization parameter.
        modelPara (float):              Parameter for the non-convex regularization term (0 < p < 1).
        MaxIter (int):                  Maximum number of iterations.
        tolerance (float):              Tolerance for the stopping criterion.
        alpha(float):                   Dampled step para, which belongs to (0,1).
        Sigma(np.ndarray):              **(useless para)** Σ of original question, expected np.ndarray format, it will be ignored if `Sigma = None`
        msg(bool):                      print message if `msg=True`
    """

    # Initialization
    iterate = np.copy(IteratesInit)
    iterate = iterate.astype(np.float64)

    # Using Cholesky decomposition
    L = np.linalg.cholesky(iterate)
    L_inv = np.linalg.inv(L)

    # Compute the inverse of iterate
    iterate_inv = L_inv.T @ L_inv

    perturbation = np.copy(perturbationInit)
    cnt = 0

    f_val_list = []
    KKT_list = []
    time_list = []

    f_val_list.append(
        objective_function(
            SampleCov, iterate, regularizationPara, modelPara, perturbation
        )
    )

    start_time = time.time()
    # Iterative loop
    while True:
        # Step 1: Compute the new weights (tuning parameter) how to adjust this weights?
        weight = update_weights(iterate, perturbation, modelPara)

        # Step 2: Solve the subproblem
        msg_quic = 0
        if msg:
            print(f"============ iter: {cnt} ============")
            msg_quic = 2

        [iterateNext, iterateNextInv, _, _, _, _, _, _, _, _] = IRL_core.quic(
            S=SampleCov,
            L=regularizationPara * weight,
            mode="default",
            max_iter=2000,
            X0=iterate,
            W0=iterate_inv,
            msg=msg_quic,
        )

        iterateNext = (1.0 - alpha) * iterate + alpha * iterateNext

        # Check for convergence: KKT condition
        optRes = KKT_condition(
            SampleCov, iterateNext, iterateNextInv, regularizationPara, modelPara
        )
        KKT_list.append(optRes)
        time_list.append(time.time() - start_time)

        if msg:
            print(f"===== f_val: {f_val_list[-1]}, KKT: {optRes} =====")

        if (optRes < tolerance) or cnt >= MaxIter:
            break

        # Step 3: Update the perturbation
        perturbation = (1.0 - alpha) * perturbation + alpha * reduce_para * perturbation

        f_val_list.append(
            objective_function(
                SampleCov, iterateNext, regularizationPara, modelPara, perturbation
            )
        )

        # Step 4: Prepare for the next iteration
        iterate = np.copy(iterateNext)
        iterate_inv = np.copy(iterateNextInv)
        cnt += 1

    output = (iterate, cnt, f_val_list, KKT_list, time_list)

    return output


def iRL1_scad(
    SampleCov: np.ndarray,
    reduce_para: float,
    perturbationInit: np.ndarray,
    IteratesInit: np.ndarray,
    regularizationPara: float,
    modelPara: float,
    MaxIter: int = 3000,
    tolerance: float = 1e-5,
    alpha=0.5,
    Sigma=None,
    msg=True,
) -> tuple[np.ndarray, int, list[float], list[float], list[float]]:
    """
    Main loop for the Extrapolated Proximal Iteratively Reweighted L1 algorithm. (SCAD)

    Args:
        SampleCov(np.ndarray):          The empirical nxn covariance matrix.
        reducePara (float):             Epsilon decay factor, which belongs to (0,1).
        perturbationInit (np.ndarray):  Initial epsilon values.
        IteratesInit (np.ndarray):      Initial point for the iterates.
        regularizationPara (float):     Regularization parameter.
        modelPara (float):              **(useless para)** Parameter for the non-convex regularization term (0 < p < 1).
        MaxIter (int):                  Maximum number of iterations.
        tolerance (float):              Tolerance for the stopping criterion.
        alpha(float):                   Dampled step para, which belongs to (0,1).
        Sigma(np.ndarray):              **(useless para)** Σ of original question, expected np.ndarray format, it will be ignored if `Sigma = None`
        msg(bool):                      print message if `msg=True`
    """

    # Initialization
    iterate = np.copy(IteratesInit)
    iterate = iterate.astype(np.float64)

    # Using Cholesky decomposition
    L = np.linalg.cholesky(iterate)
    L_inv = np.linalg.inv(L)

    # Compute the inverse of iterate
    iterate_inv = L_inv.T @ L_inv

    perturbation = np.copy(perturbationInit)
    cnt = 0

    KKT_list = []
    time_list = []
    f_val_list = []

    f_val_list.append(
        objective_function_scad(SampleCov, iterate, regularizationPara, perturbation)
    )

    start_time = time.time()
    # Iterative loop
    while True:
        # Step 1: Compute the new weights (tuning parameter) how to adjust this weights?
        weight = update_weights_scad(iterate, perturbation, regularizationPara)

        # Step 2: Solve the subproblem
        msg_quic = 0
        if msg:
            print(f"============ iter: {cnt} ============")
            msg_quic = 2

        [iterateNext, iterateNextInv, _, _, _, _, _, _, _, _] = IRL_core.quic(
            S=SampleCov,
            L=regularizationPara * weight,
            mode="default",
            max_iter=2000,
            X0=iterate,
            W0=iterate_inv,
            msg=msg_quic,
        )

        iterateNext = (1.0 - alpha) * iterate + alpha * iterateNext

        # Check for convergence: KKT condition
        optRes = KKT_condition_scad(
            SampleCov, iterateNext, iterateNextInv, regularizationPara
        )
        KKT_list.append(optRes)
        time_list.append(time.time() - start_time)

        if msg:
            print(f"===== KKT: {optRes} =====")

        if (optRes < tolerance) or cnt >= MaxIter:
            break

        # Step 3: Update the perturbation
        perturbation = (1.0 - alpha) * perturbation + alpha * reduce_para * perturbation

        f_val_list.append(
            objective_function_scad(
                SampleCov, iterateNext, regularizationPara, perturbation
            )
        )

        # Step 4: Prepare for the next iteration
        iterate = np.copy(iterateNext)
        iterate_inv = np.copy(iterateNextInv)
        cnt += 1

    output = (iterate, cnt, KKT_list, time_list, f_val_list)

    return output


def iRL1_mcp(
    SampleCov: np.ndarray,
    reduce_para: float,
    perturbationInit: np.ndarray,
    IteratesInit: np.ndarray,
    regularizationPara: float,
    modelPara: float,
    MaxIter: int = 3000,
    tolerance: float = 1e-5,
    alpha=0.5,
    Sigma=None,
    msg=True,
) -> tuple[np.ndarray, int, list[float], list[float], list[float]]:
    """
    Main loop for the Extrapolated Proximal Iteratively Reweighted L1 algorithm. (SCAD)

    Args:
        SampleCov(np.ndarray):          The empirical nxn covariance matrix.
        reducePara (float):             Epsilon decay factor, which belongs to (0,1).
        perturbationInit (np.ndarray):  Initial epsilon values.
        IteratesInit (np.ndarray):      Initial point for the iterates.
        regularizationPara (float):     Regularization parameter.
        modelPara (float):              **(useless para)** Parameter for the non-convex regularization term (0 < p < 1).
        MaxIter (int):                  Maximum number of iterations.
        tolerance (float):              Tolerance for the stopping criterion.
        alpha(float):                   Dampled step para, which belongs to (0,1).
        Sigma(np.ndarray):              **(useless para)** Σ of original question, expected np.ndarray format, it will be ignored if `Sigma = None`
        msg(bool):                      print message if `msg=True`
    """

    # Initialization
    iterate = np.copy(IteratesInit)
    iterate = iterate.astype(np.float64)

    # Using Cholesky decomposition
    L = np.linalg.cholesky(iterate)
    L_inv = np.linalg.inv(L)

    # Compute the inverse of iterate
    iterate_inv = L_inv.T @ L_inv

    perturbation = np.copy(perturbationInit)
    cnt = 0

    KKT_list = []
    time_list = []
    f_val_list = []

    f_val_list.append(
        objective_function_mcp(SampleCov, iterate, regularizationPara, perturbation)
    )

    start_time = time.time()
    # Iterative loop
    while True:
        # Step 1: Compute the new weights (tuning parameter) how to adjust this weights?
        weight = update_weights_mcp(iterate, perturbation, regularizationPara)

        # Step 2: Solve the subproblem
        msg_quic = 0
        if msg:
            print(f"============ iter: {cnt} ============")
            msg_quic = 2

        [iterateNext, iterateNextInv, _, _, _, _, _, _, _, _] = IRL_core.quic(
            S=SampleCov,
            L=regularizationPara * weight,
            mode="default",
            max_iter=2000,
            X0=iterate,
            W0=iterate_inv,
            msg=msg_quic,
        )

        iterateNext = (1.0 - alpha) * iterate + alpha * iterateNext

        # Check for convergence: KKT condition
        optRes = KKT_condition_mcp(
            SampleCov, iterateNext, iterateNextInv, regularizationPara
        )
        KKT_list.append(optRes)
        time_list.append(time.time() - start_time)

        if msg:
            print(f"===== KKT: {optRes} =====")

        if (optRes < tolerance) or cnt >= MaxIter:
            break

        # Step 3: Update the perturbation
        perturbation = (1.0 - alpha) * perturbation + alpha * reduce_para * perturbation

        f_val_list.append(
            objective_function_mcp(
                SampleCov, iterateNext, regularizationPara, perturbation
            )
        )

        # Step 4: Prepare for the next iteration
        iterate = np.copy(iterateNext)
        iterate_inv = np.copy(iterateNextInv)
        cnt += 1

    output = (iterate, cnt, KKT_list, time_list, f_val_list)

    return output
