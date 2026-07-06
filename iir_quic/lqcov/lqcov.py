"""
Lq-Regularized Sparse Inverse Covariance Estimation
============================================================

Implementation of the Lq-optimization approach for sparse inverse covariance selection,
based on the paper:
    On Lq Optimization and Sparse Inverse Covariance Selection (Goran Marjanovic et al., 2014)
"""

import numpy as np
from sklearn.covariance import graphical_lasso as glasso
import time

from ..config import NONZERO, debugPrint, importantPrint


def lq_threshold(z: float, lambda_i: float, q: float, max_iter=2000, tol=1e-4):
    """l_q thresholding function `tau`."""
    if q == 0:
        return z * indicator_function(abs(z) > np.sqrt(2 * lambda_i))
    elif q == 1:
        return np.sign(z) * max(abs(z) - lambda_i, 0)
    else:
        beta_lambda = (2 * lambda_i * (1 - q)) ** (1 / (2 - q))
        h_lambda = beta_lambda + lambda_i * q * beta_lambda ** (q - 1)
        if abs(z) <= h_lambda:
            return 0
        else:
            _beta = beta_lambda
            beta = beta_lambda
            
            for _ in range(max_iter):  # Fixed-point iteration
                beta = abs(z) - lambda_i * q * _beta ** (q - 1)
                if abs(_beta - beta) < tol:
                    break
                _beta = beta

                if _ == max_iter - 1:
                    importantPrint(
                        f"(lq_threshold) Max iterations reached: {max_iter}"
                    )
            return np.sign(z) * beta


def update_u(
    u: np.ndarray,
    V_inv: np.ndarray,
    gamma: np.ndarray,
    gamma_0: float,
    lambda_: float,
    q: float,
    max_iter=2000,
    tol=1e-4,
):
    """Update u by using cyclic descent."""
    u_hat = u.copy()
    u_hat_next = u.copy()

    for _ in range(max_iter):
        for i in range(len(u)):
            _u = u_hat_next.copy()
            _u[i] = 0

            _denominator = gamma_0 * V_inv[i, i]

            # if _denominator < NONZERO:
            #     _denominator = NONZERO
            #     importantPrint(
            #         f"(update u) Division by zero: gamma_0: {gamma_0}, V_inv[i, i]: {V_inv[i, i]}"
            #     )

            z_i = -(gamma_0 * _u @ V_inv[:, i] + gamma[i]) / _denominator
            lambda_i = lambda_ / _denominator
            u_hat_next[i] = lq_threshold(z_i, lambda_i, q)

        # stop rule:
        if np.linalg.norm(u_hat - u_hat_next, ord=2) < tol:
            break

        u_hat = u_hat_next.copy()

        if _ == max_iter - 1:
            importantPrint(f"(update u) Max iterations reached: {max_iter}")

    return u_hat_next


def get_Omega_inv(V_inv: np.ndarray, u_plus: np.ndarray, w_plus: float):
    z_plus = np.dot(V_inv, u_plus.reshape(-1, 1))
    c_plus = w_plus - np.dot(u_plus, z_plus)

    # if c_plus < NONZERO:
    #     importantPrint(
    #         f"(get Omega_inv) Expected c_plus > 0: c_plus: {c_plus}, dot(u_plus, z_plus): {np.dot(u_plus, z_plus)}"
    #     )
    #     c_plus = NONZERO

    Omega_plus_inv = np.block(
        [
            [V_inv + np.outer(z_plus, z_plus) / c_plus, -z_plus / c_plus],
            [-z_plus.T / c_plus, 1 / c_plus],
        ]
    )

    return Omega_plus_inv


def get_V_inv(Omega_inv_perm: np.ndarray):
    P = Omega_inv_perm[:-1, :-1]
    s = Omega_inv_perm[:-1, -1]
    r = Omega_inv_perm[-1, -1]

    _denominator = r

    # if _denominator < NONZERO:
    #     _denominator = NONZERO
    #     importantPrint(f"(get V_inv) Division by zero: r: {r}")

    V_plus_inv = P - np.outer(s, s) / _denominator
    return V_plus_inv


def objective_function(
    Omega: np.ndarray, S: np.ndarray, lambda_: float, q: float
) -> float:
    """-log(det(Omega)) + trace(S @ Omega) + lambda * sum(abs(Omega) ** q)"""
    log_det = np.linalg.slogdet(Omega)[1]
    trace = np.trace(Omega @ S)
    Omega_d = Omega.copy()
    np.fill_diagonal(Omega_d, 0.0)
    lq_term = lambda_ * np.sum(np.power(np.abs(Omega_d), q))

    L = log_det - trace - lq_term

    return -L


def objective_function2(
    Omega: np.ndarray, S: np.ndarray, lambda_: float, q: float
) -> float:
    """-log(det(Omega)) + trace(S @ Omega) + lambda * sum(abs(Omega) ** q)"""
    log_det = np.linalg.slogdet(Omega)[1]
    trace = np.trace(Omega @ S)
    Omega_d = Omega.copy()
    # np.fill_diagonal(Omega_d, 0.0)
    lq_term = lambda_ * np.sum(np.power(np.abs(Omega_d), q))

    L = log_det - trace - lq_term

    return -L


def indicator_function(condition):
    return 1 if condition else 0


def loss_KL(Omega: np.ndarray, Sigma: np.ndarray):
    trace = np.trace(Sigma @ Omega)
    log_det = np.linalg.slogdet(Sigma @ Omega)[1]
    return trace - log_det - Omega.shape[0]


def check_positive_definite(matrix: np.ndarray) -> bool:
    """Check if a matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

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


def lq_cov(
    S: np.ndarray, lambda_: float, q_f: float, max_iter=3000, tol=1e-4, warm_start=True
):
    """
    l_q COV algorithm for sparse inverse covariance selection.

    Parameters:
        S(np.ndarray):      Sample covariance matrix (p x p), positive definite
        lambda\_(float):    Penalty parameter
        q_f(float):         Target q value (0 <= q_f < 1)
        max_iter(int):      Maximum number of iterations
        tol(float):         Convergence tolerance
        warm_start(bool):   Whether to use warm-starting

    Returns
    ------
    Omega : np.ndarray
        Estimated sparse precision matrix

    f_val_list : list
        Objective values
    """
    assert lambda_ > 0, "Expected `lambda_` > 0"
    assert q_f >= 0 and q_f < 1, "Expected 0 <= q_f < 1"
    assert type(lambda_) == float, "Expected `lambda_` to be a float"
    assert type(q_f) == float, "Expected `q_f` to be a float"

    p = S.shape[0]
    Omega = np.diag(1 / np.diag(S))  # Initialize Omega as diagonal matrix
    Omega_inv = np.zeros_like(Omega)

    f_val_list = []


    if warm_start:
        K = (
            2 * indicator_function(q_f >= 0.9)
            + 3 * indicator_function(0.7 <= q_f and q_f < 0.9)
            + 4 * indicator_function(0.4 <= q_f and q_f < 0.7)
            + 5 * indicator_function(0.2 <= q_f and q_f < 0.4)
            + 6 * indicator_function(0.0 <= q_f and q_f < 0.2)
        )

        q_values = np.linspace(1.0, q_f, num=K)  # Warm-starting with K steps
    else:
        q_values = [q_f]

        f_val_list.append(objective_function(Omega, S, lambda_, q_f))

    for q_current in q_values:
        debugPrint(f"{' ' * 0}# q_current: {q_current}")

        if q_current == 1.0:
            ### sovle q=1 by using glasso

            Omega = glasso(S, lambda_, tol=1e-4, max_iter=1000)[1]

            assert check_positive_definite(Omega), "Omega expected positive definite"

            f_val_list.append(objective_function(Omega, S, lambda_, q_f))
        else:
            ### 0<= q < 1, using lq-Cov alg

            for cnt in range(max_iter):

                k = cnt % p  # current entry

                ### BCA
                # Permute matrix to bring k-th row/column to the end
                perm = list(range(p))
                perm[k], perm[-1] = perm[-1], perm[k]

                # Extract submatrices
                S_perm = S[np.ix_(perm, perm)]
                # Gamma = S_perm[:-1, :-1]
                gamma = S_perm[:-1, -1]
                gamma_0 = S_perm[-1, -1]

                Omega_perm = Omega[np.ix_(perm, perm)]
                V = Omega_perm[:-1, :-1]
                u = Omega_perm[:-1, -1]
                # w = Omega_perm[-1, -1]

                Omega_inv_perm = Omega_inv[np.ix_(perm, perm)]

                if cnt == 0:
                    V_inv = np.linalg.inv(V)
                else:
                    V_inv = get_V_inv(Omega_inv_perm)

                ### CD
                # Update u and w
                u_next = update_u(u, V_inv, gamma, gamma_0, lambda_, q_current)
                w_next = u_next @ V_inv @ u_next + 1 / gamma_0

                # if w_next < NONZERO:
                #     w_next = NONZERO
                #     importantPrint(
                #         f"Expected w > 0: w: {w_next}, u @ V_inv @ u: {u_next @ V_inv @ u_next}"
                #     )

                ### update for next iter
                # Omega_perm[:-1, :-1] = V
                Omega_perm[:-1, -1] = u_next
                Omega_perm[-1, :-1] = u_next
                Omega_perm[-1, -1] = w_next
                Omega = Omega_perm[np.ix_(perm, perm)]

                Omega_inv_perm = get_Omega_inv(V_inv, u_next, w_next)
                Omega_inv = Omega_inv_perm[np.ix_(perm, perm)]

                f_val_list.append(objective_function(Omega, S, lambda_, q_f))

                ### Check for convergence
                diff = abs(f_val_list[-1] - f_val_list[-2])

                diff2 = 0.0
                if len(f_val_list) > 2:
                    diff2 = abs(f_val_list[-2] - f_val_list[-3])

                debugPrint(
                    f"{' ' * 2}# q_current: {q_current:.2f}, iter: {cnt}, k: {k}, diff: {diff}"
                )

                # stop rule
                if (diff < tol and diff > 0.0) and (diff2 < tol and diff2 > 0.0):
                    debugPrint(f"{' ' * 4}# Converged: {diff}")

                    break

    return Omega, f_val_list


def lq_cov2(
    S: np.ndarray, lambda_: float, q_f: float, max_iter=3000, tol=1e-5, warm_start=True, tol_ws=1e-4
):
    """
    l_q COV algorithm for sparse inverse covariance selection. (using stationary condition)

    Parameters:
        S(np.ndarray):      Sample covariance matrix (p x p)
        lambda\_(float):    Penalty parameter
        q_f(float):         Target q value (0 <= q_f < 1)
        max_iter(int):      Maximum number of iterations
        tol(float):         Convergence tolerance
        warm_start(bool):   Whether to use warm-starting
        tol_ws(float):      Convergence tolerance in warm-up stages

    Returns
    ------
    Omega : np.ndarray
        Estimated sparse precision matrix

    f_val_list : list
        Objective values
    
    KKT_list : list
        Stationarity residuals
    
    time_list : list
        Time taken for each iteration
    """
    assert lambda_ > 0, "Expected `lambda_` > 0"
    assert q_f >= 0 and q_f < 1, "Expected 0 <= q_f < 1"
    assert type(lambda_) == float or type(lambda_) == np.float64, "Expected `lambda_` to be a float"
    assert type(q_f) == float, "Expected `q_f` to be a float"

    p = S.shape[0]
    Omega = np.diag(1 / (np.diag(S)  + lambda_))  # Initialize Omega as diagonal matrix
    Omega_inv = np.zeros_like(Omega)

    f_val_list = []
    f_val_list_out = []
    KKT_list = []

    time_list = []

    start_time = time.time()


    if warm_start:
        K = (
            2 * indicator_function(q_f >= 0.9)
            + 3 * indicator_function(0.7 <= q_f and q_f < 0.9)
            + 4 * indicator_function(0.4 <= q_f and q_f < 0.7)
            + 5 * indicator_function(0.2 <= q_f and q_f < 0.4)
            + 6 * indicator_function(0.0 <= q_f and q_f < 0.2)
        )

        q_values = np.linspace(1.0, q_f, num=K)  # Warm-starting with K steps
    else:
        q_values = [q_f]


    for q_current in q_values:
        debugPrint(f"{' ' * 0}# q_current: {q_current}")

        if q_current == 1.0:
            ### sovle q=1 by using glasso

            Omega = glasso(S, lambda_, tol=1e-4, max_iter=1000)[1]

            assert check_positive_definite(Omega), "Omega expected positive definite"

            f_val_list.append(objective_function(Omega, S, lambda_, q_current))
        else:
            ### 0<= q < 1, using lq-Cov alg

            for cnt in range(max_iter):

                k = cnt % p  # current entry

                ### BCA
                # Permute matrix to bring k-th row/column to the end
                perm = list(range(p))
                perm[k], perm[-1] = perm[-1], perm[k]

                # Extract submatrices
                S_perm = S[np.ix_(perm, perm)]
                # Gamma = S_perm[:-1, :-1]
                gamma = S_perm[:-1, -1]
                gamma_0 = S_perm[-1, -1]

                Omega_perm = Omega[np.ix_(perm, perm)]
                V = Omega_perm[:-1, :-1]
                u = Omega_perm[:-1, -1]
                # w = Omega_perm[-1, -1]

                Omega_inv_perm = Omega_inv[np.ix_(perm, perm)]

                if cnt == 0:
                    V_inv = np.linalg.inv(V)
                else:
                    V_inv = get_V_inv(Omega_inv_perm)

                ### CD
                # Update u and w
                u_next = update_u(u, V_inv, gamma, gamma_0, lambda_, q_current)
                w_next = u_next @ V_inv @ u_next + 1 / gamma_0

                # if w_next < NONZERO:
                #     w_next = NONZERO
                #     importantPrint(
                #         f"Expected w > 0: w: {w_next}, u @ V_inv @ u: {u_next @ V_inv @ u_next}"
                #     )


                ### update for next iter
                # Omega_perm[:-1, :-1] = V
                Omega_perm[:-1, -1] = u_next
                Omega_perm[-1, :-1] = u_next
                Omega_perm[-1, -1] = w_next
                Omega = Omega_perm[np.ix_(perm, perm)]

                Omega_inv_perm = get_Omega_inv(V_inv, u_next, w_next)
                Omega_inv = Omega_inv_perm[np.ix_(perm, perm)]


                ### Check for convergence

                # stop rule
                if q_current == q_f:
                    # f_val_list_out.append(objective_function(Omega, S, lambda_, q_current))
                    f_val_list_out.append(objective_function2(Omega, S, lambda_, q_current))

                    non_zero_indices = np.where(np.abs(Omega) > NONZERO)

                    elementwise_product_all = np.power(
                        np.abs(Omega[non_zero_indices]), q_f - 1
                    ) * np.sign(Omega[non_zero_indices])

                    elementwise_product = np.zeros_like(Omega)

                    rows, cols = non_zero_indices
                    is_diagonal = (rows == cols)
                    off_diag_mask = ~is_diagonal

                    elementwise_product[rows[off_diag_mask], cols[off_diag_mask]] = elementwise_product_all[off_diag_mask]

                    optRes = np.max(
                        np.abs(
                            S[non_zero_indices]
                            - Omega_inv[non_zero_indices]
                            + lambda_ * q_f * elementwise_product[non_zero_indices]
                        )
                    )


                    optRes = optRes * p
                    KKT_list.append(KKT_condition(S, Omega, Omega_inv, lambda_, q_f))

                    time_list.append(time.time() - start_time)

                    if optRes < tol:
                        break
                else:
                    f_val_list.append(objective_function(Omega, S, lambda_, q_current))

                    diff = abs(f_val_list[-1] - f_val_list[-2])

                    diff2 = 0.0
                    if len(f_val_list) > 2:
                        diff2 = abs(f_val_list[-2] - f_val_list[-3])


                    if (diff < tol_ws and diff > 0.0) and (diff2 < tol_ws and diff2 > 0.0):

                        break


    return Omega, f_val_list_out, KKT_list, time_list


def lq_cov3(
    S: np.ndarray, lambda_: float, q_f: float, max_time=1800, max_iter=3000, tol=1e-5, warm_start=True, tol_ws=1e-5
):
    """
    l_q COV algorithm for sparse inverse covariance selection. (using stationary condition and time limitation)

    Parameters:
        S(np.ndarray):      Sample covariance matrix (p x p)
        lambda\_(float):    Penalty parameter
        q_f(float):         Target q value (0 <= q_f < 1)
        max_time(int):      Maximum number of time (second)
        max_iter(int):      Maximum number of iterations in warm-start
        tol(float):         Convergence tolerance
        warm_start(bool):   Whether to use warm-starting
        tol_ws(float):      Convergence tolerance in warm-up stages

    Returns
    ------
    Omega : np.ndarray
        Estimated sparse precision matrix

    f_val_list : list
        Objective values
    
    KKT_list : list
        Stationarity residuals
    
    time_list : list
        Time taken for each iteration
    """
    assert lambda_ > 0, "Expected `lambda_` > 0"
    assert q_f >= 0 and q_f < 1, "Expected 0 <= q_f < 1"
    assert type(lambda_) == float or type(lambda_) == np.float64, "Expected `lambda_` to be a float"
    assert type(q_f) == float, "Expected `q_f` to be a float"

    p = S.shape[0]
    Omega = np.diag(1 / (np.diag(S)  + lambda_))  # Initialize Omega as diagonal matrix
    Omega_inv = np.zeros_like(Omega)

    f_val_list = []
    f_val_list_out = []
    KKT_list = []

    time_list = []

    start_time = time.time()


    if warm_start:
        K = (
            2 * indicator_function(q_f >= 0.9)
            + 3 * indicator_function(0.7 <= q_f and q_f < 0.9)
            + 4 * indicator_function(0.4 <= q_f and q_f < 0.7)
            + 5 * indicator_function(0.2 <= q_f and q_f < 0.4)
            + 6 * indicator_function(0.0 <= q_f and q_f < 0.2)
        )

        q_values = np.linspace(1.0, q_f, num=K)  # Warm-starting with K steps
    else:
        q_values = [q_f]


    for q_current in q_values:
        debugPrint(f"{' ' * 0}# q_current: {q_current}")

        if q_current == 1.0:
            ### sovle q=1 by using glasso

            Omega = glasso(S, lambda_, tol=1e-4, max_iter=1000)[1]

            assert check_positive_definite(Omega), "Omega expected positive definite"

            f_val_list.append(objective_function(Omega, S, lambda_, q_current))
        else:
            ### 0<= q < 1, using lq-Cov alg

            cnt = 0

            while True:

                k = cnt % p  # current entry

                ### BCA
                # Permute matrix to bring k-th row/column to the end
                perm = list(range(p))
                perm[k], perm[-1] = perm[-1], perm[k]

                # Extract submatrices
                S_perm = S[np.ix_(perm, perm)]
                # Gamma = S_perm[:-1, :-1]
                gamma = S_perm[:-1, -1]
                gamma_0 = S_perm[-1, -1]

                Omega_perm = Omega[np.ix_(perm, perm)]
                V = Omega_perm[:-1, :-1]
                u = Omega_perm[:-1, -1]
                # w = Omega_perm[-1, -1]

                Omega_inv_perm = Omega_inv[np.ix_(perm, perm)]

                if cnt == 0:
                    V_inv = np.linalg.inv(V)
                else:
                    V_inv = get_V_inv(Omega_inv_perm)
                
                cnt += 1

                ### CD
                # Update u and w
                u_next = update_u(u, V_inv, gamma, gamma_0, lambda_, q_current)
                w_next = u_next @ V_inv @ u_next + 1 / gamma_0

                # if w_next < NONZERO:
                #     w_next = NONZERO
                #     importantPrint(
                #         f"Expected w > 0: w: {w_next}, u @ V_inv @ u: {u_next @ V_inv @ u_next}"
                #     )


                ### update for next iter
                # Omega_perm[:-1, :-1] = V
                Omega_perm[:-1, -1] = u_next
                Omega_perm[-1, :-1] = u_next
                Omega_perm[-1, -1] = w_next
                Omega = Omega_perm[np.ix_(perm, perm)]

                Omega_inv_perm = get_Omega_inv(V_inv, u_next, w_next)
                Omega_inv = Omega_inv_perm[np.ix_(perm, perm)]


                ### Check for convergence

                # stop rule
                if q_current == q_f:
                    # f_val_list_out.append(objective_function(Omega, S, lambda_, q_current))
                    f_val_list_out.append(objective_function2(Omega, S, lambda_, q_current))

                    non_zero_indices = np.where(np.abs(Omega) > NONZERO)

                    elementwise_product_all = np.power(
                        np.abs(Omega[non_zero_indices]), q_f - 1
                    ) * np.sign(Omega[non_zero_indices])

                    elementwise_product = np.zeros_like(Omega)

                    rows, cols = non_zero_indices
                    is_diagonal = (rows == cols)
                    off_diag_mask = ~is_diagonal

                    elementwise_product[rows[off_diag_mask], cols[off_diag_mask]] = elementwise_product_all[off_diag_mask]

                    optRes = np.max(
                        np.abs(
                            S[non_zero_indices]
                            - Omega_inv[non_zero_indices]
                            + lambda_ * q_f * elementwise_product[non_zero_indices]
                        )
                    )


                    optRes = optRes * p
                    KKT_list.append(KKT_condition(S, Omega, Omega_inv, lambda_, q_f))

                    time_list.append(time.time() - start_time)

                    if optRes < tol or time_list[-1] >= max_time:
                        break
                else:
                    f_val_list.append(objective_function(Omega, S, lambda_, q_current))

                    diff = abs(f_val_list[-1] - f_val_list[-2])

                    diff2 = 0.0
                    if len(f_val_list) > 2:
                        diff2 = abs(f_val_list[-2] - f_val_list[-3])


                    if ((diff < tol_ws and diff > 0.0) and (diff2 < tol_ws and diff2 > 0.0)) or cnt >= max_iter:

                        break


    return Omega, f_val_list_out, KKT_list, time_list
