import numpy as np
cimport numpy as np
np.import_array()  # initialize NumPy C-API

from libcpp.vector cimport vector
from libcpp cimport bool

from sklearn.covariance import graphical_lasso as glasso

cdef extern from "<Eigen/Dense>" namespace "Eigen":
    cdef cppclass MatrixXd:
        MatrixXd() except +
        MatrixXd(int rows, int cols) except +
        double* data()
        int rows() const
        int cols() const

    cdef cppclass Map[MatrixXd]:
        Map(double* data, int rows, int cols) except +

cdef extern from "numpy_eigen.hpp":
    MatrixXd numpy_to_eigen(np.ndarray[np.float64_t, ndim=2] arr)
    np.ndarray[np.float64_t, ndim=2] eigen_to_numpy(const MatrixXd& mat)

cdef extern from "lqcov.hpp":
    cdef cppclass LqCov:
        LqCov(double lambda_, double q, int max_iter, int max_time, double tol, double tol_ws, bool warm_start, bool msg) except +
        vector[double] get_fval_list()
        vector[double] get_fval_list_out()
        vector[double] get_KKT_list()
        vector[double] get_time_list()
        MatrixXd fit(const MatrixXd& S, const MatrixXd& Omega_ini)
        MatrixXd fit2(const MatrixXd& S, const MatrixXd& Omega_ini)
        MatrixXd fit3(const MatrixXd& S, const MatrixXd& Omega_ini)


def lq_cov(np.ndarray[np.float64_t, ndim=2] S, double lambda_, double q_f, 
           int max_iter=3000, double tol=1e-4, bool warm_start=True):
    """
    C++ implementation of lq_cov algorithm
    
    Parameters:
        S(np.ndarray):      Sample covariance matrix (p x p)
        lambda\_(float):    Penalty parameter
        q_f(float):         Target q value (0 <= q_f < 1)
        max_iter(int):      Maximum number of iterations
        tol(float):         Convergence tolerance
        warm_start(bool):   Whether to use warm-starting

    Returns
    ------
    Omega : np.ndarray
        Estimated sparse precision matrix

    f_val_list : np.ndarray
        Objective values
    """
    cdef np.ndarray Omega = np.diag(1 / np.diag(S))
    if warm_start:
        Omega = glasso(S, lambda_, tol=1e-4, max_iter=1000)[1]

    cdef LqCov* solver = new LqCov(lambda_, q_f, max_iter, 0, tol, tol, warm_start, False)
    cdef np.ndarray Omega_est = eigen_to_numpy(solver.fit(numpy_to_eigen(S), numpy_to_eigen(Omega)))
    cdef vector[double] fvals = solver.get_fval_list()
    
    # Convert vector to numpy array
    cdef np.ndarray f_val_list = np.array(fvals)
    
    return Omega_est, f_val_list


def lq_cov2(np.ndarray[np.float64_t, ndim=2] S, double lambda_, double q_f, 
           int max_iter=3000, double tol=1e-5, bool warm_start=True, bool msg=False, double tol_ws=1e-4):
    """
    C++ implementation of lq_cov algorithm. (using stationary condition)
    
    Parameters:
        S(np.ndarray):      Sample covariance matrix (p x p)
        lambda\_(float):    Penalty parameter
        q_f(float):         Target q value (0 <= q_f < 1)
        max_iter(int):      Maximum number of iterations
        tol(float):         Convergence tolerance
        warm_start(bool):   Whether to use warm-starting
        msg(bool):          Whether to print messages

    Returns
    ------
    Omega : np.ndarray
        Estimated sparse precision matrix

    f_val_list : np.ndarray
        Objective values

    KKT_list : np.ndarray
        stationarity residuals

    time_list : np.ndarray
        Time taken for each iteration
    """
    cdef np.ndarray Omega = np.diag(1 / (np.diag(S) + lambda_))
    if warm_start:
        Omega = glasso(S, lambda_, tol=1e-4, max_iter=1000)[1]

    cdef LqCov* solver = new LqCov(lambda_, q_f, max_iter, 0, tol, tol_ws, warm_start, msg)
    cdef np.ndarray Omega_est = eigen_to_numpy(solver.fit2(numpy_to_eigen(S), numpy_to_eigen(Omega)))
    cdef vector[double] fvals = solver.get_fval_list_out()
    cdef vector[double] res = solver.get_KKT_list()
    cdef vector[double] times = solver.get_time_list()
    
    # Convert vector to numpy array
    cdef np.ndarray f_val_list = np.array(fvals)
    cdef np.ndarray KKT_list = np.array(res)
    cdef np.ndarray time_list = np.array(times)
    
    return Omega_est, f_val_list, KKT_list, time_list


def lq_cov3(np.ndarray[np.float64_t, ndim=2] S, double lambda_, double q_f, int max_time=1800, 
           int max_iter=3000, double tol=1e-5, bool warm_start=True, bool msg=False, double tol_ws=1e-4
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

    Returns
    ------
    Omega : np.ndarray
        Estimated sparse precision matrix

    f_val_list : np.ndarray
        Objective values
    
    KKT_list : np.ndarray
        Stationarity residuals
    
    time_list : np.ndarray
        Time taken for each iteration
    """
    cdef np.ndarray Omega = np.diag(1 / (np.diag(S) + lambda_))
    if warm_start:
        Omega = glasso(S, lambda_, tol=1e-4, max_iter=1000)[1]

    cdef LqCov* solver = new LqCov(lambda_, q_f, max_iter, max_time, tol, tol_ws, warm_start, msg)
    cdef np.ndarray Omega_est = eigen_to_numpy(solver.fit3(numpy_to_eigen(S), numpy_to_eigen(Omega)))
    cdef vector[double] fvals = solver.get_fval_list_out()
    cdef vector[double] res = solver.get_KKT_list()
    cdef vector[double] times = solver.get_time_list()
    
    # Convert vector to numpy array
    cdef np.ndarray f_val_list = np.array(fvals)
    cdef np.ndarray KKT_list = np.array(res)
    cdef np.ndarray time_list = np.array(times)
    
    return Omega_est, f_val_list, KKT_list, time_list