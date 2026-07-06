'''
c++ part of IRL core module

modified quic function (c++)
'''

import numpy as np

def quic(
    mode: bytes, p: int, 
    S: np.ndarray, 
    L: np.ndarray,
    pathLen: int,
    path: np.ndarray,
    tol: float, msg: int, max_iter: int, 
    X: np.ndarray, 
    W: np.ndarray,
    opt: np.ndarray,
    cputime: np.ndarray,
    iter: np.ndarray,
    dGap: np.ndarray,
    info_list: np.ndarray
) -> None:

    pass