'''
This module provides SCAD and MCP penalty terms
'''

import numpy as np
from .config import SCAD_A, MCP_GAMMA


def scad_penalty(x: np.ndarray, lambda_val: float, a: float = SCAD_A):
    """
    Computes the SCAD penalty
    Args:
        x: input value (array)
        lambda_val: sparsity parameter (λ)
        a: decay parameter (usually > 2)
    Returns:
        SCAD penalty value
    """
    penalty = np.zeros_like(x)

    # Case 1: x <= lambda
    mask1 = x <= lambda_val
    penalty[mask1] = lambda_val * x[mask1]

    # Case 2: lambda < x <= a*lambda
    mask2 = (x > lambda_val) & (x <= a * lambda_val)
    term = 2 * a * lambda_val * x[mask2] - x[mask2] ** 2 - lambda_val**2
    penalty[mask2] = term / (2 * (a - 1))

    # Case 3: x > a*lambda
    mask3 = x > a * lambda_val
    penalty[mask3] = (lambda_val**2 * (a + 1)) / 2

    return penalty


def matrix_scad(X: np.ndarray, lambda_val: float, a: float = SCAD_A) -> float:
    """
    Compute the SCAD penalty sum of matrix X: sum_{i,j} phi_SCAD(|X_ij|)
    Args:
        X: input matrix
        lambda_val: sparsity parameter (λ)
        a: decay parameter (usually > 2)
    Returns:
        Scalar penalty sum
    """
    return np.sum(scad_penalty(X, lambda_val, a))


def scad_derivative(x: np.ndarray, lambda_val: float, a: float = SCAD_A):
    """
    Computes the SCAD derivative
    Args:
        x: input value (scalar or array)
        lambda_val: sparsity parameter (λ)
        a: decay parameter (usually > 2)
    Returns:
        SCAD derivative value
    """
    derivative = np.zeros_like(x)

    # Case 1: x <= lambda
    mask1 = x <= lambda_val
    derivative[mask1] = lambda_val

    # Case 2: lambda < x <= a*lambda
    mask2 = (x > lambda_val) & (x <= a * lambda_val)
    derivative[mask2] = (a * lambda_val - x[mask2]) / (a - 1)

    # Case 3: x > a*lambda
    mask3 = x > a * lambda_val
    derivative[mask3] = 0

    return derivative


def matrix_scad_derivative(X: np.ndarray, lambda_val: float, a: float = SCAD_A) -> float:
    """
    Computes the sum of SCAD derivatives of matrix X: sum_{i,j} phi_SCAD'(|X_ij|)
    Args:
        X: input matrix
        lambda_val: λ
        a: decay parameter
    Returns:
        Sum of scalar derivatives
    """
    return np.sum(scad_derivative(X, lambda_val, a))


def mcp_penalty(x: np.ndarray, lambda_val: float, gamma: float = MCP_GAMMA):
    """
    Computes the MCP penalty for a single value
    Args:
        x: input value (scalar or array)
        lambda_val: sparsity parameter (λ)
        gamma: concavity parameter (> 1)
    Returns:
        MCP penalty value
    """
    abs_x = np.abs(x)
    penalty = np.zeros_like(abs_x)
    
    # Case 1: |x| <= γλ
    mask = abs_x <= gamma * lambda_val
    penalty[mask] = lambda_val * abs_x[mask] - (abs_x[mask]**2) / (2 * gamma)
    
    # Case 2: |x| > γλ
    penalty[~mask] = gamma * lambda_val**2 / 2
    
    return penalty


def matrix_mcp(X: np.ndarray, lambda_val: float, gamma: float = MCP_GAMMA) -> float:
    """
    Calculate the MCP penalty sum of matrix X: sum_{i,j} ϕ_MCP(|X_ij|)
    """
    return np.sum(mcp_penalty(X, lambda_val, gamma))


def mcp_derivative(x: np.ndarray, lambda_val: float, gamma: float = MCP_GAMMA):
    """
    Compute MCP derivative
    Args:
        x: input value (array)
        lambda_val: λ
        gamma: concavity parameter (> 1)
    Returns:
        MCP derivative value
    """
    abs_x = np.abs(x)
    derivative = np.zeros_like(abs_x)
    
    # Case 1: |x| <= γλ
    mask = abs_x <= gamma * lambda_val
    derivative[mask] = lambda_val - abs_x[mask] / gamma
    
    # Case 2: |x| > γλ (导数为0)
    return derivative

def matrix_mcp_derivative(X: np.ndarray, lambda_val: float, gamma: float = MCP_GAMMA) -> float:
    """
    Calculate the sum of the MCP derivatives of matrix X: sum_{i,j} ϕ_MCP'(|X_ij|)
    """
    return np.sum(mcp_derivative(X, lambda_val, gamma))