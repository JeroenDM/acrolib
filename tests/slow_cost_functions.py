""" Cost functions for the dynamic programming algorithms.

All cost functions take two matrices as input with an equal number of columns.
Then the cost function is applied to al combinations of rows.
"""

import numpy as np


def _apply_cost_function_to_rows(A, B, fun):
    C = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = fun(A[i] - B[j])
    return C


def _weighted_sum_squared(x, w):
    assert len(x) == len(w)
    return np.sum(w * (x ** 2))


def _sum_squared(x):
    return np.sum(x ** 2)


def _l2_norm(x):
    return np.sqrt(_sum_squared(x))


def _l1_norm(x):
    return np.sum(np.abs(x))


def _linfinity_norm(x):
    return np.max(np.abs(x))


def cost_function_weighted_sum_squared(A, B, weights):
    return _apply_cost_function_to_rows(
        A, B, lambda x: _weighted_sum_squared(x, weights)
    )


def cost_function_sum_squared(A, B):
    return _apply_cost_function_to_rows(A, B, _sum_squared)


def cost_function_l2_norm(A, B):
    return _apply_cost_function_to_rows(A, B, _l2_norm)


def cost_function_l1_norm(A, B):
    return _apply_cost_function_to_rows(A, B, _l1_norm)


def cost_function_linfinity_norm(A, B):
    return _apply_cost_function_to_rows(A, B, _linfinity_norm)
