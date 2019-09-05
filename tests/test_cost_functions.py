from acrolib.cost_functions import (  # pylint: disable=no-name-in-module
    norm_l1,
    norm_l2,
    sum_squared,
    norm_infinity,
    weighted_sum_squared,
)

import numpy as np
from numpy.testing import assert_almost_equal
from .slow_cost_functions import (
    cost_function_l1_norm,
    cost_function_l2_norm,
    cost_function_sum_squared,
    cost_function_linfinity_norm,
    cost_function_weighted_sum_squared,
)

np.random.seed(42)


class TestCythonFunctions:
    def test_norm_l1(self):
        A = np.random.rand(5, 3)
        B = np.random.rand(6, 3)
        C_desired = cost_function_l1_norm(A, B)
        C = norm_l1(A, B)

        assert C.shape == C_desired.shape
        assert_almost_equal(C, C_desired)

    def test_norm_l2(self):
        A = np.random.rand(5, 3)
        B = np.random.rand(6, 3)
        C_desired = cost_function_l2_norm(A, B)
        C = norm_l2(A, B)

        assert C.shape == C_desired.shape
        assert_almost_equal(C, C_desired)

    def test_sum_squared(self):
        A = np.random.rand(5, 3)
        B = np.random.rand(6, 3)
        C_desired = cost_function_sum_squared(A, B)
        C = sum_squared(A, B)

        assert C.shape == C_desired.shape
        assert_almost_equal(C, C_desired)

    def test_weighted_sum_squared(self):
        A = np.random.rand(5, 3)
        B = np.random.rand(6, 3)
        w = np.random.rand(3)
        C_desired = cost_function_weighted_sum_squared(A, B, w)
        C = weighted_sum_squared(A, B, w)

        assert C.shape == C_desired.shape
        assert_almost_equal(C, C_desired)

    def test_norm_infinity(self):
        A = np.random.rand(5, 3)
        B = np.random.rand(6, 3)
        C_desired = cost_function_linfinity_norm(A, B)
        C = norm_infinity(A, B)

        assert C.shape == C_desired.shape
        assert_almost_equal(C, C_desired)

