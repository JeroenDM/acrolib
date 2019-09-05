from .slow_cost_functions import (
    _l1_norm,
    _l2_norm,
    _linfinity_norm,
    _sum_squared,
    _weighted_sum_squared,
    cost_function_l1_norm,
    cost_function_l2_norm,
    cost_function_sum_squared,
    cost_function_linfinity_norm,
    cost_function_weighted_sum_squared,
)

import numpy as np
from numpy.testing import assert_almost_equal


class TestMathFunctions:
    def test_all(self):
        x = np.array([1, 2, -3])
        x_l1 = 1 + 2 + 3
        x_l2 = np.sqrt(1 + 2 ** 2 + 3 ** 2)
        x_sq = 1 + 2 ** 2 + 3 ** 2
        x_inf = 3
        assert_almost_equal(_l1_norm(x), x_l1)
        assert_almost_equal(_l2_norm(x), x_l2)
        assert_almost_equal(_sum_squared(x), x_sq)
        assert_almost_equal(_linfinity_norm(x), x_inf)

        w = np.array([0.5, 0, 1])
        x_sq_w = 0.5 * 1 ** 2 + (-3) ** 2
        assert_almost_equal(_weighted_sum_squared(x, w), x_sq_w)


class TestSlowFunctions:
    def test_l1_norm(self):
        a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        b = np.array([[3, 3, 3], [2, 2, 2]])
        c_desired = np.array([[6, 3], [3, 0], [0, 3]])

        c = cost_function_l1_norm(a, b)

        assert c.shape == c_desired.shape
        assert_almost_equal(c, c_desired)

    def test_l2_norm(self):
        a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        b = np.array([[3, 3, 3], [2, 2, 2]])
        v1 = np.sqrt(4 + 4 + 4)
        v2 = np.sqrt(1 + 1 + 1)
        c_desired = np.array([[v1, v2], [v2, 0], [0, v2]])

        c = cost_function_l2_norm(a, b)

        assert c.shape == c_desired.shape
        assert_almost_equal(c, c_desired)

    def test_sum_squared(self):
        a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        b = np.array([[3, 3, 3], [2, 2, 2]])
        v1 = 4 + 4 + 4
        c_desired = np.array([[v1, 3], [3, 0], [0, 3]])

        c = cost_function_sum_squared(a, b)

        assert c.shape == c_desired.shape
        assert_almost_equal(c, c_desired)

    def test_infinity_norm(self):
        a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        b = np.array([[3, 3, 3], [2, 2, 2]])
        v1 = 2
        c_desired = np.array([[v1, 1], [1, 0], [0, 1]])

        c = cost_function_linfinity_norm(a, b)

        assert c.shape == c_desired.shape
        assert_almost_equal(c, c_desired)

    def test_weighted_sum_squared(self):
        a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        b = np.array([[3, 3, 3], [2, 2, 2]])
        w = np.array([0.5, 0, 1])

        c_desired = np.zeros((3, 2))
        for ia in range(len(a)):
            for ib in range(len(b)):
                c_desired[ia, ib] = np.sum(w * (a[ia] - b[ib]) ** 2)

        c = cost_function_weighted_sum_squared(a, b, w)

        assert c.shape == c_desired.shape
        assert_almost_equal(c, c_desired)
