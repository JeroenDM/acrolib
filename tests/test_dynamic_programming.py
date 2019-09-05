import pytest
import numpy as np

from acrolib.dynamic_programming import (
    apply_cost_function,
    calculate_value_function,
    calculate_value_function_with_state_cost,
    extract_shortest_path,
    shortest_path_with_state_cost,
    shortest_path,
)
from acrolib.cost_functions import norm_l1  # pylint: disable=no-name-in-module
from numpy.testing import assert_almost_equal


class TestBookExample:
    """
    Example from Operations research book.
    """

    def test_calculate_value_function(self):
        C1 = np.array([550, 900, 770], ndmin=2)
        C2 = np.array([[680, 790, 1050], [580, 760, 660], [510, 700, 830]])
        C3 = np.array([[610, 790], [540, 940], [790, 270]])
        C4 = np.array([[1030], [1390]])
        transition_costs = [C1, C2, C3, C4]

        vind, v = calculate_value_function(transition_costs)

        v_exact = [
            np.array([2870.0]),
            np.array([2320, 2220, 2150]),
            np.array([1640, 1570, 1660]),
            np.array([1030, 1390]),
            np.array([0.0]),
        ]

        v_ind_exact = [[0], [0, 0, 0], [0, 0, 1], [0, 0], [0]]

        print(v)
        assert len(v) == len(v_exact)
        for v1, v2 in zip(v, v_exact):
            assert_almost_equal(v1, v2)

        assert len(vind) == len(v_ind_exact)
        for v1, v2 in zip(vind, v_ind_exact):
            assert_almost_equal(v1, v2)

    def test_state_cost(self):
        C1 = np.array([550, 900, 770], ndmin=2)
        C2 = np.array([[680, 790, 1050], [580, 760, 660], [510, 700, 830]])
        C3 = np.array([[610, 790], [540, 940], [790, 270]])
        C4 = np.array([[1030], [1390]])

        v_exact = [
            np.array([3020]),
            np.array([2470, 2370, 2300]),
            np.array([1790, 1720, 1910]),
            np.array([1180, 1640]),
            np.array([50]),
        ]

        v_ind_exact = [[0], [0, 0, 0], [0, 0, 1], [0, 0], [0]]

        transition_costs = [C1, C2, C3, C4]
        state_costs = np.array([[0], [0, 0, 0], [0, 0, 0], [100, 200], [50]])

        v_ind, v = calculate_value_function_with_state_cost(
            transition_costs, state_costs
        )

        print(v)
        print(v_ind)

        assert len(v) == len(v_exact)
        for v1, v2 in zip(v, v_exact):
            assert_almost_equal(v1, v2)

        assert len(v_ind) == len(v_ind_exact)
        for v1, v2 in zip(v_ind, v_ind_exact):
            assert_almost_equal(v1, v2)


@pytest.fixture
def simple_graph():
    """ Define a simple graph with three stages and (2, 3, 3) states in
    those stages. Also define exact solutions for transition_costs,
    value function and indices to retreive shortest path, and the
    shortest path itself.
    """
    data1 = np.array([[0, 0], [0, 1]], dtype=np.double)
    data2 = np.array([[1, -1], [1, 0], [1, 1]], dtype=np.double)
    data3 = np.array([[0, 2], [2, 2]], dtype=np.double)
    data = [data1, data2, data3]
    C_exact = [np.array([[2, 1, 2], [3, 2, 1]]), np.array([[4, 4], [3, 3], [2, 2]])]
    v_exact = [np.array([4, 3]), np.array([4, 3, 2]), np.array([0, 0])]
    v_ind_exact = [[1, 2], [0, 0, 0], [0, 0]]
    path = [[0, 1], [1, 1], [0, 2]]
    return data, C_exact, v_exact, v_ind_exact, path


class TestSimpleGraph:
    def test_apply_cost_function(self, simple_graph):
        def f(d1, d2):
            """L1 norm cost function. """
            ci = np.zeros((len(d1), len(d2)))
            for i in range(len(d1)):
                for j in range(len(d2)):
                    ci[i, j] = np.sum(np.abs(d1[i] - d2[j]))
            return ci

        C = apply_cost_function(simple_graph[0], f)

        assert len(C) == len(simple_graph[1])
        for ca, cb in zip(C, simple_graph[1]):
            assert ca.shape == cb.shape
            assert_almost_equal(ca, cb)

    def test_calculate_value_function(self, simple_graph):
        v_ind, v = calculate_value_function(simple_graph[1])

        print(v)
        assert len(v) == len(simple_graph[2])
        for v1, v2 in zip(v, simple_graph[2]):
            assert_almost_equal(v1, v2)

        assert len(v_ind) == len(simple_graph[3])
        for v1, v2 in zip(v_ind, simple_graph[3]):
            assert_almost_equal(v1, v2)

    def test_extract_shortest_path(self, simple_graph):
        shortest_path = extract_shortest_path(
            simple_graph[0], simple_graph[3], simple_graph[2]
        )

        assert len(shortest_path) == len(simple_graph[4])
        for v1, v2 in zip(shortest_path, simple_graph[4]):
            assert_almost_equal(v1, v2)

    def test_shortest_path_l1_norm(self, simple_graph):

        res = shortest_path(simple_graph[0], norm_l1)
        assert res["success"]
        assert_almost_equal(res["path"], simple_graph[4])

    def test_shortest_path_state_cost(self, simple_graph):
        state_costs = [np.array([3, 1]), np.array([3, 1, 4]), np.array([2, 3])]
        transition_costs = [
            np.array([[2, 1, 2], [3, 2, 1]]),
            np.array([[4, 4], [3, 3], [2, 2]]),
        ]

        V_exact = [np.array([10, 9]), np.array([9, 6, 8]), np.array([2, 3])]
        V_ind_exact = [np.array([1, 1]), np.array([0, 0, 0]), np.array([0, 0])]

        V_ind, V = calculate_value_function_with_state_cost(
            transition_costs, state_costs
        )

        for v1, v2 in zip(V_ind, V_ind_exact):
            assert_almost_equal(v1, v2)

        for v1, v2 in zip(V, V_exact):
            assert_almost_equal(v1, v2)

        path_exact = [[0, 1], [1, 0], [0, 2]]

        res = shortest_path_with_state_cost(simple_graph[0], state_costs, norm_l1)
        assert res["success"]
        assert_almost_equal(res["path"], path_exact)
