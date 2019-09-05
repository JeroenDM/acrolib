import numpy as np
from acrolib.quaternion import Quaternion


def quat_distance(qa: Quaternion, qb: Quaternion):
    """ Half of the rotation angle to bring qa to qb."""
    return np.arccos(np.abs(qa.elements @ qb.elements))


def test_create_extended_quat():
    q = Quaternion()
    assert isinstance(q, Quaternion)


def test_max_distance():
    for _ in range(10):
        dist = quat_distance(Quaternion.random(), Quaternion.random())
        assert dist <= (0.5 * np.pi)


def test_random_near():
    q = Quaternion()
    for _ in range(5):
        q_near = q.random_near(0.11)
        assert quat_distance(q, q_near) <= 0.11
        assert np.any(np.not_equal(q.rotation_matrix, q_near.rotation_matrix))
    for _ in range(5):
        q_near = q.random_near(0.01)
        assert quat_distance(q, q_near) <= 0.01
        assert np.any(np.not_equal(q.rotation_matrix, q_near.rotation_matrix))


def test_random_near_large_dist():
    q = Quaternion()
    for _ in range(5):
        q_near = q.random_near(10.0)
        assert quat_distance(q, q_near) <= (0.5 * np.pi)
        assert np.any(np.not_equal(q.rotation_matrix, q_near.rotation_matrix))
