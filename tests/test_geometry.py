
import numpy as np
import matplotlib
import mpl_toolkits

from numpy.testing import assert_almost_equal
from acrolib.geometry import (
    rot_x,
    rot_y,
    rot_z,
    pose_x,
    pose_y,
    pose_z,
    translation,
    tf_inverse,
    quat_distance,
    rpy_to_rot_mat,
    rotation_matrix_to_rpy
)
from acrolib.quaternion import Quaternion

IDENTITY = np.eye(4)

class TestMatrixCreation:
    def test_rot_x(self):
        A = rot_x(0.6)
        B = rot_x(-0.6)
        assert_almost_equal(A @ B, np.eye(3))
        assert_almost_equal(A[0, :3], np.array([1, 0, 0]))
        assert_almost_equal(A[:3, 0], np.array([1, 0, 0]))

    def test_rot_y(self):
        A = rot_y(0.6)
        B = rot_y(-0.6)
        assert_almost_equal(A @ B, np.eye(3))
        assert_almost_equal(A[1, :3], np.array([0, 1, 0]))
        assert_almost_equal(A[:3, 1], np.array([0, 1, 0]))

    def test_rot_z(self):
        A = rot_z(0.6)
        B = rot_z(-0.6)
        assert_almost_equal(A @ B, np.eye(3))
        assert_almost_equal(A[2, :3], np.array([0, 0, 1]))
        assert_almost_equal(A[:3, 2], np.array([0, 0, 1]))

    def test_pose_x(self):
        A = pose_x(0.6, 1, 2, 3)
        assert_almost_equal(A[0, :3], np.array([1, 0, 0]))
        assert_almost_equal(A[:3, 0], np.array([1, 0, 0]))
        assert_almost_equal(A[:3, 3], np.array([1, 2, 3]))
        assert_almost_equal(A[3, :], np.array([0, 0, 0, 1]))

    def test_pose_y(self):
        A = pose_y(0.6, 1, 2, 3)
        assert_almost_equal(A[1, :3], np.array([0, 1, 0]))
        assert_almost_equal(A[:3, 1], np.array([0, 1, 0]))
        assert_almost_equal(A[:3, 3], np.array([1, 2, 3]))
        assert_almost_equal(A[3, :], np.array([0, 0, 0, 1]))

    def test_pose_z(self):
        A = pose_z(0.6, 1, 2, 3)
        assert_almost_equal(A[2, :3], np.array([0, 0, 1]))
        assert_almost_equal(A[:3, 2], np.array([0, 0, 1]))
        assert_almost_equal(A[:3, 3], np.array([1, 2, 3]))
        assert_almost_equal(A[3, :], np.array([0, 0, 0, 1]))
    
    def test_translation(self):
        A = translation(1, 2, -3)
        B = pose_z(0, 1, 2, -3)
        assert_almost_equal(A, B)


class TestOperations:
    def test_inverse(self):
        A = pose_z(0.6, 1, 2, 3)
        assert_almost_equal(tf_inverse(A) @ A, IDENTITY)
        assert_almost_equal(A @ tf_inverse(A), IDENTITY)
        
    def test_quat_distance(self):
        q1 = Quaternion()
        assert_almost_equal(quat_distance(q1, q1), 0.0)
        q2 = Quaternion(axis=[0, 0, 1], angle=np.pi)
        assert_almost_equal(quat_distance(q1, q2), np.pi / 2)

class TestConversions:
    def test_matrix_to_rxyz(self):
        for _ in range(100):
            R_random = Quaternion.random().rotation_matrix
            rpy = rotation_matrix_to_rpy(R_random)
            R_converted = rpy_to_rot_mat(rpy)
            assert_almost_equal(R_random, R_converted)
