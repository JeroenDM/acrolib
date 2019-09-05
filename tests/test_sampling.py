import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from acrolib.sampling import Sampler, SampleMethod


def assert_in_range(x, lower=0, upper=1):
    assert max(x) <= upper
    assert min(x) >= lower


class TestSampler:
    def test_random(self):
        s = Sampler()
        samples = s.sample(5, 3, method=SampleMethod.random_uniform)
        assert samples.shape == (5, 3)
        assert_in_range(samples.flatten())

    def test_deterministic(self):
        s = Sampler()
        samples = s.sample(5, 3, method=SampleMethod.deterministic_uniform)
        assert samples.shape == (5, 3)
        assert_in_range(samples.flatten())
        desired_samples = np.array(
            [
                [0.33333333, 0.2, 0.14285714],
                [0.66666667, 0.4, 0.28571429],
                [0.11111111, 0.6, 0.42857143],
                [0.44444444, 0.8, 0.57142857],
                [0.77777778, 0.04, 0.71428571],
            ]
        )

        assert_almost_equal(samples, desired_samples)

        # we should get different samples on a second run
        desired_samples_2 = np.array(
            [
                [0.22222222, 0.24, 0.85714286],
                [0.55555556, 0.44, 0.02040816],
                [0.88888889, 0.64, 0.16326531],
                [0.03703704, 0.84, 0.30612245],
                [0.37037037, 0.08, 0.44897959],
            ]
        )
        samples_2 = s.sample(5, 3, method=SampleMethod.deterministic_uniform)
        assert_almost_equal(samples_2, desired_samples_2)

    def test_raise_exception(self):
        s = Sampler()
        with pytest.raises(Exception) as info:
            s.sample(5, 3, method=3)
        msg = "Unkown sampling method"
        assert msg in str(info.value)

    def test_get_many_samples(self):
        s = Sampler()
        samples = s.sample(1000, 3, method=SampleMethod.deterministic_uniform)
        assert len(samples) == 1000
