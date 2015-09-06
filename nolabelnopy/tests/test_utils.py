import numpy as np
from numpy.testing import assert_array_equal

from utils import to_zero_one


def test_to_zero_one():
    labels = np.array([1, -1])
    cast_labels = to_zero_one(labels)
    assert_array_equal(np.array([1, 0]), cast_labels)
