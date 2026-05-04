from __future__ import annotations

import numpy as np
import pytest

from mnemo.core.statistics import (
    one_sided_t_test_lower,
    remove_outliers_to_mean,
    sidak_combine,
)


def test_remove_outliers_replaces_extremes_with_middle_mean():
    arr = np.array([100.0, 1.0, 2.0, 3.0, 4.0, 5.0, -100.0])
    cleaned = remove_outliers_to_mean(arr, fraction=1 / 7)
    middle_mean = np.mean([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cleaned[0] == pytest.approx(middle_mean)
    assert cleaned[-1] == pytest.approx(middle_mean)
    assert cleaned[1:-1].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_remove_outliers_zero_fraction_is_identity():
    arr = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(remove_outliers_to_mean(arr, fraction=0.0), arr)


def test_remove_outliers_rejects_bad_fraction():
    with pytest.raises(ValueError, match="fraction"):
        remove_outliers_to_mean(np.array([1.0, 2.0]), fraction=0.5)
    with pytest.raises(ValueError, match="fraction"):
        remove_outliers_to_mean(np.array([1.0, 2.0]), fraction=-0.1)


def test_one_sided_t_test_low_p_when_suspect_lower():
    rng = np.random.default_rng(0)
    sus = rng.normal(0.0, 1.0, size=200)
    val = rng.normal(2.0, 1.0, size=200)
    p = one_sided_t_test_lower(sus, val)
    assert p < 0.001


def test_one_sided_t_test_high_p_when_no_difference():
    rng = np.random.default_rng(0)
    sus = rng.normal(0.0, 1.0, size=200)
    val = rng.normal(0.0, 1.0, size=200)
    p = one_sided_t_test_lower(sus, val)
    assert p > 0.05


def test_one_sided_t_test_requires_min_samples():
    with pytest.raises(ValueError, match="at least 2"):
        one_sided_t_test_lower([1.0], [2.0, 3.0])


def test_sidak_combine_independent_uniform_p_values():
    p = sidak_combine([0.1, 0.1, 0.1])
    expected = 1.0 - 0.9**3
    assert p == pytest.approx(expected, rel=1e-9)


def test_sidak_combine_single_value_passthrough():
    assert sidak_combine([0.42]) == pytest.approx(0.42)


def test_sidak_combine_preserves_significance_for_tiny_p_values():
    p = sidak_combine([1e-10, 1e-10, 1e-10])
    assert 0 < p < 1e-8


def test_sidak_combine_saturates_to_one_when_all_p_are_one():
    p = sidak_combine([1.0, 1.0, 1.0])
    assert p == pytest.approx(1.0, abs=1e-9)


def test_sidak_combine_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        sidak_combine([])
