import numpy as np
import pytest

from projections import metrics


def test_log_loss_matches_per_event_definition():
    p, k, n = [0.25], [1], [4]
    expected = -(np.log(0.25) + 3 * np.log(0.75)) / 4
    assert metrics.log_loss(p, k, n) == pytest.approx(expected)


def test_brier_matches_per_event_definition():
    assert metrics.brier([0.25], [1], [4]) == pytest.approx((0.75**2 + 3 * 0.25**2) / 4)


def test_perfect_rate_beats_wrong_rate():
    k, n = [30, 10], [100, 100]
    assert metrics.log_loss([0.3, 0.1], k, n) < metrics.log_loss([0.2, 0.2], k, n)
    assert metrics.weighted_rmse([0.3, 0.1], k, n) == pytest.approx(0)


def test_paired_bootstrap_sign_and_interval():
    rng = np.random.default_rng(1)
    n = np.full(200, 100.0)
    a = rng.normal(50, 1, 200)
    b = a + 5
    pt, lo, hi = metrics.paired_bootstrap(a, b, n)
    assert pt == pytest.approx(-0.05)
    assert lo <= pt <= hi < 0


def test_interval_coverage_is_near_nominal_when_model_is_right():
    rng = np.random.default_rng(2)
    p_true = rng.uniform(0.1, 0.3, 2000)
    n = np.full(2000, 400)
    k = rng.binomial(n, p_true)
    draws = np.tile(p_true, (500, 1))
    cov = metrics.interval_coverage(draws, k, n, 0.8)
    assert 0.77 < cov < 0.88  # discrete counts make intervals slightly conservative


def test_calibration_table_bins():
    p = np.linspace(0.1, 0.3, 100)
    tab = metrics.calibration_table(p, (p * 100).round(), np.full(100, 100), bins=5)
    assert len(tab) == 5
    assert np.allclose(tab.projected, tab.realized, atol=0.01)
