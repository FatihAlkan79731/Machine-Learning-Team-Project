import numpy as np
import pytest

from uarps.evaluation.splits import negative_subsample
from uarps.models.calibration import RiskCalibrator, prior_correction


def test_prior_correction_identity_when_no_subsampling():
    p = np.array([0.0, 0.2, 0.7, 1.0])
    np.testing.assert_allclose(prior_correction(p, 1.0), p)


def test_prior_correction_recovers_true_rate():
    # True rate 1%; keeping 5% of negatives raises the sample rate to about 16.8%.
    true_rate, beta = 0.01, 0.05
    p_sample = true_rate / (true_rate + beta * (1 - true_rate))
    assert prior_correction([p_sample], beta)[0] == pytest.approx(true_rate)


def test_prior_correction_is_monotonic():
    p = np.linspace(0, 1, 101)
    assert np.all(np.diff(prior_correction(p, 0.05)) >= 0)


def test_calibrator_end_to_end_on_subsampled_model():
    rng = np.random.default_rng(3)
    n = 200_000
    x = rng.random(n)
    true_p = 0.002 + 0.02 * x
    y = (rng.random(n) < true_p).astype(int)
    beta = 0.05
    keep = negative_subsample(y, beta, seed=4)
    # the subsample used for training is enriched with positives
    assert y[keep].mean() > 5 * y.mean()
    # a model trained on that subsample outputs probabilities on the subsample scale
    p_s = true_p / (true_p + beta * (1 - true_p))
    # prior correction alone already restores the true scale
    np.testing.assert_allclose(prior_correction(p_s, beta), true_p)
    # calibrate on a validation half with the natural class ratio, check on the other half
    val, test = np.arange(n) < n // 2, np.arange(n) >= n // 2
    cal = RiskCalibrator(beta=beta).fit(p_s[val], y[val])
    p_hat = cal.transform(p_s[test])
    assert p_hat.mean() == pytest.approx(y[test].mean(), rel=0.15)


def test_calibrator_requires_fit():
    with pytest.raises(RuntimeError):
        RiskCalibrator().transform([0.5])
