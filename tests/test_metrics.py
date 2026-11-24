import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from uarps.evaluation.metrics import (
    block_bootstrap_ci,
    evaluate,
    recall_at_k,
    threshold_for_budget,
    threshold_max_f1,
)


def test_recall_at_k_perfect_and_random():
    y = np.array([0] * 90 + [1] * 10)
    perfect = y.astype(float)
    assert recall_at_k(y, perfect, 10) == 1.0
    reversed_scores = 1 - perfect
    assert recall_at_k(y, reversed_scores, 10) == 0.0


def test_recall_at_k_rejects_bad_k():
    with pytest.raises(ValueError):
        recall_at_k([0, 1], [0.1, 0.9], 0)


def test_threshold_max_f1_separates_classes():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert threshold_max_f1(y, s) == pytest.approx(0.8)


def test_threshold_for_budget():
    s = np.arange(100) / 100
    t = threshold_for_budget(s, 5)
    assert (s >= t).sum() == 5


def test_evaluate_keys_and_values():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    s = np.clip(y * 0.3 + rng.random(5000) * 0.5, 0, 1)
    res = evaluate(y, s, threshold=0.5)
    for key in ["pr_auc", "roc_auc", "brier", "f1", "recall_at_5pct", "tp", "fn"]:
        assert key in res
    assert res["tp"] + res["fn"] == y.sum()
    assert 0 <= res["pr_auc"] <= 1


def test_block_bootstrap_ci_contains_point_and_paired_difference():
    rng = np.random.default_rng(2)
    n = 3650
    days = np.repeat(np.arange(365), 10)
    y = (rng.random(n) < 0.05).astype(int)
    good = y * 0.5 + rng.random(n) * 0.5
    noise = rng.random(n)
    point, lo, hi = block_bootstrap_ci(y, good, days, n_resamples=200)
    assert lo <= point <= hi
    assert point == pytest.approx(average_precision_score(y, good))
    diff, dlo, dhi = block_bootstrap_ci(y, (good, noise), days, n_resamples=200)
    assert diff > 0 and dlo > 0  # the informative model is clearly better
