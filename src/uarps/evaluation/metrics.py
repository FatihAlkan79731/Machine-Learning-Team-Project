"""Evaluation metrics for rare-event risk prediction (Section "Proposed Evaluation Strategy")."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recall_at_k(y_true: Sequence[int], y_score: Sequence[float], k_percent: float) -> float:
    """Share of all positives captured in the top ``k_percent`` % of ranked samples (Eq. 24)."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    if not 0 < k_percent <= 100:
        raise ValueError("k_percent must be in (0, 100]")
    total = y.sum()
    if total == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(len(s) * k_percent / 100.0)))
    top_idx = np.argsort(-s, kind="stable")[:n_top]
    return float(y[top_idx].sum() / total)


def threshold_max_f1(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Decision threshold that maximises F1 (to be chosen on the validation year only)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    # the last precision/recall pair has no threshold
    return float(thresholds[int(np.argmax(f1[:-1]))])


def threshold_for_budget(y_score: Sequence[float], k_percent: float) -> float:
    """Score threshold that flags the top ``k_percent`` % of samples."""
    s = np.asarray(y_score)
    return float(np.quantile(s, 1.0 - k_percent / 100.0))


def evaluate(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    k_percents: Sequence[float] = (1, 5, 10),
) -> dict[str, float]:
    """Compute all evaluation metrics of the paper for one set of predictions."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    y_pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    results = {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "positive_rate": float(y.mean()),
    }
    for k in k_percents:
        results[f"recall_at_{k:g}pct"] = recall_at_k(y, s, k)
    return results


def _block_indices(groups: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Sample row indices by a moving-block bootstrap over ordered groups (for example days)."""
    unique = np.unique(groups)
    n_groups = len(unique)
    block_len = max(1, min(block_len, n_groups))
    n_blocks = int(np.ceil(n_groups / block_len))
    starts = rng.integers(0, n_groups - block_len + 1, size=n_blocks)
    chosen = np.concatenate([unique[s : s + block_len] for s in starts])[:n_groups]
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    left = np.searchsorted(sorted_groups, chosen, side="left")
    right = np.searchsorted(sorted_groups, chosen, side="right")
    return np.concatenate([order[a:b] for a, b in zip(left, right, strict=True)])


def block_bootstrap_ci(
    y_true: Sequence[int],
    scores: Sequence[float] | tuple[Sequence[float], Sequence[float]],
    groups: Sequence,
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    n_resamples: int = 1000,
    block_len: int = 7,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Moving-block bootstrap confidence interval for a metric or a paired difference.

    If ``scores`` is a tuple ``(scores_a, scores_b)``, the interval is computed for
    ``metric(y, scores_a) - metric(y, scores_b)`` on the same resampled rows.

    Returns ``(point_estimate, lower, upper)``.
    """
    y = np.asarray(y_true)
    g = np.asarray(groups)
    paired = isinstance(scores, tuple)
    if paired:
        a, b = (np.asarray(x, dtype=float) for x in scores)

        def stat(idx: np.ndarray) -> float:
            return metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    else:
        s = np.asarray(scores, dtype=float)

        def stat(idx: np.ndarray) -> float:
            return metric(y[idx], s[idx])

    point = stat(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        idx = _block_indices(g, block_len, rng)
        if y[idx].min() == y[idx].max():  # resample without both classes
            continue
        values.append(stat(idx))
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lower), float(upper)
