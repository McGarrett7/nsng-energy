"""Threshold selection, metrics, and paired inference for Section V."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLDS = np.arange(30, 71, 5) / 100


def counts(labels: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.bincount(2 * labels.astype(int) + predicted, minlength=4)


def classification(confusion: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = (int(value) for value in confusion)
    return {
        "accuracy": (tp + tn) / (tn + fp + fn + tp),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        raise ValueError("Validation requires normal and anomalous observations.")
    return max(
        THRESHOLDS, key=lambda threshold: (
            classification(counts(labels, scores > threshold))["f1"],
            -abs(round(float(threshold), 2) * 100 - 45),
            threshold,
        ),
    ).item()


def summarize(frames: list[pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    rows = []
    for frame in frames:
        y, predicted, score = (frame[column].to_numpy() for column in ("label", "prediction", "score"))
        confusion = counts(y, predicted)
        rows.append({
            "sequence": frame["sequence"].iat[0], "group": frame["group"].iat[0],
            "fold": frame["fold"].iat[0], "fault": frame["fault"].iat[0],
            "threshold": frame["threshold"].iat[0], "samples": len(frame),
            "anomaly_ratio": y.mean(), **classification(confusion), "roc_auc": roc_auc(y, score),
            **dict(zip(("tn", "fp", "fn", "tp"), confusion)),
        })
    per_sequence = pd.DataFrame(rows)
    labels = np.concatenate([frame["label"] for frame in frames])
    scores = np.concatenate([frame["score"] for frame in frames])
    confusion = per_sequence[["tn", "fp", "fn", "tp"]].sum().to_numpy()
    pooled = classification(confusion)
    pooled["roc_auc"] = roc_auc(labels, scores)
    macro = per_sequence[list(pooled)].mean()
    metrics = {**pooled, **{f"macro_{name}": float(value) for name, value in macro.items()}}
    metrics.update(dict(zip(("tn", "fp", "fn", "tp"), confusion)))
    return metrics, per_sequence


def threshold_grid(frames: list[pd.DataFrame]) -> list[dict]:
    labels = np.concatenate([frame["label"] for frame in frames])
    scores = np.concatenate([frame["score"] for frame in frames])
    auc = roc_auc(labels, scores)
    return [
        {"threshold": threshold, **classification(counts(labels, scores > threshold)), "roc_auc": auc}
        for threshold in THRESHOLDS
    ]


def paired_group_test(full: list[pd.DataFrame], baseline: list[pd.DataFrame], seed: int) -> dict:
    """Swap methods within acquisition groups and re-evaluate the pooled F1 difference."""
    a = pd.concat(full).set_index(["sequence", "sample"])
    b = pd.concat(baseline).set_index(["sequence", "sample"]).reindex(a.index)
    if not a["label"].equals(b["label"]):
        raise ValueError("Paired testing requires predictions for the same labeled observations.")
    pairs = []
    for _, frame in a.groupby("group"):
        y = frame["label"].to_numpy()
        pairs.append((counts(y, frame["prediction"].to_numpy()),
                      counts(y, b.loc[frame.index, "prediction"].to_numpy())))
    first, second = np.asarray(pairs).transpose(1, 0, 2)

    def pooled_f1(confusion):
        _, fp, fn, tp = confusion.T
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    observed = pooled_f1(first.sum(axis=0)) - pooled_f1(second.sum(axis=0))
    exact = len(pairs) <= 16
    if exact:
        swaps = np.array(list(product((0, 1), repeat=len(pairs))))
    else:
        swaps = np.random.default_rng(seed).integers(0, 2, size=(10000, len(pairs)))
    adjustment = swaps @ (second - first)
    differences = pooled_f1(first.sum(axis=0) + adjustment) - pooled_f1(second.sum(axis=0) - adjustment)
    extreme = np.count_nonzero(np.abs(differences) >= abs(observed) - 1e-12)
    return {
        "statistic": "pooled F1 difference", "difference": float(observed),
        "p_value_two_sided": (int(extreme) + (not exact)) / (len(swaps) + (not exact)),
        "acquisition_groups": len(pairs), "permutations": len(swaps), "exact": exact,
    }
