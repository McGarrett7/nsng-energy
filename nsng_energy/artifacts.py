"""Export the dataset inventory, Figure 1, and consensus knowledge graph."""

from dataclasses import asdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agents import NAMES
from .data import Sequence
from .evaluation import classification, counts


def inventory(sequences: list[Sequence], destination: Path) -> pd.DataFrame:
    table = pd.DataFrame([{
        "sequence": sequence.name, "group": sequence.group, "fault": sequence.fault,
        "samples": len(sequence.power), "anomalies": sequence.labels.sum(),
        "anomaly_ratio": sequence.labels.mean(), "sha256": sequence.digest,
        "has_voltage": sequence.voltage is not None and np.isfinite(sequence.voltage).any(),
        "normal_reference": sequence.normal_key,
    } for sequence in sequences])
    table.to_csv(destination / "table_i_dataset.csv", index=False)
    return table


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_trace(path: Path, sequence_number: int, detail: dict, settings) -> dict:
    relative = Path("traces") / f"sequence_{sequence_number:03d}.npz"
    (path / relative).parent.mkdir(exist_ok=True)
    np.savez_compressed(
        path / relative,
        agents=np.asarray(NAMES), properties=np.asarray(["score", "confidence", "trust"]),
        rules=np.asarray(settings.rules), settings=json.dumps(asdict(settings)),
        states=detail["states"], outgoing=detail["outgoing"].astype(np.float16),
        conflict_edges=detail["conflict_edges"], rule_hits=detail["rule_hits"],
        rounds=detail["rounds"], final_trust=detail["trust"],
    )
    return {
        "file": relative.as_posix(),
        **{key: detail[key] for key in ("initial_conflicts", "final_conflicts", "initial_direct_conflicts",
                                        "final_direct_conflicts", "rounds", "trust")},
        "rule_counts": detail["rule_hits"].sum(axis=(0, 1, 2)).tolist(),
    }


def export_consensus(path: Path, sequences, full, trace_files) -> None:
    by_name = {frame["sequence"].iat[0]: frame for frame in full}
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        def emit(value: dict):
            handle.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")

        for sequence in sequences:
            frame = by_name[sequence.name]
            series_id = f"energy:dryer/{sequence.name}"
            detail = trace_files[sequence.name]
            emit({
                "id": series_id, "type": "DryerPowerSequence",
                "group": sequence.group, "fault": sequence.fault, "score": float(frame["score"].max()),
            })
            for i, (source, confidence, detected, threshold) in enumerate(
                frame[["score", "confidence", "prediction", "threshold"]].itertuples(index=False, name=None)
            ):
                measurement_id = f"{series_id}/{i}"
                emit({
                    "id": measurement_id, "type": "PowerObservation", "partOf": series_id,
                    "ctime": sequence.time[i], "activePower": float(sequence.power[i]),
                    "hasAnomalyAssessment": {
                        "id": f"{measurement_id}/assessment", "score": source, "confidence": confidence,
                        "threshold": threshold, "anomalous": bool(detected), "rounds": int(detail["rounds"][i]),
                        "initialConflicts": int(detail["initial_conflicts"][i]),
                        "finalConflicts": int(detail["final_conflicts"][i]),
                        "initialDirectContradictions": int(detail["initial_direct_conflicts"][i]),
                        "finalDirectContradictions": int(detail["final_direct_conflicts"][i]),
                        "agentTrust": dict(zip(NAMES, map(float, detail["trust"][i]))),
                        "decisionTrace": {"file": detail["file"], "row": i},
                    },
                })


def _anomaly_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, labels.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def plot_detection(path: Path, sequence: Sequence, frame: pd.DataFrame, beliefs: np.ndarray,
                   threshold: float) -> dict:
    """Figure 1 on the first 1,000 samples, at the best grid threshold used for every reported metric."""
    limit = min(1000, len(sequence.power))
    x = np.arange(limit)
    labels = sequence.labels[:limit]
    scores = frame["score"].to_numpy()[:limit]
    predicted = scores > threshold
    confusion = counts(labels, predicted)
    metrics = classification(confusion)

    figure = plt.figure(figsize=(12, 11), layout="constrained")
    grid = figure.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.05], width_ratios=[1, 1])
    axes = [figure.add_subplot(grid[row, :]) for row in range(3)]
    matrix_axis, text_axis = figure.add_subplot(grid[3, 0]), figure.add_subplot(grid[3, 1])
    spans = _anomaly_spans(labels)

    axes[0].plot(x, sequence.power[:limit], c="#2455a5", lw=1.0, label="Power")
    anomaly = labels == 1
    axes[0].scatter(x[anomaly], sequence.power[:limit][anomaly], s=13, c="#bb3437", label="Labeled anomaly")
    axes[0].set(title=f"(a) {sequence.name} · {sequence.fault.replace('_', ' ')}", ylabel="Power (W)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    for name, agent in zip(NAMES, beliefs[:limit].T):
        axes[1].plot(x, agent, lw=0.9, alpha=0.8, label=name)
    axes[1].set(title="(b) Individual agents", ylabel="Anomaly score")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    axes[2].plot(x, scores, c="#207856", lw=1.25, label="NSNG-Energy")
    axes[2].axhline(threshold, c="#b66c14", linestyle="--", label=f"Threshold {threshold:.2f}")
    axes[2].fill_between(x, threshold, scores, where=predicted, color="#65a991", alpha=0.35,
                         label="Detected")
    axes[2].set(title="(c) Consensus detection", ylabel="Anomaly score", xlabel="Observation index")
    for ax in axes[1:]:
        for first, last in spans:
            ax.axvspan(first - 0.5, last - 0.5, color="#bb3437", alpha=0.12, lw=0)
    axes[2].axvspan(np.nan, np.nan, color="#bb3437", alpha=0.12, lw=0, label="Labeled anomaly")
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    for ax in axes:
        ax.set_xlim(x[0], max(x[-1], 1))
        ax.grid(alpha=0.2)
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)

    matrix = confusion.reshape(2, 2)
    image = matrix_axis.imshow(matrix, cmap="Blues", vmin=0)
    for i, j in np.ndindex(matrix.shape):
        color = "white" if matrix[i, j] > matrix.max() / 2 else "#202830"
        matrix_axis.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color=color, fontsize=12)
    matrix_axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"], xlabel="Predicted", ylabel="Labeled",
                    title="(d) Confusion matrix")
    figure.colorbar(image, ax=matrix_axis, shrink=0.8, label="Observations")
    text_axis.axis("off")
    rows = [("Samples", f"{limit:,}"), ("Threshold", f"{threshold:.2f}"),
            ("Accuracy", f"{metrics['accuracy']:.3f}"), ("Precision", f"{metrics['precision']:.3f}"),
            ("Recall", f"{metrics['recall']:.3f}"), ("F1", f"{metrics['f1']:.3f}"),
            ("False positive rate", f"{metrics['false_positive_rate']:.3f}")]
    table = text_axis.table(cellText=[[value] for _, value in rows], rowLabels=[name for name, _ in rows],
                            colWidths=[0.35], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    text_axis.set_title("(e) Metrics on these samples")
    figure.savefig(path, dpi=155)
    plt.close(figure)
    return {
        "sequence": sequence.name, "samples": limit, "threshold": threshold, **metrics,
        **dict(zip(("tn", "fp", "fn", "tp"), map(int, confusion))),
    }
