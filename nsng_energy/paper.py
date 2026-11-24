"""Numbers published in docs/CSA_Tung.pdf, Section V, used only for side-by-side comparison."""

TABLE_I = {
    "sequences": 15, "average_length": 1492, "anomaly_ratio_min": 0.020,
    "anomaly_ratio_max": 0.071, "total_samples": 22380,
}

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")

TABLE_III = {
    "NSNG Energy": {"accuracy": 0.8113, "f1": 0.2379, "precision": 0.1679, "recall": 0.5625, "roc_auc": 0.7837},
    "Smart Ensemble": {"accuracy": 0.8093, "f1": 0.2254, "precision": 0.1540, "recall": 0.5664, "roc_auc": 0.7856},
    "Confidence Aggregation": {"accuracy": 0.8093, "f1": 0.2254, "precision": 0.1540, "recall": 0.5664,
                               "roc_auc": 0.7856},
}

# Keys follow the experiment names; values are Table IV of the paper.
TABLE_IV = {
    "NSNG Energy": {"precision": 0.1679, "recall": 0.5625, "f1": 0.2379},
    "w/o R1": {"precision": 0.1612, "recall": 0.5512, "f1": 0.2263},
    "w/o R2": {"precision": 0.1589, "recall": 0.5489, "f1": 0.2234},
    "w/o R3": {"precision": 0.1634, "recall": 0.5587, "f1": 0.2331},
    "w/o R4": {"precision": 0.1601, "recall": 0.5598, "f1": 0.2289},
    "w/o All symbolic rules": {"precision": 0.1540, "recall": 0.5664, "f1": 0.2254},
}

TABLE_V = {
    "NSNG Energy": {"f1": 0.2379, "roc_auc": 0.7837},
    "w/o LSTM Agent": {"f1": 0.2156, "roc_auc": 0.7712},
    "w/o Trust Propagation": {"f1": 0.2254, "roc_auc": 0.7718},
    "w/o Symbolic Rules": {"f1": 0.2289, "roc_auc": 0.7789},
    "w/o Multi Head Attention": {"f1": 0.2218, "roc_auc": 0.7756},
}

# Section V.I: only these two points are numeric in the text.
SENSITIVITY = {("rounds", "5"): {"f1": 0.182}, ("rounds", "12"): {"f1": 0.238}}

FIGURE_1 = {"threshold": 0.45, "samples": 1000, "tn": 775, "fp": 158, "fn": 67, "tp": 0,
            "accuracy": 0.811, "f1": 0.238}

LATENCY_MS_PER_SAMPLE = 0.5

# Section V.F: per-sequence F1 quoted by sequence number.
PER_SEQUENCE_F1 = {9: 0.738, 10: 0.533}
LOW_F1_SEQUENCES = (1, 5, 11, 15)
