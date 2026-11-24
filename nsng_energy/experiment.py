"""Run the experiment with disjoint training, threshold-selection, and test groups."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
from torch import nn

from . import paper
from .agents import Agents, NAMES
from .artifacts import export_consensus, export_trace, inventory, plot_detection, write_json
from .data import load_manifest
from .evaluation import choose_threshold, paired_group_test, summarize, threshold_grid
from .negotiation import GraphInputs, Negotiator, Settings, graph_inputs

FULL = "NSNG Energy"
REFERENCE_METHODS = ("Smart Ensemble", "Confidence Aggregation")
ALL_AGENTS = list(range(len(NAMES)))
NO_LSTM = [index for index in ALL_AGENTS if NAMES[index] != "LSTM Autoencoder"]
TRACE_KEYS = ("states", "outgoing", "conflict_edges", "rule_hits")


def _score(scores, confidence):
    """Section V.B: confidence-weighted average of the agent scores."""
    return (scores * confidence).sum(axis=1) / confidence.sum(axis=1)


def _batch(inputs, members, indices):
    batch = GraphInputs.stack([inputs[i] for i in members])
    return batch if indices == ALL_AGENTS else batch.select_agents(indices)


@torch.no_grad()
def _chunk_states(model, batch, chunk_size):
    """Carried state (trust, R2 history) at the start of every chunk under the current parameters."""
    training = model.training
    model.eval()
    states, state = {}, None
    for start in range(0, batch.score.shape[1], chunk_size):
        states[start] = state
        state = model(batch.take(slice(start, start + chunk_size)), state=state)[3]
    model.train(training)
    return states


def _fit_negotiator(settings, indices, training, sequences, inputs, args, seed, shuffle=True):
    """Truncated backpropagation through time over the training sequences in parallel.

    With ``shuffle`` the chunks of an epoch are visited in random order. Each chunk starts
    from the trust state reached at its first time step, recomputed once per epoch, so
    Eq. (9) still carries trust through time while gradient steps are decorrelated.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    batch = _batch(inputs, training, indices)
    device = batch.score.device
    model = Negotiator(settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    target = torch.zeros(batch.valid.shape, device=device)
    for row, i in enumerate(training):
        target[row, :len(sequences[i].labels)] = torch.tensor(np.array(sequences[i].labels), dtype=torch.float32, device=device)
    samples, positives = batch.valid.sum().item(), target.sum().item()
    if not 0 < positives < samples:
        raise ValueError("Each training split requires both classes for fitting graph attention.")
    positive_weight = (samples - positives) / positives
    model.train()
    starts = np.arange(0, target.shape[1], args.chunk_size)
    for _ in range(args.graph_epochs):
        state = None
        carried = _chunk_states(model, batch, args.chunk_size) if shuffle else None
        for start in (rng.permutation(starts) if shuffle else starts):
            window = slice(start, start + args.chunk_size)
            chunk = batch.take(window)
            if shuffle:
                state = carried[start]
            optimizer.zero_grad()
            scores, confidence, _, state = model(chunk, state=state)
            mask, y = chunk.valid.float(), target[:, window]
            count = mask.sum().clamp_min(1)
            weights = mask * (1 + y * (positive_weight - 1))
            loss = nn.functional.binary_cross_entropy(scores, y, weights, reduction="sum") / count
            correctness = 1 - (scores.detach() - y).abs()
            calibration = nn.functional.binary_cross_entropy(confidence, correctness, reduction="none")
            loss = loss + 0.1 * (calibration * mask).sum() / count
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = state.detach()
    return model.eval()


def variants(settings):
    for rule in settings.rules:
        yield "rule", f"w/o {rule}", replace(settings, rules=tuple(r for r in settings.rules if r != rule)), ALL_AGENTS
    yield "rule", "w/o All symbolic rules", replace(settings, rules=()), ALL_AGENTS
    yield "component", "w/o LSTM Agent", settings, NO_LSTM
    yield "component", "w/o Trust Propagation", replace(settings, trust=False), ALL_AGENTS
    yield "component", "w/o Symbolic Rules", replace(settings, rules=()), ALL_AGENTS
    yield "component", "w/o Multi Head Attention", replace(settings, heads=1), ALL_AGENTS
    for axis, values in (
        ("rounds", (1, 3, 5, 8, 10, 12, 15, 20)),
        ("momentum", (0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)),
        ("symbolic_strength", (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1)),
    ):
        for value in values:
            config = replace(settings, **{axis: value})
            if axis == "symbolic_strength":
                config = replace(config, voltage_rule="soft")
            yield axis, str(value), config, ALL_AGENTS


def _frame(sequence, fold, scores, confidence, threshold):
    return pd.DataFrame({
        "sequence": sequence.name, "group": sequence.group, "fault": sequence.fault,
        "fold": fold, "sample": np.arange(len(scores)), "ctime": sequence.time,
        "label": sequence.labels, "score": scores, "confidence": confidence,
        "prediction": (scores > threshold).astype(int), "threshold": threshold,
    })


@torch.inference_mode()
def _predict(model, batch, config, members, sequences, trace=False):
    """Scan whole sequences and split the padded batch into one record per sequence."""
    scores, confidence, details, _ = model(batch, config, trace=trace)
    results = []
    for row, i in enumerate(members):
        n = len(sequences[i].labels)
        detail = {key: value[row, :n].cpu().numpy() for key, value in details.items()}
        for key in TRACE_KEYS:
            if key in detail:  # [time, round, ...] -> [round, time, ...]
                detail[key] = np.swapaxes(detail[key], 0, 1)
        results.append((scores[row, :n].cpu().numpy(), confidence[row, :n].cpu().numpy(), detail))
    return results


def _evaluate_graph(model, config, indices, validation, test, inputs, sequences, fold, save_trace, timing):
    held_out = [*validation, *test]
    if save_trace:
        validation_results = _predict(model, _batch(inputs, validation, indices), config, validation, sequences)
        started = perf_counter()
        test_results = _predict(model, _batch(inputs, test, indices), config, test, sequences, trace=True)
        timing.append((perf_counter() - started, sum(len(sequences[i].labels) for i in test),
                       max(len(sequences[i].labels) for i in test)))
    else:
        results = _predict(model, _batch(inputs, held_out, indices), config, held_out, sequences)
        validation_results, test_results = results[:len(validation)], results[len(validation):]
    threshold = choose_threshold(
        np.concatenate([sequences[i].labels for i in validation]),
        np.concatenate([score for score, _, _ in validation_results]),
    )
    frames = [_frame(sequences[i], fold, score, confidence, threshold)
              for i, (score, confidence, _) in zip(test, test_results)]
    return frames, {i: detail for i, (_, _, detail) in zip(test, test_results)}


def _run_fold(fold, test, validation, sequences, args, settings, figure_name, device):
    torch.set_num_threads(args.threads)
    # Spawned workers do not inherit these flags from the parent process.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda" and device.index is None and torch.cuda.device_count() > 1:
        # Spread folds over all visible GPUs: fold k runs on cuda:(k - 1) mod count.
        device = torch.device("cuda", (fold - 1) % torch.cuda.device_count())
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training = np.setdiff1d(np.arange(len(sequences)), np.union1d(test, validation)).tolist()
    test, validation = list(map(int, test)), list(map(int, validation))
    groups = [sequence.group for sequence in sequences]
    split_rows = [{"fold": fold, "role": role, "sequence": sequences[i].name, "group": groups[i]}
                  for role, members in (("train", training), ("validation", validation), ("test", test))
                  for i in members]
    print(f"Fold {fold}/{args.folds} | {device} | training {len(training)} sequences | "
          f"validation {len(validation)} | testing {len(test)}", flush=True)
    seed = args.seed + fold
    torch.manual_seed(seed)
    detectors = Agents(seed, device, args.window, args.lstm_epochs)
    detectors.fit(sequences[i] for i in training)
    beliefs, inputs, agent_timing, belief_for_figure = {}, {}, [], None
    for i, sequence in enumerate(sequences):
        started = perf_counter()
        beliefs[i] = detectors.score(sequence.power)
        inputs[i] = graph_inputs(beliefs[i], sequence.voltage, args.nominal_voltage, device)
        if i in test:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            agent_timing.append((perf_counter() - started, len(sequence.power)))
            if sequence.name == figure_name:
                belief_for_figure = beliefs[i].scores

    experiments = defaultdict(list)
    negotiation_timing = []
    full = _fit_negotiator(settings, ALL_AGENTS, training, sequences, inputs, args, seed)
    predictions, details = _evaluate_graph(
        full, settings, ALL_AGENTS, validation, test, inputs, sequences, fold, True, negotiation_timing,
    )
    experiments["method", FULL].extend(predictions)

    validation_labels = np.concatenate([sequences[i].labels for i in validation])
    for name in (*REFERENCE_METHODS, *NAMES):
        agent = NAMES.index(name) if name in NAMES else None

        def get_scores(i):
            belief = beliefs[i]
            if agent is None:
                return _score(belief.scores, belief.confidence), belief.confidence.mean(axis=1)
            return belief.scores[:, agent], belief.confidence[:, agent]

        threshold = choose_threshold(validation_labels, np.concatenate([get_scores(i)[0] for i in validation]))
        experiments["method", name].extend(_frame(sequences[i], fold, *get_scores(i), threshold) for i in test)

    for study, name, config, indices in variants(settings):
        if config == settings and indices == ALL_AGENTS:
            experiments[study, name].extend(predictions)
            continue
        if study == "component":
            print(f"  fold {fold} | {name}", flush=True)
            model = _fit_negotiator(config, indices, training, sequences, inputs, args, seed)
        else:
            model = full
        result, _ = _evaluate_graph(model, config, indices, validation, test, inputs, sequences, fold, False, [])
        experiments[study, name].extend(result)
    print(f"Fold {fold} finished", flush=True)
    return {
        "experiments": dict(experiments), "splits": split_rows, "details": details,
        "negotiation_timing": negotiation_timing, "agent_timing": agent_timing,
        "belief_for_figure": belief_for_figure,
    }


def run(args):
    sequences, figure_name = load_manifest(args.manifest)
    groups = np.array([sequence.group for sequence in sequences])
    if not 3 <= args.folds <= len(set(groups)):
        raise ValueError(f"Fold count must be between 3 and the number of acquisition groups ({len(set(groups))}).")
    settings = Settings(nominal_voltage=args.nominal_voltage, voltage_rule=args.voltage_rule,
                        detector_residual=args.detector_residual)
    data_has_voltage = any(s.voltage is not None and np.isfinite(s.voltage).any() for s in sequences)
    if data_has_voltage and settings.nominal_voltage is None:
        raise ValueError("Supply --nominal-voltage to use the voltage measurements in the manifest.")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access a GPU.")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    table = inventory(sequences, output)
    test_folds = [test for _, test in GroupKFold(n_splits=args.folds).split(sequences, groups=groups)]
    jobs = [(fold, test, test_folds[fold % len(test_folds)], sequences, args, settings, figure_name, device)
            for fold, test in enumerate(test_folds, 1)]
    if args.jobs > 1:
        # CUDA cannot be used in forked workers (the Linux default), so always spawn.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs)), mp_context=context) as pool:
            folds = list(pool.map(_run_fold, *zip(*jobs)))
    else:
        folds = [_run_fold(*job) for job in jobs]

    frames_by_experiment = defaultdict(list)
    trace_files, split_rows, negotiation_timing, agent_timing = {}, [], [], []
    belief_for_figure = None
    for result in folds:
        for key, frames in result["experiments"].items():
            frames_by_experiment[key].extend(frames)
        for i, detail in result["details"].items():
            trace_files[sequences[i].name] = export_trace(output, i + 1, detail, settings)
        split_rows.extend(result["splits"])
        negotiation_timing.extend(result["negotiation_timing"])
        agent_timing.extend(result["agent_timing"])
        if result["belief_for_figure"] is not None:
            belief_for_figure = result["belief_for_figure"]

    pd.DataFrame(split_rows).to_csv(output / "splits.csv", index=False)
    all_summary, sequence_tables, sweep_rows = [], [], []
    for (study, name), frames in frames_by_experiment.items():
        metrics, per_sequence = summarize(frames)
        all_summary.append({"study": study, "name": name, **metrics})
        sequence_tables.append(per_sequence.assign(study=study, name=name))
        sweep_rows.extend({"study": study, "name": name, **row} for row in threshold_grid(frames))
    pd.concat(sequence_tables, ignore_index=True).to_csv(output / "per_sequence.csv", index=False)
    summary = pd.DataFrame(all_summary)
    grid = pd.DataFrame(sweep_rows)
    grid.to_csv(output / "threshold_grid_test.csv", index=False)
    # Paper protocol (Section V.D): best threshold of the [0.3, 0.7] grid on the test predictions.
    grid["distance_to_figure_threshold"] = (grid["threshold"].mul(100).round() - 45).abs()
    oracle = grid.sort_values(["f1", "distance_to_figure_threshold", "threshold"], ascending=[False, True, False])
    oracle = oracle.drop_duplicates(["study", "name"]).drop(columns="distance_to_figure_threshold")
    oracle.to_csv(output / "best_test_grid.csv", index=False)
    full_row = summary.query("study == 'method' and name == @FULL").iloc[0]
    summary.query("study == 'method'").drop(columns="study").to_csv(output / "table_iii_methods.csv", index=False)
    for study, filename in (("rule", "table_iv_rules.csv"), ("component", "table_v_components.csv")):
        ablation = pd.concat([full_row.to_frame().T, summary[summary["study"] == study]]).drop(columns="study")
        ablation["delta_f1_percent"] = 100 * (ablation["f1"] - full_row["f1"]) / max(full_row["f1"], 1e-12)
        ablation.to_csv(output / filename, index=False)
    sensitivity = summary[summary["study"].isin(("rounds", "momentum", "symbolic_strength"))]
    sensitivity.rename(columns={"study": "parameter", "name": "value"}).to_csv(output / "sensitivity.csv", index=False)
    _qualitative_comparison(output / "table_vi_qualitative.csv")
    full_frames = frames_by_experiment["method", FULL]
    reference = frames_by_experiment["method", REFERENCE_METHODS[0]]
    write_json(output / "paired_significance.json", paired_group_test(full_frames, reference, args.seed))
    full_scores = pd.concat(full_frames).sort_values(["sequence", "sample"])
    full_scores.to_csv(output / "predictions.csv.gz", index=False)
    export_consensus(output / "consensus_kg.jsonl.gz", sequences, full_frames, trace_files)
    figure = next(sequence for sequence in sequences if sequence.name == figure_name)
    figure_metrics = plot_detection(
        output / "figure_1.png", figure, full_scores[full_scores["sequence"] == figure_name], belief_for_figure,
    )
    write_json(output / "figure_1_metrics.json", figure_metrics)
    latency = _latency(negotiation_timing, agent_timing, args.jobs)
    _paper_comparison(output / "paper_comparison.csv", summary, oracle, table, figure_metrics, latency)
    _write_run(output, args, settings, device, table, trace_files, latency)


def _qualitative_comparison(path):
    rows = [
        ("Specific to anomaly detection", "Yes", "No", "No"),
        ("Convergence guarantee claimed in the paper", "Yes", "No", "No"),
        ("Trust propagation", "Yes", "No", "No"),
        ("Interpretable decision trace", "Yes", "Partial", "Partial"),
        ("Latency quoted in the paper (ms/sample)", "0.5", "500–2000", "500–2000"),
        ("F1 quoted in the paper on Power Anomaly", "0.238", "N/A", "N/A"),
    ]
    table = pd.DataFrame(rows, columns=["feature", "NSNG Energy", "AutoGen", "MetaGPT"])
    table.assign(source="Paper Table VI (qualitative; no matched benchmark)").to_csv(path, index=False)


def _latency(timing, agent_timing, jobs):
    def per_sample(measurements):
        return 1000 * sum(item[0] for item in measurements) / sum(item[1] for item in measurements)

    negotiation = per_sample(timing)
    return {
        "scope": "test sequences of each fold negotiated together as one padded batch, "
                 "excluding fitting and artifact export",
        "parallel_fold_workers": jobs,
        "negotiation_ms_per_sample": negotiation,
        "negotiation_ms_per_time_step": 1000 * sum(t for t, _, _ in timing) / sum(n for _, _, n in timing),
        "agent_scoring_and_graph_input_ms_per_sample": per_sample(agent_timing),
        "end_to_end_ms_per_sample": negotiation + per_sample(agent_timing),
    }


def _paper_comparison(path, summary, oracle, table, figure_metrics, latency):
    """Place each published number next to the validation-threshold and best-test-grid results."""
    measured = summary.set_index(["study", "name"])
    best = oracle.set_index(["study", "name"])
    rows = []

    def add(section, study, name, values, label=None):
        for metric, published in values.items():
            row = {"section": section, "configuration": label or name, "metric": metric, "paper": published}
            if (study, name) in measured.index:
                row["reproduced_pooled"] = measured.loc[(study, name), metric]
                macro = f"macro_{metric}"
                row["reproduced_macro"] = measured.loc[(study, name), macro] if macro in measured else np.nan
                row["reproduced_best_test_grid"] = best.loc[(study, name), metric]
            rows.append(row)

    rows += [
        {"section": "Table I", "configuration": "dataset", "metric": "sequences",
         "paper": paper.TABLE_I["sequences"], "reproduced_pooled": len(table)},
        {"section": "Table I", "configuration": "dataset", "metric": "average_length",
         "paper": paper.TABLE_I["average_length"], "reproduced_pooled": table["samples"].mean()},
        {"section": "Table I", "configuration": "dataset", "metric": "anomaly_ratio_min",
         "paper": paper.TABLE_I["anomaly_ratio_min"], "reproduced_pooled": table["anomaly_ratio"].min()},
        {"section": "Table I", "configuration": "dataset", "metric": "anomaly_ratio_max",
         "paper": paper.TABLE_I["anomaly_ratio_max"], "reproduced_pooled": table["anomaly_ratio"].max()},
        {"section": "Table I", "configuration": "dataset", "metric": "total_samples",
         "paper": paper.TABLE_I["total_samples"], "reproduced_pooled": table["samples"].sum()},
    ]
    for name, values in paper.TABLE_III.items():
        add("Table III", "method", name, values)
    for name, values in paper.TABLE_IV.items():
        add("Table IV", "method" if name == FULL else "rule", name, values)
    for name, values in paper.TABLE_V.items():
        add("Table V", "method" if name == FULL else "component", name, values)
    for (axis, value), values in paper.SENSITIVITY.items():
        add("Section V.I", axis, value, values, f"{axis}={value}")
    rows += [{"section": "Figure 1", "configuration": "first 1,000 samples at 0.45", "metric": metric,
              "paper": published, "reproduced_pooled": figure_metrics[metric]}
             for metric, published in paper.FIGURE_1.items() if metric not in ("threshold", "samples")]
    rows.append({"section": "Table VI", "configuration": FULL, "metric": "ms_per_sample",
                 "paper": paper.LATENCY_MS_PER_SAMPLE, "reproduced_pooled": latency["end_to_end_ms_per_sample"]})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_run(output, args, settings, device, table, traces, latency):
    provenance = args.manifest.with_name("source.json")
    run_parameters = {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items() if key != "command"}
    write_json(output / "run.json", {
        "protocol": "grouped out-of-fold test with disjoint validation for threshold selection",
        "parameters": run_parameters, "negotiation": asdict(settings),
        "device": ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                   if device.type == "cuda" else "CPU"),
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "torch", "scikit-learn", "matplotlib")},
        "dataset_source": pd.read_json(provenance, typ="series").to_dict() if provenance.exists() else None,
        "dataset": {"sequences": len(table), "samples": int(table["samples"].sum()),
                    "groups": int(table["group"].nunique()), "anomalies": int(table["anomalies"].sum()),
                    "anomaly_ratio_min": float(table["anomaly_ratio"].min()), "anomaly_ratio_max": float(table["anomaly_ratio"].max())},
        "full_model_rule_applications": dict(zip(
            settings.rules, np.sum([item["rule_counts"] for item in traces.values()], axis=0).tolist(),
        )),
        "full_model_conflicts": {
            key: int(sum(item[key].sum() for item in traces.values()))
            for key in ("initial_conflicts", "final_conflicts", "initial_direct_conflicts", "final_direct_conflicts")
        },
        "latency": latency,
    })
