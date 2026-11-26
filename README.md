# NSNG-Energy

Anomaly detection on dryer power consumption using multiple detector agents combined through a trust-aware neuro-symbolic negotiation graph.

## Tech stack

| Area | Technology | Used for |
| --- | --- | --- |
| Language, packaging | Python ≥ 3.11, setuptools (`pyproject.toml`) | Package install, `nsng-energy` entry point |
| Deep learning | PyTorch (CPU/CUDA) | LSTM autoencoder, multi-head Graph Attention Network, truncated BPTT training (chunks in random order from carried trust states, gradient clipping 1.0) |
| Machine learning | scikit-learn | Isolation Forest, One-Class SVM, LOF, `StandardScaler`, `GroupKFold`, ROC-AUC |
| Numerics | NumPy, SciPy | Sliding-window features, sigmoid (`expit`), EMA (`lfilter`) |
| Data handling | pandas | Reading sequence CSVs and manifests, result tables |
| Visualization | Matplotlib (`Agg` backend) | Rendering `figure_1.png` |
| Parallelism | `concurrent.futures.ProcessPoolExecutor` | Running folds concurrently (`--jobs`) |
| Tracing | JSON, gzipped JSONL, compressed NPZ | `run.json`, consensus knowledge graph, per-round traces |
| Testing | `unittest` | Tests for the negotiation module |

**How it works:** 7 detector agents produce a score and a confidence for each observation. The agents are connected as a graph, and a GAT runs several rounds so they can exchange beliefs. After each round, symbolic rules R1–R4 adjust confidence and trust, and trust is carried over to the next observation. The final consensus score is compared against a decision threshold.

**Evaluation protocol:** grouped 5-fold cross-validation; the next fold serves as validation for early stopping of the GAT. As in Section V.D of the paper, every threshold-dependent metric uses the threshold with the best F1 among 0.30, 0.35, …, 0.70 on the test predictions, chosen separately for each method, ablation and sensitivity setting.

## Installation and usage

Requires Python 3.11 or later.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m nsng_energy prepare             # download and prepare data into data/public-dryer/
.venv\Scripts\python.exe -m nsng_energy run --device auto   # train, evaluate, write results to results/public-dryer/
```

Run tests:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

### Common options

| Command / flag | Meaning |
| --- | --- |
| `prepare --archive <zip>` | Use an already downloaded data ZIP instead of downloading |
| `prepare --output <dir>` | Data output directory (default `data/public-dryer`) |
| `run --device auto\|cpu\|cuda` | `auto` picks CUDA if PyTorch can access a GPU |
| `run --jobs N` | Run N folds in parallel (measured inference time is affected by CPU contention) |
| `run --folds N` | Number of folds (default 5) |
| `run --manifest <csv> --output <dir>` | Use your own data |
| `run --seed`, `--window`, `--lstm-epochs`, `--graph-epochs`, `--chunk-size` | Seed (42), window (24), LSTM epochs (8), maximum GAT epochs (8), truncated BPTT length (32) |
| `run --patience N` | Early stopping: keep the GAT epoch with the best validation ROC-AUC, stop after N epochs without gain (3) |
| `run --restarts N` | Re-initialise the GAT (up to N fits, default 3) while its best validation ROC-AUC is below 0.5; the best fit on validation is kept |
| `run --extra-seeds 43 44 …` | Repeat Table III with these seeds; writes mean, std and 95% CI |
| `run --nominal-voltage <V>` `--voltage-rule soft\|hard` | Enable voltage rule R1 when the data has a `voltage` column |
| `run --detector-residual w` | Add `w·logit` of the initial detector score to the prediction layer |

### Using your own data

The manifest CSV has columns `sequence,path,group`, with optional `fault,normal_path,figure`. Paths may be relative to the manifest's directory.

- Each sequence CSV needs `ctime,activePower,label` (0/1 labels); add a `voltage` column to use R1.
- `group`: sequences from the same measurement or with overlapping collection periods must share a group. At least 3 groups are required, and both classes must appear in the train/validation split of every fold.
- `normal_path`: extra normal-operation data for training the detectors.
- `figure=1`: the sequence plotted in `figure_1.png`.

<!-- ## Code structure

| Module | Role |
| --- | --- |
| [`__main__.py`](nsng_energy/__main__.py) | CLI with `prepare` and `run` commands |
| [`data.py`](nsng_energy/data.py) | Downloads the public dataset (pinned commit), selects 15 sequences, writes the manifest; loads manifests |
| [`agents.py`](nsng_energy/agents.py) | 7 detector agents (Isolation Forest ×2, One-Class SVM ×2, LOF, LSTM autoencoder, Adaptive Z-Score); maps raw scores to score and confidence |
| [`negotiation.py`](nsng_energy/negotiation.py) | Negotiation graph: edge construction, multi-round GAT, symbolic rules R1–R4, trust updates over time, consensus |
| [`experiment.py`](nsng_energy/experiment.py) | Main pipeline: group-based folds, agent and negotiator training, baselines, ablation, sensitivity, inference timing |
| [`evaluation.py`](nsng_energy/evaluation.py) | Best grid threshold, metrics, paired permutation test |
| [`artifacts.py`](nsng_energy/artifacts.py) | Exports the dataset table, Figure 1, consensus knowledge graph, and traces |
| [`paper.py`](nsng_energy/paper.py) | Reference numbers used in the comparison table |
| [`tests/`](tests/test_negotiation.py) | Unit tests for consensus, edge matrix, and trust | -->

## Outputs

Written to `results/public-dryer/` by default:

| File | Contents |
| --- | --- |
| `table_i_dataset.csv`, `splits.csv` | Sequence list and fold splits |
| `table_iii_methods.csv`, `per_sequence.csv` | Metrics for NSNG-Energy, baselines, and each agent |
| `table_iv_rules.csv`, `table_v_components.csv` | Rule and component ablations |
| `sensitivity.csv` | Hyperparameter sweeps (rounds, momentum, symbolic strength) |
| `best_test_grid.csv`, `threshold_grid_test.csv` | Chosen threshold and metrics for every experiment; metrics at each grid threshold |
| `figure_1.png`, `figure_1_metrics.json` | Plot of power, labels, agent scores, consensus |
| `paired_significance.json` | Paired permutation test, NSNG-Energy vs Smart Ensemble |
| `paper_comparison.csv`, `table_vi_qualitative.csv` | Comparison against reference numbers |
| `run.json` | Parameters, library versions, data source, inference time |
| `training_history.csv` | Validation ROC-AUC per GAT restart and epoch, and the selected one |
| `seeds_table_iii.csv`, `seeds_summary.csv`, `seeds_folds.csv` | Table III per seed; mean, std, 95% CI; NSNG per-fold F1, AUC, score range |
| `predictions.csv.gz` | Per-observation NSNG-Energy score and prediction |
| `consensus_kg.jsonl.gz` | Per observation: score, confidence, prediction, rounds, conflict counts, final trust |


## Data

[Power Consumption Anomaly Dataset](https://github.com/AlexisPapaioannou/Power-Consumption-Anomaly-Dataset): normal power is measured; fault labels are simulated. License information is kept in the data directory after `prepare`.
