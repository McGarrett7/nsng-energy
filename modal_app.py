"""Run the Kaggle notebook pipeline (prepare -> run) on Modal.

    modal run modal_app.py                 # full run (5 folds, seeds 42 + 43..46)
    modal run modal_app.py --quick         # pipeline check: 3 folds, 1 epoch
    modal run modal_app.py --residual      # also run --detector-residual 0.25
    modal volume get nsng-results public-dryer ./results_modal
"""
import subprocess
import sys
import time
from pathlib import Path

import modal

GPU = "L4"            # "T4", "T4:2", "A10G" ... or None for CPU only
CPUS = 8
REMOTE = "/root/nsng-energy"
RESULTS = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26,<3", "scipy>=1.12,<2", "pandas>=2.2,<4",
        "scikit-learn>=1.5,<2", "torch>=2.3,<3", "matplotlib>=3.8,<4",
    )
    .add_local_dir("nsng_energy", f"{REMOTE}/nsng_energy", ignore=["**/__pycache__"])
    .add_local_dir("tests", f"{REMOTE}/tests", ignore=["**/__pycache__"])
    .add_local_dir("data", f"{REMOTE}/data")
)

app = modal.App("nsng-energy", image=image)
volume = modal.Volume.from_name("nsng-results", create_if_missing=True)


def _run(*args: str) -> None:
    command = [sys.executable, "-u", "-m", *args]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=REMOTE, check=True)


@app.function(gpu=GPU, cpu=CPUS, memory=16384, timeout=24 * 3600, volumes={RESULTS: volume})
def pipeline(quick: bool = False, residual: bool = False, folds: int = 5,
             seed: int = 42, extra_seeds: str = "43,44,45,46", tests: bool = False) -> None:
    import torch

    device = "cpu"
    if torch.cuda.is_available():
        torch.ones(4, device="cuda").sum().item()
        device = "cuda"
        print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    folds_used = 3 if quick else folds
    jobs = folds_used if device == "cuda" else min(folds_used, CPUS)
    threads = max(1, CPUS // jobs)
    print(f"device={device} jobs={jobs} threads={threads}", flush=True)

    if tests:
        _run("unittest", "discover", "-s", "tests", "-v")
    if not Path(REMOTE, "data/public-dryer/manifest.csv").exists():
        _run("nsng_energy", "prepare")

    common = ["--device", device, "--jobs", str(jobs), "--threads", str(threads), "--seed", str(seed)]
    if quick:
        common += ["--folds", "3", "--lstm-epochs", "1", "--graph-epochs", "1", "--extra-seeds", "43"]
    else:
        common += ["--folds", str(folds), "--extra-seeds", *extra_seeds.split(",")]

    suffix = "-quick" if quick else ""
    started = time.time()
    _run("nsng_energy", "run", *common, "--output", f"{RESULTS}/public-dryer{suffix}")
    print(f"Main run: {(time.time() - started) / 60:.1f} min", flush=True)
    volume.commit()

    if residual:
        started = time.time()
        _run("nsng_energy", "run", *common, "--detector-residual", "0.25",
             "--output", f"{RESULTS}/public-dryer-residual{suffix}")
        print(f"Residual run: {(time.time() - started) / 60:.1f} min", flush=True)
        volume.commit()


@app.local_entrypoint()
def main(quick: bool = False, residual: bool = False, folds: int = 5,
         seed: int = 42, extra_seeds: str = "43,44,45,46", tests: bool = False) -> None:
    pipeline.remote(quick=quick, residual=residual, folds=folds,
                    seed=seed, extra_seeds=extra_seeds, tests=tests)
    print("Done. Download with: modal volume get nsng-results / ./results_modal")
