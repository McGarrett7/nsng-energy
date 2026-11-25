"""Command-line entry point for the project."""

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NSNG-Energy dryer experiments.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="Prepare a candidate benchmark from the pinned public archive.")
    prepare.add_argument("--output", type=Path, default=Path("data/public-dryer"))
    prepare.add_argument("--archive", type=Path, help="Read an already downloaded upstream ZIP.")

    run = subcommands.add_parser("run", help="Fit and evaluate the benchmark from a sequence manifest.")
    run.add_argument("--manifest", type=Path, default=Path("data/public-dryer/manifest.csv"))
    run.add_argument("--output", type=Path, default=Path("results/public-dryer"))
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--folds", type=positive_int, default=5)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--threads", type=positive_int, default=2)
    run.add_argument("--window", type=positive_int, default=24)
    run.add_argument("--lstm-epochs", type=positive_int, default=8)
    run.add_argument("--graph-epochs", type=positive_int, default=8,
                     help="Maximum epochs for the graph; the epoch with the best validation ROC-AUC is kept.")
    run.add_argument("--patience", type=positive_int, default=3,
                     help="Stop fitting the graph after this many epochs without validation ROC-AUC gain.")
    run.add_argument("--extra-seeds", type=int, nargs="*", default=[],
                     help="Repeat Table III (main model, baselines, agents) with these seeds for mean/std/95%% CI.")
    run.add_argument("--chunk-size", type=positive_int, default=32,
                     help="Time steps per truncated backpropagation window when fitting the graph.")
    run.add_argument("--jobs", type=positive_int, default=1, help="Folds evaluated in parallel processes.")
    run.add_argument("--detector-residual", type=float, default=0.0,
                     help="Weight of the detector logit added to Eq. 13 (0 follows the paper).")
    run.add_argument("--nominal-voltage", type=float, help="Nominal voltage (V) for Rule R1.")
    run.add_argument("--voltage-rule", choices=("soft", "hard"), default="soft",
                     help="Use Eq. 16 (soft) or the hard floor in Table II.")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare_public

        print(prepare_public(args.output, args.archive))
    else:
        import math

        from .experiment import run as run_benchmark

        if args.nominal_voltage is not None and (
            not math.isfinite(args.nominal_voltage) or args.nominal_voltage <= 0
        ):
            parser.error("--nominal-voltage must be finite and positive.")
        run_benchmark(args)


if __name__ == "__main__":
    main()
