#!/usr/bin/env python3
"""
CLI entry point for the LIBERO VLA memory labeler.

Usage examples
--------------
# Process all demos in a single task file:
python main.py \
    --hdf5  data/libero/libero_90/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.hdf5 \
    --bddl  data/libero/libero_90/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl \
    --config config/config.yaml \
    --output output/

# Process a specific subset of demos:
python main.py --hdf5 ... --bddl ... --demos demo_0 demo_1 demo_2

# Process an entire dataset directory (one HDF5+BDDL pair per task):
python main.py --dataset_dir data/libero/libero_90 --output output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline import run_pipeline, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LIBERO VLA memory labeler: segment demos and annotate with symbolic actions."
    )

    # Single-task mode
    parser.add_argument("--hdf5", type=str, help="Path to a LIBERO .hdf5 file.")
    parser.add_argument("--bddl", type=str, help="Path to the corresponding .bddl file.")
    parser.add_argument(
        "--demos",
        nargs="*",
        help="Specific demo keys to process (e.g. demo_0 demo_1). Default: all.",
    )

    # Batch mode
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory containing .hdf5 and .bddl file pairs (batch mode).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Process only this suite subdirectory (e.g. libero_90). Default: all suites.",
    )

    # Common
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml. Default: config/config.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for JSON files. Overrides config value.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.dataset_dir:
        _run_batch(args, config)
    elif args.hdf5 and args.bddl:
        _run_single(args, config)
    else:
        print(
            "Error: provide either --hdf5 + --bddl (single task) "
            "or --dataset_dir (batch mode).",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_single(args: argparse.Namespace, config: dict) -> None:
    run_pipeline(
        hdf5_path=args.hdf5,
        bddl_path=args.bddl,
        config=config,
        output_dir=args.output,
        demo_keys=args.demos,
    )


def _run_batch(args: argparse.Namespace, config: dict) -> None:
    dataset_dir = Path(args.dataset_dir)

    # Discover subdirectory pairs: libero_90 / libero_90_bddl
    hdf5_dirs = sorted(
        d for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.endswith("_bddl")
        and (args.suite is None or d.name == args.suite)
    )

    if not hdf5_dirs:
        print(f"No task subdirectories found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    for hdf5_dir in hdf5_dirs:
        bddl_dir = dataset_dir / f"{hdf5_dir.name}_bddl"
        if not bddl_dir.exists():
            print(f"  Skipping {hdf5_dir.name}: no matching _bddl directory found.")
            continue

        hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"  Skipping {hdf5_dir.name}: no .hdf5 files found.")
            continue

        print(f"\n=== Suite: {hdf5_dir.name} ({len(hdf5_files)} tasks) ===")
        for hdf5_path in hdf5_files:
            # abc_demo.hdf5 -> abc.bddl
            bddl_stem = hdf5_path.stem.removesuffix("_demo")
            bddl_path = bddl_dir / f"{bddl_stem}.bddl"
            if not bddl_path.exists():
                print(f"  Skipping {hdf5_path.name}: no matching .bddl found in {bddl_dir.name}.")
                continue
            print(f"\n--- Task: {hdf5_path.stem} ---")
            run_pipeline(
                hdf5_path=str(hdf5_path),
                bddl_path=str(bddl_path),
                config=config,
                output_dir=args.output,
            )


if __name__ == "__main__":
    main()
