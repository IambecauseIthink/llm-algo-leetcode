"""Create v1 sidecar results for legacy training-memory measurements.

Usage:
    python tools/normalize_training_memory_results.py \
      benchmarks/results/73_real_gpu_training.json \
      benchmarks/results/76_real_gpu_memory.json \
      benchmarks/results/75_memory_budget_decision.json

The command never overwrites an input unless an explicit output path is
provided for that input.  Existing experiment results therefore remain the
source of truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training_memory_result_schema import normalize_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Legacy result JSON files")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated sidecar files")
    args = parser.parse_args()

    for source in args.inputs:
        output = None
        if args.output_dir:
            output = args.output_dir / f"{source.stem}_v1.json"
        print(normalize_file(source, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
