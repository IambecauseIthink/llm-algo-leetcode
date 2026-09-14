"""Result schema for project 74, profiling-driven end-to-end optimization.

Project 74 shares the common experiment envelope with projects 73, 76 and 75,
but adds evidence about the diagnosed bottleneck.  The profiling fields are
optional so a CPU-first notebook can still produce a valid report before a
real GPU trace is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "profiling-optimization-benchmark/v1"


def make_result(
    *,
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    tuned: Mapping[str, Any],
    decision: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
    bottleneck: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Create a portable 74 report while preserving raw candidate metrics."""

    normalized_config = {
        "model": config.get("model", config.get("model_id")),
        "device": config.get("device"),
        "torch": config.get("torch"),
        "cuda": config.get("cuda", config.get("torch_cuda")),
        "dtype": config.get("dtype", config.get("amp_dtype")),
        "batch_size": config.get("batch_size"),
        "seq_len": config.get("seq_len"),
        "warmup": config.get("warmup"),
        "iters": config.get("iters"),
        "workload": config.get("workload"),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": "74",
        "stage": "profiling_driven_optimization",
        "config": normalized_config,
        "candidates": [
            {"name": "baseline", "metrics": dict(baseline)},
            {"name": "tuned", "metrics": dict(tuned)},
        ],
        "profile": dict(profile or {}),
        "bottleneck": dict(bottleneck or {}),
        "validation": dict(validation or {}),
        "decision": dict(decision),
    }
    if source:
        result["source"] = source
    return result


def save_result(path: str | Path, result: Mapping[str, Any]) -> Path:
    """Save a profiling result for notebook and local-GPU workflows."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
