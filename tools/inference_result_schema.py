"""Shared result schema for the Part 02 inference projects.

The project notebooks may add strategy-specific fields, but the fields built by
this module are intentionally stable across projects 66--70.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "inference-benchmark/v1"


def make_result(
    *,
    project: str,
    strategy: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    quality: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    strategy_metrics: Mapping[str, Any] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Create one portable project result.

    ``config`` contains the common experiment dimensions while ``metrics``
    contains measured values. Strategy-specific measurements stay under
    ``strategy_metrics`` instead of changing the shared contract.
    """

    common_config = {
        "model": config.get("model"),
        "backend": config.get("backend", "local"),
        "dtype": config.get("dtype", "unknown"),
        "prompt_tokens": config.get("prompt_tokens"),
        "generated_tokens": config.get("generated_tokens"),
        "batch": config.get("batch", 1),
        "concurrency": config.get("concurrency", 1),
        "cache_policy": config.get("cache_policy", "default"),
    }
    # Preserve useful dimensions without making them mandatory for every
    # project (for example, speculative decoding has draft-model fields).
    common_config.update(
        {
            key: value
            for key, value in config.items()
            if key not in common_config and value is not None
        }
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": str(project),
        "strategy": strategy,
        "config": common_config,
        "metrics": dict(metrics),
        "quality": dict(quality or {}),
        "decision": dict(decision or {"decision": "not_evaluated"}),
    }
    if strategy_metrics:
        result["strategy_metrics"] = dict(strategy_metrics)
    if source:
        result["source"] = source
    return result


def from_backend_report(
    report: Mapping[str, Any],
    *,
    project: str = "66",
    strategy: str | None = None,
    backend: str = "vllm",
    dtype: str = "unknown",
    batch: int = 1,
    cache_policy: str = "default",
    peak_memory_mb: float | None = None,
    quality: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    strategy_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize the legacy backend benchmark report into the shared schema."""

    workload = report.get("workload", {})
    raw_metrics = report.get("metrics", {})
    requests = report.get("request_results") or report.get("results") or []
    prompt_tokens = None
    if requests:
        values = [item.get("prompt_tokens") for item in requests if isinstance(item, Mapping)]
        values = [value for value in values if isinstance(value, (int, float))]
        if values:
            prompt_tokens = round(sum(values) / len(values), 3)
    if prompt_tokens is None:
        prompt_tokens = raw_metrics.get("prompt_tokens_mean")
    e2e = raw_metrics.get("e2e_ms", {})
    metrics = {
        "ttft_ms": raw_metrics.get("ttft_ms", {}),
        "tpot_ms": raw_metrics.get("tpot_ms", {}),
        "e2e_latency_ms": e2e,
        "throughput_requests_per_s": raw_metrics.get("request_throughput_per_s"),
        "throughput_tokens_per_s": raw_metrics.get("output_token_throughput_per_s"),
        "p99_ms": e2e.get("p99") if isinstance(e2e, Mapping) else None,
        "peak_memory_mb": peak_memory_mb,
        "successful_requests": raw_metrics.get("successful_requests", 0),
        "failed_requests": raw_metrics.get("failed_requests", 0),
        "duration_s": raw_metrics.get("duration_s"),
    }
    config = {
        "model": report.get("model"),
        "backend": backend,
        "dtype": dtype,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": workload.get("max_tokens"),
        "batch": batch,
        "concurrency": workload.get("concurrency", 1),
        "cache_policy": cache_policy,
        "workload": workload.get("source"),
    }
    return make_result(
        project=project,
        strategy=strategy or report.get("label", "backend"),
        config=config,
        metrics=metrics,
        quality=quality,
        decision=decision,
        strategy_metrics=strategy_metrics,
        source=report.get("base_url"),
    )


def save_result(path: str | Path, result: Mapping[str, Any]) -> Path:
    """Write a result atomically enough for notebook workflows and return path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
