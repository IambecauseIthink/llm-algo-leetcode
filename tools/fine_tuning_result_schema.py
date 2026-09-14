"""Shared result contract for fine-tuning projects 60--65.

Project notebooks may keep project-specific fields, but their report envelope
uses the same top-level sections. This module does not invent measurements.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "fine-tuning-project/v1"
PROJECTS = {"60", "61", "62", "63", "64", "65"}
DECISIONS = {"accept", "tune", "reject", "not_evaluated"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def normalize_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a portable envelope without changing project measurements."""

    config = _mapping(report.get("config"))
    quality = _mapping(report.get("quality"))
    resources = _mapping(report.get("resources"))
    decision = _mapping(report.get("decision"))

    normalized = dict(report)
    normalized["schema_version"] = report.get("schema_version", SCHEMA_VERSION)
    normalized["config"] = {
        "model": config.get("model", config.get("model_id")),
        "dataset": config.get("dataset"),
        "dtype": config.get("dtype", config.get("amp_dtype")),
        "batch_size": config.get("batch_size"),
        "seq_len": config.get("seq_len"),
        "seed": config.get("seed"),
        **dict(config),
    }
    normalized["quality"] = {
        "train_loss": quality.get("train_loss", quality.get("loss")),
        "val_loss": quality.get("val_loss", quality.get("eval_loss")),
        "task_metrics": quality.get("task_metrics", {}),
        **dict(quality),
    }
    normalized["resources"] = {
        "trainable_params": resources.get("trainable_params"),
        "trainable_ratio": resources.get("trainable_ratio"),
        "peak_memory_mb": resources.get("peak_memory_mb", resources.get("memory_mb")),
        "step_time_ms": resources.get("step_time_ms"),
        "tokens_per_s": resources.get("tokens_per_s"),
        **dict(resources),
    }
    normalized["baseline"] = report.get("baseline", {})
    normalized["candidates"] = report.get("candidates", [])
    normalized["artifacts"] = report.get("artifacts", {})
    normalized["decision"] = {
        "decision": decision.get("decision", "not_evaluated"),
        "reason": decision.get("reason"),
        "next_action": decision.get("next_action"),
        **dict(decision),
    }
    normalized["environment"] = report.get("environment", {})
    return normalized


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return validation errors; an empty list means the contract is valid."""

    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    project = str(report.get("project", ""))
    if project and project.split("_", 1)[0] not in PROJECTS:
        errors.append("project must start with one of 60, 61, 62, 63, 64, 65")
    for section in ("config", "baseline", "candidates", "quality", "resources", "artifacts", "decision"):
        if section not in report:
            errors.append(f"missing section: {section}")
    if not isinstance(report.get("candidates"), list):
        errors.append("candidates must be a list")
    decision = _mapping(report.get("decision"))
    if decision.get("decision") not in DECISIONS:
        errors.append("decision.decision must be accept, tune, reject, or not_evaluated")
    return errors


def normalize_and_validate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a report and raise ValueError when its contract fails."""

    normalized = normalize_report(report)
    errors = validate_report(normalized)
    if errors:
        raise ValueError("; ".join(errors))
    return normalized
