"""Reference construction and official GriTS evaluation."""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .data import DataError


_METRICS = ("top", "con", "loc")
_EVALUATION_IDENTITY_FIELDS = (
    "package",
    "version",
    "content_attachment_minimum_overlap",
    "table_metrics",
    "headline_metrics",
    "top_reference",
    "con_reference",
    "loc_reference",
    "standalone_failure_score",
)
_EVALUATION_IDENTITY_LIST_FIELDS = {"table_metrics", "headline_metrics"}
_TERMINAL_SOURCE_STATUSES = {
    "success",
    "failure",
    "parse_failed",
    "inference_failed",
    "inference_timeout",
    "out_of_memory",
    "worker_failed",
}


def evaluation_semantic_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached identity for the protocol's specified GriTS settings."""

    settings = protocol.get("evaluation")
    if not isinstance(settings, Mapping) or set(settings) != set(_EVALUATION_IDENTITY_FIELDS):
        raise DataError("protocol evaluation settings are incomplete")
    identity: dict[str, Any] = {}
    for field in _EVALUATION_IDENTITY_FIELDS:
        value = settings[field]
        if field in _EVALUATION_IDENTITY_LIST_FIELDS:
            if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
                raise DataError(f"protocol evaluation {field} is invalid")
            identity[field] = list(value)
        elif field in {"content_attachment_minimum_overlap", "standalone_failure_score"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise DataError(f"protocol evaluation {field} is invalid")
            identity[field] = float(value) if isinstance(value, float) else value
        else:
            if not isinstance(value, str) or not value:
                raise DataError(f"protocol evaluation {field} is invalid")
            identity[field] = value
    return identity


def _grits_classes() -> tuple[Any, Any]:
    try:
        from grits import GritsEvaluator, TableCell
    except Exception as exc:
        raise DataError("GritsEvaluator and TableCell are not importable") from exc
    return GritsEvaluator, TableCell


def validate_evaluation_environment(protocol: Mapping[str, Any]) -> dict[str, str]:
    """Validate the installed GriTS package against the protocol before scoring."""

    identity = evaluation_semantic_identity(protocol)
    package = identity["package"]
    expected_version = identity["version"]
    try:
        installed_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed_version = "<missing>"
    if installed_version != expected_version:
        raise DataError(
            f"installed {package} version {installed_version} does not match "
            f"protocol version {expected_version}"
        )
    _grits_classes()
    return {"grits_metric": installed_version}


def _validate_evaluation_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_EVALUATION_IDENTITY_FIELDS):
        raise DataError("evaluation record identity has unexpected fields")
    return evaluation_semantic_identity({"evaluation": dict(value)})


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DataError(f"{context} must be finite")
    return number


def _bbox(value: Any, *, context: str) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        values = [value.get(key) for key in ("l", "t", "r", "b")]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    if len(values) != 4:
        raise DataError(f"{context} must contain four coordinates")
    coordinates = [_number(item, context=f"{context}[{index}]") for index, item in enumerate(values)]
    if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
        raise DataError(f"{context} must have positive area")
    return coordinates


def prediction_cells_to_grits(
    cells: Sequence[Mapping[str, Any]],
    *,
    allow_overlaps: bool = False,
) -> list[TableCell]:
    """Validate cells and convert them to ``TableCell`` objects.

    Primary TableFormer predictions may contain cross-cell overlaps under the
    specified conversion semantics. Expert predictions retain strict global
    non-overlap validation through the default.
    """

    _grits_evaluator, table_cell_class = _grits_classes()
    if isinstance(cells, (str, bytes, Mapping)):
        raise DataError("prediction cells must be a sequence of objects")
    try:
        values = list(cells)
    except TypeError as exc:
        raise DataError("prediction cells must be a sequence") from exc
    prepared: list[tuple[tuple[Any, ...], TableCell]] = []
    occupied: set[tuple[int, int]] = set()
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise DataError(f"prediction cell {index} must be an object")
        ranges = tuple(item.get(name) for name in ("row_start", "row_end", "column_start", "column_end"))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ranges):
            raise DataError(f"prediction cell {index} has invalid integer ranges")
        row_start, row_end, column_start, column_end = ranges
        if row_start < 0 or column_start < 0 or row_end <= row_start or column_end <= column_start:
            raise DataError(f"prediction cell {index} has invalid ranges")
        text = item.get("text")
        if not isinstance(text, str):
            raise DataError(f"prediction cell {index}.text must be a string")
        for row in range(row_start, row_end):
            for column in range(column_start, column_end):
                coordinate = (row, column)
                if not allow_overlaps and coordinate in occupied:
                    raise DataError(f"prediction cells overlap at {coordinate}")
                occupied.add(coordinate)
        for name in ("is_column_header", "is_row_header"):
            if name in item and not isinstance(item[name], bool):
                raise DataError(f"prediction cell {index}.{name} must be boolean")
        box = _bbox(item.get("bbox"), context=f"prediction cell {index}.bbox")
        grits_cell = table_cell_class(
            row_nums=list(range(row_start, row_end)),
            column_nums=list(range(column_start, column_end)),
            cell_text=text,
            bbox=box,
            is_column_header=item.get("is_column_header", False),
            is_row_header=item.get("is_row_header", False),
        )
        sort_key = (
            row_start,
            column_start,
            row_end,
            column_end,
            tuple(box) if box is not None else (),
            text,
        )
        prepared.append((sort_key, grits_cell))
    return [cell for _, cell in sorted(prepared, key=lambda value: value[0])]


def _official_metric(
    reference_cells: Sequence[TableCell],
    prediction_cells: Sequence[TableCell],
    metric: str,
) -> float:
    if metric not in _METRICS:
        raise DataError(f"unsupported GriTS metric: {metric}")
    grits_evaluator, _table_cell_class = _grits_classes()
    evaluator = grits_evaluator(metrics=[metric])
    evaluator.eval_table_cell_lists([list(reference_cells)], [list(prediction_cells)])
    return float(evaluator.compute_grits()[f"grits_{metric}"])


def score_table(
    reference: Mapping[str, Any],
    prediction_cells: Sequence[TableCell],
) -> dict[str, Any]:
    """Score one canonical prediction against official structure/content references."""

    structure = reference.get("structure_cells")
    content = reference.get("content_cells")
    if not isinstance(structure, Sequence) or not isinstance(content, Sequence):
        raise DataError("reference lacks structure_cells or content_cells")
    result: dict[str, Any] = {
        "reference_cell_count": len(structure),
        "prediction_cell_count": len(prediction_cells),
        "grits_top": None,
        "grits_con": None,
        "grits_loc": None,
        "grits_loc_available": all(cell.bbox is not None for cell in prediction_cells),
        "metric_errors": [],
    }
    for metric, reference_cells in (("top", structure), ("con", content)):
        try:
            result[f"grits_{metric}"] = _official_metric(reference_cells, prediction_cells, metric)
        except Exception as exc:
            result["metric_errors"].append(
                {"metric": metric, "error_type": type(exc).__name__, "error_message": " ".join(str(exc).split())[:500]}
            )
    if result["grits_loc_available"]:
        try:
            result["grits_loc"] = _official_metric(structure, prediction_cells, "loc")
        except Exception as exc:
            result["metric_errors"].append(
                {"metric": "loc", "error_type": type(exc).__name__, "error_message": " ".join(str(exc).split())[:500]}
            )
    return result


def aggregate_scores(
    pairs: Sequence[tuple[Sequence[TableCell], Sequence[TableCell]]],
    *,
    metrics: Sequence[str] = ("top", "con", "loc"),
) -> dict[str, Any]:
    """Aggregate official GriTS scores across reference/prediction pairs."""

    selected = tuple(metrics)
    if any(metric not in _METRICS for metric in selected) or len(set(selected)) != len(selected):
        raise DataError("metrics must be distinct names from top, con, and loc")
    result: dict[str, Any] = {"document_count": len(pairs), "micro": {}, "macro": {}}
    for metric in selected:
        grits_evaluator, _table_cell_class = _grits_classes()
        evaluator = grits_evaluator(metrics=[metric])
        values: list[float] = []
        for reference, prediction in pairs:
            evaluator.eval_table_cell_lists([list(reference)], [list(prediction)])
            values.append(_official_metric(reference, prediction, metric))
        if values:
            official = evaluator.compute_grits()
            macro = evaluator.compute_mean_grits_per_sample()
            result["micro"][f"grits_{metric}"] = float(official[f"grits_{metric}"])
            result["macro"][f"grits_{metric}"] = float(macro[f"mean_grits_{metric}_per_sample"])
    return result


def score_output_record(
    reference: Mapping[str, Any],
    source_record: Mapping[str, Any],
    *,
    allow_prediction_overlaps: bool = False,
) -> dict[str, Any]:
    """Score one recognition record with explicit coverage semantics."""

    status = source_record.get("status")
    base: dict[str, Any] = {
        "source_status": status,
        "scoring_status": "failed_source" if status != "success" else "scored",
        "grits_top": None,
        "grits_con": None,
        "grits_loc": None,
        "grits_loc_available": False,
        "reference_cell_count": len(reference.get("structure_cells", [])),
        "prediction_cell_count": 0,
        "coverage_aware_grits_top": 0.0,
        "coverage_aware_grits_con": 0.0,
        "content_attachment_diagnostics": reference.get("content_attachment_diagnostics"),
        "failure_stage": None,
        "error_type": None,
        "error_message": None,
    }
    if status != "success":
        base["failure_stage"] = source_record.get("failure_stage", "recognition")
        base["error_type"] = source_record.get("error_type", "RecognitionFailure")
        base["error_message"] = source_record.get("error_message", "source recognition did not succeed")
        return base
    try:
        prediction = prediction_cells_to_grits(
            source_record.get("cells", []),
            allow_overlaps=allow_prediction_overlaps,
        )
        if not prediction:
            raise DataError("successful recognition has no canonical cells")
        scores = score_table(reference, prediction)
    except Exception as exc:
        base["scoring_status"] = "metric_failed"
        base["failure_stage"] = "scoring"
        base["error_type"] = type(exc).__name__
        base["error_message"] = " ".join(str(exc).split())[:500]
        return base
    base.update({key: scores[key] for key in (
        "grits_top", "grits_con", "grits_loc", "grits_loc_available",
        "reference_cell_count", "prediction_cell_count",
    )})
    base["metric_errors"] = scores.get("metric_errors", [])
    if base["grits_top"] is not None and base["grits_con"] is not None:
        base["coverage_aware_grits_top"] = base["grits_top"]
        base["coverage_aware_grits_con"] = base["grits_con"]
    if base["metric_errors"]:
        base["scoring_status"] = "metric_failed"
        base["failure_stage"] = "scoring"
    return base


def validate_evaluation_record(
    record: Mapping[str, Any],
    *,
    sample_id: str | None = None,
    system_id: str | None = None,
    expected_split: str | None = None,
    expected_issuer: str | None = None,
    expected_dataset_id: str | None = None,
    expected_revision: str | None = None,
    expected_source_status: str | None = None,
    expected_source_identity: Mapping[str, Any] | None = None,
    expected_evaluation_identity: Mapping[str, Any] | None = None,
) -> None:
    """Validate a resumable evaluation record without loading a model."""

    if not isinstance(record, Mapping):
        raise DataError("evaluation record must be an object")
    for key in (
        "sample_id",
        "split",
        "issuer",
        "system_id",
        "dataset_id",
        "revision",
        "source_status",
        "source_recognition_status",
        "scoring_status",
    ):
        if not isinstance(record.get(key), str) or not record[key]:
            raise DataError(f"evaluation record lacks {key}")
    if sample_id is not None and record["sample_id"] != sample_id:
        raise DataError("evaluation record sample ID changed")
    if system_id is not None and record["system_id"] != system_id:
        raise DataError("evaluation record system ID changed")
    for key, expected in (
        ("split", expected_split),
        ("issuer", expected_issuer),
        ("dataset_id", expected_dataset_id),
        ("revision", expected_revision),
        ("source_status", expected_source_status),
    ):
        if expected is not None and record[key] != expected:
            raise DataError(f"evaluation record {key} changed")
    if record["source_recognition_status"] != record["source_status"]:
        raise DataError("evaluation source status fields disagree")
    source_identity = record.get("source_identity")
    if not isinstance(source_identity, Mapping) or not source_identity:
        raise DataError("evaluation record source identity is invalid")
    kind = source_identity.get("kind")
    if kind == "primary":
        if set(source_identity) != {"kind", "primary_identity"}:
            raise DataError("Primary evaluation source identity has unexpected fields")
        if not isinstance(source_identity.get("primary_identity"), Mapping):
            raise DataError("Primary evaluation source identity is incomplete")
    elif kind == "expert":
        if set(source_identity) != {"kind", "expert_id", "checkpoint_model_id", "checkpoint_revision"}:
            raise DataError("expert evaluation source identity has unexpected fields")
        for key in ("expert_id", "checkpoint_model_id", "checkpoint_revision"):
            if not isinstance(source_identity.get(key), str) or not source_identity[key]:
                raise DataError(f"expert evaluation source identity lacks {key}")
    else:
        raise DataError("evaluation source identity has an unknown kind")
    if expected_source_identity is not None and dict(source_identity) != dict(expected_source_identity):
        raise DataError("evaluation record source identity changed")
    evaluation_identity = _validate_evaluation_identity(record.get("evaluation_identity"))
    if expected_evaluation_identity is not None and evaluation_identity != dict(expected_evaluation_identity):
        raise DataError("evaluation record identity does not match the specified GriTS protocol")
    if record["source_status"] not in _TERMINAL_SOURCE_STATUSES:
        raise DataError("evaluation record source status is not terminal")
    if record["scoring_status"] not in {"scored", "failed_source", "metric_failed"}:
        raise DataError("evaluation record scoring status is invalid")
    for key in ("grits_top", "grits_con", "grits_loc", "coverage_aware_grits_top", "coverage_aware_grits_con"):
        value = record.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
            raise DataError(f"evaluation record {key} is not finite or null")
    if record["source_status"] != "success":
        if record["scoring_status"] != "failed_source":
            raise DataError("failed source has an invalid scoring status")
        if record.get("grits_top") is not None or record.get("grits_con") is not None:
            raise DataError("failed source has an actual score")
        if record.get("coverage_aware_grits_top") != 0.0 or record.get("coverage_aware_grits_con") != 0.0:
            raise DataError("failed source does not have zero coverage-aware scores")
    elif record["scoring_status"] == "failed_source":
        raise DataError("successful source has an invalid scoring status")
    if record["scoring_status"] == "scored" and (
        record.get("grits_top") is None or record.get("grits_con") is None
    ):
        raise DataError("scored evaluation lacks headline metrics")


__all__ = [
    "aggregate_scores",
    "evaluation_semantic_identity",
    "prediction_cells_to_grits",
    "score_output_record",
    "score_table",
    "validate_evaluation_environment",
    "validate_evaluation_record",
]
