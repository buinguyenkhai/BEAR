#!/usr/bin/env python3
"""Run the BEAR reproduction studies."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bear.figure import FigureError, render_figure2
from bear.tinix import TinixError, run_tinix_stage

from bear.data import DataError, VietFinTabDataset, load_protocol
from bear.evaluation import (
    evaluation_semantic_identity,
    score_output_record,
    validate_evaluation_environment,
    validate_evaluation_record,
)
from bear.recognition import (
    EXPERT_IDS,
    ExpertRecognizer,
    PrimaryRecognizer,
    RecognitionError,
    validate_expert_protocol,
    validate_expert_runtime_summary,
    expert_semantic_identity,
    validate_primary_environment,
    primary_semantic_identity,
)
from bear.routing import (
    call_benefit_grits_con,
    DIFFICULTY_MODEL_FILENAME,
    FEATURE_NAMES,
    MODEL_IDS,
    RoutingError,
    TIE_TOLERANCE,
    _candidate_params,
    _matrix,
    _model_specs,
    build_risk_rankings,
    deterministic_random_route,
    evaluate_route,
    fit_final_router_families,
    fit_difficulty_router,
    load_difficulty_router,
    risk_populations,
    nested_oof_predictions,
    paired_bootstrap_mean_difference,
    rank_by_predicted_benefit,
    route_score_vector,
    random_summary,
    random_mean_vector,
    route_for_budget,
    predict_router,
    routing_settings,
    spearman_summary,
    standalone_summary,
)


ROOT = Path(__file__).resolve().parent


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _split_summary(dataset: VietFinTabDataset) -> dict[str, Any]:
    records = dataset.validate_records()
    entries = dataset.split_entries
    development_issuers = {item["issuer"] for item in entries["development"]}
    final_issuers = {item["issuer"] for item in entries["final"]}
    return {
        "dataset_id": dataset.dataset_id,
        "revision": dataset.revision,
        "development_count": len(entries["development"]),
        "final_count": len(entries["final"]),
        "reserve_count": len(entries["reserve"]),
        "calibration_count": len(dataset.calibration_entries),
        "total_protocol_count": len(records),
        "development_issuer_count": len(development_issuers),
        "final_issuer_count": len(final_issuers),
        "shared_issuer_count": len(development_issuers & final_issuers),
        "final_spanning_count": sum(
            bool(records[item["sample_id"]]["spanning"]) for item in entries["final"]
        ),
        "final_non_spanning_count": sum(
            not bool(records[item["sample_id"]]["spanning"]) for item in entries["final"]
        ),
        "split_sample_ids": {
            name: [item["sample_id"] for item in values]
            for name, values in entries.items()
        },
        "calibration_sample_ids": dataset.calibration_sample_ids(),
    }


_BATCH_COHORTS = ("calibration", "development", "final")


def _cohort_ids(dataset: VietFinTabDataset, cohort: str) -> list[str]:
    if cohort not in _BATCH_COHORTS:
        raise DataError(
            f"unsupported batch cohort: {cohort}; reserve is not available in this run"
        )
    ids = (
        dataset.calibration_sample_ids()
        if cohort == "calibration"
        else dataset.split_sample_ids(cohort)
    )
    if len(ids) != len(set(ids)):
        raise DataError(f"batch cohort {cohort} contains duplicate sample IDs")
    return sorted(ids)


def _readable_image_paths(
    dataset: VietFinTabDataset,
    ids: list[str],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    """Resolve and validate every image before any recognizer is initialized."""

    missing_metadata = [sample_id for sample_id in ids if sample_id not in records]
    if missing_metadata:
        raise DataError(f"selected sample is absent from the metadata: {missing_metadata[0]}")

    def resolve(sample_id: str) -> tuple[str, Path]:
        return sample_id, dataset.image_path(sample_id)

    resolved: dict[str, Path] = {}
    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(resolve, sample_id): sample_id for sample_id in ids}
            for future in as_completed(futures):
                sample_id, image_path = future.result()
                resolved[sample_id] = image_path
    except Exception as exc:
        sample_id = futures[future] if "future" in locals() and future in futures else "unknown"
        raise DataError(f"selected image could not be resolved: {sample_id}") from exc

    paths: dict[str, Path] = {}
    try:
        from PIL import Image

        for sample_id in ids:
            image_path = resolved[sample_id]
            with Image.open(image_path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("image dimensions are not positive")
            paths[sample_id] = image_path
    except Exception as exc:
        sample_id = sample_id if "sample_id" in locals() else "unknown"
        raise DataError(f"selected image is not readable: {sample_id}") from exc
    return paths


def _validate_existing(
    path: Path,
    sample: dict[str, Any],
    protocol: dict[str, Any],
    *,
    expected_cohort: str,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataError(f"existing output is not valid JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("sample_id") != sample["sample_id"]:
        raise DataError(f"existing output has the wrong sample ID: {path}")
    if (
        value.get("split") != expected_cohort
        or value.get("issuer") != sample["issuer"]
        or value.get("spanning") != sample["spanning"]
        or value.get("dataset_id") != protocol["dataset"]["id"]
        or value.get("revision") != protocol["dataset"]["revision"]
    ):
        raise DataError(f"existing Primary output provenance does not match the protocol: {path}")
    try:
        expected_identity = primary_semantic_identity(protocol)
    except RecognitionError as exc:
        raise DataError(str(exc)) from exc
    if value.get("primary_identity") != expected_identity:
        raise DataError(
            "existing Primary output has an incompatible semantic identity; "
            f"remove it before rerunning: {path}"
        )
    if value.get("status") not in {"success", "failure"}:
        raise DataError(f"existing output has no terminal status: {path}")
    if value.get("feature_names") != list(FEATURE_NAMES):
        raise DataError(f"existing output has the wrong feature order: {path}")
    features = value.get("features")
    vector = value.get("feature_vector")
    if not isinstance(features, dict) or list(features) != list(FEATURE_NAMES):
        raise DataError(f"existing output has an invalid feature record: {path}")
    if not isinstance(vector, list) or len(vector) != len(FEATURE_NAMES):
        raise DataError(f"existing output has an invalid feature vector: {path}")
    for index, name in enumerate(FEATURE_NAMES):
        feature_value = features[name]
        vector_value = vector[index]
        if feature_value is None:
            if vector_value is not None:
                raise DataError(f"feature/vector mismatch for {name}: {path}")
            continue
        if isinstance(feature_value, bool) or not isinstance(feature_value, (int, float)):
            raise DataError(f"feature {name} is not a number or null: {path}")
        if not math.isfinite(float(feature_value)):
            raise DataError(f"feature {name} is not finite: {path}")
        if isinstance(vector_value, bool) or not isinstance(vector_value, (int, float)):
            raise DataError(f"feature vector {name} is not a number or null: {path}")
        if not math.isfinite(float(vector_value)) or float(vector_value) != float(feature_value):
            raise DataError(f"feature/vector mismatch for {name}: {path}")
    if value["status"] == "success":
        required_features = (
            ("ocr_confidence_p10", "unmatched_ocr_ratio", "empty_cell_ratio")
            if expected_cohort == "calibration"
            else FEATURE_NAMES
        )
        if any(features[name] is None for name in required_features):
            raise DataError(f"successful output has unavailable features: {path}")
        if expected_cohort != "calibration":
            width = value.get("image_width")
            height = value.get("image_height")
            if (
                isinstance(width, bool)
                or not isinstance(width, (int, float))
                or not math.isfinite(float(width))
                or width <= 0
                or isinstance(height, bool)
                or not isinstance(height, (int, float))
                or not math.isfinite(float(height))
                or height <= 0
            ):
                raise DataError(f"successful output has invalid image dimensions: {path}")
            cells = value.get("cells")
            if not isinstance(cells, list) or not cells:
                raise DataError(f"successful output has no canonical cells: {path}")
        if "failure_stage" in value:
            raise DataError(f"successful output contains a failure stage: {path}")
    else:
        for field in ("failure_stage", "error_type", "error_message"):
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise DataError(f"failed output has no {field}: {path}")
    return value


def _primary_cohort_stage(
    dataset: VietFinTabDataset,
    cohort: str,
    output_dir: Path,
    device: str,
    models_dir: Path,
) -> None:
    """Run one persistent Primary recognizer over one configured cohort."""

    try:
        validate_primary_environment(dataset.protocol)
    except RecognitionError as exc:
        raise DataError(str(exc)) from exc
    ids = _cohort_ids(dataset, cohort)
    records = dataset.validate_records()
    output_root = output_dir / "vietfintab" / "primary"
    output_paths = {sample_id: output_root / f"{sample_id}.json" for sample_id in ids}
    resumed = 0
    pending: list[str] = []
    validated_existing: dict[str, dict[str, Any]] = {}
    for sample_id in ids:
        path = output_paths[sample_id]
        if path.is_file():
            validated_existing[sample_id] = _validate_existing(
                path,
                records[sample_id],
                dataset.protocol,
                expected_cohort=cohort,
            )
            resumed += 1
        else:
            pending.append(sample_id)
    paths = _readable_image_paths(dataset, ids, records)
    recognizer = (
        PrimaryRecognizer(
            device=device,
            models_dir=models_dir,
            protocol=dataset.protocol,
        )
        if pending
        else None
    )
    generated = 0
    for sample_id in pending:
        assert recognizer is not None
        item = records[sample_id]
        result = recognizer.predict(paths[sample_id])
        output = {
            "sample_id": sample_id,
            "split": cohort,
            "issuer": item["issuer"],
            "spanning": item["spanning"],
            "dataset_id": dataset.dataset_id,
            "revision": dataset.revision,
            "primary_identity": primary_semantic_identity(dataset.protocol),
            **result,
        }
        _atomic_json_write(output_paths[sample_id], output)
        _validate_existing(
            output_paths[sample_id],
            item,
            dataset.protocol,
            expected_cohort=cohort,
        )
        generated += 1
    status_counts: dict[str, int] = {}
    first_failed: str | None = None
    for sample_id in ids:
        value = (
            validated_existing[sample_id]
            if sample_id in validated_existing
            else _validate_existing(
                output_paths[sample_id],
                records[sample_id],
                dataset.protocol,
                expected_cohort=cohort,
            )
        )
        status = str(value["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "success" and first_failed is None:
            first_failed = sample_id
    print(
        json.dumps(
            {
                "stage": "primary",
                "cohort": cohort,
                "selected_count": len(ids),
                "resumed": resumed,
                "newly_generated": generated,
                "status_counts": dict(sorted(status_counts.items())),
                "first_failed_sample": first_failed,
                "model_initialized": recognizer is not None,
                "artifact_resolution": (
                    recognizer.artifact_resolution
                    if recognizer is not None
                    else {"runtime_checked": False, "reason": "all selected records were resumed"}
                ),
                "output_directory": str(output_root.relative_to(ROOT))
                if output_root.is_relative_to(ROOT)
                else str(output_root),
            },
            indent=2,
        )
    )


def _validate_expert_cells(value: Any, path: Path) -> None:
    if not isinstance(value, list) or not value:
        raise DataError(f"successful expert output has no canonical cells: {path}")
    occupied: set[tuple[int, int]] = set()
    for index, cell in enumerate(value):
        if not isinstance(cell, dict):
            raise DataError(f"expert cell {index} is not an object: {path}")
        ranges = [cell.get("row_start"), cell.get("row_end"), cell.get("column_start"), cell.get("column_end")]
        if any(isinstance(item, bool) or not isinstance(item, int) for item in ranges):
            raise DataError(f"expert cell {index} has invalid ranges: {path}")
        row_start, row_end, column_start, column_end = ranges
        if row_start < 0 or column_start < 0 or row_end <= row_start or column_end <= column_start:
            raise DataError(f"expert cell {index} has invalid ranges: {path}")
        if not isinstance(cell.get("text"), str):
            raise DataError(f"expert cell {index} has invalid text: {path}")
        for row in range(row_start, row_end):
            for column in range(column_start, column_end):
                coordinate = (row, column)
                if coordinate in occupied:
                    raise DataError(f"expert cells overlap at {coordinate}: {path}")
                occupied.add(coordinate)


def _validate_expert_existing(
    path: Path,
    sample: dict[str, Any],
    expert_id: str,
    protocol: dict[str, Any],
    *,
    expected_cohort: str,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataError(f"existing expert output is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DataError(f"existing expert output is not an object: {path}")
    expected_checkpoint = protocol["experts"][expert_id]["checkpoint"]
    if (
        value.get("sample_id") != sample["sample_id"]
        or value.get("split") != expected_cohort
        or value.get("issuer") != sample["issuer"]
        or value.get("spanning") != sample["spanning"]
        or value.get("expert_id") != expert_id
        or value.get("dataset_id") != protocol["dataset"]["id"]
        or value.get("revision") != protocol["dataset"]["revision"]
        or value.get("checkpoint_model_id") != expected_checkpoint["model_id"]
        or value.get("checkpoint_revision") != expected_checkpoint["revision"]
    ):
        raise DataError(f"existing expert output provenance does not match the protocol: {path}")
    try:
        expected_identity = expert_semantic_identity(protocol, expert_id)
    except RecognitionError as exc:
        raise DataError(str(exc)) from exc
    if value.get("expert_identity") != expected_identity:
        raise DataError(
            "existing expert output has an incompatible semantic identity; "
            f"remove it before rerunning: {path}"
        )
    status = value.get("status")
    if status not in {"success", "parse_failed", "inference_failed", "inference_timeout", "out_of_memory", "worker_failed"}:
        raise DataError(f"existing expert output has no terminal status: {path}")
    if not isinstance(value.get("raw_output"), str):
        raise DataError(f"existing expert output has no raw output field: {path}")
    if not isinstance(value.get("runtime_environment"), dict):
        raise DataError(f"existing expert output has no runtime environment summary: {path}")
    try:
        validate_expert_runtime_summary(value["runtime_environment"], expert_id)
    except RecognitionError as exc:
        raise DataError(f"existing expert output has invalid CUDA-only runtime placement: {path}") from exc
    raw_bytes = value.get("raw_output_byte_count")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes < 0:
        raise DataError(f"existing expert output has an invalid raw-output byte count: {path}")
    for field in ("inference_elapsed_seconds", "parsing_elapsed_seconds"):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)) or number < 0:
            raise DataError(f"existing expert output has an invalid {field}: {path}")
    if status == "success":
        if not value["raw_output"].strip():
            raise DataError(f"successful expert output is empty: {path}")
        _validate_expert_cells(value.get("cells"), path)
        if value.get("failure_stage") is not None or value.get("error_type") is not None or value.get("error_message") is not None:
            raise DataError(f"successful expert output contains failure information: {path}")
        if value.get("native_output_format") not in {"html", "markdown"}:
            raise DataError(f"successful expert output has an invalid native format: {path}")
    else:
        if value.get("cells") not in (None, []):
            raise DataError(f"failed expert output contains canonical cells: {path}")
        for field in ("failure_stage", "error_type", "error_message"):
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise DataError(f"failed expert output has no {field}: {path}")
    return value


def _expert_cohort_stage(
    dataset: VietFinTabDataset,
    expert_id: str,
    cohort: str,
    output_dir: Path,
    device: str,
    models_dir: Path,
) -> None:
    """Run one selected expert serially over an allowed cohort."""

    if device != "cuda":
        raise DataError("expert inference requires --device cuda")
    allowed_cohorts = {
        "hunyuanocr_1_5": {"development", "final"},
        "glm_ocr": {"final"},
    }
    if expert_id not in EXPERT_IDS:
        raise DataError(f"unsupported expert: {expert_id}")
    if cohort not in allowed_cohorts[expert_id]:
        raise DataError(f"{expert_id} is not available for the {cohort} cohort")
    try:
        validate_expert_protocol(dataset.protocol, expert_id)
    except RecognitionError as exc:
        raise DataError(str(exc)) from exc
    ids = _cohort_ids(dataset, cohort)
    records = dataset.validate_records()
    output_root = output_dir / "vietfintab" / "experts" / expert_id
    output_paths = {sample_id: output_root / f"{sample_id}.json" for sample_id in ids}
    resumed = 0
    pending: list[str] = []
    validated_existing: dict[str, dict[str, Any]] = {}
    for sample_id in ids:
        path = output_paths[sample_id]
        if path.is_file():
            validated_existing[sample_id] = _validate_expert_existing(
                path,
                records[sample_id],
                expert_id,
                dataset.protocol,
                expected_cohort=cohort,
            )
            resumed += 1
        else:
            pending.append(sample_id)
    paths = _readable_image_paths(dataset, ids, records)
    recognizer = (
        ExpertRecognizer(
            expert_id,
            dataset.protocol,
            models_dir=models_dir,
            device=device,
        )
        if pending
        else None
    )
    generated = 0
    for sample_id in pending:
        assert recognizer is not None
        item = records[sample_id]
        result = recognizer.predict(paths[sample_id])
        output = {
            "sample_id": sample_id,
            "split": cohort,
            "issuer": item["issuer"],
            "spanning": item["spanning"],
            "dataset_id": dataset.dataset_id,
            "revision": dataset.revision,
            "expert_id": expert_id,
            "checkpoint_model_id": dataset.protocol["experts"][expert_id]["checkpoint"]["model_id"],
            "checkpoint_revision": dataset.protocol["experts"][expert_id]["checkpoint"]["revision"],
            "expert_identity": expert_semantic_identity(dataset.protocol, expert_id),
            "runtime_environment": recognizer.runtime_summary,
            "initialization_elapsed_seconds": recognizer.initialization_elapsed_seconds,
            **result,
        }
        _atomic_json_write(output_paths[sample_id], output)
        _validate_expert_existing(
            output_paths[sample_id],
            item,
            expert_id,
            dataset.protocol,
            expected_cohort=cohort,
        )
        generated += 1
    status_counts: dict[str, int] = {}
    first_failed: str | None = None
    for sample_id in ids:
        value = (
            validated_existing[sample_id]
            if sample_id in validated_existing
            else _validate_expert_existing(
                output_paths[sample_id],
                records[sample_id],
                expert_id,
                dataset.protocol,
                expected_cohort=cohort,
            )
        )
        status = str(value["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "success" and first_failed is None:
            first_failed = sample_id
    print(
        json.dumps(
            {
                "stage": "expert",
                "expert": expert_id,
                "cohort": cohort,
                "selected_count": len(ids),
                "resumed": resumed,
                "newly_generated": generated,
                "status_counts": dict(sorted(status_counts.items())),
                "first_failed_sample": first_failed,
                "model_initialized": recognizer is not None,
                "initialization_seconds": (
                    recognizer.initialization_elapsed_seconds if recognizer is not None else None
                ),
                "output_directory": str(output_root.relative_to(ROOT))
                if output_root.is_relative_to(ROOT)
                else str(output_root),
            },
            indent=2,
        )
    )


def _read_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"could not read {context}: {path}") from exc
    if not isinstance(value, dict):
        raise DataError(f"{context} must contain an object: {path}")
    return value


_CALIBRATION_POPULATION_NAMES = (
    "confidence_tail_risk",
    "unmatched_ocr_ratio",
    "empty_cell_ratio",
)
_DEVELOPMENT_ROUTE_METHODS = (
    "confidence_tail_only",
    "combined_risk",
    "ridge_benefit",
    "hist_gradient_boosting_benefit",
    "oracle_benefit",
)
_FINAL_ROUTE_METHODS = (
    "confidence_tail_only",
    "combined_risk",
    "ridge_benefit",
    "hist_gradient_boosting_benefit",
    "random_matched_budget",
)
_EXPERT_TERMINAL_STATUSES = {
    "success",
    "parse_failed",
    "inference_failed",
    "inference_timeout",
    "out_of_memory",
    "worker_failed",
}
_REPLACEMENT_POLICY = (
    "successful Hunyuan output replaces the primary; expert failure retains primary; "
    "no post-expert quality gate"
)
_FIGURE_CONTROL_BOOTSTRAP_SEED = "cg-hybrid-ocr-phaseb-feature-ablation-bootstrap-v1"


def _validate_calibration_populations(value: Any) -> None:
    """Validate the 50-value populations before they drive routing."""

    if not isinstance(value, dict) or set(value) != set(_CALIBRATION_POPULATION_NAMES):
        raise DataError("calibration populations have the wrong feature set")
    for name in _CALIBRATION_POPULATION_NAMES:
        population = value[name]
        if not isinstance(population, list) or len(population) != 50:
            raise DataError(f"calibration population is not exactly 50 values: {name}")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in population
        ):
            raise DataError(f"calibration population contains a non-finite value: {name}")


def _require_finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{context} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DataError(f"{context} is not finite")
    return number


def _validate_exact_id_list(
    value: Any,
    expected: list[str],
    *,
    context: str,
) -> None:
    if value != expected:
        raise DataError(f"{context} does not contain the exact specified ID order")


def _validate_calibration_artifact(
    protocol: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    expected_populations: Mapping[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Validate the derived calibration artifact without fitting or inference."""

    settings = routing_settings(protocol)
    expected_ids = [
        str(item["sample_id"])
        for item in protocol["vietfintab"]["calibration"]
    ]
    required = {
        "cohort",
        "sample_ids",
        "active_features",
        "inactive_features",
        "raw_populations",
        "percentile_rule",
        "primary_configuration_id",
        "primary_name",
    }
    if set(artifact) != required:
        raise DataError("calibration artifact has unexpected fields")
    if (
        artifact["cohort"] != "calibration"
        or artifact["sample_ids"] != expected_ids
        or artifact["active_features"] != list(settings["active_risk_features"])
        or artifact["inactive_features"] != list(settings["inactive_risk_features"])
        or artifact["percentile_rule"] != settings["percentile_rule"]
        or artifact["primary_configuration_id"] != protocol["primary"]["configuration_id"]
        or artifact["primary_name"] != protocol["primary"]["name"]
    ):
        raise DataError("calibration artifact does not match the protocol")
    populations = artifact["raw_populations"]
    _validate_calibration_populations(populations)
    if expected_populations is not None and dict(populations) != dict(expected_populations):
        raise DataError("calibration populations changed")
    return populations


def _validate_feature_mapping(
    value: Any,
    *,
    context: str,
    require_finite: bool = True,
) -> dict[str, float | None]:
    if not isinstance(value, Mapping) or list(value) != list(FEATURE_NAMES):
        raise DataError(f"{context} has the wrong feature order")
    result: dict[str, float | None] = {}
    for name in FEATURE_NAMES:
        feature = value[name]
        if feature is None:
            if require_finite:
                raise DataError(f"{context}.{name} is unavailable")
            result[name] = None
            continue
        result[name] = _require_finite(feature, context=f"{context}.{name}")
    return result


def _validate_rank_permutation(
    value: Any,
    expected_ids: list[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DataError(f"{context} is not an ID list")
    if len(value) != len(expected_ids) or len(set(value)) != len(value) or set(value) != set(expected_ids):
        raise DataError(f"{context} is not a complete ID permutation")


def _validate_development_report_shape(
    protocol: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    """Validate a completed development report using only serialized values."""

    settings = routing_settings(protocol)
    development_budgets = settings["development_budgets"]
    if not isinstance(report, Mapping):
        raise DataError("development router report must be an object")
    expected_ids = sorted(_protocol_ids(protocol, "development"))
    final_ids = set(_protocol_ids(protocol, "final"))
    reserve_ids = set(_protocol_ids(protocol, "reserve"))
    calibration_ids = set(
        str(item["sample_id"])
        for item in protocol["vietfintab"]["calibration"]
    )
    if report.get("run_status") != "success":
        raise DataError("development router report is not complete")
    if (
        report.get("dataset_id") != protocol["dataset"]["id"]
        or report.get("pinned_revision") != protocol["dataset"]["revision"]
    ):
        raise DataError("development router report dataset provenance changed")
    _validate_exact_id_list(report.get("sample_ids"), expected_ids, context="development sample_ids")
    if calibration_ids.intersection(expected_ids) or final_ids.intersection(expected_ids) or reserve_ids.intersection(expected_ids):
        raise DataError("development router report contains an ID from another cohort")
    if report.get("feature_order") != list(FEATURE_NAMES):
        raise DataError("development router report feature order changed")

    feature_rows = report.get("features")
    if not isinstance(feature_rows, list) or len(feature_rows) != len(expected_ids):
        raise DataError("development router report does not contain 600 feature records")
    feature_ids: list[str] = []
    for index, row in enumerate(feature_rows):
        if not isinstance(row, Mapping) or set(row) != {"sample_id", "issuer", "values"}:
            raise DataError(f"development feature record {index} is malformed")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise DataError(f"development feature record {index} has an unknown sample ID")
        feature_ids.append(sample_id)
        entry = next(item for item in protocol["vietfintab"]["splits"]["development"] if item["sample_id"] == sample_id)
        if row.get("issuer") != entry["issuer"]:
            raise DataError(f"development feature issuer changed: {sample_id}")
        _validate_feature_mapping(row.get("values"), context=f"development features {sample_id}")
    if feature_ids != expected_ids:
        raise DataError("development feature records are not in the exact specified order")

    targets = report.get("targets")
    if not isinstance(targets, list) or len(targets) != len(expected_ids):
        raise DataError("development router report does not contain 600 benefit targets")
    target_ids: list[str] = []
    for index, row in enumerate(targets):
        if not isinstance(row, Mapping) or set(row) != {"sample_id", "issuer", "call_benefit_grits_con"}:
            raise DataError(f"development target record {index} is malformed")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise DataError(f"development target record {index} has an unknown sample ID")
        target_ids.append(sample_id)
        entry = next(item for item in protocol["vietfintab"]["splits"]["development"] if item["sample_id"] == sample_id)
        if row.get("issuer") != entry["issuer"]:
            raise DataError(f"development target issuer changed: {sample_id}")
        _require_finite(row.get("call_benefit_grits_con"), context=f"development target {sample_id}")
    if target_ids != expected_ids:
        raise DataError("development targets are not in the exact specified order")

    status_counts = report.get("status_counts")
    if not isinstance(status_counts, Mapping) or set(status_counts) != {"primary", "hunyuan"}:
        raise DataError("development status counts are malformed")
    primary_counts = status_counts.get("primary")
    if primary_counts != {"success": 600}:
        raise DataError("development Primary status counts changed")
    hunyuan_counts = status_counts.get("hunyuan")
    if (
        not isinstance(hunyuan_counts, Mapping)
        or not hunyuan_counts
        or not set(hunyuan_counts).issubset(_EXPERT_TERMINAL_STATUSES)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in hunyuan_counts.values()
        )
    ):
        raise DataError("development Hunyuan status counts are malformed")
    if sum(hunyuan_counts.values()) != 600:
        raise DataError("development Hunyuan status counts do not total 600")

    router_models = report.get("router_models")
    if not isinstance(router_models, Mapping) or set(router_models) != set(MODEL_IDS):
        raise DataError("development router model identities changed")
    try:
        specs = _model_specs(protocol)
        expected_parameter_names = {
            model_id: set(_candidate_params(model_id, specs)[0])
            for model_id in MODEL_IDS
        }
    except (RoutingError, IndexError) as exc:
        raise DataError("protocol router grids are not usable") from exc
    for model_id in MODEL_IDS:
        model_record = router_models.get(model_id)
        if not isinstance(model_record, Mapping):
            raise DataError(f"development router model record is malformed: {model_id}")
        selected = model_record.get("selected_params")
        if not isinstance(selected, Mapping) or set(selected) != expected_parameter_names[model_id]:
            raise DataError(f"development selected parameters are malformed: {model_id}")
        _require_finite(
            model_record.get("selected_mean_validation_mse"),
            context=f"development selected validation MSE {model_id}",
        )
        if not isinstance(model_record.get("selected_canonical_params"), str):
            raise DataError(f"development canonical parameters are missing: {model_id}")
        if not isinstance(model_record.get("validation"), Mapping):
            raise DataError(f"development validation details are missing: {model_id}")

    def validate_prediction_rows(value: Any, *, context: str) -> None:
        if not isinstance(value, list) or len(value) != len(expected_ids):
            raise DataError(f"{context} does not contain 600 predictions")
        ids: list[str] = []
        for index, row in enumerate(value):
            if not isinstance(row, Mapping) or set(row) != {"sample_id", "prediction"}:
                raise DataError(f"{context} record {index} is malformed")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in expected_ids:
                raise DataError(f"{context} has an unknown sample ID")
            ids.append(sample_id)
            _require_finite(row.get("prediction"), context=f"{context} {sample_id}")
        if ids != expected_ids:
            raise DataError(f"{context} is not in the exact specified order")

    oof_predictions = report.get("oof_predictions")
    if not isinstance(oof_predictions, Mapping) or set(oof_predictions) != set(MODEL_IDS):
        raise DataError("development OOF prediction identities changed")
    for model_id in MODEL_IDS:
        validate_prediction_rows(oof_predictions[model_id], context=f"development OOF {model_id}")
    oof_rankings = report.get("oof_rankings")
    if not isinstance(oof_rankings, Mapping) or set(oof_rankings) != set(MODEL_IDS):
        raise DataError("development OOF ranking identities changed")
    for model_id in MODEL_IDS:
        _validate_rank_permutation(oof_rankings[model_id], expected_ids, context=f"development OOF ranking {model_id}")

    predictions = report.get("predictions")
    if not isinstance(predictions, Mapping) or set(predictions) != set(MODEL_IDS):
        raise DataError("development fitted prediction identities changed")
    for model_id in MODEL_IDS:
        model_predictions = predictions[model_id]
        if not isinstance(model_predictions, Mapping) or set(model_predictions) != set(expected_ids):
            raise DataError(f"development fitted predictions are incomplete: {model_id}")
        for sample_id in expected_ids:
            _require_finite(model_predictions[sample_id], context=f"development fitted prediction {model_id}:{sample_id}")

    risk = report.get("risk")
    if not isinstance(risk, Mapping) or set(risk) != {"active_features", "inactive_features", "calibration_sample_ids", "populations", "records"}:
        raise DataError("development risk data is malformed")
    if risk["active_features"] != list(settings["active_risk_features"]) or risk["inactive_features"] != list(settings["inactive_risk_features"]):
        raise DataError("development risk feature activity changed")
    expected_calibration_ids = [
        str(item["sample_id"])
        for item in protocol["vietfintab"]["calibration"]
    ]
    if risk["calibration_sample_ids"] != expected_calibration_ids:
        raise DataError("development calibration IDs changed")
    _validate_calibration_populations(risk["populations"])
    risk_records = risk["records"]
    if not isinstance(risk_records, list) or len(risk_records) != len(expected_ids):
        raise DataError("development risk records are incomplete")
    risk_ids: list[str] = []
    for index, row in enumerate(risk_records):
        if not isinstance(row, Mapping):
            raise DataError(f"development risk record {index} is malformed")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise DataError(f"development risk record {index} has an unknown sample ID")
        risk_ids.append(sample_id)
        for name in (
            "confidence_tail_risk",
            "unmatched_ocr_ratio",
            "confidence_tail_risk_percentile",
            "unmatched_ocr_ratio_percentile",
            "combined_risk",
        ):
            _require_finite(row.get(name), context=f"development risk {sample_id}:{name}")
    if risk_ids != expected_ids:
        raise DataError("development risk records are not in the exact specified order")

    rankings = report.get("rankings")
    if not isinstance(rankings, Mapping) or set(rankings) != set(_DEVELOPMENT_ROUTE_METHODS):
        raise DataError("development ranking methods changed")
    for method in _DEVELOPMENT_ROUTE_METHODS:
        _validate_rank_permutation(rankings[method], expected_ids, context=f"development ranking {method}")
    if report.get("budgets") != list(development_budgets) or report.get("main_budget") != settings["development_main_budget"]:
        raise DataError("development routing budgets changed")
    routes = report.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != set(_DEVELOPMENT_ROUTE_METHODS):
        raise DataError("development route methods changed")
    for method in _DEVELOPMENT_ROUTE_METHODS:
        method_routes = routes[method]
        if not isinstance(method_routes, Mapping) or set(method_routes) != {str(item) for item in development_budgets}:
            raise DataError(f"development route budgets are incomplete: {method}")
        for budget in development_budgets:
            route = method_routes[str(budget)]
            if route != rankings[method][:budget] or len(route) != budget or len(set(route)) != budget:
                raise DataError(f"development route changed: {method}:{budget}")

    routing_results = report.get("routing_results")
    if not isinstance(routing_results, Mapping) or set(routing_results) != set(_DEVELOPMENT_ROUTE_METHODS):
        raise DataError("development routing results changed")
    for method in _DEVELOPMENT_ROUTE_METHODS:
        method_results = routing_results[method]
        if not isinstance(method_results, Mapping) or set(method_results) != {str(item) for item in development_budgets}:
            raise DataError(f"development routing result budgets are incomplete: {method}")
        for budget in development_budgets:
            result = method_results[str(budget)]
            if not isinstance(result, Mapping) or result.get("requested_budget") != budget or result.get("expert_call_count") != budget:
                raise DataError(f"development routing result changed: {method}:{budget}")

    random_report = report.get("random")
    if not isinstance(random_report, Mapping) or random_report.get("seed") != settings["random_seed"] or random_report.get("repetitions") != settings["random_repetitions"]:
        raise DataError("development random-routing configuration changed")
    random_budgets = random_report.get("budgets")
    if not isinstance(random_budgets, Mapping) or set(random_budgets) != {str(item) for item in development_budgets}:
        raise DataError("development random-routing budgets are incomplete")
    for budget in development_budgets:
        item = random_budgets[str(budget)]
        if not isinstance(item, Mapping) or item.get("budget") != budget:
            raise DataError(f"development random-routing budget changed: {budget}")

    bootstrap = report.get("bootstrap")
    expected_bootstrap = {
        "hgb_minus_primary",
        "ridge_minus_primary",
        "hgb_minus_ridge",
        "hgb_minus_random",
        "hgb_minus_confidence_tail",
        "hgb_minus_combined_risk",
    }
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != expected_bootstrap:
        raise DataError("development bootstrap comparison identities changed")
    for name, value in bootstrap.items():
        if not isinstance(value, Mapping) or value.get("replicates") != settings["bootstrap_replicates"]:
            raise DataError(f"development bootstrap record is malformed: {name}")
        for field in ("observed", "lower", "upper"):
            _require_finite(value.get(field), context=f"development bootstrap {name}:{field}")

    final_fit = report.get("final_fit")
    if not isinstance(final_fit, Mapping) or final_fit.get("feature_order") != list(FEATURE_NAMES) or final_fit.get("fit_sample_count") != 600 or final_fit.get("fit_is_development_only") is not True or final_fit.get("final_test_used") is not False:
        raise DataError("development final-fit scope changed")
    if not isinstance(final_fit.get("model_paths"), Mapping) or set(final_fit["model_paths"]) != set(MODEL_IDS):
        raise DataError("development final-fit model paths are incomplete")
    if report.get("replacement_policy") != _REPLACEMENT_POLICY:
        raise DataError("development replacement policy changed")
    for forbidden in ("final_labels", "final_test_labels", "final_test_sample_ids"):
        if forbidden in report:
            raise DataError(f"development report contains forbidden final data: {forbidden}")


def _validate_development_source_consistency(
    protocol: Mapping[str, Any],
    report: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    calibration_records: Mapping[str, Mapping[str, Any]],
) -> None:
    """Compare validated source records to the serialized report without refitting."""

    expected_ids = sorted(_protocol_ids(protocol, "development"))
    feature_rows = report["features"]
    for row, sample_id in zip(feature_rows, expected_ids, strict=True):
        if row["values"] != records[sample_id]["features"] or row["issuer"] != records[sample_id]["issuer"]:
            raise DataError(f"development report feature source changed: {sample_id}")
    target_values = {
        row["sample_id"]: row["call_benefit_grits_con"]
        for row in report["targets"]
    }
    for sample_id in expected_ids:
        if float(target_values[sample_id]) != float(records[sample_id]["call_benefit_grits_con"]):
            raise DataError(f"development report target source changed: {sample_id}")
    status_counts = {
        status: count
        for status, count in report["status_counts"]["hunyuan"].items()
    }
    source_counts = {
        status: sum(record["hunyuan_status"] == status for record in records.values())
        for status in sorted({record["hunyuan_status"] for record in records.values()})
    }
    if status_counts != source_counts:
        raise DataError("development Hunyuan status counts do not match source records")
    populations = risk_populations(calibration_records, protocol)
    if populations != report["risk"]["populations"]:
        raise DataError("development risk populations do not match calibration records")


def _evaluation_source_identity(
    source: Mapping[str, Any],
    system: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the semantic identity of the recognition record being scored."""

    if system == "primary":
        identity = source.get("primary_identity")
        if not isinstance(identity, Mapping):
            raise DataError("Primary recognition record lacks semantic identity")
        return {"kind": "primary", "primary_identity": dict(identity)}
    try:
        identity = expert_semantic_identity(protocol, system)
    except RecognitionError as exc:
        raise DataError(str(exc)) from exc
    if source.get("expert_identity") != identity:
        raise DataError("expert recognition record lacks the exact checkpoint identity")
    return {"kind": "expert", **identity}


def _evaluation_cohort_stage(
    dataset: VietFinTabDataset,
    system: str,
    cohort: str,
    output_dir: Path,
) -> None:
    """Score one cohort without initializing a recognition model."""

    if system not in {"primary", *EXPERT_IDS}:
        raise DataError(f"unsupported evaluation system: {system}")
    if cohort == "calibration":
        raise DataError("evaluation is not defined for calibration samples")
    if cohort == "reserve":
        raise DataError("evaluation is not available for reserve samples")
    if system == "glm_ocr" and cohort != "final":
        raise DataError("GLM-OCR evaluation is final-only")
    evaluation_identity = evaluation_semantic_identity(dataset.protocol)
    validate_evaluation_environment(dataset.protocol)
    ids = _cohort_ids(dataset, cohort)
    records = dataset.validate_records()
    evaluation_paths: dict[str, Path] = {}
    sources: dict[str, dict[str, Any]] = {}
    existing_evaluations: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for sample_id in ids:
        source_path = (
            output_dir / "vietfintab" / "primary" / f"{sample_id}.json"
            if system == "primary"
            else output_dir / "vietfintab" / "experts" / system / f"{sample_id}.json"
        )
        evaluation_path = output_dir / "vietfintab" / "evaluation" / system / f"{sample_id}.json"
        evaluation_paths[sample_id] = evaluation_path
        if not source_path.is_file():
            raise DataError(f"evaluation source record is missing: {source_path}")
        sample = records[sample_id]
        source = _read_object(source_path, context="recognition output")
        if system == "primary":
            _validate_existing(
                source_path,
                sample,
                dataset.protocol,
                expected_cohort=cohort,
            )
        else:
            _validate_expert_existing(
                source_path,
                sample,
                system,
                dataset.protocol,
                expected_cohort=cohort,
            )
        if source.get("sample_id") != sample_id or source.get("split") != cohort or source.get("issuer") != sample["issuer"]:
            raise DataError(f"evaluation source provenance does not match the configured protocol: {sample_id}")
        sources[sample_id] = source
        source_identity = _evaluation_source_identity(source, system, dataset.protocol)
        if evaluation_path.is_file():
            existing = _read_object(evaluation_path, context="evaluation record")
            try:
                validate_evaluation_record(
                    existing,
                    sample_id=sample_id,
                    system_id=system,
                    expected_split=cohort,
                    expected_issuer=sample["issuer"],
                    expected_dataset_id=dataset.dataset_id,
                    expected_revision=dataset.revision,
                    expected_source_status=source.get("status"),
                    expected_source_identity=source_identity,
                    expected_evaluation_identity=evaluation_identity,
                )
            except DataError as exc:
                raise DataError(
                    "existing evaluation record is stale; remove it before rerunning: "
                    f"{evaluation_path}"
                ) from exc
            existing_evaluations[sample_id] = existing
        else:
            pending.append(sample_id)
    for sample_id in pending:
        sample = records[sample_id]
        source = sources[sample_id]
        source_identity = _evaluation_source_identity(source, system, dataset.protocol)
        reference = dataset.reference(sample_id)
        result = score_output_record(
            reference,
            source,
            allow_prediction_overlaps=system == "primary",
        )
        output = {
            "sample_id": sample_id,
            "split": cohort,
            "issuer": sample["issuer"],
            "system_id": system,
            "dataset_id": dataset.dataset_id,
            "revision": dataset.revision,
            "source_recognition_status": source["status"],
            "source_identity": source_identity,
            "evaluation_identity": evaluation_identity,
            **result,
        }
        validate_evaluation_record(
            output,
            sample_id=sample_id,
            system_id=system,
            expected_split=cohort,
            expected_issuer=sample["issuer"],
            expected_dataset_id=dataset.dataset_id,
            expected_revision=dataset.revision,
            expected_source_status=source.get("status"),
            expected_source_identity=source_identity,
            expected_evaluation_identity=evaluation_identity,
        )
        _atomic_json_write(evaluation_paths[sample_id], output)
    status_counts: dict[str, int] = {}
    for sample_id in ids:
        value = (
            existing_evaluations[sample_id]
            if sample_id in existing_evaluations
            else _read_object(evaluation_paths[sample_id], context="evaluation record")
        )
        status = str(value["scoring_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    output_root = output_dir / "vietfintab" / "evaluation" / system
    print(
        json.dumps(
            {
                "stage": "evaluate",
                "system": system,
                "cohort": cohort,
                "selected_count": len(ids),
                "resumed": len(ids) - len(pending),
                "newly_scored": len(pending),
                "status_counts": dict(sorted(status_counts.items())),
                "model_initialized": False,
                "output_directory": str(output_root.relative_to(ROOT))
                if output_root.is_relative_to(ROOT)
                else str(output_root),
            },
            indent=2,
        )
    )


def _protocol_ids(protocol: Mapping[str, Any], split: str) -> list[str]:
    entries = protocol.get("vietfintab", {}).get("splits", {}).get(split)
    if not isinstance(entries, list):
        raise DataError(f"protocol has no {split} split")
    return [str(item["sample_id"]) for item in entries]


def _router_source_record(
    path: Path,
    *,
    sample_id: str,
    issuer: str,
    expected_system: str | None = None,
    expected_split: str | None = None,
    expected_dataset_id: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    value = _read_object(path, context="router source record")
    if value.get("sample_id") != sample_id or value.get("issuer") != issuer:
        raise DataError(f"router source provenance does not match {sample_id}: {path}")
    if expected_system is not None and value.get("system_id") != expected_system:
        raise DataError(f"router evaluation has the wrong system ID: {path}")
    if expected_split is not None and value.get("split") != expected_split:
        raise DataError(f"router source has the wrong cohort for {sample_id}: {path}")
    if expected_dataset_id is not None and value.get("dataset_id") != expected_dataset_id:
        raise DataError(f"router source has the wrong dataset for {sample_id}: {path}")
    if expected_revision is not None and value.get("revision") != expected_revision:
        raise DataError(f"router source has the wrong dataset revision for {sample_id}: {path}")
    return value


def _load_router_calibration_records(
    protocol: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Load only Primary feature records for the separate 50-sample calibration cohort."""

    entries = protocol["vietfintab"]["calibration"]
    if not isinstance(entries, list) or len(entries) != 50:
        raise DataError("calibration requires exactly 50 protocol records")
    missing: list[str] = []
    paths: dict[str, Path] = {}
    for item in entries:
        sample_id = str(item["sample_id"])
        path = output_dir / "vietfintab" / "primary" / f"{sample_id}.json"
        paths[sample_id] = path
        if not path.is_file():
            missing.append(sample_id)
    if missing:
        raise DataError(
            "router development inputs are incomplete "
            f"(calibration_primary={len(missing)}); no fitting was started"
        )
    records: dict[str, dict[str, Any]] = {}
    for item in entries:
        sample_id = str(item["sample_id"])
        issuer = str(item["issuer"])
        value = _validate_existing(
            paths[sample_id],
            {"sample_id": sample_id, "issuer": issuer, "spanning": item["spanning"]},
            protocol,
            expected_cohort="calibration",
        )
        if value.get("status") != "success":
            raise DataError(f"calibration Primary record is not successful: {sample_id}")
        features = value.get("features")
        if not isinstance(features, dict) or list(features) != list(FEATURE_NAMES):
            raise DataError(f"calibration features have the wrong order: {sample_id}")
        for name in FEATURE_NAMES:
            feature_value = features[name]
            if feature_value is not None and (
                isinstance(feature_value, bool)
                or not isinstance(feature_value, (int, float))
                or not math.isfinite(float(feature_value))
            ):
                raise DataError(f"calibration feature is not finite: {sample_id}:{name}")
        records[sample_id] = {
            "sample_id": sample_id,
            "issuer": issuer,
            "primary_status": value["status"],
            "features": {name: features[name] for name in FEATURE_NAMES},
        }
    if set(records) != {str(item["sample_id"]) for item in entries}:
        raise DataError("calibration record set does not match the protocol")
    return records


def _load_router_development_records(
    protocol: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    evaluation_identity = evaluation_semantic_identity(protocol)
    validate_evaluation_environment(protocol)
    entries = protocol["vietfintab"]["splits"]["development"]
    if not isinstance(entries, list) or len(entries) != 600:
        raise DataError("router development requires exactly 600 protocol IDs")
    missing = {"primary": [], "hunyuan": [], "primary_evaluation": [], "hunyuan_evaluation": []}
    for item in entries:
        sample_id = str(item["sample_id"])
        primary_path = output_dir / "vietfintab" / "primary" / f"{sample_id}.json"
        expert_path = output_dir / "vietfintab" / "experts" / "hunyuanocr_1_5" / f"{sample_id}.json"
        primary_eval_path = output_dir / "vietfintab" / "evaluation" / "primary" / f"{sample_id}.json"
        expert_eval_path = output_dir / "vietfintab" / "evaluation" / "hunyuanocr_1_5" / f"{sample_id}.json"
        for key, path in (("primary", primary_path), ("hunyuan", expert_path), ("primary_evaluation", primary_eval_path), ("hunyuan_evaluation", expert_eval_path)):
            if not path.is_file():
                missing[key].append(sample_id)
    if any(missing.values()):
        detail = ", ".join(f"{key}={len(values)}" for key, values in missing.items())
        raise DataError(f"router development inputs are incomplete ({detail}); no fitting was started")
    records: dict[str, dict[str, Any]] = {}
    for item in entries:
        sample_id = str(item["sample_id"])
        issuer = str(item["issuer"])
        sample = {
            "sample_id": sample_id,
            "issuer": issuer,
            "spanning": item["spanning"],
        }
        primary = _validate_existing(
            output_dir / "vietfintab" / "primary" / f"{sample_id}.json",
            sample,
            protocol,
            expected_cohort="development",
        )
        expert = _validate_expert_existing(
            output_dir / "vietfintab" / "experts" / "hunyuanocr_1_5" / f"{sample_id}.json",
            sample,
            "hunyuanocr_1_5",
            protocol,
            expected_cohort="development",
        )
        primary_eval_path = output_dir / "vietfintab" / "evaluation" / "primary" / f"{sample_id}.json"
        primary_eval = _read_object(primary_eval_path, context="router Primary evaluation")
        validate_evaluation_record(
            primary_eval,
            sample_id=sample_id,
            system_id="primary",
            expected_split="development",
            expected_issuer=issuer,
            expected_dataset_id=str(protocol["dataset"]["id"]),
            expected_revision=str(protocol["dataset"]["revision"]),
            expected_source_status=primary.get("status"),
            expected_source_identity=_evaluation_source_identity(primary, "primary", protocol),
            expected_evaluation_identity=evaluation_identity,
        )
        expert_eval_path = output_dir / "vietfintab" / "evaluation" / "hunyuanocr_1_5" / f"{sample_id}.json"
        expert_eval = _read_object(expert_eval_path, context="router Hunyuan evaluation")
        validate_evaluation_record(
            expert_eval,
            sample_id=sample_id,
            system_id="hunyuanocr_1_5",
            expected_split="development",
            expected_issuer=issuer,
            expected_dataset_id=str(protocol["dataset"]["id"]),
            expected_revision=str(protocol["dataset"]["revision"]),
            expected_source_status=expert.get("status"),
            expected_source_identity=_evaluation_source_identity(expert, "hunyuanocr_1_5", protocol),
            expected_evaluation_identity=evaluation_identity,
        )
        if primary.get("status") != "success" or primary_eval.get("source_status") != "success":
            raise DataError(f"development primary record is not a successful scored record: {sample_id}")
        features = primary.get("features")
        if not isinstance(features, dict) or list(features) != list(FEATURE_NAMES):
            raise DataError(f"development primary features have the wrong order: {sample_id}")
        for name in FEATURE_NAMES:
            value = features[name]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                raise DataError(f"development feature is not finite: {sample_id}:{name}")
        primary_con = primary_eval.get("grits_con")
        primary_top = primary_eval.get("grits_top")
        if isinstance(primary_con, bool) or not isinstance(primary_con, (int, float)) or not math.isfinite(float(primary_con)):
            raise DataError(f"development Primary GriTS-Con is not finite: {sample_id}")
        if isinstance(primary_top, bool) or not isinstance(primary_top, (int, float)) or not math.isfinite(float(primary_top)):
            raise DataError(f"development Primary GriTS-Top is not finite: {sample_id}")
        expert_status = expert.get("status")
        allowed_statuses = {"success", "parse_failed", "inference_failed", "inference_timeout", "out_of_memory", "worker_failed"}
        if expert_status not in allowed_statuses:
            raise DataError(f"development expert record has no terminal status: {sample_id}")
        if expert_eval.get("source_status") != expert_status:
            raise DataError(f"expert evaluation status does not match recognition status: {sample_id}")
        expert_success = expert_status == "success"
        expert_con = expert_eval.get("grits_con") if expert_success else None
        expert_top = expert_eval.get("grits_top") if expert_success else None
        if expert_success:
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in (expert_con, expert_top)):
                raise DataError(f"successful expert evaluation lacks finite headline scores: {sample_id}")
        elif expert_eval.get("grits_con") is not None or expert_eval.get("grits_top") is not None:
            raise DataError(f"failed expert has an actual score: {sample_id}")
        records[sample_id] = {
            "sample_id": sample_id,
            "issuer": issuer,
            "primary_status": primary["status"],
            "primary_grits_con": float(primary_con),
            "primary_grits_top": float(primary_top),
            "hunyuan_status": expert_status,
            "hunyuan_success": expert_success,
            "hunyuan_standalone_grits_con": float(expert_con) if expert_success else 0.0,
            "hunyuan_standalone_grits_top": float(expert_top) if expert_success else 0.0,
            "hunyuan_inference_seconds": expert.get("inference_elapsed_seconds", 0.0),
            "features": {name: features[name] for name in FEATURE_NAMES},
        }
        from bear.routing import call_benefit_grits_con

        records[sample_id]["call_benefit_grits_con"] = call_benefit_grits_con(
            float(primary_con),
            float(expert_con) if expert_success else None,
            expert_success,
        )
    if set(records) != {str(item["sample_id"]) for item in entries}:
        raise DataError("router development record set does not match the protocol")
    return records


def _validate_router_development_inputs_present(
    protocol: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Report missing router inputs by cohort and system before any fitting."""

    missing: dict[str, list[str]] = {
        "calibration_primary": [],
        "development_primary": [],
        "development_hunyuan": [],
        "development_primary_evaluation": [],
        "development_hunyuan_evaluation": [],
    }
    calibration_entries = protocol["vietfintab"]["calibration"]
    for item in calibration_entries:
        sample_id = str(item["sample_id"])
        path = output_dir / "vietfintab" / "primary" / f"{sample_id}.json"
        if not path.is_file():
            missing["calibration_primary"].append(sample_id)
    development_entries = protocol["vietfintab"]["splits"]["development"]
    for item in development_entries:
        sample_id = str(item["sample_id"])
        paths = {
            "development_primary": output_dir / "vietfintab" / "primary" / f"{sample_id}.json",
            "development_hunyuan": output_dir / "vietfintab" / "experts" / "hunyuanocr_1_5" / f"{sample_id}.json",
            "development_primary_evaluation": output_dir / "vietfintab" / "evaluation" / "primary" / f"{sample_id}.json",
            "development_hunyuan_evaluation": output_dir / "vietfintab" / "evaluation" / "hunyuanocr_1_5" / f"{sample_id}.json",
        }
        for name, path in paths.items():
            if not path.is_file():
                missing[name].append(sample_id)
    if any(missing.values()):
        detail = ", ".join(f"{name}={len(values)}" for name, values in missing.items())
        raise DataError(f"router development inputs are incomplete ({detail}); no fitting was started")


def _router_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for distribution in ("numpy", "scipy", "scikit-learn", "joblib"):
        try:
            values[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            values[distribution] = "unavailable"
    return values


def _validate_router_artifact(path: Path, model_id: str, selected_params: Mapping[str, Any]) -> Any:
    try:
        import joblib
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = joblib.load(path)
    except Exception as exc:
        raise DataError(f"could not load router artifact {path}; rerun with --force") from exc
    if type(model) is not Pipeline or not hasattr(model, "predict"):
        raise DataError(f"router artifact is not a fitted pipeline: {path}")
    if model_id not in MODEL_IDS or not isinstance(selected_params, Mapping):
        raise DataError(f"router artifact has an unsupported model identity: {path}")
    expected_steps = ["imputer", "scaler", "model"] if model_id == "ridge_benefit" else ["imputer", "model"]
    if list(model.named_steps) != expected_steps:
        raise DataError(f"router artifact has unexpected steps: {path}")
    imputer = model.named_steps["imputer"]
    if type(imputer) is not SimpleImputer or imputer.strategy != "median":
        raise DataError(f"router artifact does not use median imputation: {path}")
    estimator = model.named_steps["model"]
    if model_id == "ridge_benefit":
        scaler = model.named_steps["scaler"]
        if type(scaler) is not StandardScaler:
            raise DataError(f"Ridge router artifact does not use StandardScaler: {path}")
        if type(estimator) is not Ridge:
            raise DataError(f"Ridge router artifact has an unexpected estimator: {path}")
        expected_parameter_names = {"alpha"}
    else:
        if type(estimator) is not HistGradientBoostingRegressor:
            raise DataError(f"HGB router artifact has an unexpected estimator: {path}")
        expected_parameter_names = set(selected_params)
    if set(selected_params) != expected_parameter_names:
        raise DataError(f"router artifact selected parameters are incomplete: {path}")
    parameters = estimator.get_params()
    for key, expected in selected_params.items():
        if parameters.get(key) != expected:
            raise DataError(f"router artifact selected parameters changed: {path}")
    for step in model.named_steps.values():
        count = getattr(step, "n_features_in_", None)
        if count is not None and int(count) != len(FEATURE_NAMES):
            raise DataError(f"router artifact does not have 13 input features: {path}")
    count = getattr(model, "n_features_in_", None)
    if count is not None and int(count) != len(FEATURE_NAMES):
        raise DataError(f"router artifact does not have 13 input features: {path}")
    return model


def _model_prediction_map(model: Any, records: Mapping[str, Mapping[str, Any]], sample_ids: list[str]) -> dict[str, float]:
    feature_records = {
        sample_id: records[sample_id]["features"]
        for sample_id in sample_ids
    }
    matrix = _matrix(feature_records, sample_ids)
    try:
        prediction = model.predict(matrix)
    except Exception as exc:
        raise DataError("router prediction failed for the configured feature record") from exc
    result: dict[str, float] = {}
    for sample_id, value in zip(sample_ids, prediction, strict=True):
        number = float(value)
        if not math.isfinite(number):
            raise DataError(f"router prediction is not finite: {sample_id}")
        result[sample_id] = number
    return result


def _final_model_prediction_map(
    model: Any,
    records: Mapping[str, Mapping[str, Any]],
    sample_ids: list[str],
) -> dict[str, float]:
    """Apply a fitted router to each table's feature vector."""

    result: dict[str, float] = {}
    for sample_id in sample_ids:
        matrix = _matrix(
            {sample_id: records[sample_id]["features"]},
            [sample_id],
        )
        try:
            prediction = model.predict(matrix)
        except Exception as exc:
            raise DataError("final router prediction failed for the configured feature record") from exc
        if len(prediction) != 1:
            raise DataError(f"final router prediction returned an unexpected shape: {sample_id}")
        number = float(prediction[0])
        if not math.isfinite(number):
            raise DataError(f"final router prediction is not finite: {sample_id}")
        result[sample_id] = number
    return result


def validate_router_artifact_predictions(
    model: Any,
    model_id: str,
    development_records: Mapping[str, Mapping[str, Any]],
    development_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a fitted router artifact to the report's 600 fit-on-all predictions."""

    sample_ids = sorted(str(item) for item in development_records)
    if len(sample_ids) != 600 or len(set(sample_ids)) != 600:
        raise DataError(f"{model_id} artifact binding requires exactly 600 development records")
    if development_report.get("sample_ids") != sample_ids:
        raise DataError(f"{model_id} artifact binding sample IDs do not match the development report")
    if development_report.get("feature_order") != list(FEATURE_NAMES):
        raise DataError(f"{model_id} artifact binding feature order changed")
    feature_rows = development_report.get("features")
    if not isinstance(feature_rows, list) or len(feature_rows) != len(sample_ids):
        raise DataError(f"{model_id} artifact binding lacks the serialized development features")
    for sample_id, row in zip(sample_ids, feature_rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("sample_id") != sample_id
            or row.get("issuer") != development_records[sample_id]["issuer"]
            or row.get("values") != development_records[sample_id]["features"]
        ):
            raise DataError(f"{model_id} artifact binding source features changed: {sample_id}")

    if not isinstance(development_report.get("predictions"), Mapping):
        raise DataError(f"{model_id} artifact binding lacks the fitted prediction mapping")
    stored_predictions = development_report["predictions"].get(model_id)
    if not isinstance(stored_predictions, Mapping) or list(stored_predictions) != sample_ids:
        raise DataError(f"{model_id} artifact binding lacks the ordered fitted predictions")
    stored: dict[str, float] = {}
    for sample_id in sample_ids:
        stored[sample_id] = _require_finite(
            stored_predictions[sample_id],
            context=f"{model_id} stored development prediction {sample_id}",
        )

    recomputed = _model_prediction_map(model, development_records, sample_ids)
    differences = {
        sample_id: abs(recomputed[sample_id] - stored[sample_id])
        for sample_id in sample_ids
    }
    maximum_difference = max(differences.values())
    first_difference = next(
        (sample_id for sample_id in sample_ids if not math.isclose(
            recomputed[sample_id],
            stored[sample_id],
            rel_tol=0.0,
            abs_tol=1e-15,
        )),
        None,
    )
    if first_difference is not None:
        raise DataError(
            f"{model_id} artifact predictions do not match the validated development report "
            f"at {first_difference}"
        )
    stored_ranking = rank_by_predicted_benefit(sample_ids, stored)
    recomputed_ranking = rank_by_predicted_benefit(sample_ids, recomputed)
    if recomputed_ranking != stored_ranking:
        raise DataError(f"{model_id} artifact ranking differs from the development report")
    return {
        "model_id": model_id,
        "sample_count": len(sample_ids),
        "exact_equal_count": sum(recomputed[sample_id] == stored[sample_id] for sample_id in sample_ids),
        "maximum_absolute_difference": maximum_difference,
        "first_differing_sample": first_difference,
        "first_stored_value": stored[first_difference] if first_difference is not None else None,
        "first_recomputed_value": recomputed[first_difference] if first_difference is not None else None,
        "ranking_matches": True,
    }


def _load_and_bind_router_artifacts(
    models_dir: Path,
    development_records: Mapping[str, Mapping[str, Any]],
    development_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load structurally valid routers and bind them to the development report."""

    models: dict[str, Any] = {}
    binding: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        selected = development_report["router_models"][model_id]["selected_params"]
        model = _validate_router_artifact(
            models_dir / "router" / f"{model_id}.joblib",
            model_id,
            selected,
        )
        binding[model_id] = validate_router_artifact_predictions(
            model,
            model_id,
            development_records,
            development_report,
        )
        models[model_id] = model
    return models, binding


def build_final_route_plan(
    protocol: Mapping[str, Any],
    final_primary_records: Mapping[str, Mapping[str, Any]],
    router_models: Mapping[str, Any],
    calibration_populations: Mapping[str, list[float]],
) -> dict[str, Any]:
    """Build the final route from Primary features and specified router inputs only."""

    settings = routing_settings(protocol)
    final_budgets = settings["final_budgets"]
    final_main_budget = settings["final_main_budget"]

    expected_ids = sorted(
        str(item["sample_id"])
        for item in protocol["vietfintab"]["splits"]["final"]
    )
    sample_ids = sorted(str(item) for item in final_primary_records)
    if sample_ids != expected_ids or len(sample_ids) != 300:
        raise DataError("final route construction requires the exact 300 final Primary records")
    if set(router_models) != set(MODEL_IDS):
        raise DataError("final route construction requires exactly the specified Ridge and HGB models")
    _validate_calibration_populations(calibration_populations)
    expected_issuers = {
        str(item["sample_id"]): str(item["issuer"])
        for item in protocol["vietfintab"]["splits"]["final"]
    }
    for sample_id in sample_ids:
        record = final_primary_records[sample_id]
        if not isinstance(record, Mapping):
            raise DataError(f"final route Primary record is not an object: {sample_id}")
        if (
            record.get("sample_id", sample_id) != sample_id
            or record.get("issuer") != expected_issuers[sample_id]
            or record.get("primary_status") != "success"
        ):
            raise DataError(f"final route Primary provenance is invalid: {sample_id}")
        _validate_feature_mapping(
            record.get("features"),
            context=f"final route Primary features {sample_id}",
        )

    feature_records = {
        sample_id: final_primary_records[sample_id]["features"]
        for sample_id in sample_ids
    }
    predictions = {
        model_id: _final_model_prediction_map(router_models[model_id], final_primary_records, sample_ids)
        for model_id in MODEL_IDS
    }
    risk_records, risk_rankings = build_risk_rankings(
        sample_ids,
        feature_records,
        calibration_populations,
        protocol,
    )
    rankings = {
        "confidence_tail_only": risk_rankings["confidence_tail_only"],
        "combined_risk": risk_rankings["combined_risk"],
        "ridge_benefit": rank_by_predicted_benefit(
            sample_ids,
            predictions["ridge_benefit"],
        ),
        "hist_gradient_boosting_benefit": rank_by_predicted_benefit(
            sample_ids,
            predictions["hist_gradient_boosting_benefit"],
        ),
    }
    main_routes = {
        "always_primary": [],
        **{
            name: route_for_budget(ranking, final_main_budget, final_budgets)
            for name, ranking in rankings.items()
        },
    }
    random_route = deterministic_random_route(
        sample_ids,
        final_main_budget,
        0,
        seed=settings["random_seed"],
        allowed_budgets=final_budgets,
    )
    main_routes["random_matched_budget"] = random_route
    return {
        "run_status": "ready",
        "dataset_id": protocol["dataset"]["id"],
        "revision": protocol["dataset"]["revision"],
        "final_test_sample_ids": sample_ids,
        "feature_order": list(FEATURE_NAMES),
        "primary_status": {
            item: final_primary_records[item]["primary_status"]
            for item in sample_ids
        },
        "features": [
            {
                "sample_id": item,
                "issuer": final_primary_records[item]["issuer"],
                "values": final_primary_records[item]["features"],
            }
            for item in sample_ids
        ],
        "router_predictions": predictions,
        "risk_features": risk_records,
        "active_risk_features": list(settings["active_risk_features"]),
        "inactive_risk_features": list(settings["inactive_risk_features"]),
        "risk_populations": calibration_populations,
        "percentile_rule": settings["percentile_rule"],
        "rankings": rankings,
        "ranking_rule": "descending score, then ascending sample_id",
        "main_budget": final_main_budget,
        "diagnostic_budgets": list(final_budgets),
        "main_20_percent_routes": main_routes,
        "random_routing": {
            "seed": settings["random_seed"],
            "repetition": 0,
            "budget": final_main_budget,
            "route": random_route,
            "rule": protocol["routing"]["random"]["selection"],
        },
        "final_route_transfers_to": "glm_ocr",
    }


def validate_final_route_plan(
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    final_primary_records: Mapping[str, Mapping[str, Any]],
    router_models: Mapping[str, Any],
    calibration_populations: Mapping[str, list[float]],
) -> None:
    """Reject any final route plan that differs from a fresh pure reconstruction."""

    if not isinstance(plan, Mapping):
        raise DataError("final route plan must be an object")
    expected = build_final_route_plan(
        protocol,
        final_primary_records,
        router_models,
        calibration_populations,
    )
    if dict(plan) != expected:
        raise DataError("final route plan differs from the deterministic Primary-only reconstruction")


def _development_router_report(
    protocol: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    calibration_records: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    models_dir: Path,
    force: bool,
) -> dict[str, Any]:
    settings = routing_settings(protocol)
    development_budgets = settings["development_budgets"]
    development_main_budget = settings["development_main_budget"]
    if len(records) != 600:
        raise DataError("development router report requires 600 records")
    sample_ids = sorted(records)
    groups = {sample_id: str(records[sample_id]["issuer"]) for sample_id in sample_ids}
    features = {sample_id: records[sample_id]["features"] for sample_id in sample_ids}
    targets = {sample_id: float(records[sample_id]["call_benefit_grits_con"]) for sample_id in sample_ids}
    specs = _model_specs(protocol)
    oof_predictions: dict[str, dict[str, float]] = {}
    validation: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        try:
            predictions, details = nested_oof_predictions(
                model_id,
                sample_ids=sample_ids,
                feature_records=features,
                targets=targets,
                groups=groups,
                specs=specs,
                outer_folds=settings["outer_folds"],
                inner_folds=settings["inner_folds"],
            )
        except RoutingError as exc:
            raise DataError(str(exc)) from exc
        oof_predictions[model_id] = predictions
        validation[model_id] = details
    try:
        models, tunings = fit_final_router_families(
            sample_ids=sample_ids,
            feature_records=features,
            targets=targets,
            groups=groups,
            specs=specs,
            folds=settings["inner_folds"],
            models_dir=models_dir / "router",
            force=force,
        )
    except RoutingError as exc:
        raise DataError(str(exc)) from exc
    predictions = {model_id: _model_prediction_map(models[model_id], records, sample_ids) for model_id in MODEL_IDS}
    populations = risk_populations(calibration_records, protocol)
    risk_records, risk_rankings = build_risk_rankings(sample_ids, features, populations, protocol)
    rankings = {
        "confidence_tail_only": risk_rankings["confidence_tail_only"],
        "combined_risk": risk_rankings["combined_risk"],
        # Development routing is evaluated with one nested-OOF prediction per
        # sample. The fit-on-all-600 predictions remain diagnostic only; final
        # routing uses those fitted models in _router_final_stage.
        "ridge_benefit": rank_by_predicted_benefit(sample_ids, oof_predictions["ridge_benefit"]),
        "hist_gradient_boosting_benefit": rank_by_predicted_benefit(sample_ids, oof_predictions["hist_gradient_boosting_benefit"]),
        "oracle_benefit": rank_by_predicted_benefit(sample_ids, {sample_id: records[sample_id]["call_benefit_grits_con"] for sample_id in sample_ids}),
    }
    route_methods = {
        "confidence_tail_only": rankings["confidence_tail_only"],
        "combined_risk": rankings["combined_risk"],
        "ridge_benefit": rankings["ridge_benefit"],
        "hist_gradient_boosting_benefit": rankings["hist_gradient_boosting_benefit"],
        "oracle_benefit": rankings["oracle_benefit"],
    }
    route_results: dict[str, dict[str, Any]] = {}
    for method, ranking in route_methods.items():
        route_results[method] = {}
        for budget in development_budgets:
            route = route_for_budget(ranking, budget, development_budgets)
            route_results[method][str(budget)] = evaluate_route(
                records,
                route,
                budget=budget,
                routing_method=method,
                allowed_budgets=development_budgets,
            )
    random_report, random_vector = random_summary(
        records,
        budgets=development_budgets,
        main_budget=development_main_budget,
        repetitions=settings["random_repetitions"],
        seed=settings["random_seed"],
    )
    main_hgb = route_score_vector(records, route_for_budget(rankings["hist_gradient_boosting_benefit"], development_main_budget, development_budgets))[0]
    main_ridge = route_score_vector(records, route_for_budget(rankings["ridge_benefit"], development_main_budget, development_budgets))[0]
    main_conf = route_score_vector(records, route_for_budget(rankings["confidence_tail_only"], development_main_budget, development_budgets))[0]
    main_combined = route_score_vector(records, route_for_budget(rankings["combined_risk"], development_main_budget, development_budgets))[0]
    primary_vector = [float(records[item]["primary_grits_con"]) for item in sample_ids]
    bootstrap = {
        "hgb_minus_primary": paired_bootstrap_mean_difference(main_hgb, primary_vector, seed=f"{settings['bootstrap_seed']}:hist_gradient_boosting_benefit:always_primary", replicates=settings["bootstrap_replicates"]),
        "ridge_minus_primary": paired_bootstrap_mean_difference(main_ridge, primary_vector, seed=f"{settings['bootstrap_seed']}:ridge_benefit:always_primary", replicates=settings["bootstrap_replicates"]),
        "hgb_minus_ridge": paired_bootstrap_mean_difference(main_hgb, main_ridge, seed=f"{settings['bootstrap_seed']}:hist_gradient_boosting_minus_ridge", replicates=settings["bootstrap_replicates"]),
        "hgb_minus_random": paired_bootstrap_mean_difference(main_hgb, random_vector or primary_vector, seed=f"{settings['bootstrap_seed']}:hist_gradient_boosting_benefit:random_mean", replicates=settings["bootstrap_replicates"]),
        "hgb_minus_confidence_tail": paired_bootstrap_mean_difference(main_hgb, main_conf, seed=f"{settings['bootstrap_seed']}:hist_gradient_boosting_benefit:confidence_tail_only", replicates=settings["bootstrap_replicates"]),
        "hgb_minus_combined_risk": paired_bootstrap_mean_difference(main_hgb, main_combined, seed=f"{settings['bootstrap_seed']}:hist_gradient_boosting_benefit:combined_risk", replicates=settings["bootstrap_replicates"]),
    }
    quality = {
        method: {
            str(budget): {
                "budget": budget,
                "hybrid_full_set_grits_con": route_results[method][str(budget)]["hybrid_full_set_grits_con"],
                "hybrid_full_set_grits_top": route_results[method][str(budget)]["hybrid_full_set_grits_top"],
            }
            for budget in development_budgets
        }
        for method in route_methods
    }
    report = {
        "run_status": "success",
        "dataset_id": protocol["dataset"]["id"],
        "pinned_revision": protocol["dataset"]["revision"],
        "sample_ids": sample_ids,
        "feature_order": list(FEATURE_NAMES),
        "features": [{"sample_id": item, "issuer": records[item]["issuer"], "values": records[item]["features"]} for item in sample_ids],
        "targets": [{"sample_id": item, "issuer": records[item]["issuer"], "call_benefit_grits_con": records[item]["call_benefit_grits_con"]} for item in sample_ids],
        "status_counts": {
            "primary": {"success": sum(records[item]["primary_status"] == "success" for item in sample_ids)},
            "hunyuan": {status: sum(records[item]["hunyuan_status"] == status for item in sample_ids) for status in sorted({records[item]["hunyuan_status"] for item in sample_ids})},
        },
        "router_models": {
            model_id: {
                "selected_params": tunings[model_id]["selected_params"],
                "selected_canonical_params": tunings[model_id]["selected_canonical_params"],
                "selected_mean_validation_mse": tunings[model_id]["selected_mean_validation_mse"],
                "validation": validation[model_id],
            }
            for model_id in MODEL_IDS
        },
        "oof_predictions": {model_id: [{"sample_id": item, "prediction": oof_predictions[model_id][item]} for item in sample_ids] for model_id in MODEL_IDS},
        "oof_rankings": {model_id: rank_by_predicted_benefit(sample_ids, oof_predictions[model_id]) for model_id in MODEL_IDS},
        "final_fit": {
            "feature_order": list(FEATURE_NAMES),
            "fit_sample_count": 600,
            "fit_is_development_only": True,
            "final_test_used": False,
            "software_versions": _router_versions(),
            "model_paths": {model_id: f"models/router/{model_id}.joblib" for model_id in MODEL_IDS},
        },
        "predictions": {model_id: predictions[model_id] for model_id in MODEL_IDS},
        "risk": {
            "active_features": list(settings["active_risk_features"]),
            "inactive_features": list(settings["inactive_risk_features"]),
            "calibration_sample_ids": [
                str(item["sample_id"])
                for item in protocol["vietfintab"]["calibration"]
            ],
            "populations": populations,
            "records": risk_records,
        },
        "rankings": rankings,
        "main_budget": development_main_budget,
        "budgets": list(development_budgets),
        "routes": {method: {str(budget): route_for_budget(ranking, budget, development_budgets) for budget in development_budgets} for method, ranking in route_methods.items()},
        "routing_results": route_results,
        "random": random_report,
        "bootstrap": bootstrap,
        "quality_budget_curves": quality,
        "standalone": {
            "primary": standalone_summary(records, system="primary"),
            "hunyuan": standalone_summary(records, system="hunyuan"),
        },
        "replacement_policy": "successful Hunyuan output replaces the primary; expert failure retains primary; no post-expert quality gate",
    }
    return report


def _router_development_stage(protocol: Mapping[str, Any], output_dir: Path, models_dir: Path, force: bool) -> None:
    settings = routing_settings(protocol)
    output_path = output_dir / "vietfintab" / "router" / "development.json"
    calibration_path = output_dir / "vietfintab" / "router" / "calibration.json"
    _validate_router_development_inputs_present(protocol, output_dir)
    if output_path.is_file() and not force:
        existing = _read_object(output_path, context="router development report")
        _validate_development_report_shape(protocol, existing)
        calibration = _read_object(calibration_path, context="calibration artifact")
        existing_populations = existing["risk"]["populations"]
        _validate_calibration_artifact(
            protocol,
            calibration,
            expected_populations=existing_populations,
        )
        calibration_records = _load_router_calibration_records(protocol, output_dir)
        records = _load_router_development_records(protocol, output_dir)
        try:
            recomputed_populations = risk_populations(calibration_records, protocol)
        except RoutingError as exc:
            raise DataError(str(exc)) from exc
        if recomputed_populations != existing_populations:
            raise DataError("existing calibration populations are invalid; use --force")
        _validate_development_source_consistency(
            protocol,
            existing,
            records,
            calibration_records,
        )
        _models, binding = _load_and_bind_router_artifacts(
            models_dir,
            records,
            existing,
        )
        fit_difficulty_router(
            sample_ids=sorted(records),
            feature_records={sample_id: records[sample_id]["features"] for sample_id in records},
            targets={sample_id: 1.0 - float(records[sample_id]["primary_grits_con"]) for sample_id in records},
            models_dir=models_dir / "router",
            force=False,
        )
        print(json.dumps({"stage": "router", "split": "development", "resumed": True, "sample_count": 600, "artifact_binding": binding, "difficulty_control": DIFFICULTY_MODEL_FILENAME}, indent=2))
        return
    calibration_records = _load_router_calibration_records(protocol, output_dir)
    records = _load_router_development_records(protocol, output_dir)
    report = _development_router_report(protocol, records, calibration_records, output_dir, models_dir, force)
    _models, binding = _load_and_bind_router_artifacts(models_dir, records, report)
    calibration = {
        "cohort": "calibration",
        "sample_ids": [str(item["sample_id"]) for item in protocol["vietfintab"]["calibration"]],
        "active_features": list(settings["active_risk_features"]),
        "inactive_features": list(settings["inactive_risk_features"]),
        "raw_populations": report["risk"]["populations"],
        "percentile_rule": settings["percentile_rule"],
        "primary_configuration_id": protocol["primary"]["configuration_id"],
        "primary_name": protocol["primary"]["name"],
    }
    _atomic_json_write(calibration_path, calibration)
    _atomic_json_write(output_path, report)
    fit_difficulty_router(
        sample_ids=sorted(records),
        feature_records={sample_id: records[sample_id]["features"] for sample_id in records},
        targets={sample_id: 1.0 - float(records[sample_id]["primary_grits_con"]) for sample_id in records},
        models_dir=models_dir / "router",
        force=force,
    )
    print(json.dumps({"stage": "router", "split": "development", "resumed": False, "sample_count": 600, "model_paths": report["final_fit"]["model_paths"], "difficulty_control": DIFFICULTY_MODEL_FILENAME, "artifact_binding": binding}, indent=2))


def _router_final_primary_records(protocol: Mapping[str, Any], output_dir: Path) -> dict[str, dict[str, Any]]:
    entries = protocol["vietfintab"]["splits"]["final"]
    sample_ids = sorted(str(item["sample_id"]) for item in entries)
    missing: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    for item in entries:
        sample_id = str(item["sample_id"])
        path = output_dir / "vietfintab" / "primary" / f"{sample_id}.json"
        if not path.is_file():
            missing.append(sample_id)
            continue
        value = _validate_existing(
            path,
            {
                "sample_id": sample_id,
                "issuer": item["issuer"],
                "spanning": item["spanning"],
            },
            protocol,
            expected_cohort="final",
        )
        if value.get("status") != "success":
            raise DataError(f"final primary record is not successful: {sample_id}")
        features = value.get("features")
        if not isinstance(features, dict) or list(features) != list(FEATURE_NAMES):
            raise DataError(f"final primary feature order is invalid: {sample_id}")
        for name in FEATURE_NAMES:
            number = features[name]
            if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number))):
                raise DataError(f"final primary feature is not finite: {sample_id}:{name}")
        records[sample_id] = {"sample_id": sample_id, "issuer": str(item["issuer"]), "primary_status": value["status"], "features": {name: features[name] for name in FEATURE_NAMES}}
    if missing:
        raise DataError(f"router final inputs are incomplete (primary={len(missing)}); no expert or reference paths were inspected")
    if set(records) != set(sample_ids) or len(records) != 300:
        raise DataError("router final requires exactly 300 Primary records")
    return records


def _reject_pre_route_final_outputs(
    output_dir: Path,
    final_sample_ids: set[str],
) -> None:
    """Reject only final-ID JSON filenames before the route manifest exists.

    The check intentionally examines path names only. Development expert and
    evaluation records may remain beside the inputs needed for router fitting.
    """

    for directory in (
        output_dir / "vietfintab" / "experts",
        output_dir / "vietfintab" / "evaluation",
    ):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.json"):
            if path.stem in final_sample_ids:
                raise DataError(
                    "final route creation found a final-ID output before the route manifest "
                    f"(filename only): {path.name}"
                )


def _router_final_stage(protocol: Mapping[str, Any], output_dir: Path, models_dir: Path, force: bool) -> None:
    settings = routing_settings(protocol)
    router_root = output_dir / "vietfintab" / "router"
    route_path = router_root / "final_routes.json"
    final_sample_ids = set(_protocol_ids(protocol, "final"))
    if route_path.is_file():
        if force:
            raise DataError("an existing final route plan cannot be overwritten; use a clean output directory")
        existing = _read_object(route_path, context="final route plan")
        development_path = router_root / "development.json"
        calibration_path = router_root / "calibration.json"
        if not development_path.is_file():
            raise DataError("final route resumption requires the development router report")
        development = _read_object(development_path, context="development router report")
        _validate_development_report_shape(protocol, development)
        development_records = _load_router_development_records(protocol, output_dir)
        calibration_records = _load_router_calibration_records(protocol, output_dir)
        _validate_development_source_consistency(
            protocol,
            development,
            development_records,
            calibration_records,
        )
        if not calibration_path.is_file():
            raise DataError("final route resumption requires the calibration artifact")
        calibration = _read_object(calibration_path, context="calibration artifact")
        populations = _validate_calibration_artifact(protocol, calibration)
        final_primary_records = _router_final_primary_records(protocol, output_dir)
        models, binding = _load_and_bind_router_artifacts(
            models_dir,
            development_records,
            development,
        )
        load_difficulty_router(models_dir / "router")
        validate_final_route_plan(
            existing,
            protocol,
            final_primary_records,
            models,
            populations,
        )
        print(json.dumps({"stage": "router", "split": "final", "resumed": True, "sample_count": 300, "artifact_binding": binding}, indent=2))
        return
    # Before the route manifest exists, inspect only names. Development outputs
    # are permitted; a final-ID file indicates a stale pre-route result.
    _reject_pre_route_final_outputs(output_dir, final_sample_ids)
    development_path = output_dir / "vietfintab" / "router" / "development.json"
    calibration_path = output_dir / "vietfintab" / "router" / "calibration.json"
    if not development_path.is_file():
        raise DataError("final router artifacts are missing; run the development router split first")
    development = _read_object(development_path, context="development router report")
    _validate_development_report_shape(protocol, development)
    development_records = _load_router_development_records(protocol, output_dir)
    calibration_records = _load_router_calibration_records(protocol, output_dir)
    _validate_development_source_consistency(
        protocol,
        development,
        development_records,
        calibration_records,
    )
    models, binding = _load_and_bind_router_artifacts(
        models_dir,
        development_records,
        development,
    )
    load_difficulty_router(models_dir / "router")
    if not calibration_path.is_file():
        raise DataError("final router artifacts are missing the calibration artifact")
    calibration = _read_object(calibration_path, context="calibration artifact")
    populations = _validate_calibration_artifact(protocol, calibration)
    final_primary_records = _router_final_primary_records(protocol, output_dir)
    route = build_final_route_plan(
        protocol,
        final_primary_records,
        models,
        populations,
    )
    _atomic_json_write(route_path, route)
    print(json.dumps({"stage": "router", "split": "final", "resumed": False, "sample_count": 300, "route_count": settings["final_main_budget"], "path": str(route_path), "artifact_binding": binding}, indent=2))


def _validate_final_route_shape(
    protocol: Mapping[str, Any],
    route: Mapping[str, Any],
) -> None:
    """Validate a final route without loading router artifacts."""

    settings = routing_settings(protocol)
    final_budgets = settings["final_budgets"]
    final_main_budget = settings["final_main_budget"]
    expected_ids = sorted(_protocol_ids(protocol, "final"))
    if route.get("run_status") != "ready":
        raise DataError("final route is not marked as ready")
    if route.get("dataset_id") != protocol["dataset"]["id"] or route.get("revision") != protocol["dataset"]["revision"]:
        raise DataError("final route dataset provenance changed")
    _validate_exact_id_list(route.get("final_test_sample_ids"), expected_ids, context="final route sample IDs")
    if route.get("feature_order") != list(FEATURE_NAMES):
        raise DataError("final route feature order changed")
    if route.get("main_budget") != final_main_budget or route.get("diagnostic_budgets") != list(final_budgets):
        raise DataError("final route budget settings changed")
    if route.get("active_risk_features") != list(settings["active_risk_features"]) or route.get("inactive_risk_features") != list(settings["inactive_risk_features"]):
        raise DataError("final route risk feature activity changed")
    if route.get("percentile_rule") != settings["percentile_rule"]:
        raise DataError("final route percentile rule changed")
    if route.get("final_route_transfers_to") != "glm_ocr":
        raise DataError("final route GLM transfer identity changed")

    predictions = route.get("router_predictions")
    if not isinstance(predictions, Mapping) or set(predictions) != set(MODEL_IDS):
        raise DataError("final route router predictions are incomplete")
    for model_id in MODEL_IDS:
        values = predictions[model_id]
        if not isinstance(values, Mapping) or list(values) != expected_ids:
            raise DataError(f"final route {model_id} predictions are not ordered by sample ID")
        for sample_id in expected_ids:
            _require_finite(values[sample_id], context=f"final route {model_id} prediction {sample_id}")

    feature_rows = route.get("features")
    if not isinstance(feature_rows, list) or len(feature_rows) != len(expected_ids):
        raise DataError("final route does not contain 300 feature records")
    expected_issuers = {
        str(item["sample_id"]): str(item["issuer"])
        for item in protocol["vietfintab"]["splits"]["final"]
    }
    for sample_id, row in zip(expected_ids, feature_rows, strict=True):
        if not isinstance(row, Mapping) or row.get("sample_id") != sample_id or row.get("issuer") != expected_issuers[sample_id]:
            raise DataError(f"final route feature provenance changed: {sample_id}")
        _validate_feature_mapping(row.get("values"), context=f"final route features {sample_id}")

    risk_rows = route.get("risk_features")
    if not isinstance(risk_rows, list) or len(risk_rows) != len(expected_ids):
        raise DataError("final route does not contain 300 risk records")
    for sample_id, row in zip(expected_ids, risk_rows, strict=True):
        if not isinstance(row, Mapping) or row.get("sample_id") != sample_id:
            raise DataError(f"final route risk ordering changed: {sample_id}")
        for field in (
            "confidence_tail_risk",
            "unmatched_ocr_ratio",
            "confidence_tail_risk_percentile",
            "unmatched_ocr_ratio_percentile",
            "combined_risk",
        ):
            _require_finite(row.get(field), context=f"final route risk {sample_id}:{field}")
    _validate_calibration_populations(route.get("risk_populations"))

    rankings = route.get("rankings")
    expected_ranking_names = {
        "confidence_tail_only",
        "combined_risk",
        "ridge_benefit",
        "hist_gradient_boosting_benefit",
    }
    if not isinstance(rankings, Mapping) or set(rankings) != expected_ranking_names:
        raise DataError("final route rankings are incomplete")
    for method in expected_ranking_names:
        _validate_rank_permutation(rankings[method], expected_ids, context=f"final route ranking {method}")

    main_routes = route.get("main_20_percent_routes")
    expected_route_names = {"always_primary", *expected_ranking_names, "random_matched_budget"}
    if not isinstance(main_routes, Mapping) or set(main_routes) != expected_route_names:
        raise DataError("final route main-budget routes are incomplete")
    if main_routes["always_primary"] != []:
        raise DataError("final route always-primary route is not empty")
    for method in expected_ranking_names:
        if main_routes[method] != list(rankings[method][:final_main_budget]):
            raise DataError(f"final route main route differs from ranking: {method}")
    random_route = route.get("random_routing")
    if not isinstance(random_route, Mapping):
        raise DataError("final route random route is missing")
    if (
        random_route.get("seed") != settings["random_seed"]
        or random_route.get("repetition") != 0
        or random_route.get("budget") != final_main_budget
        or random_route.get("rule")
        != (
            "rank sorted final-test IDs by SHA256(seed + ':' + repetition + ':' + sample_id), "
            f"then select the first {final_main_budget}"
        )
        or main_routes["random_matched_budget"] != random_route.get("route")
    ):
        raise DataError("final route random route changed")


def _final_score(value: Any, *, context: str, required: bool = True) -> float | None:
    if value is None and not required:
        return None
    return _require_finite(value, context=context)


def _load_final_report_records(
    protocol: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate all final recognition/evaluation records and return score-only rows."""

    evaluation_identity = evaluation_semantic_identity(protocol)
    validate_evaluation_environment(protocol)
    route_path = output_dir / "vietfintab" / "router" / "final_routes.json"
    if not route_path.is_file():
        raise DataError(f"final report requires the prepared route plan: {route_path}")
    route = _read_object(route_path, context="final route plan")
    _validate_final_route_shape(protocol, route)
    expected_ids = sorted(_protocol_ids(protocol, "final"))
    entries = {
        str(item["sample_id"]): item
        for item in protocol["vietfintab"]["splits"]["final"]
    }
    records: dict[str, dict[str, Any]] = {}
    for sample_id in expected_ids:
        item = entries[sample_id]
        sample = {
            "sample_id": sample_id,
            "issuer": item["issuer"],
            "spanning": item["spanning"],
        }
        primary_path = output_dir / "vietfintab" / "primary" / f"{sample_id}.json"
        hunyuan_path = output_dir / "vietfintab" / "experts" / "hunyuanocr_1_5" / f"{sample_id}.json"
        glm_path = output_dir / "vietfintab" / "experts" / "glm_ocr" / f"{sample_id}.json"
        primary_eval_path = output_dir / "vietfintab" / "evaluation" / "primary" / f"{sample_id}.json"
        hunyuan_eval_path = output_dir / "vietfintab" / "evaluation" / "hunyuanocr_1_5" / f"{sample_id}.json"
        glm_eval_path = output_dir / "vietfintab" / "evaluation" / "glm_ocr" / f"{sample_id}.json"
        paths = (primary_path, hunyuan_path, glm_path, primary_eval_path, hunyuan_eval_path, glm_eval_path)
        if any(not path.is_file() for path in paths):
            missing = [str(path) for path in paths if not path.is_file()]
            raise DataError(f"final report inputs are incomplete for {sample_id}: {', '.join(missing)}")

        primary = _validate_existing(primary_path, sample, dict(protocol), expected_cohort="final")
        hunyuan = _validate_expert_existing(hunyuan_path, sample, "hunyuanocr_1_5", dict(protocol), expected_cohort="final")
        glm = _validate_expert_existing(glm_path, sample, "glm_ocr", dict(protocol), expected_cohort="final")
        primary_eval = _read_object(primary_eval_path, context="final Primary evaluation")
        hunyuan_eval = _read_object(hunyuan_eval_path, context="final Hunyuan evaluation")
        glm_eval = _read_object(glm_eval_path, context="final GLM evaluation")
        for evaluation, system, source in (
            (primary_eval, "primary", primary),
            (hunyuan_eval, "hunyuanocr_1_5", hunyuan),
            (glm_eval, "glm_ocr", glm),
        ):
            validate_evaluation_record(
                evaluation,
                sample_id=sample_id,
                system_id=system,
                expected_split="final",
                expected_issuer=item["issuer"],
                expected_dataset_id=str(protocol["dataset"]["id"]),
                expected_revision=str(protocol["dataset"]["revision"]),
                expected_source_status=source.get("status"),
                expected_source_identity=_evaluation_source_identity(source, system, protocol),
                expected_evaluation_identity=evaluation_identity,
            )

        if primary.get("status") != "success" or primary_eval.get("scoring_status") != "scored":
            raise DataError(f"final Primary is not a successful scored record: {sample_id}")
        features = primary.get("features")
        _validate_feature_mapping(features, context=f"final Primary features {sample_id}")
        primary_con = _final_score(primary_eval.get("grits_con"), context=f"final Primary GriTS-Con {sample_id}")
        primary_top = _final_score(primary_eval.get("grits_top"), context=f"final Primary GriTS-Top {sample_id}")
        primary_loc = _final_score(primary_eval.get("grits_loc"), context=f"final Primary GriTS-Loc {sample_id}", required=False)

        expert_values: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
            "hunyuan": (hunyuan, hunyuan_eval),
            "glm": (glm, glm_eval),
        }
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "issuer": str(item["issuer"]),
            "spanning": bool(item["spanning"]),
            "features": {name: features[name] for name in FEATURE_NAMES},
            "primary_status": str(primary["status"]),
            "primary_grits_con": float(primary_con),
            "primary_grits_top": float(primary_top),
            "primary_grits_loc": primary_loc,
            "primary_inference_seconds": _require_finite(
                primary.get("inference_elapsed_seconds"),
                context=f"final Primary timing {sample_id}",
            ),
        }
        for expert_name, (source, evaluation) in expert_values.items():
            status = str(source["status"])
            success = status == "success"
            if success:
                if evaluation.get("scoring_status") != "scored":
                    raise DataError(f"successful {expert_name} record was not scored: {sample_id}")
                expert_con = _final_score(evaluation.get("grits_con"), context=f"final {expert_name} GriTS-Con {sample_id}")
                expert_top = _final_score(evaluation.get("grits_top"), context=f"final {expert_name} GriTS-Top {sample_id}")
                expert_loc = _final_score(evaluation.get("grits_loc"), context=f"final {expert_name} GriTS-Loc {sample_id}", required=False)
            else:
                expert_con = None
                expert_top = None
                expert_loc = None
                if evaluation.get("scoring_status") != "failed_source":
                    raise DataError(f"failed {expert_name} record has an unexpected evaluation status: {sample_id}")
            row[f"{expert_name}_status"] = status
            row[f"{expert_name}_success"] = success
            row[f"{expert_name}_standalone_grits_con"] = expert_con
            row[f"{expert_name}_standalone_grits_top"] = expert_top
            row[f"{expert_name}_grits_loc"] = expert_loc
            row[f"{expert_name}_inference_seconds"] = _require_finite(
                source.get("inference_elapsed_seconds"),
                context=f"final {expert_name} timing {sample_id}",
            )
            row[f"{expert_name}_call_benefit_grits_con"] = call_benefit_grits_con(
                float(primary_con),
                expert_con,
                success,
            )
        records[sample_id] = row
    if set(records) != set(expected_ids) or len(records) != 300:
        raise DataError("final report requires exactly 300 validated sample records")
    return records, route


def _final_standalone_summary(
    records: Mapping[str, Mapping[str, Any]],
    *,
    system: str,
    expert: str | None = None,
) -> dict[str, Any]:
    ids = sorted(records)
    if system == "primary":
        statuses = [str(records[item]["primary_status"]) for item in ids]
        con_values = [float(records[item]["primary_grits_con"]) for item in ids]
        top_values = [float(records[item]["primary_grits_top"]) for item in ids]
        loc_values = [records[item].get("primary_grits_loc") for item in ids]
    else:
        if expert is None:
            raise DataError("expert standalone summary requires an expert name")
        statuses = [str(records[item][f"{expert}_status"]) for item in ids]
        con_values = [
            float(records[item][f"{expert}_standalone_grits_con"])
            if records[item][f"{expert}_success"]
            else 0.0
            for item in ids
        ]
        top_values = [
            float(records[item][f"{expert}_standalone_grits_top"])
            if records[item][f"{expert}_success"]
            else 0.0
            for item in ids
        ]
        loc_values = [
            records[item].get(f"{expert}_grits_loc")
            if records[item][f"{expert}_success"]
            else None
            for item in ids
        ]
    success_count = sum(status == "success" for status in statuses)
    successful_loc = [float(value) for value in loc_values if value is not None]
    loc_coverage = (
        float(statistics.fmean([float(value) if value is not None else 0.0 for value in loc_values]))
        if len(successful_loc) == success_count and success_count == len([value for value in loc_values if value is not None])
        else None
    )
    successful_indices = [index for index, status in enumerate(statuses) if status == "success"]
    return {
        "sample_count": len(ids),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "successful_canonicalization_count": success_count,
        "successful_canonicalization_coverage": success_count / len(ids) if ids else 0.0,
        "coverage_aware": {
            "grits_con": float(statistics.fmean(con_values)) if con_values else None,
            "grits_top": float(statistics.fmean(top_values)) if top_values else None,
            "grits_loc": loc_coverage,
        },
        "successful_only": {
            "grits_con": float(statistics.fmean([con_values[index] for index in successful_indices])) if successful_indices else None,
            "grits_top": float(statistics.fmean([top_values[index] for index in successful_indices])) if successful_indices else None,
            "grits_loc": float(statistics.fmean(successful_loc)) if successful_loc and len(successful_loc) == success_count else None,
            "sample_count": success_count,
            "note": "conditional on successful canonicalization; failures score zero only in coverage-aware means",
        },
    }


def _final_route_result(
    records: Mapping[str, Mapping[str, Any]],
    route: Sequence[str],
    *,
    budget: int,
    routing_method: str,
    expert: str,
    primary_cost: float,
    allowed_budgets: Sequence[int],
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    evaluation_budget = 0 if routing_method == "always_primary" else budget
    result = evaluate_route(
        records,
        [] if routing_method == "always_primary" else route,
        budget=evaluation_budget,
        routing_method=routing_method,
        primary_cost=primary_cost,
        expert=expert,
        allowed_budgets=allowed_budgets,
    )
    if routing_method == "always_primary":
        result["requested_budget"] = budget
        result["requested_expert_call_count"] = budget
        result["baseline_requested_budget"] = budget
    else:
        result["requested_expert_call_count"] = budget
    result["fallback_count"] = result["failed_expert_return_count"]
    result["total_simulated_hybrid_inference_seconds"] = result["total_warm_inference_cost_seconds"]
    result["mean_simulated_hybrid_time_per_sample_seconds"] = result["mean_warm_inference_cost_per_sample_seconds"]
    result["diagnostic_only"] = diagnostic_only
    result["improved_replacement_sample_ids"] = sorted(
        sample_id
        for sample_id in result["replacement_sample_ids"]
        if float(records[sample_id].get(f"{expert}_call_benefit_grits_con", 0.0)) > TIE_TOLERANCE
    )
    result["harmed_replacement_sample_ids"] = sorted(
        sample_id
        for sample_id in result["replacement_sample_ids"]
        if float(records[sample_id].get(f"{expert}_call_benefit_grits_con", 0.0)) < -TIE_TOLERANCE
    )
    result["unchanged_replacement_sample_ids"] = sorted(
        sample_id
        for sample_id in result["replacement_sample_ids"]
        if abs(float(records[sample_id].get(f"{expert}_call_benefit_grits_con", 0.0))) <= TIE_TOLERANCE
    )
    return result


def _random_position_summary(
    records: Mapping[str, Mapping[str, Any]],
    *,
    expert: str,
    target: float,
    primary_cost: float,
    budget: int,
    repetitions: int,
    seed: str,
    allowed_budgets: Sequence[int],
) -> dict[str, Any]:
    values = [
        float(
            evaluate_route(
                records,
                deterministic_random_route(sorted(records), budget, repetition, seed=seed, allowed_budgets=allowed_budgets),
                budget=budget,
                routing_method="random_matched_budget",
                primary_cost=primary_cost,
                expert=expert,
                allowed_budgets=allowed_budgets,
            )["hybrid_full_set_grits_con"]
        )
        for repetition in range(repetitions)
    ]
    return {
        "target": target,
        "random_mean": float(statistics.fmean(values)),
        "random_median": float(statistics.median(values)),
        "random_percentile_95_interval": {
            "lower": float(_linear_percentile(values, 0.025)),
            "upper": float(_linear_percentile(values, 0.975)),
        },
        "target_percentile_position": sum(value <= target for value in values) / len(values),
    }


def _final_observed_expert_delta(
    record: Mapping[str, Any],
    *,
    expert: str,
) -> float:
    """Return the deployment-aware expert benefit used by final diagnostics."""

    succeeded = record.get(f"{expert}_status") == "success"
    expert_score = float(record[f"{expert}_standalone_grits_con"]) if succeeded else None
    return call_benefit_grits_con(
        float(record["primary_grits_con"]),
        expert_score,
        succeeded,
    )


def _final_random_mean_vector(
    records: Mapping[str, Mapping[str, Any]],
    *,
    expert: str,
    budget: int,
    repetitions: int,
    seed: str,
    allowed_budgets: Sequence[int],
) -> list[float]:
    """Compute the per-sample random allocation summary."""

    sample_ids = sorted(records)
    per_sample_values: list[list[float]] = [[] for _ in sample_ids]
    for repetition in range(repetitions):
        route = deterministic_random_route(
            sample_ids,
            budget,
            repetition,
            seed=seed,
            allowed_budgets=allowed_budgets,
        )
        con_values, _ = route_score_vector(records, route, expert=expert)
        for index, value in enumerate(con_values):
            per_sample_values[index].append(value)
    return [float(statistics.fmean(values)) for values in per_sample_values]


def _linear_percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise DataError("percentile requires non-empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _build_final_report(
    protocol: Mapping[str, Any],
    output_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    settings = routing_settings(protocol)
    final_budgets = settings["final_budgets"]
    final_main_budget = settings["final_main_budget"]
    random_seed = settings["random_seed"]
    random_repetitions = settings["random_repetitions"]
    bootstrap_seed = settings["bootstrap_seed"]
    bootstrap_replicates = settings["bootstrap_replicates"]
    records, route = _load_final_report_records(protocol, output_dir)
    sample_ids = sorted(records)
    try:
        difficulty_model = load_difficulty_router(models_dir / "router")
        difficulty_predictions = predict_router(
            difficulty_model,
            {sample_id: records[sample_id]["features"] for sample_id in sample_ids},
            sample_ids,
        )
    except RoutingError as exc:
        raise DataError(str(exc)) from exc
    difficulty_ranking = rank_by_predicted_benefit(sample_ids, difficulty_predictions)
    primary_cost = float(sum(float(records[item]["primary_inference_seconds"]) for item in sample_ids))
    expected_main_routes = route["main_20_percent_routes"]
    route_rankings = route["rankings"]
    hgb_hunyuan_route = list(expected_main_routes["hist_gradient_boosting_benefit"])
    hgb_glm_route = list(expected_main_routes["hist_gradient_boosting_benefit"])

    hunyuan_method_rankings: dict[str, list[str]] = {
        "always_primary": [],
        "confidence_tail_only": list(route_rankings["confidence_tail_only"]),
        "combined_risk": list(route_rankings["combined_risk"]),
        "ridge_benefit": list(route_rankings["ridge_benefit"]),
        "hist_gradient_boosting_benefit": list(route_rankings["hist_gradient_boosting_benefit"]),
        "difficulty_control": difficulty_ranking,
        "oracle_benefit": rank_by_predicted_benefit(
            sample_ids,
            {
                sample_id: _final_observed_expert_delta(records[sample_id], expert="hunyuan")
                for sample_id in sample_ids
            },
        ),
    }
    glm_oracle_ranking = rank_by_predicted_benefit(
        sample_ids,
        {
            sample_id: _final_observed_expert_delta(records[sample_id], expert="glm")
            for sample_id in sample_ids
        },
    )
    hunyuan_route_results: dict[str, list[dict[str, Any]]] = {}
    hunyuan_routes: dict[str, dict[str, list[str]]] = {}
    for method, ranking in hunyuan_method_rankings.items():
        hunyuan_route_results[method] = []
        hunyuan_routes[method] = {}
        for budget in final_budgets:
            chosen = [] if method == "always_primary" else route_for_budget(ranking, budget, final_budgets)
            if method == "always_primary":
                chosen = []
            hunyuan_routes[method][str(budget)] = chosen
            hunyuan_route_results[method].append(
                _final_route_result(
                    records,
                    chosen,
                    budget=budget,
                    routing_method=method,
                    expert="hunyuan",
                    primary_cost=primary_cost,
                    allowed_budgets=final_budgets,
                    diagnostic_only=method == "oracle_benefit",
                )
            )

    glm_method_rankings = {
        "hist_gradient_boosting_benefit": list(route_rankings["hist_gradient_boosting_benefit"]),
        "oracle_benefit": glm_oracle_ranking,
    }
    glm_route_results: dict[str, list[dict[str, Any]]] = {}
    glm_routes: dict[str, dict[str, list[str]]] = {}
    for method, ranking in glm_method_rankings.items():
        glm_route_results[method] = []
        glm_routes[method] = {}
        for budget in final_budgets:
            chosen = route_for_budget(ranking, budget, final_budgets)
            glm_routes[method][str(budget)] = chosen
            glm_route_results[method].append(
                _final_route_result(
                    records,
                    chosen,
                    budget=budget,
                    routing_method=method,
                    expert="glm",
                    primary_cost=primary_cost,
                    allowed_budgets=final_budgets,
                    diagnostic_only=method == "oracle_benefit",
                )
            )
    glm_always_rows = [
        _final_route_result(
            records,
            [],
            budget=budget,
            routing_method="always_primary",
            expert="glm",
            primary_cost=primary_cost,
            allowed_budgets=final_budgets,
        )
        for budget in final_budgets
    ]
    glm_route_results["always_primary"] = glm_always_rows
    glm_routes["always_primary"] = {str(budget): [] for budget in final_budgets}

    oracle_180_row = next(
        row for row in hunyuan_route_results["oracle_benefit"]
        if row["requested_budget"] == 180
    )
    oracle_180_gain = float(oracle_180_row["gain_over_always_primary_grits_con"])

    random_hunyuan, random_hunyuan_vector = random_summary(
        records,
        budgets=final_budgets,
        main_budget=final_main_budget,
        repetitions=random_repetitions,
        seed=random_seed,
        primary_cost=primary_cost,
        expert="hunyuan",
    )
    random_glm, random_glm_vector = random_summary(
        records,
        budgets=final_budgets,
        main_budget=final_main_budget,
        repetitions=random_repetitions,
        seed=random_seed,
        primary_cost=primary_cost,
        expert="glm",
    )
    random_hunyuan_vector = _final_random_mean_vector(
        records,
        expert="hunyuan",
        budget=final_main_budget,
        repetitions=random_repetitions,
        seed=random_seed,
        allowed_budgets=final_budgets,
    )
    random_glm_vector = _final_random_mean_vector(
        records,
        expert="glm",
        budget=final_main_budget,
        repetitions=random_repetitions,
        seed=random_seed,
        allowed_budgets=final_budgets,
    )
    hgb_hunyuan_main = next(
        row for row in hunyuan_route_results["hist_gradient_boosting_benefit"]
        if row["requested_budget"] == final_main_budget
    )
    hgb_glm_main = next(
        row for row in glm_route_results["hist_gradient_boosting_benefit"]
        if row["requested_budget"] == final_main_budget
    )
    primary_con = [float(records[item]["primary_grits_con"]) for item in sample_ids]
    primary_top = [float(records[item]["primary_grits_top"]) for item in sample_ids]
    primary_mean_con = float(statistics.fmean(primary_con))
    primary_mean_top = float(statistics.fmean(primary_top))
    hgb_hun_con = route_score_vector(records, hgb_hunyuan_route, expert="hunyuan")[0]
    difficulty_hun_con = route_score_vector(
        records,
        route_for_budget(difficulty_ranking, final_main_budget, final_budgets),
        expert="hunyuan",
    )[0]
    ridge_hun_con = route_score_vector(records, expected_main_routes["ridge_benefit"], expert="hunyuan")[0]
    confidence_hun_con = route_score_vector(records, expected_main_routes["confidence_tail_only"], expert="hunyuan")[0]
    combined_hun_con = route_score_vector(records, expected_main_routes["combined_risk"], expert="hunyuan")[0]
    hgb_glm_con = route_score_vector(records, hgb_glm_route, expert="glm")[0]
    random_hunyuan_vector = random_hunyuan_vector or primary_con
    random_glm_vector = random_glm_vector or primary_con
    bootstrap_comparisons = {
        "hgb_hunyuan_minus_primary": (hgb_hun_con, primary_con),
        "hgb_hunyuan_minus_random_mean": (hgb_hun_con, random_hunyuan_vector),
        "hgb_hunyuan_minus_confidence_tail": (hgb_hun_con, confidence_hun_con),
        "hgb_hunyuan_minus_combined_risk": (hgb_hun_con, combined_hun_con),
        "hgb_hunyuan_minus_ridge": (hgb_hun_con, ridge_hun_con),
        "ridge_hunyuan_minus_primary": (ridge_hun_con, primary_con),
        "ridge_hunyuan_minus_random_mean": (ridge_hun_con, random_hunyuan_vector),
        "hgb_glm_transfer_minus_primary": (hgb_glm_con, primary_con),
        "hgb_glm_transfer_minus_random_mean": (hgb_glm_con, random_glm_vector),
    }
    bootstrap = {
        "seed": bootstrap_seed,
        "replicates": bootstrap_replicates,
        "interval": "paired percentile 95%",
        "comparisons": {
            name: {
                **paired_bootstrap_mean_difference(left, right, seed=bootstrap_seed, replicates=bootstrap_replicates),
                "interval": "paired percentile 95%",
            }
            for name, (left, right) in bootstrap_comparisons.items()
        },
    }
    difficulty_bootstrap = paired_bootstrap_mean_difference(
        hgb_hun_con,
        difficulty_hun_con,
        seed=_FIGURE_CONTROL_BOOTSTRAP_SEED,
        replicates=bootstrap_replicates,
    )

    figure_intervals = {
        "random": {
            key: bootstrap["comparisons"]["hgb_hunyuan_minus_random_mean"][key]
            for key in ("observed", "lower", "upper", "interval")
        },
        "ocr_confidence": {
            key: bootstrap["comparisons"]["hgb_hunyuan_minus_confidence_tail"][key]
            for key in ("observed", "lower", "upper", "interval")
        },
        "combined_risk": {
            key: bootstrap["comparisons"]["hgb_hunyuan_minus_combined_risk"][key]
            for key in ("observed", "lower", "upper", "interval")
        },
        "ridge": {
            key: bootstrap["comparisons"]["hgb_hunyuan_minus_ridge"][key]
            for key in ("observed", "lower", "upper", "interval")
        },
        "difficulty": {
            key: value
            for key, value in difficulty_bootstrap.items()
            if key in {"observed", "lower", "upper", "interval"}
        },
    }

    def quality_rows(route_results: Mapping[str, Sequence[Mapping[str, Any]]], random_report: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for method in sorted(route_results):
            rows.extend(dict(item) for item in route_results[method])
        for budget in final_budgets:
            random_row = dict(random_report["budgets"][str(budget)])
            random_row.update(
                {
                    "routing_method": "random_matched_budget",
                    "requested_expert_call_count": budget,
                    "expert_call_count": budget,
                    "expert_call_rate": budget / len(sample_ids),
                    "hybrid_full_set_grits_con": random_row["mean_hybrid_full_set_grits_con"],
                    "hybrid_full_set_grits_top": random_row["mean_hybrid_full_set_grits_top"],
                    "gain_over_always_primary_grits_con": random_row["mean_hybrid_full_set_grits_con"] - primary_mean_con,
                    "gain_over_always_primary_grits_top": random_row["mean_hybrid_full_set_grits_top"] - primary_mean_top,
                    "diagnostic_only": False,
                }
            )
            rows.append(random_row)
        return rows

    standalone = {
        "crop_tableformer_rapidocr_v6": _final_standalone_summary(records, system="primary"),
        "hunyuanocr_1_5": _final_standalone_summary(records, system="expert", expert="hunyuan"),
        "glm_ocr": _final_standalone_summary(records, system="expert", expert="glm"),
    }
    route_main = {
        "hunyuanocr_1_5": hgb_hunyuan_main,
        "glm_ocr": hgb_glm_main,
    }
    report = {
        "run_status": "completed",
        "dataset_id": protocol["dataset"]["id"],
        "pinned_revision": protocol["dataset"]["revision"],
        "final_test_sample_ids": sample_ids,
        "evaluation_identity": evaluation_semantic_identity(protocol),
        "route_manifest": {
            "status": route["run_status"],
            "main_budget": final_main_budget,
            "diagnostic_budgets": list(final_budgets),
            "transfers_to": "glm_ocr",
        },
        "feature_order": list(FEATURE_NAMES),
        "main_budget": final_main_budget,
        "diagnostic_budgets": list(final_budgets),
        "random": {
            "seed": random_seed,
            "repetitions": random_repetitions,
            "sampling": "rank sorted sample IDs by SHA256(seed:repetition:sample_id), then select the first budget IDs",
        },
        "bootstrap_configuration": {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "interval": "paired percentile 95%",
        },
        "replacement_policy": _REPLACEMENT_POLICY,
        "glm_transfer": {
            "expert": "glm_ocr",
            "uses_unchanged_hunyuan_trained_hgb_route": True,
            "retrained": False,
            "recalibrated": False,
        },
        "recognition_status_counts": {
            "primary": {status: sum(records[item]["primary_status"] == status for item in sample_ids) for status in sorted({records[item]["primary_status"] for item in sample_ids})},
            "hunyuanocr_1_5": {status: sum(records[item]["hunyuan_status"] == status for item in sample_ids) for status in sorted({records[item]["hunyuan_status"] for item in sample_ids})},
            "glm_ocr": {status: sum(records[item]["glm_status"] == status for item in sample_ids) for status in sorted({records[item]["glm_status"] for item in sample_ids})},
        },
        "standalone_results": standalone,
        "hunyuan_routes": {
            "route_assignments": hunyuan_routes,
            "quality_budget_curves": quality_rows(hunyuan_route_results, random_hunyuan),
            "main_results": route_main["hunyuanocr_1_5"],
        },
        "figure2": {
            "budgets": list(final_budgets),
            "oracle_gain_at_180_calls": oracle_180_gain,
            "paired_intervals": figure_intervals,
            "interval_definition": "paired percentile 95% over final table records",
        },
        "glm_transfer_results": {
            "route_assignments": glm_routes,
            "quality_budget_curves": quality_rows(glm_route_results, random_glm),
            "main_results": route_main["glm_ocr"],
        },
        "random_routing": {
            "hunyuanocr_1_5": random_hunyuan,
            "glm_ocr": random_glm,
        },
        "random_main_point_comparison": {
            "hunyuanocr_1_5": _random_position_summary(
                records,
                expert="hunyuan",
                target=float(hgb_hunyuan_main["hybrid_full_set_grits_con"]),
                primary_cost=primary_cost,
                budget=final_main_budget,
                repetitions=random_repetitions,
                seed=random_seed,
                allowed_budgets=final_budgets,
            ),
            "glm_ocr": _random_position_summary(
                records,
                expert="glm",
                target=float(hgb_glm_main["hybrid_full_set_grits_con"]),
                primary_cost=primary_cost,
                budget=final_main_budget,
                repetitions=random_repetitions,
                seed=random_seed,
                allowed_budgets=final_budgets,
            ),
        },
        "bootstrap": bootstrap,
        "main_results": {
            "hgb_hunyuan_at_20_percent": hgb_hunyuan_main,
            "hgb_glm_transfer_at_20_percent": hgb_glm_main,
        },
        "sample_records": [
            {
                key: value
                for key, value in records[sample_id].items()
                if key not in {"features"}
                and not key.endswith("_loc")
            }
            | {"features": records[sample_id]["features"]}
            for sample_id in sample_ids
        ],
        "timing": {
            "primary_inference_seconds": primary_cost,
            "expert_inference_seconds": {
                "hunyuanocr_1_5": float(sum(float(records[item]["hunyuan_inference_seconds"]) for item in sample_ids)),
                "glm_ocr": float(sum(float(records[item]["glm_inference_seconds"]) for item in sample_ids)),
            },
        },
    }
    return report


def _final_report_stage(protocol: Mapping[str, Any], output_dir: Path, models_dir: Path) -> None:
    """Reconstruct the final report from validated result records only."""

    output_path = output_dir / "vietfintab" / "results" / "final.json"
    report = _build_final_report(protocol, output_dir, models_dir)
    if output_path.is_file():
        existing = _read_object(output_path, context="final report")
        if existing != report:
            raise DataError("existing final report is stale; report assembly is deterministic")
        print(json.dumps({"stage": "report", "split": "final", "resumed": True, "path": str(output_path)}, indent=2))
        return
    _atomic_json_write(output_path, report)
    print(json.dumps({"stage": "report", "split": "final", "resumed": False, "path": str(output_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BEAR reproduction study.")
    parser.add_argument("--study", choices=("vietfintab", "tinix"), required=True)
    parser.add_argument(
        "--stage",
        choices=("split", "primary", "expert", "evaluate", "router", "route", "report", "figure", "prepare"),
        required=True,
    )
    parser.add_argument("--expert", choices=EXPERT_IDS)
    parser.add_argument("--system", choices=("primary", *EXPERT_IDS))
    parser.add_argument("--split", choices=("development", "final"))
    parser.add_argument("--cohort", choices=_BATCH_COHORTS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing TiniX or router artifact where supported.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for Primary; expert stage requires --device cuda.",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.study == "tinix":
        if args.stage not in {"prepare", "primary", "route", "expert", "evaluate", "report"}:
            raise DataError(
                "TiniX supports prepare, primary, route, expert, evaluate, and report stages"
            )
        if any(value is not None for value in (args.expert, args.system, args.split, args.cohort)):
            raise DataError("TiniX stages do not accept VietFinTab selection arguments")
        if args.stage == "expert" and args.device != "cuda":
            raise DataError("TiniX Hunyuan inference requires --device cuda")
        if args.stage not in {"primary", "expert"} and args.device != "auto":
            raise DataError("--device is used only by TiniX primary and expert stages")
        try:
            path = run_tinix_stage(
                args.stage,
                source_root=ROOT,
                output_dir=args.output_dir,
                models_dir=args.models_dir,
                protocol_path=ROOT / "protocol.json",
                device=args.device,
                force=args.force,
            )
        except TinixError as exc:
            raise DataError(str(exc)) from exc
        print(json.dumps({"study": "tinix", "stage": args.stage, "path": str(path)}, indent=2))
        return 0
    if args.stage == "figure":
        if any(value is not None for value in (args.expert, args.system, args.split, args.cohort)) or args.force or args.device != "auto":
            raise DataError("figure stage does not accept cohort, expert, split, or device options")
        source = args.output_dir / "vietfintab" / "results" / "final.json"
        destination = args.output_dir / "vietfintab" / "results" / "figure2.svg"
        try:
            path = render_figure2(source, destination)
        except FigureError as exc:
            raise DataError(str(exc)) from exc
        print(json.dumps({"study": "vietfintab", "stage": "figure", "path": str(path)}, indent=2))
        return 0
    if args.stage == "prepare":
        raise DataError("prepare stage is available only for the tinix study")
    if args.stage == "expert" and args.device != "cuda":
        raise DataError("expert inference requires --device cuda")
    protocol = load_protocol(ROOT / "protocol.json")
    dataset = VietFinTabDataset(protocol, args.data_dir)
    if args.stage == "split":
        if args.cohort is not None or args.expert is not None or args.system is not None or args.split is not None or args.force:
            raise DataError("split stage does not accept cohort, expert, or force arguments")
        _atomic_json_write(args.output_dir / "vietfintab" / "split.json", _split_summary(dataset))
        print(json.dumps({"stage": "split", "status": "success"}, indent=2))
        return 0
    if args.stage == "expert":
        if args.system is not None or args.split is not None or args.force:
            raise DataError("expert stage does not accept --system, --split, or --force")
        if args.expert is None:
            raise DataError("expert stage requires --expert")
        if args.cohort is None:
            raise DataError("expert stage requires --cohort")
        _expert_cohort_stage(
            dataset,
            args.expert,
            args.cohort,
            args.output_dir,
            args.device,
            args.models_dir,
        )
        return 0
    if args.stage == "evaluate":
        if args.expert is not None or args.split is not None or args.force:
            raise DataError("evaluate stage uses --system and does not accept --expert, --split, or --force")
        if args.system is None:
            raise DataError("evaluate stage requires --system")
        if args.cohort is None:
            raise DataError("evaluate stage requires --cohort")
        _evaluation_cohort_stage(dataset, args.system, args.cohort, args.output_dir)
        return 0
    if args.stage == "router":
        if args.expert is not None or args.system is not None or args.cohort is not None:
            raise DataError("router stage does not accept expert, system, or cohort arguments")
        if args.split is None:
            raise DataError("router stage requires --split development or --split final")
        if args.device != "auto":
            raise DataError("router stage does not use --device")
        if args.split == "development":
            _router_development_stage(protocol, args.output_dir, args.models_dir, args.force)
        else:
            _router_final_stage(protocol, args.output_dir, args.models_dir, args.force)
        return 0
    if args.stage == "report":
        if args.expert is not None or args.system is not None or args.cohort is not None or args.force:
            raise DataError("report stage does not accept expert, system, cohort, or force arguments")
        if args.split != "final":
            raise DataError("report stage requires --split final")
        if args.device != "auto":
            raise DataError("report stage does not use --device")
        _final_report_stage(protocol, args.output_dir, args.models_dir)
        return 0
    if args.expert is not None:
        raise DataError("--expert is valid only with --stage expert")
    if args.system is not None or args.split is not None:
        raise DataError("--system and --split are valid only with evaluate or router stages")
    if args.cohort is None:
        raise DataError("primary stage requires --cohort")
    if args.force:
        raise DataError("primary stage does not use --force; remove the existing cohort records to rerun it")
    _primary_cohort_stage(
        dataset,
        args.cohort,
        args.output_dir,
        args.device,
        args.models_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataError as exc:
        raise SystemExit(f"error: {exc}")
