#!/usr/bin/env python3
"""Reproduce the routing diagnostics reported for the VietFinTab study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bear.routing import (
    DIFFICULTY_MODEL_PARAMS,
    FEATURE_NAMES,
    TIE_TOLERANCE,
    evaluate_route,
    paired_bootstrap_mean_difference,
    rank_by_predicted_benefit,
    routing_settings,
    spearman_summary,
)


CONTRAST_SEED = "cg-hybrid-ocr-phaseb-feature-ablation-bootstrap-v1"
PERMUTATION_REPLICATES = 200
PERMUTATION_SEED = "cg-hybrid-ocr-phaseb-feature-importance-permutation-v1"

FEATURE_ORDER = tuple(FEATURE_NAMES)
FEATURE_GROUPS = {
    "ocr_reliability": FEATURE_ORDER[0:5],
    "predicted_structure": FEATURE_ORDER[5:9],
    "content_profile": FEATURE_ORDER[9:11],
    "geometry": FEATURE_ORDER[11:13],
}
FEATURE_VARIANTS = {
    "all_features": FEATURE_ORDER,
    "ocr_confidence_only": FEATURE_ORDER[0:3],
    "ocr_reliability_only": FEATURE_ORDER[0:5],
    "structure_only": FEATURE_ORDER[5:9],
    "content_only": FEATURE_ORDER[9:11],
    "geometry_only": FEATURE_ORDER[11:13],
    "without_ocr_reliability": FEATURE_ORDER[5:13],
    "without_structure": FEATURE_ORDER[0:5] + FEATURE_ORDER[9:13],
    "without_content": FEATURE_ORDER[0:9] + FEATURE_ORDER[11:13],
    "without_geometry": FEATURE_ORDER[0:11],
}
PERMUTATION_UNITS = {
    **FEATURE_GROUPS,
    "ocr_confidence": FEATURE_ORDER[0:3],
    **{name: (name,) for name in FEATURE_ORDER},
}


class DiagnosticError(RuntimeError):
    """Raised when the generated experiment artifacts are incomplete or stale."""


def _fail(message: str) -> None:
    raise DiagnosticError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(
            f"missing {label}: {path}. Run the required VietFinTab reproduction "
            "stage before running routing diagnostics."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"could not read {label}: {path}: {exc}")
    _require(isinstance(value, dict), f"{label} is not a JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} is not finite")
    return number


def _linear_percentile(values: Sequence[float], probability: float) -> float:
    _require(values, "percentile requires a non-empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    _require(numbers, "cannot summarize an empty sequence")
    return {
        "count": len(numbers),
        "mean": float(statistics.fmean(numbers)),
        "sample_standard_deviation": float(statistics.stdev(numbers)) if len(numbers) > 1 else None,
        "median": float(statistics.median(numbers)),
        "P05": _linear_percentile(numbers, 0.05),
        "P95": _linear_percentile(numbers, 0.95),
        "minimum": float(min(numbers)),
        "maximum": float(max(numbers)),
    }


def _analysis_settings(protocol: Mapping[str, Any]) -> dict[str, Any]:
    dataset = protocol.get("dataset")
    _require(isinstance(dataset, Mapping), "protocol dataset is missing")
    dataset_id = dataset.get("id")
    dataset_revision = dataset.get("revision")
    _require(isinstance(dataset_id, str) and dataset_id, "protocol dataset ID is missing")
    _require(isinstance(dataset_revision, str) and dataset_revision, "protocol dataset revision is missing")
    settings = routing_settings(protocol)
    return {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "main_budget": settings["final_main_budget"],
        "development_budget": settings["development_main_budget"],
        "final_budgets": settings["final_budgets"],
        "development_budgets": settings["development_budgets"],
        "outer_folds": settings["outer_folds"],
        "bootstrap_replicates": settings["bootstrap_replicates"],
    }


def _validate_identity(value: Mapping[str, Any], label: str, settings: Mapping[str, Any]) -> None:
    _require(value.get("dataset_id") == settings["dataset_id"], f"{label} dataset differs")
    revision = value.get("pinned_revision", value.get("revision"))
    _require(revision == settings["dataset_revision"], f"{label} dataset revision differs")


def _feature_values(value: Any, label: str) -> dict[str, float | None]:
    if isinstance(value, Mapping):
        _require(set(value) == set(FEATURE_ORDER), f"{label} feature names differ")
        return {
            name: None if value[name] is None else _finite(value[name], f"{label}:{name}")
            for name in FEATURE_ORDER
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        _require(len(value) == len(FEATURE_ORDER), f"{label} feature count differs")
        return {
            name: None if item is None else _finite(item, f"{label}:{name}")
            for name, item in zip(FEATURE_ORDER, value, strict=True)
        }
    _fail(f"{label} features are neither a mapping nor an ordered sequence")


def _matrix(
    records: Mapping[str, Mapping[str, Any]],
    sample_ids: Sequence[str],
    feature_names: Sequence[str] = FEATURE_ORDER,
) -> np.ndarray:
    return np.asarray(
        [
            [
                np.nan
                if records[sample_id]["features"].get(name) is None
                else float(records[sample_id]["features"][name])
                for name in feature_names
            ]
            for sample_id in sample_ids
        ],
        dtype=float,
    )


def _hgb_pipeline(params: Mapping[str, Any]) -> Pipeline:
    estimator = HistGradientBoostingRegressor(
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=bool(params["early_stopping"]),
        random_state=int(params["random_state"]),
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def _record_from_scores(
    *,
    sample_id: str,
    issuer: str,
    features: Mapping[str, float | None],
    primary: Mapping[str, Any],
    expert: Mapping[str, Any],
) -> dict[str, Any]:
    primary_con = _finite(primary.get("grits_con"), f"Primary Con:{sample_id}")
    primary_top = _finite(primary.get("grits_top"), f"Primary Top:{sample_id}")
    success = (
        expert.get("source_status") == "success"
        and expert.get("scoring_status") == "scored"
        and expert.get("grits_con") is not None
        and expert.get("grits_top") is not None
    )
    if success:
        expert_con = _finite(expert.get("grits_con"), f"Hunyuan Con:{sample_id}")
        expert_top = _finite(expert.get("grits_top"), f"Hunyuan Top:{sample_id}")
        benefit = expert_con - primary_con
    else:
        expert_con = 0.0
        expert_top = 0.0
        benefit = 0.0
    return {
        "sample_id": sample_id,
        "issuer": issuer,
        "features": dict(features),
        "primary_grits_con": primary_con,
        "primary_grits_top": primary_top,
        "hunyuan_success": bool(success),
        "hunyuan_status": "success" if success else str(expert.get("source_status", "failed")),
        "hunyuan_standalone_grits_con": expert_con,
        "hunyuan_standalone_grits_top": expert_top,
        "hunyuan_call_benefit_grits_con": benefit,
    }


def _load_evaluation_records(
    output_root: Path,
    sample_ids: Sequence[str],
    feature_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    primary_dir = output_root / "vietfintab" / "evaluation" / "primary"
    expert_dir = output_root / "vietfintab" / "evaluation" / "hunyuanocr_1_5"
    result: dict[str, dict[str, Any]] = {}
    for sample_id in sample_ids:
        primary = _load_json(primary_dir / f"{sample_id}.json", f"Primary evaluation for {sample_id}")
        expert = _load_json(expert_dir / f"{sample_id}.json", f"Hunyuan evaluation for {sample_id}")
        result[sample_id] = _record_from_scores(
            sample_id=sample_id,
            issuer=str(feature_records[sample_id]["issuer"]),
            features=feature_records[sample_id]["features"],
            primary=primary,
            expert=expert,
        )
    return result


def _load_development(
    output_root: Path,
    settings: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, float]]:
    path = output_root / "vietfintab" / "router" / "development.json"
    artifact = _load_json(path, "development router artifact")
    _validate_identity(artifact, "development router artifact", settings)
    _require(artifact.get("feature_order") == list(FEATURE_ORDER), "development feature order differs")
    ids = artifact.get("sample_ids")
    _require(isinstance(ids, list) and len(ids) == 600 and ids == sorted(ids), "development population is not 600 sorted IDs")
    _require(len(set(ids)) == 600, "development population contains duplicate IDs")
    feature_rows = artifact.get("features")
    target_rows = artifact.get("targets")
    _require(isinstance(feature_rows, list) and len(feature_rows) == 600, "development feature records are incomplete")
    _require(isinstance(target_rows, list) and len(target_rows) == 600, "development target records are incomplete")
    records: dict[str, dict[str, Any]] = {}
    for row in feature_rows:
        sample_id = str(row.get("sample_id"))
        _require(sample_id in ids and sample_id not in records, f"invalid development feature record: {sample_id}")
        records[sample_id] = {
            "sample_id": sample_id,
            "issuer": str(row.get("issuer")),
            "features": _feature_values(row.get("values"), f"development:{sample_id}"),
        }
    _require(set(records) == set(ids), "development feature IDs differ")
    targets: dict[str, float] = {}
    for row in target_rows:
        sample_id = str(row.get("sample_id"))
        _require(sample_id in ids and sample_id not in targets, f"invalid development target record: {sample_id}")
        targets[sample_id] = _finite(row.get("call_benefit_grits_con"), f"development target:{sample_id}")
    _require(set(targets) == set(ids), "development target IDs differ")
    model = artifact.get("router_models", {}).get("hist_gradient_boosting_benefit", {})
    selected = model.get("selected_params")
    _require(isinstance(selected, Mapping), "development HGB specification is missing")
    for key, expected in DIFFICULTY_MODEL_PARAMS.items():
        _require(selected.get(key) == expected, f"development HGB specification differs for {key}")
    return ids, records, targets


def _load_final(
    output_root: Path,
    settings: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, list[str]]]:
    path = output_root / "vietfintab" / "results" / "final.json"
    report = _load_json(path, "final result report")
    _validate_identity(report, "final result report", settings)
    _require(report.get("feature_order") == list(FEATURE_ORDER), "final feature order differs")
    rows = report.get("sample_records")
    _require(isinstance(rows, list) and len(rows) == 300, "final population is not 300 records")
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        _require(sample_id not in records, f"duplicate final sample: {sample_id}")
        features = _feature_values(row.get("features"), f"final:{sample_id}")
        primary_con = _finite(row.get("primary_grits_con"), f"final Primary Con:{sample_id}")
        primary_top = _finite(row.get("primary_grits_top"), f"final Primary Top:{sample_id}")
        success = bool(row.get("hunyuan_success"))
        if success:
            expert_con = _finite(row.get("hunyuan_standalone_grits_con"), f"final Hunyuan Con:{sample_id}")
            expert_top = _finite(row.get("hunyuan_standalone_grits_top"), f"final Hunyuan Top:{sample_id}")
            benefit = expert_con - primary_con
        else:
            expert_con = 0.0
            expert_top = 0.0
            benefit = 0.0
        records[sample_id] = {
            "sample_id": sample_id,
            "issuer": str(row.get("issuer")),
            "features": features,
            "primary_grits_con": primary_con,
            "primary_grits_top": primary_top,
            "hunyuan_success": success,
            "hunyuan_status": str(row.get("hunyuan_status", "success" if success else "failed")),
            "hunyuan_standalone_grits_con": expert_con,
            "hunyuan_standalone_grits_top": expert_top,
            "hunyuan_call_benefit_grits_con": benefit,
        }
    ids = sorted(records)
    _require(len(ids) == 300, "final population is not 300 unique IDs")
    assignments = report.get("hunyuan_routes", {}).get("route_assignments", {})
    _require(isinstance(assignments, Mapping), "final route assignments are missing")
    required = ("hist_gradient_boosting_benefit", "difficulty_control")
    routes: dict[str, list[str]] = {}
    for name in required:
        route = assignments.get(name, {}).get(str(settings["main_budget"]))
        _require(
            isinstance(route, list) and len(route) == settings["main_budget"],
            f"final {name} route is missing",
        )
        _require(
            len(set(route)) == settings["main_budget"] and set(route).issubset(ids),
            f"final {name} route has invalid IDs",
        )
        routes[name] = list(route)
    route_path = output_root / "vietfintab" / "router" / "final_routes.json"
    if route_path.is_file():
        route_artifact = _load_json(route_path, "final route artifact")
        _validate_identity(route_artifact, "final route artifact", settings)
        _require(route_artifact.get("final_test_sample_ids") == ids, "final route population differs")
        route = route_artifact.get("main_20_percent_routes", {}).get("hist_gradient_boosting_benefit")
        _require(route == routes["hist_gradient_boosting_benefit"], "final HGB route differs between artifacts")
    return ids, records, routes


def _route_vectors(records: Mapping[str, Mapping[str, Any]], route: Sequence[str]) -> tuple[list[float], list[float]]:
    selected = set(route)
    con: list[float] = []
    top: list[float] = []
    for sample_id in sorted(records):
        record = records[sample_id]
        if sample_id in selected and record["hunyuan_success"]:
            con.append(float(record["hunyuan_standalone_grits_con"]))
            top.append(float(record["hunyuan_standalone_grits_top"]))
        else:
            con.append(float(record["primary_grits_con"]))
            top.append(float(record["primary_grits_top"]))
    return con, top


def _fit_feature_variants(
    development_ids: Sequence[str],
    development_records: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, float],
    final_ids: Sequence[str],
    final_records: Mapping[str, Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[float]]]:
    params = DIFFICULTY_MODEL_PARAMS
    y = np.asarray([targets[sample_id] for sample_id in development_ids], dtype=float)
    results: dict[str, dict[str, Any]] = {}
    con_vectors: dict[str, list[float]] = {}
    for name, feature_names in FEATURE_VARIANTS.items():
        model = _hgb_pipeline(params)
        model.fit(_matrix(development_records, development_ids, feature_names), y)
        predictions = model.predict(_matrix(final_records, final_ids, feature_names))
        prediction_map = {sample_id: float(value) for sample_id, value in zip(final_ids, predictions, strict=True)}
        route = rank_by_predicted_benefit(final_ids, prediction_map)[: settings["main_budget"]]
        evaluated = evaluate_route(
            final_records,
            route,
            budget=settings["main_budget"],
            routing_method=name,
            expert="hunyuan",
            allowed_budgets=settings["final_budgets"],
        )
        con, _ = _route_vectors(final_records, route)
        con_vectors[name] = con
        results[name] = {
            "feature_names": list(feature_names),
            "feature_count": len(feature_names),
            "fit_population": len(development_ids),
            "route_size": len(route),
            "grits_con": float(evaluated["hybrid_full_set_grits_con"]),
            "grits_top": float(evaluated["hybrid_full_set_grits_top"]),
            "successful_expert_returns": int(evaluated["successful_expert_return_count"]),
            "failed_expert_returns": int(evaluated["failed_expert_return_count"]),
        }
    return results, con_vectors


def _feature_contrasts(
    con_vectors: Mapping[str, Sequence[float]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons = (
        ("all_features_minus_ocr_confidence_only", "ocr_confidence_only"),
        ("all_features_minus_ocr_reliability_only", "ocr_reliability_only"),
        ("all_features_minus_without_ocr_reliability", "without_ocr_reliability"),
    )
    result: dict[str, Any] = {
        "budget": settings["main_budget"],
        "resampling_unit": "final table",
        "interval": "paired percentile 95%",
        "replicates": settings["bootstrap_replicates"],
        "seed": CONTRAST_SEED,
        "comparisons": {},
    }
    for name, right_name in comparisons:
        result["comparisons"][name] = paired_bootstrap_mean_difference(
            con_vectors["all_features"],
            con_vectors[right_name],
            seed=CONTRAST_SEED,
            replicates=settings["bootstrap_replicates"],
        )
    return result


def _build_development_records(
    output_root: Path,
    development_ids: Sequence[str],
    development_features: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    evaluated = _load_evaluation_records(output_root, development_ids, development_features)
    for sample_id, record in evaluated.items():
        _require(
            abs(float(record["hunyuan_call_benefit_grits_con"]) - float(targets[sample_id])) <= 1e-12,
            f"development target differs from Hunyuan evaluation for {sample_id}",
        )
    return evaluated


def _fold_models(
    sample_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, float],
    *,
    folds: int,
) -> tuple[dict[int, Pipeline], dict[int, list[str]], dict[int, np.ndarray], dict[str, float]]:
    ordered = list(sample_ids)
    matrix = _matrix(records, ordered)
    y = np.asarray([targets[sample_id] for sample_id in ordered], dtype=float)
    groups = np.asarray([records[sample_id]["issuer"] for sample_id in ordered])
    models: dict[int, Pipeline] = {}
    validation_ids: dict[int, list[str]] = {}
    validation_matrices: dict[int, np.ndarray] = {}
    predictions: dict[str, float] = {}
    splitter = GroupKFold(n_splits=folds)
    for fold_index, (train_indices, valid_indices) in enumerate(splitter.split(matrix, y, groups)):
        valid_ids = [ordered[index] for index in valid_indices]
        train_ids = [ordered[index] for index in train_indices]
        model = _hgb_pipeline(DIFFICULTY_MODEL_PARAMS)
        model.fit(_matrix(records, train_ids), np.asarray([targets[item] for item in train_ids]))
        predicted = model.predict(_matrix(records, valid_ids))
        for sample_id, value in zip(valid_ids, predicted, strict=True):
            _require(sample_id not in predictions, f"duplicate out-of-fold prediction: {sample_id}")
            predictions[sample_id] = float(value)
        models[fold_index] = model
        validation_ids[fold_index] = valid_ids
        validation_matrices[fold_index] = _matrix(records, valid_ids)
    _require(set(predictions) == set(ordered), "out-of-fold predictions do not cover development")
    return models, validation_ids, validation_matrices, predictions


def _permutation_schedule(records: Mapping[str, Mapping[str, Any]], sample_ids: Sequence[str]) -> dict[tuple[int, str], list[int]]:
    result: dict[tuple[int, str], list[int]] = {}
    issuers = sorted({str(records[sample_id]["issuer"]) for sample_id in sample_ids})
    for repeat in range(PERMUTATION_REPLICATES):
        for issuer in issuers:
            issuer_ids = sorted(sample_id for sample_id in sample_ids if records[sample_id]["issuer"] == issuer)
            token = f"{PERMUTATION_SEED}|repeat={repeat}|issuer={issuer}"
            seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
            result[(repeat, issuer)] = [int(index) for index in np.random.default_rng(seed).permutation(len(issuer_ids))]
    return result


def _permuted_predictions_for_units(
    unit_features: Mapping[str, Sequence[str]],
    repeat: int,
    sample_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    models: Mapping[int, Pipeline],
    validation_ids: Mapping[int, Sequence[str]],
    validation_matrices: Mapping[int, np.ndarray],
    schedule: Mapping[tuple[int, str], Sequence[int]],
) -> dict[str, float]:
    positions = {name: index for index, name in enumerate(FEATURE_ORDER)}
    result: dict[str, dict[str, float]] = {name: {} for name in unit_features}
    for fold_index in sorted(validation_ids):
        valid_ids = list(validation_ids[fold_index])
        original = validation_matrices[fold_index]
        matrices: dict[str, np.ndarray] = {}
        for name, names in unit_features.items():
            matrix = original.copy()
            columns = [positions[feature] for feature in names]
            for issuer in sorted({records[sample_id]["issuer"] for sample_id in valid_ids}):
                local_positions = [
                    index
                    for index, sample_id in enumerate(valid_ids)
                    if records[sample_id]["issuer"] == issuer
                ]
                issuer_ids = sorted(sample_id for sample_id in sample_ids if records[sample_id]["issuer"] == issuer)
                _require(len(local_positions) == len(issuer_ids), f"issuer split is not complete: {issuer}")
                donor_positions = [local_positions[int(index)] for index in schedule[(repeat, issuer)]]
                for column in columns:
                    matrix[local_positions, column] = original[donor_positions, column]
            matrices[name] = matrix
        unit_names = list(unit_features)
        stacked = np.vstack([matrices[name] for name in unit_names])
        predicted = models[fold_index].predict(stacked)
        width = len(valid_ids)
        for index, name in enumerate(unit_names):
            values = predicted[index * width : (index + 1) * width]
            for sample_id, value in zip(valid_ids, values, strict=True):
                result[name][sample_id] = float(value)
    for name in unit_features:
        _require(set(result[name]) == set(sample_ids), f"permuted predictions do not cover development: {name}")
    return {
        name: {sample_id: result[name][sample_id] for sample_id in sample_ids}
        for name in unit_features
    }


def _mse(predictions: Mapping[str, float], targets: Mapping[str, float], sample_ids: Sequence[str]) -> float:
    return float(statistics.fmean((predictions[item] - targets[item]) ** 2 for item in sample_ids))


def _permutation_reliance(
    development_ids: Sequence[str],
    development_records: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, float],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    models, validation_ids, validation_matrices, baseline = _fold_models(
        development_ids,
        development_records,
        targets,
        folds=int(settings["outer_folds"]),
    )
    baseline_mse = _mse(baseline, targets, development_ids)
    baseline_spearman = spearman_summary(
        [baseline[item] for item in development_ids],
        [targets[item] for item in development_ids],
    )
    _require(baseline_spearman.get("value") is not None, "baseline Spearman is undefined")
    schedule = _permutation_schedule(development_records, development_ids)
    baseline_ranking = rank_by_predicted_benefit(development_ids, baseline)
    baseline_route = baseline_ranking[: settings["development_budget"]]
    baseline_eval = evaluate_route(
        development_records,
        baseline_route,
        budget=settings["development_budget"],
        routing_method="all_features",
        expert="hunyuan",
        allowed_budgets=settings["development_budgets"],
    )
    units: dict[str, Any] = {}
    unit_predictions = {name: list(features) for name, features in PERMUTATION_UNITS.items()}
    unit_values = {name: {"mse": [], "spearman": [], "route": []} for name in PERMUTATION_UNITS}
    for repeat in range(PERMUTATION_REPLICATES):
        predictions_by_unit = _permuted_predictions_for_units(
            unit_predictions,
            repeat,
            development_ids,
            development_records,
            models,
            validation_ids,
            validation_matrices,
            schedule,
        )
        for unit_name, predictions in predictions_by_unit.items():
            permuted_mse = _mse(predictions, targets, development_ids)
            permuted_spearman = spearman_summary(
                [predictions[item] for item in development_ids],
                [targets[item] for item in development_ids],
            )
            _require(permuted_spearman.get("value") is not None, f"permuted Spearman is undefined: {unit_name}")
            route = rank_by_predicted_benefit(development_ids, predictions)[: settings["development_budget"]]
            route_eval = evaluate_route(
                development_records,
                route,
                budget=settings["development_budget"],
                routing_method=unit_name,
                expert="hunyuan",
                allowed_budgets=settings["development_budgets"],
            )
            unit_values[unit_name]["mse"].append(permuted_mse - baseline_mse)
            unit_values[unit_name]["spearman"].append(
                float(baseline_spearman["value"]) - float(permuted_spearman["value"])
            )
            unit_values[unit_name]["route"].append(
                float(baseline_eval["hybrid_full_set_grits_con"])
                - float(route_eval["hybrid_full_set_grits_con"])
            )
    for unit_name, unit_features in PERMUTATION_UNITS.items():
        mse_increases: list[float] = []
        spearman_drops: list[float] = []
        route_drops: list[float] = []
        mse_increases.extend(unit_values[unit_name]["mse"])
        spearman_drops.extend(unit_values[unit_name]["spearman"])
        route_drops.extend(unit_values[unit_name]["route"])
        units[unit_name] = {
            "feature_names": list(unit_features),
            "feature_count": len(unit_features),
            "repetitions": PERMUTATION_REPLICATES,
            "predictive": {
                "baseline_oof_mse": baseline_mse,
                "baseline_oof_spearman": float(baseline_spearman["value"]),
                "mse_increase": _summary(mse_increases),
                "spearman_drop": _summary(spearman_drops),
            },
            "routing_utility": {
                "budget": settings["development_budget"],
                "main_budget_con_drop": _summary(route_drops),
            },
        }
    group_order = list(FEATURE_GROUPS)
    predictive_order = sorted(group_order, key=lambda name: (-units[name]["predictive"]["mse_increase"]["mean"], group_order.index(name)))
    routing_order = sorted(group_order, key=lambda name: (-units[name]["routing_utility"]["main_budget_con_drop"]["mean"], group_order.index(name)))
    return {
        "evaluation_population": len(development_ids),
        "budget": settings["development_budget"],
        "permutation": {
            "seed": PERMUTATION_SEED,
            "repetitions": PERMUTATION_REPLICATES,
            "within_issuer": True,
            "refit_models": False,
            "target_permuted": False,
            "cross_validation": f"{settings['outer_folds']} issuer-group folds; sample IDs sorted ascending",
            "score": "out-of-fold MSE increase",
            "route_score": "drop in deployment-aware hybrid GriTS-Con",
            "aggregation": "mean, sample standard deviation, median, and linear P05/P95",
        },
        "group_results": {name: units[name] for name in group_order},
        "nested_and_feature_results": {
            name: units[name]
            for name in units
            if name not in FEATURE_GROUPS
        },
        "headline_order": {
            "predictive_mse_increase": predictive_order,
            "routing_utility_con_drop": routing_order,
        },
    }


def _selection_agreement(
    final_ids: Sequence[str],
    final_records: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, Sequence[str]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    bear = set(routes["hist_gradient_boosting_benefit"])
    difficulty = set(routes["difficulty_control"])
    _require(
        len(bear) == settings["main_budget"] and len(difficulty) == settings["main_budget"],
        "selection routes do not match the configured main budget",
    )
    groups = {
        "both": bear & difficulty,
        "bear_only": bear - difficulty,
        "difficulty_only": difficulty - bear,
        "neither": set(final_ids) - (bear | difficulty),
    }
    rows: dict[str, Any] = {}
    for name, members in groups.items():
        _require(members, f"selection group is empty: {name}")
        rows[name] = {
            "table_count": len(members),
            "mean_primary_grits_con": float(
                statistics.fmean(final_records[item]["primary_grits_con"] for item in sorted(members))
            ),
            "mean_replacement_benefit": float(
                statistics.fmean(
                    final_records[item]["hunyuan_call_benefit_grits_con"]
                    for item in sorted(members)
                )
            ),
        }
    _require(sum(row["table_count"] for row in rows.values()) == len(final_ids), "selection partition is not exhaustive")
    return {
        "budget": settings["main_budget"],
        "bear_route_size": len(bear),
        "difficulty_route_size": len(difficulty),
        "overlap": len(bear & difficulty),
        "groups": rows,
        "interpretation": "The rankings are related but not identical.",
    }


def _outcome_row(
    name: str,
    route: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts = {"beneficial": 0, "harmful": 0, "neutral": 0, "failed": 0}
    for sample_id in route:
        record = records[sample_id]
        if not record["hunyuan_success"]:
            counts["failed"] += 1
            continue
        delta = float(record["hunyuan_call_benefit_grits_con"])
        if delta > TIE_TOLERANCE:
            counts["beneficial"] += 1
        elif delta < -TIE_TOLERANCE:
            counts["harmful"] += 1
        else:
            counts["neutral"] += 1
    return {"allocation": name, "calls": len(route), **counts}


def _expert_outcomes(
    final_ids: Sequence[str],
    final_records: Mapping[str, Mapping[str, Any]],
    routes: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return {
        "classification": {
            "beneficial": "replacement benefit > 1e-9",
            "harmful": "replacement benefit < -1e-9",
            "neutral": "absolute replacement benefit <= 1e-9",
            "failed": "generation or parsing failure; Primary retained",
        },
        "allocations": [
            _outcome_row("BEAR (HGB)", routes["hist_gradient_boosting_benefit"], final_records),
            _outcome_row("Difficulty (HGB)", routes["difficulty_control"], final_records),
            _outcome_row("All tables", final_ids, final_records),
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_analysis(output_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _load_json(protocol_path, "protocol")
    settings = _analysis_settings(protocol)
    development_ids, development_features, targets = _load_development(output_root, settings)
    final_ids, final_records, routes = _load_final(output_root, settings)
    development_records = _build_development_records(output_root, development_ids, development_features, targets)
    feature_results, con_vectors = _fit_feature_variants(
        development_ids,
        development_features,
        targets,
        final_ids,
        final_records,
        settings,
    )
    return {
        "study": "vietfintab",
        "analysis": "routing_diagnostics",
        "dataset": {"id": settings["dataset_id"], "revision": settings["dataset_revision"]},
        "budget": settings["main_budget"],
        "feature_groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
        "feature_set_controls": {
            "model": "HistGradientBoostingRegressor",
            "fit_population": len(development_ids),
            "evaluation_population": len(final_ids),
            "benefit_target": "deployment-aware call replacement benefit",
            "variants": feature_results,
        },
        "paired_feature_contrasts": _feature_contrasts(con_vectors, settings),
        "permutation_reliance": _permutation_reliance(development_ids, development_records, targets, settings),
        "selection_agreement": _selection_agreement(final_ids, final_records, routes, settings),
        "expert_outcomes": _expert_outcomes(final_ids, final_records, routes),
        "semantics": {
            "replacement": "successful expert output replaces Primary; failed calls retain Primary",
            "quality_gate": False,
            "routing": "rank the complete cohort and select exactly the requested call budget",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="root directory containing the VietFinTab reproduction outputs",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocol.json"),
        help="experiment configuration used to verify the dataset identity",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="optional output path; defaults to outputs/analysis/routing_diagnostics.json",
    )
    args = parser.parse_args(argv)
    result_path = args.result or args.output_dir / "analysis" / "routing_diagnostics.json"
    try:
        result = build_analysis(args.output_dir, args.protocol)
        _write_json(result_path, result)
    except DiagnosticError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "path": str(result_path),
                "feature_variants": len(result["feature_set_controls"]["variants"]),
                "selection_overlap": result["selection_agreement"]["overlap"],
                "permutation_repetitions": PERMUTATION_REPLICATES,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
