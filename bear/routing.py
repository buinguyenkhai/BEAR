"""Primary-side feature extraction for BEAR routing."""

from __future__ import annotations

import math
import hashlib
import importlib.metadata
import itertools
import re
import statistics
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


FEATURE_NAMES = (
    "ocr_confidence_p10",
    "ocr_confidence_mean",
    "ocr_confidence_std",
    "ocr_box_count",
    "unmatched_ocr_ratio",
    "predicted_row_count",
    "predicted_column_count",
    "predicted_cell_count",
    "predicted_spanning_cell_count",
    "empty_cell_ratio",
    "numeric_looking_cell_ratio",
    "image_aspect_ratio",
    "cell_area_coefficient_of_variation",
)


# The matched difficulty control uses the same 13 Primary-side features and
# the fixed HGB estimator used by the paper's main routing comparison.  It is
# kept separate from the benefit routers because its target is different.
DIFFICULTY_MODEL_FILENAME = "difficulty_hgb.joblib"
DIFFICULTY_MODEL_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "max_iter": 100,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 1601,
}


_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
_HORIZONTAL_SPACE_SEPARATOR = r"[\u0020\u00a0\u2000-\u200a\u202f\u205f\u3000]"
_NUMERIC_EXPRESSION = re.compile(
    r"(?<![\w%])"
    r"[+-]?"
    r"(?:"
    r"\d{1,3}(?:,\d{3})+"
    r"|\d{1,3}(?:\.\d{3})+"
    rf"|\d{{1,2}}(?:{_HORIZONTAL_SPACE_SEPARATOR}+\d{{3}})+"
    r"|\d+"
    r")"
    r"(?:[.,]\d+)?"
    r"%?",
    flags=re.UNICODE,
)
_NUMERIC_MINUS_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "﹣": "-",
        "－": "-",
    }
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def normalize_text_for_cer(text: str) -> str:
    """Apply the specified conservative text normalization."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_numeric_token(token: str) -> str:
    """Normalize one matched numeric expression without changing its meaning."""

    if not isinstance(token, str) or not token:
        raise ValueError("numeric token must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", token).translate(
        _NUMERIC_MINUS_TRANSLATION
    )
    normalized = _WHITESPACE_RE.sub("", normalized)
    if not normalized:
        raise ValueError("numeric token must not be empty after normalization")
    return normalized


def _is_numeric_match_embedded(text: str, match: re.Match[str]) -> bool:
    """Reject numeric substrings embedded in identifiers or larger tokens."""

    if match.start() >= 1 and text[match.start() - 1] in "+-":
        word_end = match.start() - 2
        if word_end >= 0 and (text[word_end].isalnum() or text[word_end] == "_"):
            word_start = word_end
            while word_start >= 0 and (
                text[word_start].isalnum() or text[word_start] == "_"
            ):
                word_start -= 1
            word = text[word_start + 1 : word_end + 1]
            if any(character.isalpha() or character == "_" for character in word):
                return True
    if match.end() >= len(text):
        return False
    next_character = text[match.end()]
    return next_character == "%" or next_character == "_" or next_character.isalnum()


def extract_numeric_tokens(text: str) -> list[str]:
    """Return numeric expressions under the specified feature-token rules."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).translate(
        _NUMERIC_MINUS_TRANSLATION
    )
    return [
        normalize_numeric_token(match.group(0))
        for match in _NUMERIC_EXPRESSION.finditer(normalized)
        if not _is_numeric_match_embedded(normalized, match)
    ]


def _cell_text(cell: Any) -> str:
    value = _field(cell, "cell_text", _field(cell, "text", ""))
    return normalize_text_for_cer(str(value))


def _numeric_looking(value: Any) -> bool:
    return bool(extract_numeric_tokens(_cell_text({"text": value})))


def _cell_indices(cell: Any, axis: str) -> list[int]:
    values = _field(cell, f"{axis}_nums")
    if values is not None:
        return [int(value) for value in values]
    values = _field(cell, f"{axis}s")
    if values is not None:
        return [int(value) for value in values]
    start = _field(cell, f"{axis}_start")
    end = _field(cell, f"{axis}_end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return list(range(start, end))
    return []


def _cell_bbox(cell: Any) -> list[float] | None:
    value = _field(cell, "bbox")
    if value is None:
        return None
    if isinstance(value, Mapping):
        values = [value.get(key) for key in ("l", "t", "r", "b")]
    else:
        values = list(value) if isinstance(value, (list, tuple)) else [
            _field(value, key) for key in ("l", "t", "r", "b")
        ]
    if len(values) != 4 or any(_finite(item) is None for item in values):
        return None
    left, top, right, bottom = (float(item) for item in values)
    if not left < right or not top < bottom:
        return None
    return [left, top, right, bottom]


def _linear_percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def feature_record_from_primary(
    *,
    ocr_boxes: Sequence[Sequence[float] | None] | None,
    ocr_confidences: Sequence[float | None] | None,
    primary_cells: Sequence[Mapping[str, Any]] | None,
    image_width: int | None,
    image_height: int | None,
) -> dict[str, float | None]:
    """Return the specified feature record, preserving unavailable values as ``None``."""

    result: dict[str, Any] = {name: None for name in FEATURE_NAMES}
    prepared = list(ocr_boxes) if ocr_boxes is not None else None
    if prepared is not None:
        confidences = []
        for item in ocr_confidences or ():
            confidence = _finite(item)
            if confidence is not None and -1e-6 <= confidence <= 1.0 + 1e-6:
                confidences.append(min(1.0, max(0.0, confidence)))
        result["ocr_confidence_p10"] = _linear_percentile(confidences, 0.10)
        result["ocr_confidence_mean"] = statistics.fmean(confidences) if confidences else None
        if confidences:
            mean = statistics.fmean(confidences)
            result["ocr_confidence_std"] = math.sqrt(
                statistics.fmean((value - mean) ** 2 for value in confidences)
            )
        result["ocr_box_count"] = len(prepared)

    if primary_cells is not None:
        cells = list(primary_cells)
        rows = {index for cell in cells for index in _cell_indices(cell, "row")}
        columns = {index for cell in cells for index in _cell_indices(cell, "column")}
        result["predicted_row_count"] = len(rows)
        result["predicted_column_count"] = len(columns)
        result["predicted_cell_count"] = len(cells)
        result["predicted_spanning_cell_count"] = sum(
            len(_cell_indices(cell, "row")) > 1
            or len(_cell_indices(cell, "column")) > 1
            for cell in cells
        )
        result["empty_cell_ratio"] = (
            sum(_cell_text(cell) == "" for cell in cells) / len(cells)
            if cells
            else None
        )
        result["numeric_looking_cell_ratio"] = (
            sum(bool(extract_numeric_tokens(_cell_text(cell))) for cell in cells) / len(cells)
            if cells
            else None
        )
        primary_boxes = [box for box in (_cell_bbox(cell) for cell in cells) if box is not None]
        if prepared is None or not prepared:
            result["unmatched_ocr_ratio"] = None if prepared is None or not prepared else 0.0
        else:
            unmatched = 0
            for box in prepared:
                if box is None:
                    unmatched += 1
                    continue
                center_x = (float(box[0]) + float(box[2])) / 2.0
                center_y = (float(box[1]) + float(box[3])) / 2.0
                if not any(
                    candidate[0] <= center_x <= candidate[2]
                    and candidate[1] <= center_y <= candidate[3]
                    for candidate in primary_boxes
                ):
                    unmatched += 1
            result["unmatched_ocr_ratio"] = unmatched / len(prepared)
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in primary_boxes]
        if areas and statistics.fmean(areas) > 0:
            area_mean = statistics.fmean(areas)
            result["cell_area_coefficient_of_variation"] = math.sqrt(
                statistics.fmean((value - area_mean) ** 2 for value in areas)
            ) / area_mean
    if image_width is not None and image_height is not None and image_height > 0:
        result["image_aspect_ratio"] = float(image_width) / float(image_height)
    return result


def ordered_feature_vector(features: Mapping[str, Any]) -> list[float | None]:
    """Return a vector in the specified protocol order, retaining ``None`` values."""

    if list(features) != list(FEATURE_NAMES):
        raise ValueError("feature record names do not match the specified 13-feature schema")
    vector: list[float | None] = []
    for name in FEATURE_NAMES:
        value = features[name]
        if value is None:
            vector.append(None)
            continue
        finite = _finite(value)
        if finite is None:
            raise ValueError(f"feature {name} is not finite or null")
        vector.append(finite)
    return vector


def features_from_primary(**kwargs: Any) -> tuple[dict[str, float | None], list[float | None]]:
    """Return the named feature record and its ordered vector."""

    record = feature_record_from_primary(**kwargs)
    return record, ordered_feature_vector(record)


MODEL_IDS = ("ridge_benefit", "hist_gradient_boosting_benefit")
TIE_TOLERANCE = 1e-9


def routing_settings(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the configured routing settings without duplicating their values."""

    routing = protocol.get("routing")
    if not isinstance(routing, Mapping):
        raise RoutingError("protocol routing settings are missing")
    budgets = routing.get("budgets")
    validation = routing.get("validation")
    random = routing.get("random")
    bootstrap = routing.get("bootstrap")
    calibration = routing.get("calibration")
    if not all(isinstance(value, Mapping) for value in (budgets, validation, random, bootstrap, calibration)):
        raise RoutingError("protocol routing settings are incomplete")
    development_budgets = tuple(int(value) for value in budgets["development"])
    final_budgets = tuple(int(value) for value in budgets["final"])
    main_budgets = budgets["main"]
    return {
        "development_budgets": development_budgets,
        "final_budgets": final_budgets,
        "development_main_budget": int(main_budgets["development"]),
        "final_main_budget": int(main_budgets["final"]),
        "outer_folds": int(validation["outer_folds"]),
        "inner_folds": int(validation["inner_folds"]),
        "random_seed": str(random["seed"]),
        "random_repetitions": int(random["repetitions"]),
        "bootstrap_seed": str(bootstrap["seed"]),
        "bootstrap_replicates": int(bootstrap["replicates"]),
        "active_risk_features": tuple(str(value) for value in calibration["active_features"]),
        "inactive_risk_features": tuple(str(value) for value in calibration["inactive_features"]),
        "percentile_rule": str(calibration["percentile_rule"]),
        "combined_risk": str(calibration["combined_risk"]),
    }


class RoutingError(RuntimeError):
    """Raised when routing inputs or a specified routing operation is invalid."""


def call_benefit_grits_con(
    primary_grits_con: float,
    expert_grits_con: float | None,
    expert_succeeded: bool,
) -> float:
    """Return realized replacement benefit, assigning failed calls a zero target."""

    if not expert_succeeded:
        return 0.0
    if expert_grits_con is None:
        raise RoutingError("a successful expert requires a GriTS-Con score")
    primary = _finite(primary_grits_con)
    expert = _finite(expert_grits_con)
    if primary is None or expert is None:
        raise RoutingError("successful benefit labels require finite scores")
    return expert - primary


def _import_learning_dependencies() -> tuple[Any, ...]:
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RoutingError("scikit-learn and its numerical dependencies are required") from exc
    return (
        np,
        HistGradientBoostingRegressor,
        SimpleImputer,
        Ridge,
        mean_squared_error,
        GroupKFold,
        Pipeline,
        StandardScaler,
    )


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise RoutingError("percentile requires a non-empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _model_specs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    routing = protocol.get("routing")
    specs = routing.get("models") if isinstance(routing, Mapping) else None
    if not isinstance(specs, Mapping) or set(specs) != set(MODEL_IDS):
        raise RoutingError("protocol does not define exactly the two specified router families")
    return {str(key): dict(value) for key, value in specs.items()}


def _candidate_params(model_id: str, specs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if model_id not in MODEL_IDS:
        raise RoutingError(f"unsupported router family: {model_id}")
    grid = specs[model_id].get("grid")
    if not isinstance(grid, Mapping):
        raise RoutingError(f"router family {model_id} has no grid")
    if model_id == "ridge_benefit":
        return [{"alpha": float(value)} for value in grid.get("alpha", [])]
    names = ("max_depth", "learning_rate", "max_iter", "l2_regularization")
    values = [grid.get(name, []) for name in names]
    early_stopping = grid.get("early_stopping")
    random_state = grid.get("random_state")
    if (
        not isinstance(early_stopping, list)
        or len(early_stopping) != 1
        or not isinstance(early_stopping[0], bool)
        or not isinstance(random_state, list)
        or len(random_state) != 1
        or isinstance(random_state[0], bool)
        or not isinstance(random_state[0], int)
    ):
        raise RoutingError("HGB early stopping or random state is not specified correctly")
    early_stopping_value = bool(early_stopping[0])
    random_state_value = int(random_state[0])
    return [
        {
            "max_depth": int(depth),
            "learning_rate": float(rate),
            "max_iter": int(max_iter),
            "l2_regularization": float(l2),
            "early_stopping": early_stopping_value,
            "random_state": random_state_value,
        }
        for depth, rate, max_iter, l2 in itertools.product(*values)
    ]


def _params_key(params: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(params), sort_keys=True, separators=(",", ":"))


def validate_training_matrix(matrix: Any, feature_names: Sequence[str] = FEATURE_NAMES) -> None:
    """Reject an all-missing training column before a pipeline is fitted."""

    for index, name in enumerate(feature_names):
        column = matrix[:, index]
        if all(_finite_or_none(value) is None for value in column):
            raise RoutingError(f"all-null training feature column: {name}")


def _matrix(
    feature_records: Mapping[str, Mapping[str, Any]],
    sample_ids: Sequence[str],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> Any:
    np, *_ = _import_learning_dependencies()
    return np.asarray(
        [
            [
                float(feature_records[sample_id].get(name))
                if _finite_or_none(feature_records[sample_id].get(name)) is not None
                else np.nan
                for name in feature_names
            ]
            for sample_id in sample_ids
        ],
        dtype=float,
    )


def _make_pipeline(model_id: str, params: Mapping[str, Any]) -> Any:
    _, HistGradientBoostingRegressor, SimpleImputer, Ridge, _, _, Pipeline, StandardScaler = _import_learning_dependencies()
    if model_id == "ridge_benefit":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(params["alpha"]))),
        ])
    if model_id != "hist_gradient_boosting_benefit":
        raise RoutingError(f"unsupported router family: {model_id}")
    estimator = HistGradientBoostingRegressor(
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=bool(params["early_stopping"]),
        random_state=int(params["random_state"]),
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def _group_splits(
    sample_ids: Sequence[str],
    groups: Mapping[str, str],
    folds: int,
) -> list[tuple[list[str], list[str], int]]:
    _, _, _, _, _, GroupKFold, _, _ = _import_learning_dependencies()
    ordered = sorted(str(item) for item in sample_ids)
    if len(set(ordered)) != len(ordered):
        raise RoutingError("routing sample IDs must be unique")
    if any(item not in groups for item in ordered):
        raise RoutingError("routing groups are missing a sample ID")
    splitter = GroupKFold(n_splits=folds)
    group_values = [groups[item] for item in ordered]
    result: list[tuple[list[str], list[str], int]] = []
    for fold_index, (train_idx, validation_idx) in enumerate(
        splitter.split(list(range(len(ordered))), groups=group_values)
    ):
        training = [ordered[index] for index in train_idx]
        validation = [ordered[index] for index in validation_idx]
        if set(group_values[index] for index in train_idx).intersection(group_values[index] for index in validation_idx):
            raise RoutingError("issuer groups overlap between a training and validation fold")
        result.append((training, validation, fold_index))
    return result


def _tuning_fold_summary(
    split_rows: Sequence[tuple[list[str], list[str], int]],
    groups: Mapping[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "fold": fold,
            "sample_count": len(validation_ids),
            "training_sample_count": len(training_ids),
            "issuer_groups": sorted({groups[item] for item in validation_ids}),
            "training_issuer_groups": sorted({groups[item] for item in training_ids}),
            "issuer_overlap": sorted(
                set(groups[item] for item in validation_ids).intersection(groups[item] for item in training_ids)
            ),
        }
        for training_ids, validation_ids, fold in split_rows
    ]


def tune_router_family(
    model_id: str,
    *,
    sample_ids: Sequence[str],
    feature_records: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, float],
    groups: Mapping[str, str],
    specs: Mapping[str, Mapping[str, Any]],
    folds: int,
) -> dict[str, Any]:
    """Select settings within one router family using grouped validation."""

    _, _, _, _, mean_squared_error, _, _, _ = _import_learning_dependencies()
    ids = sorted(str(item) for item in sample_ids)
    if any(item not in targets for item in ids):
        raise RoutingError("router target is missing a sample ID")
    matrix = _matrix(feature_records, ids)
    validate_training_matrix(matrix)
    split_rows = _group_splits(ids, groups, folds)
    candidates: list[dict[str, Any]] = []
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    for params in _candidate_params(model_id, specs):
        fold_scores: list[float] = []
        for training_ids, validation_ids, _fold in split_rows:
            train_matrix = matrix[[positions[item] for item in training_ids]]
            validation_matrix = matrix[[positions[item] for item in validation_ids]]
            validate_training_matrix(train_matrix)
            pipeline = _make_pipeline(model_id, params)
            pipeline.fit(train_matrix, [float(targets[item]) for item in training_ids])
            predicted = pipeline.predict(validation_matrix)
            fold_scores.append(float(mean_squared_error(
                [float(targets[item]) for item in validation_ids], predicted
            )))
        candidates.append({
            "params": dict(params),
            "canonical_params": _params_key(params),
            "fold_mse": fold_scores,
            "mean_validation_mse": float(statistics.fmean(fold_scores)),
        })
    if not candidates:
        raise RoutingError(f"router family {model_id} has no candidates")
    selected = sorted(candidates, key=lambda item: (item["mean_validation_mse"], item["canonical_params"]))[0]
    return {
        "model_id": model_id,
        "fold_count": folds,
        "selection_metric": "mean validation squared error",
        "tie_break": "mean validation MSE, then canonical JSON hyperparameter representation",
        "candidate_results": candidates,
        "selected_params": dict(selected["params"]),
        "selected_canonical_params": selected["canonical_params"],
        "selected_mean_validation_mse": selected["mean_validation_mse"],
        "folds": _tuning_fold_summary(split_rows, groups),
    }


def nested_oof_predictions(
    model_id: str,
    *,
    sample_ids: Sequence[str],
    feature_records: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, float],
    groups: Mapping[str, str],
    specs: Mapping[str, Mapping[str, Any]],
    outer_folds: int,
    inner_folds: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Produce exactly one out-of-fold prediction per development sample."""

    ids = sorted(str(item) for item in sample_ids)
    matrix = _matrix(feature_records, ids)
    outer_rows = _group_splits(ids, groups, outer_folds)
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    predictions: dict[str, float] = {}
    outer_reports: list[dict[str, Any]] = []
    for train_ids, validation_ids, outer_fold in outer_rows:
        tuning = tune_router_family(
            model_id,
            sample_ids=train_ids,
            feature_records=feature_records,
            targets=targets,
            groups=groups,
            specs=specs,
            folds=inner_folds,
        )
        train_matrix = matrix[[positions[item] for item in train_ids]]
        validation_matrix = matrix[[positions[item] for item in validation_ids]]
        validate_training_matrix(train_matrix)
        pipeline = _make_pipeline(model_id, tuning["selected_params"])
        pipeline.fit(train_matrix, [float(targets[item]) for item in train_ids])
        for sample_id, value in zip(validation_ids, pipeline.predict(validation_matrix), strict=True):
            predictions[sample_id] = float(value)
        inner_rows = _group_splits(train_ids, groups, inner_folds)
        outer_reports.append({
            "outer_fold": outer_fold,
            "training_sample_count": len(train_ids),
            "validation_sample_count": len(validation_ids),
            "training_issuer_groups": sorted({groups[item] for item in train_ids}),
            "validation_issuer_groups": sorted({groups[item] for item in validation_ids}),
            "issuer_overlap": sorted(set(groups[item] for item in train_ids).intersection(groups[item] for item in validation_ids)),
            "inner_fold_assignments": _tuning_fold_summary(inner_rows, groups),
            "selected_params": tuning["selected_params"],
            "selected_mean_inner_validation_mse": tuning["selected_mean_validation_mse"],
            "inner_candidate_results": tuning["candidate_results"],
        })
    if set(predictions) != set(ids):
        raise RoutingError(f"{model_id} did not produce one OOF prediction per sample")
    return predictions, {
        "outer_fold_count": outer_folds,
        "inner_fold_count": inner_folds,
        "outer_folds": outer_reports,
        "oof_prediction_count": len(predictions),
        "all_predictions_out_of_fold": True,
    }


def fit_final_router_families(
    *,
    sample_ids: Sequence[str],
    feature_records: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, float],
    groups: Mapping[str, str],
    specs: Mapping[str, Mapping[str, Any]],
    folds: int,
    models_dir: str | Any = "models/router",
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the two specified families on development records and save local joblib files."""

    try:
        import joblib
    except ImportError as exc:
        raise RoutingError("joblib is required to save router models") from exc
    from pathlib import Path

    ids = sorted(str(item) for item in sample_ids)
    if len(ids) != 600 or len(set(ids)) != 600:
        raise RoutingError("router development fitting requires exactly 600 unique samples")
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    tuning: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        path = root / f"{model_id}.joblib"
        selected = tune_router_family(
            model_id,
            sample_ids=ids,
            feature_records=feature_records,
            targets=targets,
            groups=groups,
            specs=specs,
            folds=folds,
        )
        tuning[model_id] = selected
        if path.is_file() and not force:
            try:
                model = joblib.load(path)
                if not hasattr(model, "predict"):
                    raise RoutingError(f"router artifact has no predict method: {path}")
                if list(getattr(model, "named_steps", {})) != (
                    ["imputer", "scaler", "model"] if model_id == "ridge_benefit" else ["imputer", "model"]
                ):
                    raise RoutingError(f"router artifact has unexpected pipeline steps: {path}")
                fitted_params = model.named_steps["model"].get_params()
                for key, expected in selected["selected_params"].items():
                    if fitted_params.get(key) != expected:
                        raise RoutingError(f"router artifact parameters changed: {path}")
                models[model_id] = model
                continue
            except Exception:
                if not force:
                    raise RoutingError(f"existing router artifact is invalid; rerun with --force: {path}")
        matrix = _matrix(feature_records, ids)
        validate_training_matrix(matrix)
        model = _make_pipeline(model_id, selected["selected_params"])
        model.fit(matrix, [float(targets[item]) for item in ids])
        joblib.dump(model, path)
        models[model_id] = model
    return models, tuning


def fit_difficulty_router(
    *,
    sample_ids: Sequence[str],
    feature_records: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, float],
    models_dir: str | Any = "models/router",
    force: bool = False,
) -> Any:
    """Fit or load the matched Primary-difficulty HGB control.

    The control is part of the canonical budget comparison.  It uses only
    development Primary scores for fitting and never reads expert outcomes.
    """

    try:
        import joblib
    except ImportError as exc:
        raise RoutingError("joblib is required to save router models") from exc
    from pathlib import Path

    ids = sorted(str(item) for item in sample_ids)
    if len(ids) != 600 or len(set(ids)) != 600:
        raise RoutingError("difficulty-router fitting requires exactly 600 unique samples")
    if set(ids) != set(str(item) for item in targets):
        raise RoutingError("difficulty-router targets do not match the development IDs")
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / DIFFICULTY_MODEL_FILENAME
    expected_steps = ["imputer", "model"]
    if path.is_file() and not force:
        try:
            model = joblib.load(path)
            if not hasattr(model, "predict") or list(getattr(model, "named_steps", {})) != expected_steps:
                raise RoutingError(f"difficulty-router artifact has unexpected structure: {path}")
            fitted = model.named_steps["model"].get_params()
            for key, expected in DIFFICULTY_MODEL_PARAMS.items():
                if fitted.get(key) != expected:
                    raise RoutingError(f"difficulty-router parameters changed: {path}")
            return model
        except Exception as exc:
            raise RoutingError(f"existing difficulty-router artifact is invalid; rerun with --force: {path}") from exc

    matrix = _matrix(feature_records, ids)
    validate_training_matrix(matrix)
    model = _make_pipeline("hist_gradient_boosting_benefit", DIFFICULTY_MODEL_PARAMS)
    model.fit(matrix, [float(targets[item]) for item in ids])
    joblib.dump(model, path)
    return model


def load_difficulty_router(models_dir: str | Any = "models/router") -> Any:
    """Load the matched difficulty control without refitting it."""

    try:
        import joblib
    except ImportError as exc:
        raise RoutingError("joblib is required to load router models") from exc
    from pathlib import Path

    path = Path(models_dir) / DIFFICULTY_MODEL_FILENAME
    if not path.is_file():
        raise RoutingError(f"difficulty-router artifact is missing: {path}")
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise RoutingError(f"could not load difficulty-router artifact: {path}") from exc
    if list(getattr(model, "named_steps", {})) != ["imputer", "model"]:
        raise RoutingError(f"difficulty-router artifact has unexpected structure: {path}")
    fitted = model.named_steps["model"].get_params()
    for key, expected in DIFFICULTY_MODEL_PARAMS.items():
        if fitted.get(key) != expected:
            raise RoutingError(f"difficulty-router parameters changed: {path}")
    return model


def predict_router(
    model: Any,
    feature_records: Mapping[str, Mapping[str, Any]],
    sample_ids: Sequence[str],
) -> dict[str, float]:
    """Predict one finite score for each sample in sorted ID order."""

    ids = sorted(str(item) for item in sample_ids)
    matrix = _matrix(feature_records, ids)
    values = model.predict(matrix)
    if len(values) != len(ids):
        raise RoutingError("router returned an unexpected prediction count")
    result = {sample_id: float(value) for sample_id, value in zip(ids, values, strict=True)}
    if any(not math.isfinite(value) for value in result.values()):
        raise RoutingError("router returned a non-finite prediction")
    return result


def midrank_percentile(value: float, population: Sequence[float]) -> float:
    if not population:
        raise RoutingError("midrank percentile requires a non-empty population")
    return (
        sum(float(item) < float(value) for item in population)
        + 0.5 * sum(float(item) == float(value) for item in population)
    ) / len(population)


_RISK_FEATURES = {
    "confidence_tail_risk": "ocr_confidence_p10",
    "unmatched_ocr_ratio": "unmatched_ocr_ratio",
    "empty_cell_ratio": "empty_cell_ratio",
}


def _risk_feature_value(features: Mapping[str, Any], risk_name: str) -> float:
    source_name = _RISK_FEATURES.get(risk_name)
    if source_name is None:
        raise RoutingError(f"unsupported calibration risk feature: {risk_name}")
    value = _finite_or_none(features.get(source_name))
    if value is None:
        return 1.0
    return 1.0 - value if risk_name == "confidence_tail_risk" else value


def risk_populations(
    calibration_records: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, list[float]]:
    settings = routing_settings(protocol)
    configured_features = (*settings["active_risk_features"], *settings["inactive_risk_features"])
    if any(name not in _RISK_FEATURES for name in configured_features):
        raise RoutingError("protocol contains an unsupported calibration risk feature")
    if settings["combined_risk"] != "maximum active-feature percentile":
        raise RoutingError("protocol contains an unsupported combined-risk definition")
    vietfintab = protocol.get("vietfintab")
    calibration = vietfintab.get("calibration") if isinstance(vietfintab, Mapping) else None
    if not isinstance(calibration, list) or len(calibration) != 50:
        raise RoutingError("protocol does not define exactly 50 calibration records")
    ids = [item.get("sample_id") for item in calibration if isinstance(item, Mapping)]
    if len(ids) != 50 or len(set(ids)) != 50 or any(not isinstance(item, str) for item in ids):
        raise RoutingError("protocol calibration records are invalid")
    if set(ids) != set(calibration_records):
        raise RoutingError("calibration records do not match the protocol")
    values = {name: [] for name in _RISK_FEATURES}
    for sample_id in ids:
        features = calibration_records[sample_id].get("features")
        if not isinstance(features, Mapping):
            raise RoutingError(f"calibration record lacks features: {sample_id}")
        for name in values:
            values[name].append(_risk_feature_value(features, name))
    return values


def _risk_records(
    sample_ids: Sequence[str],
    feature_records: Mapping[str, Mapping[str, Any]],
    populations: Mapping[str, Sequence[float]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    settings = routing_settings(protocol)
    result: list[dict[str, Any]] = []
    for sample_id in sorted(str(item) for item in sample_ids):
        features = feature_records[sample_id]
        raw = {name: _risk_feature_value(features, name) for name in _RISK_FEATURES}
        percentile = {
            name: midrank_percentile(raw[name], populations[name])
            for name in _RISK_FEATURES
        }
        result.append({
            "sample_id": sample_id,
            "confidence_tail_risk": raw["confidence_tail_risk"],
            "unmatched_ocr_ratio": raw["unmatched_ocr_ratio"],
            "confidence_tail_risk_percentile": percentile["confidence_tail_risk"],
            "unmatched_ocr_ratio_percentile": percentile["unmatched_ocr_ratio"],
            "combined_risk": max(percentile[name] for name in settings["active_risk_features"]),
        })
    return result


def rank_by_score(sample_ids: Sequence[str], scores: Mapping[str, float]) -> list[str]:
    return sorted((str(item) for item in sample_ids), key=lambda item: (-float(scores[item]), item))


def build_risk_rankings(
    sample_ids: Sequence[str],
    feature_records: Mapping[str, Mapping[str, Any]],
    populations: Mapping[str, Sequence[float]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    records = _risk_records(sample_ids, feature_records, populations, protocol)
    confidence = {item["sample_id"]: item["confidence_tail_risk_percentile"] for item in records}
    combined = {item["sample_id"]: item["combined_risk"] for item in records}
    return records, {
        "confidence_tail_only": rank_by_score(sample_ids, confidence),
        "combined_risk": rank_by_score(sample_ids, combined),
    }


def rank_by_predicted_benefit(sample_ids: Sequence[str], predictions: Mapping[str, float]) -> list[str]:
    return rank_by_score(sample_ids, predictions)


def route_for_budget(ranked_ids: Sequence[str], budget: int, allowed_budgets: Sequence[int]) -> list[str]:
    budgets = tuple(allowed_budgets)
    if budget not in budgets:
        raise RoutingError(f"unsupported routing budget: {budget}")
    if budget < 0 or budget > len(ranked_ids):
        raise RoutingError("route budget exceeds sample count")
    return list(ranked_ids[:budget])


def deterministic_random_route(
    sample_ids: Sequence[str],
    budget: int,
    repetition: int,
    *,
    seed: str,
    allowed_budgets: Sequence[int],
) -> list[str]:
    budgets = tuple(allowed_budgets)
    if budget not in budgets or budget < 0 or budget > len(sample_ids):
        raise RoutingError("unsupported random route budget")
    ranked = sorted(
        (str(item) for item in sample_ids),
        key=lambda item: (hashlib.sha256(f"{seed}:{repetition}:{item}".encode("utf-8")).hexdigest(), item),
    )
    return sorted(ranked[:budget])


def _score(value: Any) -> float:
    finite = _finite_or_none(value)
    return finite if finite is not None else 0.0


def route_score_vector(
    sample_records: Mapping[str, Mapping[str, Any]],
    route_ids: Sequence[str],
    *,
    expert: str = "hunyuan",
) -> tuple[list[float], list[float]]:
    route = set(route_ids)
    con: list[float] = []
    top: list[float] = []
    con_key = f"{expert}_standalone_grits_con"
    top_key = f"{expert}_standalone_grits_top"
    success_key = f"{expert}_success"
    for sample_id in sorted(sample_records):
        record = sample_records[sample_id]
        use_expert = sample_id in route and bool(record.get(success_key, record.get(f"{expert}_status") == "success"))
        con.append(_score(record.get(con_key) if use_expert else record.get("primary_grits_con")))
        top.append(_score(record.get(top_key) if use_expert else record.get("primary_grits_top")))
    return con, top


def _fraction_oracle_gain(hybrid: float, primary: float, oracle: float) -> float | None:
    denominator = oracle - primary
    return None if denominator == 0.0 else (hybrid - primary) / denominator


def _delta_class(delta: float) -> str:
    if delta > TIE_TOLERANCE:
        return "improved"
    if delta < -TIE_TOLERANCE:
        return "harmed"
    return "unchanged"


def evaluate_route(
    sample_records: Mapping[str, Mapping[str, Any]],
    route_ids: Sequence[str],
    *,
    budget: int,
    routing_method: str,
    primary_cost: float = 0.0,
    expert: str = "hunyuan",
    allowed_budgets: Sequence[int] | None = None,
) -> dict[str, Any]:
    sample_ids = sorted(sample_records)
    route = list(dict.fromkeys(str(item) for item in route_ids))
    if len(route) != budget or len(route) != len(set(route)):
        raise RoutingError("route does not contain exactly its requested budget")
    if set(route).difference(sample_ids):
        raise RoutingError("route contains a sample outside the input population")
    success_key = f"{expert}_success"
    con_key = f"{expert}_standalone_grits_con"
    top_key = f"{expert}_standalone_grits_top"
    replacements = [item for item in route if bool(sample_records[item].get(success_key, sample_records[item].get(f"{expert}_status") == "success"))]
    gains: list[float] = []
    losses: list[float] = []
    unchanged = 0
    for sample_id in replacements:
        benefit_key = f"{expert}_call_benefit_grits_con"
        delta = float(
            sample_records[sample_id].get(
                benefit_key,
                sample_records[sample_id].get("call_benefit_grits_con", 0.0),
            )
        )
        category = _delta_class(delta)
        if category == "improved":
            gains.append(delta)
        elif category == "harmed":
            losses.append(-delta)
        else:
            unchanged += 1
    hybrid_con, hybrid_top = route_score_vector(sample_records, route, expert=expert)
    primary_con = [_score(sample_records[item].get("primary_grits_con")) for item in sample_ids]
    primary_top = [_score(sample_records[item].get("primary_grits_top")) for item in sample_ids]
    expert_con = [_score(sample_records[item].get(con_key)) for item in sample_ids]
    expert_top = [_score(sample_records[item].get(top_key)) for item in sample_ids]
    oracle_con = [max(left, right) for left, right in zip(primary_con, expert_con, strict=True)]
    oracle_top = [max(left, right) for left, right in zip(primary_top, expert_top, strict=True)]
    primary_score = _mean(primary_con) or 0.0
    primary_top_score = _mean(primary_top) or 0.0
    expert_score = _mean(expert_con) or 0.0
    expert_top_score = _mean(expert_top) or 0.0
    hybrid_score = _mean(hybrid_con) or 0.0
    hybrid_top_score = _mean(hybrid_top) or 0.0
    oracle_score = _mean(oracle_con) or 0.0
    oracle_top_score = _mean(oracle_top) or 0.0
    expert_cost = sum(_score(sample_records[item].get(f"{expert}_inference_seconds")) for item in route)
    harm_rate = len(losses) / len(replacements) if replacements else None
    return {
        "routing_method": routing_method,
        "requested_budget": budget,
        "expert_call_count": len(route),
        "expert_call_rate": len(route) / len(sample_ids) if sample_ids else 0.0,
        "successful_expert_return_count": len(replacements),
        "failed_expert_return_count": len(route) - len(replacements),
        "replacement_count": len(replacements),
        "replacement_rate": len(replacements) / len(sample_ids) if sample_ids else 0.0,
        "improved_replacement_count": len(gains),
        "harmed_replacement_count": len(losses),
        "unchanged_replacement_count": unchanged,
        "harmed_replacement_rate": harm_rate,
        "harmed_replacement_rate_reason": None if harm_rate is not None else "no replacements",
        "mean_gain_on_improvements": _mean(gains),
        "mean_loss_on_harms": _mean(losses),
        "primary_full_set_grits_con": primary_score,
        "primary_full_set_grits_top": primary_top_score,
        "expert_only_full_set_grits_con": expert_score,
        "expert_only_full_set_grits_top": expert_top_score,
        "hybrid_full_set_grits_con": hybrid_score,
        "hybrid_full_set_grits_top": hybrid_top_score,
        "oracle_full_set_grits_con": oracle_score,
        "oracle_full_set_grits_top": oracle_top_score,
        "gain_over_always_primary_grits_con": hybrid_score - primary_score,
        "gain_over_always_primary_grits_top": hybrid_top_score - primary_top_score,
        "fraction_oracle_gain_recovered_grits_con": _fraction_oracle_gain(hybrid_score, primary_score, oracle_score),
        "fraction_oracle_gain_recovered_grits_top": _fraction_oracle_gain(hybrid_top_score, primary_top_score, oracle_top_score),
        "primary_warm_inference_cost_seconds": primary_cost,
        "expert_warm_inference_cost_seconds": expert_cost,
        "total_warm_inference_cost_seconds": primary_cost + expert_cost,
        "mean_warm_inference_cost_per_sample_seconds": (primary_cost + expert_cost) / len(sample_ids) if sample_ids else 0.0,
        "routed_sample_ids": sorted(route),
        "replacement_sample_ids": sorted(replacements),
        "replacement_rule": "successful expert output replaces the primary; failed expert calls retain the primary",
        "quality_gate_enabled": False,
    }


def random_routes(
    sample_ids: Sequence[str],
    budget: int,
    *,
    repetitions: int,
    seed: str,
    allowed_budgets: Sequence[int],
) -> list[list[str]]:
    return [
        deterministic_random_route(sample_ids, budget, repetition, seed=seed, allowed_budgets=allowed_budgets)
        for repetition in range(repetitions)
    ]


def random_mean_vector(
    sample_records: Mapping[str, Mapping[str, Any]],
    *,
    budget: int,
    repetitions: int,
    seed: str,
    expert: str = "hunyuan",
    allowed_budgets: Sequence[int],
) -> tuple[list[float], list[float]]:
    sample_ids = sorted(sample_records)
    sums = [0.0] * len(sample_ids)
    route_scores: list[float] = []
    for route in random_routes(sample_ids, budget, repetitions=repetitions, seed=seed, allowed_budgets=allowed_budgets):
        con, _ = route_score_vector(sample_records, route, expert=expert)
        route_scores.append(_mean(con) or 0.0)
        for index, value in enumerate(con):
            sums[index] += value
    return [value / repetitions for value in sums], route_scores


def random_summary(
    sample_records: Mapping[str, Mapping[str, Any]],
    *,
    budgets: Sequence[int],
    main_budget: int,
    repetitions: int,
    seed: str,
    primary_cost: float = 0.0,
    expert: str = "hunyuan",
) -> tuple[dict[str, Any], list[float] | None]:
    result: dict[str, Any] = {
        "seed": seed,
        "repetitions": repetitions,
        "sampling": "rank sorted sample IDs by SHA256(seed:repetition:sample_id), then select the first budget IDs",
        "budgets": {},
    }
    main_vector: list[float] | None = None
    for budget in budgets:
        routes = random_routes(sample_records.keys(), budget, repetitions=repetitions, seed=seed, allowed_budgets=budgets)
        evaluated = [
            evaluate_route(sample_records, route, budget=budget, routing_method="random_matched_budget", primary_cost=primary_cost, expert=expert, allowed_budgets=budgets)
            for route in routes
        ]
        con_values = [float(item["hybrid_full_set_grits_con"]) for item in evaluated]
        top_values = [float(item["hybrid_full_set_grits_top"]) for item in evaluated]
        result["budgets"][str(budget)] = {
            "budget": budget,
            "mean_hybrid_full_set_grits_con": _mean(con_values),
            "median_hybrid_full_set_grits_con": statistics.median(con_values),
            "percentile_95_interval_hybrid_full_set_grits_con": {"lower": _percentile(con_values, 0.025), "upper": _percentile(con_values, 0.975)},
            "mean_hybrid_full_set_grits_top": _mean(top_values),
            "mean_expert_cost_seconds": _mean([float(item["expert_warm_inference_cost_seconds"]) for item in evaluated]),
        }
        if budget == main_budget:
            main_vector = random_mean_vector(sample_records, budget=budget, repetitions=repetitions, seed=seed, expert=expert, allowed_budgets=budgets)[0]
            result["budgets"][str(budget)]["per_sample_mean_vector_mean"] = _mean(main_vector)
            result["budgets"][str(budget)]["route_aggregate_mean"] = result["budgets"][str(budget)]["mean_hybrid_full_set_grits_con"]
            result["budgets"][str(budget)]["per_sample_vector_matches_route_mean"] = abs(float(_mean(main_vector) or 0.0) - float(result["budgets"][str(budget)]["route_aggregate_mean"])) <= 1e-12
    return result, main_vector


def paired_bootstrap_mean_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: str,
    replicates: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise RoutingError("paired bootstrap inputs must have equal non-zero lengths")
    np, *_ = _import_learning_dependencies()
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    generator_seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(generator_seed)
    indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
    bootstrapped = difference[indices].mean(axis=1)
    return {
        "seed": seed,
        "replicates": replicates,
        "interval": "percentile_95",
        "observed": float(difference.mean()),
        "lower": float(np.percentile(bootstrapped, 2.5)),
        "upper": float(np.percentile(bootstrapped, 97.5)),
    }


def spearman_summary(predicted: Sequence[float], observed: Sequence[float]) -> dict[str, Any]:
    if len(predicted) != len(observed) or not predicted:
        raise RoutingError("Spearman inputs must have equal non-zero lengths")
    if len(set(float(value) for value in predicted)) <= 1:
        return {"value": None, "reason": "constant predicted values"}
    if len(set(float(value) for value in observed)) <= 1:
        return {"value": None, "reason": "constant observed values"}

    def ranks(values: Sequence[float]) -> list[float]:
        ordered = sorted((float(value), index) for index, value in enumerate(values))
        result = [0.0] * len(values)
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][0] == ordered[index][0]:
                end += 1
            rank = (index + 1 + end) / 2.0
            for position in range(index, end):
                result[ordered[position][1]] = rank
            index = end
        return result

    left, right = ranks(predicted), ranks(observed)
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right))
    return {"value": numerator / denominator if denominator else None, "reason": None if denominator else "constant ranked values"}


def standalone_summary(sample_records: Mapping[str, Mapping[str, Any]], *, system: str = "primary", expert: str = "hunyuan") -> dict[str, Any]:
    sample_ids = sorted(sample_records)
    if system == "primary":
        statuses = [str(sample_records[item].get("primary_status")) for item in sample_ids]
        values_con = [_score(sample_records[item].get("primary_grits_con")) for item in sample_ids]
        values_top = [_score(sample_records[item].get("primary_grits_top")) for item in sample_ids]
    else:
        statuses = [str(sample_records[item].get(f"{expert}_status")) for item in sample_ids]
        values_con = [_score(sample_records[item].get(f"{expert}_standalone_grits_con")) if sample_records[item].get(f"{expert}_success") else 0.0 for item in sample_ids]
        values_top = [_score(sample_records[item].get(f"{expert}_standalone_grits_top")) if sample_records[item].get(f"{expert}_success") else 0.0 for item in sample_ids]
    success_count = sum(status == "success" for status in statuses)
    return {
        "system": system,
        "sample_count": len(sample_ids),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "successful_canonicalization_count": success_count,
        "successful_canonicalization_coverage": success_count / len(sample_ids) if sample_ids else 0.0,
        "coverage_aware": {"mean_grits_con": _mean(values_con), "mean_grits_top": _mean(values_top)},
        "successful_only": {
            "sample_count": success_count,
            "mean_grits_con": _mean([values_con[index] for index, status in enumerate(statuses) if status == "success"]),
            "mean_grits_top": _mean([values_top[index] for index, status in enumerate(statuses) if status == "success"]),
            "note": "conditional on successful canonicalization; failures are not omitted from coverage-aware means",
        },
    }
