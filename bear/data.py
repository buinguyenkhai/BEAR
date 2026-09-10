"""VietFinTab data access and annotation parsing for BEAR."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from collections.abc import Sequence
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from grits import TableCell

from .routing import FEATURE_NAMES, MODEL_IDS, normalize_text_for_cer


class DataError(RuntimeError):
    """Raised when VietFinTab data cannot satisfy the configured protocol."""


SUPPORTED_STRUCTURE_CATEGORIES = frozenset(
    {
        "table",
        "table row",
        "table column",
        "table column header",
        "table projected row header",
        "table spanning cell",
    }
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _required_direct_child(element: ET.Element, name: str, *, context: str) -> ET.Element:
    children = _direct_children(element, name)
    if len(children) != 1:
        raise ValueError(f"{context} must contain exactly one direct <{name}> child")
    return children[0]


def _structure_number(element: ET.Element, *, context: str) -> int | float:
    value = element.text
    if value is None or not value.strip():
        raise DataError(f"{context} must contain a numeric value")
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{context} must contain a numeric value") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context} must contain a finite numeric value")
    return int(parsed) if parsed.is_integer() else parsed


def _structure_box(element: ET.Element, *, context: str) -> list[int | float]:
    box = _required_direct_child(element, "bndbox", context=context)
    values = [
        _structure_number(
            _required_direct_child(box, field, context=f"{context}.bndbox"),
            context=f"{context}.bndbox.{field}",
        )
        for field in ("xmin", "ymin", "xmax", "ymax")
    ]
    xmin, ymin, xmax, ymax = values
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"{context} must have positive area")
    return values


def parse_structure_xml(payload: bytes | str) -> dict[str, object]:
    """Parse one source VietFinTab structure annotation."""

    if not isinstance(payload, (bytes, str)):
        raise TypeError("structure XML payload must be bytes or str")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError(f"invalid structure XML: {exc}") from exc

    sizes = [element for element in root.iter() if _local_name(element.tag) == "size"]
    if len(sizes) != 1:
        raise ValueError("structure XML must contain exactly one <size>")
    size = sizes[0]
    width = _structure_number(
        _required_direct_child(size, "width", context="size"),
        context="size.width",
    )
    height = _structure_number(
        _required_direct_child(size, "height", context="size"),
        context="size.height",
    )
    if width <= 0 or height <= 0:
        raise ValueError("structure XML image dimensions must be positive")

    objects: list[dict[str, object]] = []
    for object_index, object_element in enumerate(
        element for element in root.iter() if _local_name(element.tag) == "object"
    ):
        context = f"object[{object_index}]"
        name = _required_direct_child(object_element, "name", context=context).text
        if name is None or not name.strip():
            raise ValueError(f"{context}.name must contain a non-empty category")
        category = name.strip()
        if category not in SUPPORTED_STRUCTURE_CATEGORIES:
            raise ValueError(f"{context} has unsupported category {category!r}")
        objects.append({"category": category, "bounding_box": _structure_box(object_element, context=context)})
    return {"image_size": [width, height], "objects": objects}


def parse_content_json(payload: bytes | str) -> dict[str, object]:
    """Parse one source VietFinTab content annotation."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("content JSON payload must be valid UTF-8") from exc
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError("content JSON payload must be bytes or str")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid content JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("ocr"), list):
        raise ValueError("content JSON must contain an 'ocr' array")
    records: list[dict[str, object]] = []
    for index, item in enumerate(value["ocr"]):
        context = f"ocr[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{context} must be an object")
        text_value = item.get("text")
        if not isinstance(text_value, str):
            raise ValueError(f"{context}.text must be a string")
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{context}.bbox must contain four coordinates")
        values: list[int | float] = []
        for coordinate_index, coordinate in enumerate(box):
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError(f"{context}.bbox[{coordinate_index}] must be numeric")
            if not math.isfinite(float(coordinate)):
                raise ValueError(f"{context}.bbox[{coordinate_index}] must be finite")
            values.append(coordinate)
        xmin, ymin, xmax, ymax = values
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"{context}.bbox must have positive area")
        records.append({"text": text_value, "bounding_box": values})
    return {"records": records}


def _spanning_from_structure_xml(payload: bytes, sample_id: str) -> bool:
    try:
        parsed = parse_structure_xml(payload)
    except (TypeError, ValueError) as exc:
        raise DataError(f"invalid structure XML for {sample_id}: {exc}") from exc
    return any(
        item.get("category") == "table spanning cell"
        for item in parsed["objects"]
        if isinstance(item, Mapping)
    )


_Box = tuple[int | float, int | float, int | float, int | float]


def _reference_box(value: Any, *, context: str) -> _Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise DataError(f"{context} must contain four coordinates")
    values: list[int | float] = []
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise DataError(f"{context}[{index}] must be numeric")
        if not math.isfinite(float(coordinate)):
            raise DataError(f"{context}[{index}] must be finite")
        values.append(coordinate)
    xmin, ymin, xmax, ymax = values
    if xmin >= xmax or ymin >= ymax:
        raise DataError(f"{context} must have positive area")
    return xmin, ymin, xmax, ymax


def _reference_intersection(left: _Box, right: _Box) -> _Box | None:
    result = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return None if result[0] >= result[2] or result[1] >= result[3] else result


def _reference_area(box: _Box) -> float:
    return float(box[2] - box[0]) * float(box[3] - box[1])


def _reference_extent(boxes: Sequence[_Box]) -> _Box:
    if not boxes:
        raise DataError("cannot calculate an extent for no boxes")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _reference_structure_parts(
    structure_payload: Mapping[str, object],
) -> tuple[_Box, list[tuple[str, _Box]], list[tuple[str, _Box]], list[tuple[str, _Box]], list[tuple[str, _Box]]]:
    if not isinstance(structure_payload, Mapping):
        raise DataError("structure payload must be an object")
    image_size = structure_payload.get("image_size")
    if image_size is not None:
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise DataError("structure_payload.image_size must contain width and height")
        for index, dimension in enumerate(image_size):
            if isinstance(dimension, bool) or not isinstance(dimension, (int, float)) or not math.isfinite(float(dimension)) or dimension <= 0:
                raise DataError(f"structure_payload.image_size[{index}] must be positive and finite")
    objects = structure_payload.get("objects")
    if not isinstance(objects, list):
        raise DataError("structure_payload.objects must be an array")
    tables: list[tuple[str, _Box]] = []
    rows: list[tuple[str, _Box]] = []
    columns: list[tuple[str, _Box]] = []
    spans: list[tuple[str, _Box]] = []
    headers: list[tuple[str, _Box]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise DataError(f"structure_payload.objects[{index}] must be an object")
        category = item.get("category")
        if not isinstance(category, str) or not category.strip():
            raise DataError(f"structure_payload.objects[{index}].category must be non-empty")
        category = category.strip()
        if category not in SUPPORTED_STRUCTURE_CATEGORIES:
            raise DataError(f"unsupported structure category: {category!r}")
        value = (category, _reference_box(item.get("bounding_box"), context=f"structure_payload.objects[{index}].bounding_box"))
        if category == "table":
            tables.append(value)
        elif category == "table row":
            rows.append(value)
        elif category == "table column":
            columns.append(value)
        elif category in {"table spanning cell", "table projected row header"}:
            spans.append(value)
        else:
            headers.append(value)
    if len(tables) != 1:
        raise DataError("structure payload must contain exactly one table object")
    if not rows or not columns:
        raise DataError("structure payload must contain rows and columns")
    return tables[0][1], rows, columns, spans, headers


def _reference_object_sort(item: tuple[str, _Box], axis: str) -> tuple[float, int | float, int | float, int | float, int | float]:
    _, box = item
    if axis == "row":
        return ((float(box[1]) + float(box[3])) / 2, box[1], box[0], box[3], box[2])
    return ((float(box[0]) + float(box[2])) / 2, box[0], box[1], box[2], box[3])


def _reference_cell_sort(cell: TableCell) -> tuple[Any, ...]:
    if cell.bbox is None:
        raise DataError("reference cell is missing a bounding box")
    return (
        min(cell.row_nums),
        min(cell.column_nums),
        len(cell.row_nums),
        len(cell.column_nums),
        tuple(cell.bbox),
    )


def structure_payload_to_cells(structure_payload: Mapping[str, object]) -> list[TableCell]:
    """Construct canonical structure cells from source objects."""

    table_box, row_objects, column_objects, span_objects, header_objects = _reference_structure_parts(structure_payload)
    rows = sorted(row_objects, key=lambda item: _reference_object_sort(item, "row"))
    columns = sorted(column_objects, key=lambda item: _reference_object_sort(item, "column"))
    base: dict[tuple[int, int], TableCell] = {}
    for row_index, (_, row_box) in enumerate(rows):
        for column_index, (_, column_box) in enumerate(columns):
            overlap = _reference_intersection(row_box, column_box)
            clipped = _reference_intersection(table_box, overlap) if overlap is not None else None
            if clipped is None:
                raise DataError("row-column intersection has zero area after table clipping")
            base[(row_index, column_index)] = TableCell(
                row_nums=[row_index], column_nums=[column_index], cell_text="", bbox=list(clipped)
            )
    active = dict(base)
    occupied: set[tuple[int, int]] = set()
    for category, span_box in sorted(span_objects, key=lambda item: (item[0], item[1])):
        covered: list[tuple[int, int]] = []
        for coordinates, cell in base.items():
            cell_box = _reference_box(cell.bbox, context="base cell bbox")
            overlap = _reference_intersection(span_box, cell_box)
            if overlap is not None and _reference_area(overlap) / _reference_area(cell_box) > 0.5:
                covered.append(coordinates)
        if not covered:
            raise DataError(f"{category} annotation does not cover a base cell")
        if any(item in occupied for item in covered):
            raise DataError("spanning annotations overlap or contradict one another")
        row_indices = sorted({row for row, _ in covered})
        column_indices = sorted({column for _, column in covered})
        if row_indices != list(range(row_indices[0], row_indices[-1] + 1)):
            raise DataError("spanning annotation covers non-contiguous rows")
        if column_indices != list(range(column_indices[0], column_indices[-1] + 1)):
            raise DataError("spanning annotation covers non-contiguous columns")
        expected = {(row, column) for row in row_indices for column in column_indices}
        if set(covered) != expected:
            raise DataError("spanning annotation does not cover a complete rectangle")
        row_extent = _reference_extent([rows[index][1] for index in row_indices])
        column_extent = _reference_extent([columns[index][1] for index in column_indices])
        span_extent = _reference_intersection(row_extent, column_extent)
        clipped = _reference_intersection(table_box, span_extent) if span_extent is not None else None
        if clipped is None:
            raise DataError("spanning annotation produces a zero-area cell")
        for item in covered:
            active.pop(item, None)
        active[(row_indices[0], column_indices[0])] = TableCell(
            row_nums=row_indices, column_nums=column_indices, cell_text="", bbox=list(clipped)
        )
        occupied.update(covered)
    output: list[TableCell] = []
    for cell in active.values():
        cell_box = _reference_box(cell.bbox, context="reference cell bbox")
        is_header = any(
            (overlap := _reference_intersection(cell_box, header_box)) is not None
            and _reference_area(overlap) / _reference_area(cell_box) > 0.5
            for _, header_box in header_objects
        )
        output.append(TableCell(
            row_nums=list(cell.row_nums), column_nums=list(cell.column_nums), cell_text="",
            bbox=list(cell_box), is_column_header=is_header, is_row_header=False,
        ))
    return sorted(output, key=_reference_cell_sort)


def attach_text_records(
    cells: Sequence[TableCell],
    content_payload: Mapping[str, object],
    *,
    minimum_overlap: float = 0.5,
) -> tuple[list[TableCell], dict[str, int]]:
    """Attach source OCR content to canonical structure cells."""

    if isinstance(minimum_overlap, bool) or not isinstance(minimum_overlap, (int, float)) or not math.isfinite(float(minimum_overlap)) or not 0 <= minimum_overlap <= 1:
        raise DataError("minimum_overlap must be between 0 and 1")
    try:
        canonical_cells = sorted(list(cells), key=_reference_cell_sort)
    except TypeError as exc:
        raise DataError("cells must be a sequence") from exc
    copied: list[TableCell] = []
    for index, cell in enumerate(canonical_cells):
        if not isinstance(cell, TableCell):
            raise DataError(f"cells[{index}] must be a grits.TableCell")
        if not isinstance(cell.row_nums, list) or not cell.row_nums:
            raise DataError(f"cells[{index}].row_nums must be a non-empty list")
        if not isinstance(cell.column_nums, list) or not cell.column_nums:
            raise DataError(f"cells[{index}].column_nums must be a non-empty list")
        for field, values in (("row_nums", cell.row_nums), ("column_nums", cell.column_nums)):
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise DataError(f"cells[{index}].{field} must contain non-negative integers")
            if values != sorted(set(values)):
                raise DataError(f"cells[{index}].{field} must be sorted and unique")
        if not isinstance(cell.cell_text, str):
            raise DataError(f"cells[{index}].cell_text must be a string")
        if not isinstance(cell.is_column_header, bool) or not isinstance(cell.is_row_header, bool):
            raise DataError(f"cells[{index}] header flags must be booleans")
        bbox = _reference_box(cell.bbox, context=f"cells[{index}].bbox")
        copied.append(TableCell(
            row_nums=list(cell.row_nums), column_nums=list(cell.column_nums), cell_text="",
            bbox=list(bbox), is_column_header=bool(cell.is_column_header), is_row_header=bool(cell.is_row_header),
        ))
    records = content_payload.get("records") if isinstance(content_payload, Mapping) else None
    if not isinstance(records, list):
        raise DataError("content_payload.records must be an array")
    assignments: list[list[tuple[int, _Box, str]]] = [[] for _ in copied]
    assigned = 0
    ties = 0
    for record_index, item in enumerate(records):
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
            raise DataError(f"content_payload.records[{record_index}] is malformed")
        record_box = _reference_box(item.get("bounding_box"), context=f"content_payload.records[{record_index}].bounding_box")
        overlaps: list[float] = []
        for cell in copied:
            cell_box = _reference_box(cell.bbox, context="canonical cell bbox")
            overlap = _reference_intersection(record_box, cell_box)
            overlaps.append(0.0 if overlap is None else _reference_area(overlap) / _reference_area(record_box))
        if not overlaps:
            continue
        best = max(overlaps)
        choices = [index for index, value in enumerate(overlaps) if value == best]
        if best >= float(minimum_overlap):
            if len(choices) > 1:
                ties += 1
            assignments[choices[0]].append((record_index, record_box, item["text"]))
            assigned += 1
    output: list[TableCell] = []
    nonempty = 0
    for cell, assigned_records in zip(copied, assignments, strict=True):
        fragments = []
        for _, _, text in sorted(assigned_records, key=lambda value: (value[1][1], value[1][0], value[1][3], value[1][2], value[0])):
            normalized = normalize_text_for_cer(text)
            if normalized:
                fragments.append(normalized)
        cell_text = " ".join(fragments)
        if cell_text:
            nonempty += 1
        output.append(TableCell(
            row_nums=list(cell.row_nums), column_nums=list(cell.column_nums), cell_text=cell_text,
            bbox=list(cell.bbox), is_column_header=cell.is_column_header, is_row_header=cell.is_row_header,
        ))
    return output, {
        "record_count": len(records),
        "assigned_record_count": assigned,
        "unassigned_record_count": len(records) - assigned,
        "ambiguous_tie_count": ties,
        "cell_count": len(output),
        "nonempty_cell_count": nonempty,
    }


def load_protocol(path: str | Path = "protocol.json") -> dict[str, Any]:
    """Load and validate the scientific protocol."""

    protocol_path = Path(path)
    try:
        value = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"could not load protocol: {protocol_path}") from exc
    _load_membership_manifest(value, protocol_path)
    validate_protocol(value)
    return value


def _load_membership_manifest(protocol: dict[str, Any], protocol_path: Path) -> None:
    """Attach the cohort memberships stored beside the experiment configuration."""

    vietfintab = protocol.get("vietfintab")
    if not isinstance(vietfintab, dict):
        raise DataError("protocol.vietfintab must be an object")
    manifest_reference = vietfintab.get("manifest")
    if not isinstance(manifest_reference, str) or not manifest_reference:
        raise DataError("protocol.vietfintab.manifest must name a membership manifest")
    manifest_path = protocol_path.parent / PurePosixPath(manifest_reference)
    if PurePosixPath(manifest_reference).is_absolute() or ".." in PurePosixPath(manifest_reference).parts:
        raise DataError("protocol.vietfintab.manifest must be a relative repository path")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"could not load VietFinTab membership manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise DataError("VietFinTab membership manifest must be an object")
    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise DataError("protocol.dataset must be an object")
    if manifest.get("dataset_id") != dataset.get("id"):
        raise DataError("VietFinTab membership manifest has the wrong dataset ID")
    if manifest.get("revision") != dataset.get("revision"):
        raise DataError("VietFinTab membership manifest has the wrong dataset revision")
    splits = manifest.get("splits")
    calibration = manifest.get("calibration")
    if not isinstance(splits, Mapping) or not isinstance(calibration, list):
        raise DataError("VietFinTab membership manifest is missing cohort memberships")
    vietfintab["splits"] = splits
    vietfintab["calibration"] = calibration


def _parse_protocol_entries(
    entries: Any,
    *,
    name: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise DataError(f"protocol cohort {name!r} must contain {expected_count} records")
    result: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise DataError(f"protocol cohort {name!r} contains a non-object record")
        sample_id = item.get("sample_id")
        issuer = item.get("issuer")
        spanning = item.get("spanning")
        if not isinstance(sample_id, str) or not sample_id or "/" in sample_id:
            raise DataError(f"invalid sample ID in protocol cohort {name!r}")
        if not isinstance(issuer, str) or not issuer:
            raise DataError(f"invalid issuer for {sample_id}")
        if not isinstance(spanning, bool):
            raise DataError(f"invalid spanning flag for {sample_id}")
        result.append({"sample_id": sample_id, "issuer": issuer, "spanning": spanning})
    return result


def _split_entries(protocol: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    splits = protocol.get("vietfintab", {}).get("splits")
    if not isinstance(splits, Mapping):
        raise DataError("protocol.vietfintab.splits must be an object")
    result = {
        name: _parse_protocol_entries(splits.get(name), name=name, expected_count=count)
        for name, count in (("development", 600), ("final", 300), ("reserve", 35))
    }
    identifiers = [item["sample_id"] for entries in result.values() for item in entries]
    if len(identifiers) != len(set(identifiers)):
        raise DataError("protocol split memberships overlap or contain duplicates")
    final = result["final"]
    if sum(bool(item["spanning"]) for item in final) != 69:
        raise DataError("final split spanning count is not 69")
    if sum(not bool(item["spanning"]) for item in final) != 231:
        raise DataError("final split non-spanning count is not 231")
    development_issuers = {item["issuer"] for item in result["development"]}
    final_issuers = {item["issuer"] for item in final}
    if len(development_issuers) != 13 or development_issuers != final_issuers:
        raise DataError("development and final issuer sets do not match the protocol")
    return result


def _calibration_entries(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = protocol.get("vietfintab", {}).get("calibration")
    return _parse_protocol_entries(value, name="calibration", expected_count=50)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the scientific inputs needed by the BEAR workflow."""

    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise DataError("protocol.dataset must be an object")
    if not isinstance(dataset.get("id"), str) or not dataset["id"]:
        raise DataError("protocol.dataset.id must be a non-empty string")
    if not isinstance(dataset.get("revision"), str) or not dataset["revision"]:
        raise DataError("protocol.dataset.revision must be a non-empty string")
    features = protocol.get("features")
    if features != list(FEATURE_NAMES):
        raise DataError("protocol feature order does not match the specified 13-feature schema")
    splits = _split_entries(protocol)
    _validate_routing_settings(protocol, splits)


def _validate_routing_settings(
    protocol: Mapping[str, Any],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Validate routing and evaluation shapes while leaving values in the protocol."""

    def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise DataError(f"{context} must be an object")
        return value

    def require_string(value: Any, context: str) -> None:
        if not isinstance(value, str) or not value:
            raise DataError(f"{context} must be a non-empty string")

    def require_positive_integer(value: Any, context: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DataError(f"{context} must be a positive integer")

    def require_finite_number(value: Any, context: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DataError(f"{context} must be a finite number")

    evaluation = require_mapping(protocol.get("evaluation"), "protocol.evaluation")
    for field in (
        "package",
        "version",
        "top_reference",
        "con_reference",
        "loc_reference",
    ):
        require_string(evaluation.get(field), f"protocol.evaluation.{field}")
    for field in ("content_attachment_minimum_overlap", "standalone_failure_score"):
        require_finite_number(evaluation.get(field), f"protocol.evaluation.{field}")
    overlap = float(evaluation["content_attachment_minimum_overlap"])
    if not 0.0 <= overlap <= 1.0:
        raise DataError("protocol.evaluation.content_attachment_minimum_overlap must be in [0, 1]")
    for field in ("table_metrics", "headline_metrics"):
        values = evaluation.get(field)
        if not isinstance(values, list) or not values or any(not isinstance(item, str) or not item for item in values):
            raise DataError(f"protocol.evaluation.{field} must be a non-empty string list")
    if not {"top", "con", "loc"}.issubset(evaluation["table_metrics"]):
        raise DataError("protocol.evaluation.table_metrics must include top, con, and loc")
    if not {"top", "con"}.issubset(evaluation["headline_metrics"]):
        raise DataError("protocol.evaluation.headline_metrics must include top and con")

    routing = require_mapping(protocol.get("routing"), "protocol.routing")
    for field in ("training_expert", "transfer_expert", "target", "replacement_rule"):
        require_string(routing.get(field), f"protocol.routing.{field}")
    require_finite_number(routing.get("expert_failure_target"), "protocol.routing.expert_failure_target")
    if not isinstance(routing.get("post_expert_quality_gate"), bool):
        raise DataError("protocol.routing.post_expert_quality_gate must be boolean")

    models = require_mapping(routing.get("models"), "protocol.routing.models")
    if set(models) != set(MODEL_IDS):
        raise DataError("protocol.routing.models must define the configured router families")
    required_grid_fields = {
        "ridge_benefit": {"alpha"},
        "hist_gradient_boosting_benefit": {
            "max_depth",
            "learning_rate",
            "max_iter",
            "l2_regularization",
            "early_stopping",
            "random_state",
        },
    }
    for model_id in MODEL_IDS:
        model = require_mapping(models.get(model_id), f"protocol.routing.models.{model_id}")
        for field in ("family", "estimator"):
            require_string(model.get(field), f"protocol.routing.models.{model_id}.{field}")
        require_mapping(model.get("preprocessing"), f"protocol.routing.models.{model_id}.preprocessing")
        grid = require_mapping(model.get("grid"), f"protocol.routing.models.{model_id}.grid")
        if not required_grid_fields[model_id].issubset(grid):
            raise DataError(f"protocol.routing.models.{model_id}.grid is incomplete")
        for name, values in grid.items():
            if not isinstance(values, list) or not values:
                raise DataError(f"protocol.routing.models.{model_id}.grid.{name} must be a non-empty list")

    validation = require_mapping(routing.get("validation"), "protocol.routing.validation")
    for field in ("outer_folds", "inner_folds"):
        require_positive_integer(validation.get(field), f"protocol.routing.validation.{field}")
    for field in ("group", "sample_order", "selection_metric", "tie_break"):
        require_string(validation.get(field), f"protocol.routing.validation.{field}")

    budgets = require_mapping(routing.get("budgets"), "protocol.routing.budgets")
    for name in ("development", "final"):
        values = budgets.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values)
            or values != sorted(set(values))
        ):
            raise DataError(f"protocol.routing.budgets.{name} must be sorted unique non-negative integers")
    main_budgets = require_mapping(budgets.get("main"), "protocol.routing.budgets.main")
    if set(main_budgets) != {"development", "final"}:
        raise DataError("protocol.routing.budgets.main must define both cohorts")
    for name in ("development", "final"):
        main = main_budgets[name]
        if isinstance(main, bool) or not isinstance(main, int) or main not in budgets[name]:
            raise DataError(f"protocol.routing.budgets.main.{name} is not a configured budget")

    random_settings = require_mapping(routing.get("random"), "protocol.routing.random")
    require_string(random_settings.get("seed"), "protocol.routing.random.seed")
    require_positive_integer(random_settings.get("repetitions"), "protocol.routing.random.repetitions")
    require_string(random_settings.get("selection"), "protocol.routing.random.selection")
    bootstrap = require_mapping(routing.get("bootstrap"), "protocol.routing.bootstrap")
    require_string(bootstrap.get("seed"), "protocol.routing.bootstrap.seed")
    require_positive_integer(bootstrap.get("replicates"), "protocol.routing.bootstrap.replicates")
    require_string(bootstrap.get("interval"), "protocol.routing.bootstrap.interval")

    calibration_settings = require_mapping(routing.get("calibration"), "protocol.routing.calibration")
    supported_risk_features = {
        "confidence_tail_risk",
        "unmatched_ocr_ratio",
        "empty_cell_ratio",
    }
    for field in ("active_features", "inactive_features"):
        values = calibration_settings.get(field)
        if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
            raise DataError(f"protocol.routing.calibration.{field} must be a string list")
        if not set(values).issubset(supported_risk_features):
            raise DataError(f"protocol.routing.calibration.{field} names an unsupported risk feature")
    if set(calibration_settings["active_features"]).intersection(calibration_settings["inactive_features"]):
        raise DataError("protocol.routing.calibration active and inactive features overlap")
    for field in ("percentile_rule", "combined_risk"):
        require_string(calibration_settings.get(field), f"protocol.routing.calibration.{field}")

    calibration = _calibration_entries(protocol)
    calibration_ids = [item["sample_id"] for item in calibration]
    if len(calibration_ids) != len(set(calibration_ids)):
        raise DataError("calibration cohort contains duplicate sample IDs")
    split_ids = {item["sample_id"] for entries in splits.values() for item in entries}
    if set(calibration_ids).intersection(split_ids):
        raise DataError("calibration cohort overlaps a learned/final/reserve split")


class VietFinTabDataset:
    """Resolve VietFinTab metadata and selected image/record files."""

    repository_id = "VietFinTabGroup/VietFinTab"

    def __init__(self, protocol: Mapping[str, Any], data_dir: str | Path = "data") -> None:
        validate_protocol(protocol)
        self.protocol = protocol
        dataset = protocol["dataset"]
        self.dataset_id = str(dataset["id"])
        self.revision = str(dataset["revision"])
        self.data_dir = Path(data_dir)
        self._records: dict[str, dict[str, Any]] | None = None
        self._revision_checked = False
        self._structure_checked = False

    @property
    def split_entries(self) -> dict[str, list[dict[str, Any]]]:
        return _split_entries(self.protocol)

    @property
    def calibration_entries(self) -> list[dict[str, Any]]:
        return _calibration_entries(self.protocol)

    @property
    def protocol_entries(self) -> dict[str, list[dict[str, Any]]]:
        entries = self.split_entries
        entries["calibration"] = self.calibration_entries
        return entries

    def resolve_revision(self) -> None:
        """Confirm that the requested dataset revision resolves exactly."""

        if self._revision_checked:
            return
        try:
            from huggingface_hub import HfApi

            info = HfApi().dataset_info(self.dataset_id, revision=self.revision)
        except Exception as exc:
            raise DataError(
                f"could not resolve {self.dataset_id} at the pinned revision"
            ) from exc
        resolved = str(getattr(info, "sha", "") or "")
        if resolved != self.revision:
            raise DataError(
                f"pinned revision resolved to {resolved or 'no revision'}, not {self.revision}"
            )
        if bool(getattr(info, "private", False)) or bool(getattr(info, "gated", False)):
            raise DataError("the pinned VietFinTab revision is not accessible")
        self._revision_checked = True

    def _download(self, repository_path: str) -> Path:
        path = PurePosixPath(repository_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise DataError(f"invalid repository path: {repository_path!r}")
        local_path = self.data_dir / "vietfintab" / Path(*path.parts)
        if local_path.is_file():
            return local_path
        try:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id=self.dataset_id,
                repo_type="dataset",
                revision=self.revision,
                filename=repository_path,
                local_dir=str(self.data_dir / "vietfintab"),
            )
        except Exception as exc:
            raise DataError(f"could not retrieve object {repository_path!r}") from exc
        return Path(local_path)

    def _local_path(self, repository_path: str) -> Path | None:
        path = PurePosixPath(repository_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise DataError(f"invalid repository path: {repository_path!r}")
        candidate = self.data_dir / "vietfintab" / Path(*path.parts)
        return candidate if candidate.is_file() else None

    def _ensure_structure_files(self, records: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
        """Materialize all selected structure files with targeted downloads."""

        root = self.data_dir / "vietfintab"
        local_paths: dict[str, Path] = {}
        missing = False
        for sample_id, record in records.items():
            structure_file = record.get("structure_file")
            if not isinstance(structure_file, str) or not structure_file:
                raise DataError(f"sample {sample_id} has no structure annotation")
            repository_path = PurePosixPath(structure_file)
            if (
                repository_path.is_absolute()
                or ".." in repository_path.parts
                or not repository_path.parts
            ):
                raise DataError(f"invalid structure repository path for {sample_id}")
            local_path = root.joinpath(*repository_path.parts)
            local_paths[sample_id] = local_path
            missing = missing or not local_path.is_file()
        if missing:
            missing_records = [
                (sample_id, str(records[sample_id]["structure_file"]))
                for sample_id, path in local_paths.items()
                if not path.is_file()
            ]
            try:
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {
                        executor.submit(self._download, repository_path): sample_id
                        for sample_id, repository_path in missing_records
                    }
                    for future in as_completed(futures):
                        sample_id = futures[future]
                        downloaded_path = Path(future.result())
                        if not downloaded_path.is_file():
                            raise DataError(
                                f"structure annotation did not materialize for {sample_id}"
                            )
                        local_paths[sample_id] = downloaded_path
            except DataError:
                raise
            except Exception as exc:
                raise DataError("could not retrieve the structure annotations") from exc
        unavailable = sorted(sample_id for sample_id, path in local_paths.items() if not path.is_file())
        if unavailable:
            raise DataError(
                "structure annotations did not materialize for: "
                + ", ".join(unavailable[:5])
            )
        return local_paths

    def load_records(self) -> dict[str, dict[str, Any]]:
        """Load source metadata for the configured VietFinTab cohorts."""

        if self._records is not None:
            return dict(self._records)
        self.resolve_revision()
        metadata_path = self._download("metadata.jsonl")
        metadata: dict[str, dict[str, Any]] = {}
        try:
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DataError("could not read VietFinTab metadata") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"invalid metadata at line {line_number}") from exc
            if not isinstance(item, Mapping):
                raise DataError(f"metadata line {line_number} is not an object")
            sample_id = item.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise DataError(f"metadata line {line_number} has no sample ID")
            if sample_id in metadata:
                raise DataError(f"metadata contains duplicate sample ID {sample_id}")
            file_name = item.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                raise DataError(f"metadata has no image path for {sample_id}")
            metadata[sample_id] = dict(item)

        expected = {
            item["sample_id"]: item
            for entries in self.protocol_entries.values()
            for item in entries
        }
        missing = sorted(set(expected) - set(metadata))
        if missing:
            raise DataError(f"metadata is missing {len(missing)} protocol samples")
        records: dict[str, dict[str, Any]] = {}
        for sample_id, protocol_item in expected.items():
            source = metadata[sample_id]
            if source.get("company") != protocol_item["issuer"]:
                raise DataError(f"issuer mismatch for sample {sample_id}")
            records[sample_id] = {
                "sample_id": sample_id,
                "issuer": str(source["company"]),
                "year": source.get("year"),
                "quarter": source.get("quarter"),
                "spanning": bool(protocol_item["spanning"]),
                "image_file": str(source["file_name"]),
                "structure_file": source.get("structure_file"),
                "content_file": source.get("content_file"),
                "has_content_label": bool(source.get("has_content_label", False)),
            }
        if len(records) != 985:
            raise DataError("data did not resolve exactly 985 protocol samples")
        self._records = records
        return dict(records)

    def validate_records(self) -> dict[str, dict[str, Any]]:
        """Run all metadata checks used before split or Primary execution."""

        records = self.load_records()
        if not self._structure_checked:
            structure_paths = self._ensure_structure_files(records)
            for sample_id in sorted(records):
                structure_path = structure_paths[sample_id]
                try:
                    payload = structure_path.read_bytes()
                except OSError as exc:
                    raise DataError(f"could not read structure annotation for {sample_id}") from exc
                derived_spanning = _spanning_from_structure_xml(payload, sample_id)
                records[sample_id]["annotation_spanning"] = derived_spanning
            self._structure_checked = True
        for name, entries in self.protocol_entries.items():
            for item in entries:
                sample = records.get(item["sample_id"])
                if sample is None:
                    raise DataError(f"protocol sample disappeared from the records: {item['sample_id']}")
                derived_spanning = sample.get("annotation_spanning")
                if sample["issuer"] != item["issuer"] or derived_spanning is not item["spanning"]:
                    raise DataError(f"protocol metadata mismatch for {item['sample_id']} in {name}")
                sample["spanning"] = bool(derived_spanning)
        return records

    def get(self, sample_id: str) -> dict[str, Any]:
        records = self.validate_records()
        try:
            return dict(records[sample_id])
        except KeyError as exc:
            raise DataError(f"sample ID is not in the configured protocol: {sample_id}") from exc

    def image_path(self, sample_id: str) -> Path:
        record = self.get(sample_id)
        image_file = record.get("image_file")
        if not isinstance(image_file, str) or not image_file:
            raise DataError(f"sample {sample_id} has no source image")
        path = self._download(image_file)
        if not path.is_file():
            raise DataError(f"source image did not materialize for {sample_id}")
        return path

    def structure_path(self, sample_id: str) -> Path | None:
        value = self.get(sample_id).get("structure_file")
        if not isinstance(value, str) or not value:
            return None
        return self._download(value)

    def structure_text(self, sample_id: str) -> str | None:
        """Retrieve the source structure record as text when one exists."""

        path = self.structure_path(sample_id)
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DataError(f"could not read the structure record for {sample_id}") from exc

    def content_path(self, sample_id: str) -> Path | None:
        value = self.get(sample_id).get("content_file")
        if not isinstance(value, str) or not value:
            return None
        return self._download(value)

    def content_record(self, sample_id: str) -> Any:
        """Retrieve and decode the source content record when one exists."""

        path = self.content_path(sample_id)
        if path is None:
            return None
        try:
            return parse_content_json(path.read_bytes())
        except (OSError, TypeError, ValueError) as exc:
            raise DataError(f"could not decode the content record for {sample_id}") from exc

    def structure_record(self, sample_id: str) -> dict[str, object]:
        """Retrieve and strictly parse the source structure annotation."""

        path = self.structure_path(sample_id)
        if path is None:
            raise DataError(f"sample {sample_id} has no structure annotation")
        try:
            return parse_structure_xml(path.read_bytes())
        except (OSError, TypeError, ValueError) as exc:
            raise DataError(f"could not decode the structure record for {sample_id}") from exc

    def reference(self, sample_id: str) -> dict[str, Any]:
        """Construct the canonical reference for one table."""

        sample = self.get(sample_id)
        structure = self.structure_record(sample_id)
        content = self.content_record(sample_id)
        if content is None:
            raise DataError(f"sample {sample_id} has no content annotation")
        try:
            structure_cells = structure_payload_to_cells(structure)
            content_cells, diagnostics = attach_text_records(
                structure_cells,
                content,
                minimum_overlap=0.5,
            )
        except (TypeError, ValueError, DataError) as exc:
            raise DataError(f"could not construct the reference for {sample_id}: {exc}") from exc
        source_objects = structure.get("objects")
        if not isinstance(source_objects, list):
            raise DataError(f"could not construct the reference for {sample_id}: structure objects are invalid")
        spanning_count = sum(
            isinstance(item, Mapping) and item.get("category") == "table spanning cell"
            for item in source_objects
        )
        characteristics = {
            "sample_id": sample_id,
            "table_count": 1,
            "reference_row_count": len({row for cell in structure_cells for row in cell.row_nums}),
            "reference_column_count": len({column for cell in structure_cells for column in cell.column_nums}),
            "reference_spanning_cell_count": spanning_count,
            "reference_cell_count": len(structure_cells),
            "content_record_count": diagnostics["record_count"],
        }
        return {
            "sample_id": sample_id,
            "issuer": sample["issuer"],
            "structure_cells": structure_cells,
            "content_cells": content_cells,
            "content_attachment_diagnostics": diagnostics,
            "reference_table_characteristics": characteristics,
        }

    def split_sample_ids(self, split: str) -> list[str]:
        entries = self.split_entries.get(split)
        if entries is None:
            raise DataError(f"unknown split: {split}")
        return [item["sample_id"] for item in entries]

    def cohort_for_sample(self, sample_id: str) -> str:
        for name, entries in self.protocol_entries.items():
            if sample_id in {item["sample_id"] for item in entries}:
                return name
        raise DataError(f"sample ID is not in the configured protocol: {sample_id}")

    def calibration_sample_ids(self) -> list[str]:
        return [item["sample_id"] for item in self.calibration_entries]
