"""Executable TiniX reproduction stages for BEAR.

The stages use the pinned TiniX report and OCR files. Each expensive
stage writes ordinary, resumable files below outputs/tinix.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from html.parser import HTMLParser
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import statistics
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.request import Request, urlopen

from .routing import FEATURE_NAMES


DATASET_ID = "tinixai/ocr_annual_financials"
DATASET_REVISION = "efd2d198d0e0c9f467bbc7a4682cc27aeb4d290f"
DATASET_URL = (
    "https://huggingface.co/datasets/tinixai/ocr_annual_financials/"
    f"tree/{DATASET_REVISION}"
)
DATASET_RESOLVE_URL = (
    "https://huggingface.co/datasets/tinixai/ocr_annual_financials/"
    f"resolve/{DATASET_REVISION}/"
)
MANIFEST_PATH = Path("manifests/tinix.json")
EXPECTED_REPORT_COUNT = 52
EXPECTED_ENTITY_COUNT = 50
EXPECTED_PAGE_COUNT = 2_319
EXPECTED_DETECTED_TABLE_COUNT = 2_275
EXPECTED_CALL_COUNT = 455
EXPECTED_ONE_HTML_ONE_DETECTION_COUNT = 833
EXPECTED_SCORED_TABLE_COUNT = 832
EXPECTED_ALIGNED_REPORT_COUNT = 51
ROUTE_FRACTION = 0.20
BOOTSTRAP_SEED = "cg-hybrid-ocr-tinix-silver-grits-report-bootstrap-v1"
BOOTSTRAP_REPLICATES = 5_000

PAGE_MARKER_RE = re.compile(
    r"^[ \t]*=+[ \t]*PAGE[ \t]+([0-9]+)[ \t]*=+[ \t]*[^\S\r\n]*=+[ \t\r]*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
PAGE_MARKER_LIKE_RE = re.compile(
    r"^[ \t]*=+[ \t]*PAGE\b.*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

__all__ = [
    "DATASET_ID",
    "DATASET_REVISION",
    "DATASET_URL",
    "TinixError",
    "run_tinix_stage",
]


class TinixError(RuntimeError):
    """Raised when a TiniX stage cannot satisfy the configured study definition."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TinixError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TinixError(f"{label} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TinixError(f"could not read {label}: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TinixError(f"malformed {label} at line {line_number}") from exc
        if not isinstance(value, dict):
            raise TinixError(f"{label} at line {line_number} is not an object")
        records.append(value)
    return records


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
    )


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TinixError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TinixError(f"{label} must be finite")
    return result


def _safe_repository_path(value: Any, *, root: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TinixError(f"{root} path is missing")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root:
        raise TinixError(f"unsafe dataset path: {value!r}")
    return path.as_posix()


def _load_manifest(source_root: Path) -> list[dict[str, Any]]:
    manifest = _read_json(source_root / MANIFEST_PATH, label="TiniX study manifest")
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("revision") != DATASET_REVISION:
        raise TinixError("TiniX study manifest does not identify the pinned dataset")
    raw_reports = manifest.get("reports")
    if not isinstance(raw_reports, list) or len(raw_reports) != EXPECTED_REPORT_COUNT:
        raise TinixError("TiniX study manifest must contain exactly 52 reports")
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    entities: set[str] = set()
    for index, raw in enumerate(raw_reports, start=1):
        if not isinstance(raw, Mapping):
            raise TinixError(f"TiniX study manifest record {index} is malformed")
        report_id = raw.get("report_id")
        entity = raw.get("reporting_entity")
        year = raw.get("year")
        if not isinstance(report_id, str) or not report_id or report_id in seen:
            raise TinixError(f"TiniX report identity is invalid at record {index}")
        if not isinstance(entity, str) or not entity:
            raise TinixError(f"TiniX reporting entity is missing at record {index}")
        if not isinstance(year, (str, int)) or isinstance(year, bool):
            raise TinixError(f"TiniX report year is invalid at record {index}")
        pdf_path = _safe_repository_path(raw.get("pdf_repository_path"), root="pdf_files")
        ocr_path = _safe_repository_path(raw.get("ocr_repository_path"), root="ocr_results")
        if not pdf_path.casefold().endswith(".pdf") or not ocr_path.casefold().endswith("_extracted.txt"):
            raise TinixError(f"TiniX report input paths are invalid at record {index}")
        seen.add(report_id)
        entities.add(entity)
        reports.append({
            "report_id": report_id,
            "reporting_entity": entity,
            "year": str(year),
            "pdf_repository_path": pdf_path,
            "ocr_repository_path": ocr_path,
        })
    if len(entities) != EXPECTED_ENTITY_COUNT:
        raise TinixError("TiniX study manifest must contain 50 reporting entities")
    return reports


def _input_key(report_id: str) -> str:
    return hashlib.sha256(report_id.encode("utf-8")).hexdigest()[:20]


def _download(url: str, target: Path, *, kind: str, force: bool) -> None:
    if target.is_file() and target.stat().st_size > 0 and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        request = Request(url, headers={"User-Agent": "BEAR-reproduction/1.0"})
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if temporary.stat().st_size <= 0:
            raise TinixError(f"downloaded {kind} is empty: {url}")
        if kind == "PDF" and temporary.read_bytes()[:4] != b"%PDF":
            raise TinixError(f"downloaded file is not a PDF: {url}")
        if kind == "OCR":
            temporary.read_text(encoding="utf-8")
        temporary.replace(target)
    except TinixError:
        raise
    except Exception as exc:
        raise TinixError(f"could not download pinned TiniX {kind}: {url}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path), strict=False).pages)
    except Exception as exc:
        raise TinixError(f"could not decode TiniX PDF: {pdf_path}") from exc


def _cohort_path(output_root: Path) -> Path:
    return output_root / "tinix" / "cohort.json"


def _load_cohort(output_root: Path) -> dict[str, Any]:
    cohort = _read_json(_cohort_path(output_root), label="TiniX cohort")
    if cohort.get("dataset_id") != DATASET_ID or cohort.get("revision") != DATASET_REVISION:
        raise TinixError("TiniX cohort is not for the pinned dataset")
    reports = cohort.get("reports")
    if not isinstance(reports, list) or len(reports) != EXPECTED_REPORT_COUNT:
        raise TinixError("TiniX cohort report count changed")
    if (
        cohort.get("report_count") != EXPECTED_REPORT_COUNT
        or cohort.get("reporting_entity_count") != EXPECTED_ENTITY_COUNT
        or cohort.get("page_count") != EXPECTED_PAGE_COUNT
    ):
        raise TinixError("TiniX cohort coverage changed")
    if len({item.get("report_id") for item in reports if isinstance(item, Mapping)}) != EXPECTED_REPORT_COUNT:
        raise TinixError("TiniX cohort report identities are not unique")
    return cohort


def parse_explicit_page_markers(text: str) -> list[dict[str, Any]]:
    """Split OCR text using only explicit, ordered one-based page markers."""

    if not isinstance(text, str):
        raise TinixError("OCR payload must be text")
    markers = list(PAGE_MARKER_RE.finditer(text))
    marker_like = list(PAGE_MARKER_LIKE_RE.finditer(text))
    if not markers:
        raise TinixError("OCR has no valid explicit page markers")
    if len(marker_like) != len(markers):
        raise TinixError("OCR page markers are malformed or ambiguous")
    numbers = [int(match.group(1)) for match in markers]
    if any(number <= 0 for number in numbers) or len(numbers) != len(set(numbers)):
        raise TinixError("OCR page markers are not unique positive page numbers")
    if any(left >= right for left, right in zip(numbers, numbers[1:])):
        raise TinixError("OCR page markers are not strictly increasing")
    pages: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        pages.append({
            "page_number": numbers[index],
            "page_index": numbers[index] - 1,
            "text": text[marker.end():end],
        })
    return pages


def prepare_tinix(*, source_root: Path, output_dir: Path, force: bool = False) -> Path:
    """Download and validate the exact 52-report TiniX cohort."""

    reports = _load_manifest(source_root)
    output_root = output_dir / "tinix"
    pdf_root = output_root / "inputs" / "pdf"
    ocr_root = output_root / "inputs" / "ocr"
    prepared: list[dict[str, Any]] = []
    total_pages = 0
    total_marker_pages = 0
    for report in reports:
        key = _input_key(report["report_id"])
        pdf_path = pdf_root / f"{key}.pdf"
        ocr_path = ocr_root / f"{key}.txt"
        _download(DATASET_RESOLVE_URL + report["pdf_repository_path"], pdf_path, kind="PDF", force=force)
        _download(DATASET_RESOLVE_URL + report["ocr_repository_path"], ocr_path, kind="OCR", force=force)
        pages = _page_count(pdf_path)
        markers = parse_explicit_page_markers(ocr_path.read_text(encoding="utf-8"))
        if pages != len(markers):
            raise TinixError(
                f"PDF/OCR page counts differ for {report['report_id']}: "
                f"{pages} PDF pages versus {len(markers)} OCR markers"
            )
        total_pages += pages
        total_marker_pages += len(markers)
        prepared.append({
            **report,
            "input_key": key,
            "pdf_file": str(pdf_path.relative_to(output_root)),
            "ocr_file": str(ocr_path.relative_to(output_root)),
            "page_count": pages,
            "ocr_page_marker_count": len(markers),
        })
    if total_pages != EXPECTED_PAGE_COUNT:
        raise TinixError(f"TiniX page count changed: expected {EXPECTED_PAGE_COUNT}, got {total_pages}")
    if total_marker_pages != EXPECTED_PAGE_COUNT:
        raise TinixError(
            f"TiniX OCR page-marker count changed: expected {EXPECTED_PAGE_COUNT}, got {total_marker_pages}"
        )
    cohort = {
        "study": "tinix",
        "status": "prepared",
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "report_count": len(prepared),
        "reporting_entity_count": len({item["reporting_entity"] for item in prepared}),
        "page_count": total_pages,
        "reports": prepared,
    }
    path = _cohort_path(output_dir)
    _atomic_json(path, cohort)
    return path


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normal_label(value: Any) -> str:
    value = getattr(value, "value", value)
    return "".join(char if char.isalnum() else "_" for char in str(value).casefold()).strip("_")


def _build_discovery_converter(device: str, protocol: Mapping[str, Any]) -> Any:
    try:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.object_detection_engine_options import (
            TransformersObjectDetectionEngineOptions,
        )
        from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions, PdfPipelineOptions
        from docling.datamodel.stage_model_specs import ObjectDetectionModelSpec
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise TinixError("TiniX page detection requires the documented Docling dependencies") from exc
    layout_config = (
        protocol.get("primary", {}).get("docling", {}).get("layout", {})
        if isinstance(protocol.get("primary"), Mapping)
        else {}
    )
    layout_repository = layout_config.get("model_repository")
    layout_revision = layout_config.get("revision")
    if not isinstance(layout_repository, str) or not layout_repository:
        raise TinixError("BEAR protocol lacks the Layout Heron repository")
    if not isinstance(layout_revision, str) or not layout_revision:
        raise TinixError("BEAR protocol lacks the Layout Heron revision")
    layout = LayoutObjectDetectionOptions(
        keep_empty_clusters=False,
        skip_cell_assignment=False,
        create_orphan_clusters=True,
        engine_options=TransformersObjectDetectionEngineOptions(
            score_threshold=0.3,
            torch_dtype=None,
            compile_model=False,
        ),
        model_spec=ObjectDetectionModelSpec(
            name="layout_heron",
            repo_id=layout_repository,
            revision=layout_revision,
        ),
    )
    options = PdfPipelineOptions(
        document_timeout=None,
        do_ocr=False,
        do_table_structure=False,
        generate_page_images=True,
        generate_picture_images=False,
        generate_table_images=False,
        images_scale=1.0,
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        generate_parsed_pages=False,
        enable_remote_services=False,
        allow_external_plugins=False,
        force_backend_text=False,
        ocr_batch_size=4,
        layout_batch_size=4,
        table_batch_size=4,
        batch_polling_interval_seconds=0.5,
        queue_max_size=100,
        stage_shutdown_timeout_seconds=15.0,
        layout_options=layout,
        accelerator_options=AcceleratorOptions(
            num_threads=4,
            device="auto" if device == "auto" else device,
            cuda_use_flash_attention2=False,
        ),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def _layout_clusters(page: Any) -> list[Any]:
    predictions = _field(page, "predictions")
    layout = _field(predictions, "layout")
    clusters = _field(layout, "clusters", []) or []
    try:
        return [
            cluster for cluster in list(clusters)
            if _normal_label(_field(cluster, "label")) == "table"
        ]
    except TypeError as exc:
        raise TinixError("Docling returned non-iterable layout clusters") from exc


def _page_image(document: Any, page_number: int) -> Any:
    pages = _field(document, "pages", {})
    page_item = pages.get(page_number) if isinstance(pages, Mapping) else None
    image_ref = _field(page_item, "image")
    image = _field(image_ref, "pil_image")
    return image() if callable(image) else image


def _crop_box(cluster: Any, *, page: Any, image: Any) -> tuple[int, int, int, int]:
    bbox = _field(cluster, "bbox")
    if bbox is None:
        raise TinixError("table cluster has no bounding box")
    left = _finite(_field(bbox, "l"), label="table left")
    first_y = _finite(_field(bbox, "t"), label="table y1")
    right = _finite(_field(bbox, "r"), label="table right")
    second_y = _finite(_field(bbox, "b"), label="table y2")
    page_size = _field(page, "size")
    page_width = _finite(_field(page_size, "width", image.width), label="page width")
    page_height = _finite(_field(page_size, "height", image.height), label="page height")
    origin = _normal_label(_field(bbox, "coord_origin"))
    x_min, x_max = sorted((left, right))
    if origin in {"bottomleft", "bottom_left"}:
        y_min, y_max = sorted((page_height - first_y, page_height - second_y))
    elif origin in {"topleft", "top_left"}:
        y_min, y_max = sorted((first_y, second_y))
    else:
        raise TinixError(f"unsupported table coordinate origin: {origin or 'missing'}")
    if not x_min < x_max or not y_min < y_max:
        raise TinixError("table bounding box has no positive area")
    scale_x = float(image.width) / page_width
    scale_y = float(image.height) / page_height
    x0 = max(0, min(int(math.floor(x_min * scale_x)), image.width - 1))
    y0 = max(0, min(int(math.floor(y_min * scale_y)), image.height - 1))
    x1 = max(0, min(int(math.ceil(x_max * scale_x)), image.width))
    y1 = max(0, min(int(math.ceil(y_max * scale_y)), image.height))
    if x1 <= x0 or y1 <= y0:
        raise TinixError("table crop has no positive pixel area")
    return x0, y0, x1, y1


def _stable_table_id(report_id: str, page_index: int, detection_index: int) -> str:
    if page_index < 0 or detection_index < 0:
        raise TinixError("table indices must be non-negative")
    return f"{report_id}::page-{page_index:04d}::table-{detection_index:04d}"


def _crop_name(table_id: str) -> str:
    return hashlib.sha256(table_id.encode("utf-8")).hexdigest() + ".png"


def _load_protocol(source_root: Path) -> dict[str, Any]:
    return _read_json(source_root / "protocol.json", label="BEAR protocol")


def _primary_records_path(output_root: Path) -> Path:
    return output_root / "tinix" / "primary" / "tables.jsonl"


def _primary_summary_path(output_root: Path) -> Path:
    return output_root / "tinix" / "primary" / "summary.json"


def _primary_report_path(output_root: Path, input_key: str) -> Path:
    return output_root / "tinix" / "primary" / "reports" / f"{input_key}.json"


def _primary_semantic_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from bear.recognition import primary_semantic_identity

        return primary_semantic_identity(protocol)
    except Exception as exc:
        raise TinixError("could not determine the TiniX Primary semantic identity") from exc


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TinixError(f"{label} must be a non-negative integer")
    return value


def _validate_primary_record(
    record: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    output_root: Path | None = None,
) -> None:
    report_id = str(report["report_id"])
    page_count = int(report["page_count"])
    table_id = record.get("table_id")
    page_index = _require_nonnegative_int(record.get("page_index"), label="Primary page index")
    detection_index = _require_nonnegative_int(
        record.get("detection_index"),
        label="Primary detection index",
    )
    if (
        record.get("report_id") != report_id
        or record.get("reporting_entity") != report.get("reporting_entity")
        or record.get("year") != report.get("year")
        or not isinstance(table_id, str)
        or table_id != _stable_table_id(report_id, page_index, detection_index)
        or page_index >= page_count
        or record.get("status") != "success"
    ):
        raise TinixError(f"TiniX Primary record identity changed: {table_id}")
    if not isinstance(record.get("cells"), list):
        raise TinixError(f"TiniX Primary cells are missing: {table_id}")
    crop_file = record.get("crop_file")
    crop_path = PurePosixPath(str(crop_file)) if isinstance(crop_file, str) else None
    if (
        crop_path is None
        or crop_path.is_absolute()
        or ".." in crop_path.parts
        or crop_path.as_posix() != f"tables/crops/{_crop_name(table_id)}"
        or (output_root is not None and not (output_root / "tinix" / Path(*crop_path.parts)).is_file())
    ):
        raise TinixError(f"TiniX Primary crop artifact is missing or stale: {table_id}")
    features = record.get("features")
    if not isinstance(features, Mapping) or set(features) != set(FEATURE_NAMES):
        raise TinixError(f"TiniX Primary feature schema changed: {table_id}")
    feature_names = record.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not all(isinstance(name, str) for name in feature_names)
        or len(feature_names) != len(FEATURE_NAMES)
        or len(set(feature_names)) != len(FEATURE_NAMES)
        or set(feature_names) != set(FEATURE_NAMES)
    ):
        raise TinixError(f"TiniX Primary feature schema changed: {table_id}")


def _load_primary_report(
    output_root: Path,
    report: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    input_key = str(report["input_key"])
    payload = _read_json(
        _primary_report_path(output_root, input_key),
        label=f"TiniX Primary report {report['report_id']}",
    )
    if (
        payload.get("study") != "tinix"
        or payload.get("status") != "completed"
        or payload.get("dataset_id") != DATASET_ID
        or payload.get("revision") != DATASET_REVISION
        or payload.get("report_id") != report.get("report_id")
        or payload.get("input_key") != input_key
        or payload.get("reporting_entity") != report.get("reporting_entity")
        or payload.get("year") != report.get("year")
        or payload.get("page_count") != report.get("page_count")
        or payload.get("primary_identity") != dict(expected_identity)
    ):
        raise TinixError(f"TiniX Primary report artifact is stale: {report['report_id']}")
    page_indices = payload.get("processed_page_indices")
    if page_indices != list(range(int(report["page_count"]))):
        raise TinixError(f"TiniX Primary report is incomplete: {report['report_id']}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TinixError(f"TiniX Primary report records are missing: {report['report_id']}")
    for record in records:
        if not isinstance(record, Mapping):
            raise TinixError(f"TiniX Primary report record is malformed: {report['report_id']}")
        _validate_primary_record(record, report=report, output_root=output_root)
    table_ids = [str(record["table_id"]) for record in records]
    coordinates = [
        (int(record["page_index"]), int(record["detection_index"]))
        for record in records
    ]
    if (
        payload.get("detected_table_count") != len(records)
        or len(set(table_ids)) != len(table_ids)
        or len(set(coordinates)) != len(coordinates)
        or table_ids != sorted(table_ids)
        or payload.get("detected_page_count") != len({item[0] for item in coordinates})
    ):
        raise TinixError(f"TiniX Primary report table records are inconsistent: {report['report_id']}")
    return [dict(record) for record in records]


def _validate_primary_summary(
    summary: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> None:
    if (
        summary.get("study") != "tinix"
        or summary.get("status") != "completed"
        or summary.get("dataset_id") != DATASET_ID
        or summary.get("revision") != DATASET_REVISION
        or summary.get("report_count") != EXPECTED_REPORT_COUNT
        or summary.get("page_count") != EXPECTED_PAGE_COUNT
        or summary.get("detected_table_count") != EXPECTED_DETECTED_TABLE_COUNT
        or summary.get("reports_with_detected_tables") != 51
        or summary.get("primary_valid_count") != EXPECTED_DETECTED_TABLE_COUNT
        or summary.get("feature_order") != list(FEATURE_NAMES)
        or summary.get("recognizer") != "crop_tableformer_rapidocr_v6"
        or summary.get("report_artifact_count") != EXPECTED_REPORT_COUNT
        or not isinstance(summary.get("primary_identity"), Mapping)
    ):
        raise TinixError("TiniX Primary summary identity or coverage changed")
    if expected_identity is not None and summary["primary_identity"] != dict(expected_identity):
        raise TinixError("TiniX Primary summary was generated with a different Primary identity")


def _load_primary_records(
    output_root: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary = _read_json(_primary_summary_path(output_root), label="TiniX Primary summary")
    _validate_primary_summary(summary, expected_identity=expected_identity)
    records = _read_jsonl(_primary_records_path(output_root), label="TiniX Primary records")
    if len(records) != EXPECTED_DETECTED_TABLE_COUNT:
        raise TinixError("TiniX Primary table count changed")
    ids = [record.get("table_id") for record in records]
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise TinixError("TiniX Primary table IDs are not unique")
    for record in records:
        if (
            record.get("status") != "success"
            or not isinstance(record.get("cells"), list)
            or not isinstance(record.get("feature_names"), list)
            or not all(isinstance(name, str) for name in record["feature_names"])
            or set(record["feature_names"]) != set(FEATURE_NAMES)
            or not isinstance(record.get("features"), Mapping)
            or set(record["features"]) != set(FEATURE_NAMES)
        ):
            raise TinixError(f"TiniX Primary record schema changed: {record.get('table_id')}")
    return sorted(records, key=lambda record: str(record["table_id"]))


def _assemble_primary_outputs(
    output_root: Path,
    reports: Sequence[Mapping[str, Any]],
    *,
    primary_identity: Mapping[str, Any],
) -> Path:
    records: list[dict[str, Any]] = []
    total_pages = 0
    for report in reports:
        records.extend(
            _load_primary_report(
                output_root,
                report,
                expected_identity=primary_identity,
            )
        )
        total_pages += int(report["page_count"])
    records.sort(key=lambda record: str(record["table_id"]))
    if (
        len(reports) != EXPECTED_REPORT_COUNT
        or total_pages != EXPECTED_PAGE_COUNT
        or len(records) != EXPECTED_DETECTED_TABLE_COUNT
        or len({record["report_id"] for record in records}) != 51
    ):
        raise TinixError("TiniX Primary cohort assembly counts changed")
    summary = {
        "study": "tinix",
        "status": "completed",
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "report_count": len(reports),
        "page_count": total_pages,
        "detected_table_count": len(records),
        "reports_with_detected_tables": len({record["report_id"] for record in records}),
        "primary_valid_count": len(records),
        "feature_order": list(FEATURE_NAMES),
        "recognizer": "crop_tableformer_rapidocr_v6",
        "report_artifact_count": len(reports),
        "primary_identity": dict(primary_identity),
    }
    _atomic_jsonl(_primary_records_path(output_root), records)
    _atomic_json(_primary_summary_path(output_root), summary)
    return _primary_summary_path(output_root)


def primary_tinix(
    *,
    source_root: Path,
    output_dir: Path,
    models_dir: Path,
    device: str,
    force: bool = False,
) -> Path:
    """Detect every table page and run the Primary recognizer."""

    output_root = output_dir / "tinix"
    destination = _primary_records_path(output_dir)
    cohort = _load_cohort(output_dir)
    protocol = _load_protocol(source_root)
    primary_identity = _primary_semantic_identity(protocol)
    summary_path = _primary_summary_path(output_dir)
    if destination.is_file() or summary_path.is_file():
        if not destination.is_file() or not summary_path.is_file():
            raise TinixError("TiniX Primary global artifacts are incomplete")
        if not force:
            _load_primary_records(output_dir, expected_identity=primary_identity)
            return summary_path

    report_root = output_root / "primary" / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    pending: list[Mapping[str, Any]] = []
    for report in cohort["reports"]:
        report_path = _primary_report_path(output_dir, str(report["input_key"]))
        if report_path.is_file() and not force:
            _load_primary_report(
                output_dir,
                report,
                expected_identity=primary_identity,
            )
        else:
            pending.append(report)
    if not pending:
        return _assemble_primary_outputs(
            output_dir,
            cohort["reports"],
            primary_identity=primary_identity,
        )

    try:
        from bear.recognition import PrimaryRecognizer
    except Exception as exc:
        raise TinixError("Primary recognition dependencies are unavailable") from exc
    discovery = _build_discovery_converter(device, protocol)
    primary = PrimaryRecognizer(device=device, models_dir=models_dir, protocol=protocol)
    output_root.joinpath("tables", "crops").mkdir(parents=True, exist_ok=True)
    for report in pending:
        pdf_path = output_root / str(report["pdf_file"])
        if not pdf_path.is_file():
            raise TinixError(f"prepared PDF is missing: {report['report_id']}")
        conversion = discovery.convert(str(pdf_path), raises_on_error=False)
        pages = list(_field(conversion, "pages", []) or [])
        if len(pages) != int(report["page_count"]):
            raise TinixError(f"Docling page count changed for {report['report_id']}")
        document = _field(conversion, "document")
        report_records: list[dict[str, Any]] = []
        for page_position, runtime_page in enumerate(pages):
            raw_page_number = _field(runtime_page, "page_no", page_position + 1)
            page_number = raw_page_number if type(raw_page_number) is int else page_position + 1
            page_index = page_number - 1
            clusters = _layout_clusters(runtime_page)
            image = _page_image(document, page_number)
            if clusters and image is None:
                raise TinixError(f"Docling did not expose page image for {report['report_id']}")
            for detection_index, cluster in enumerate(clusters):
                table_id = _stable_table_id(str(report["report_id"]), page_index, detection_index)
                box = _crop_box(cluster, page=runtime_page, image=image)
                crop_path = output_root / "tables" / "crops" / _crop_name(table_id)
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not crop_path.is_file() or force:
                    image.crop(box).save(crop_path, format="PNG")
                prediction = primary.predict(crop_path)
                report_records.append({
                    "table_id": table_id,
                    "report_id": report["report_id"],
                    "reporting_entity": report["reporting_entity"],
                    "year": report["year"],
                    "page_index": page_index,
                    "detection_index": detection_index,
                    "crop_file": str(crop_path.relative_to(output_root)),
                    "crop_box": list(box),
                    "crop_dimensions": {
                        "width": int(box[2] - box[0]),
                        "height": int(box[3] - box[1]),
                    },
                    "status": prediction.get("status"),
                    "cells": prediction.get("cells") or [],
                    "features": prediction.get("features"),
                    "feature_vector": prediction.get("feature_vector"),
                    "feature_names": list(FEATURE_NAMES),
                    "inference_elapsed_seconds": prediction.get("inference_elapsed_seconds"),
                    "failure_stage": prediction.get("failure_stage"),
                    "error_type": prediction.get("error_type"),
                    "error_message": prediction.get("error_message"),
                })
        report_records.sort(key=lambda record: str(record["table_id"]))
        if any(record.get("status") != "success" for record in report_records):
            raise TinixError(f"Primary failed for at least one table in {report['report_id']}")
        report_payload = {
            "study": "tinix",
            "status": "completed",
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "report_id": report["report_id"],
            "input_key": report["input_key"],
            "reporting_entity": report["reporting_entity"],
            "year": report["year"],
            "page_count": report["page_count"],
            "processed_page_indices": list(range(int(report["page_count"]))),
            "detected_page_count": len({record["page_index"] for record in report_records}),
            "detected_table_count": len(report_records),
            "primary_identity": dict(primary_identity),
            "records": report_records,
        }
        _atomic_json(_primary_report_path(output_dir, str(report["input_key"])), report_payload)
        _load_primary_report(
            output_dir,
            report,
            expected_identity=primary_identity,
        )
    return _assemble_primary_outputs(
        output_dir,
        cohort["reports"],
        primary_identity=primary_identity,
    )


def _route_path(output_root: Path) -> Path:
    return output_root / "tinix" / "route.json"


def _load_route(output_root: Path) -> dict[str, Any]:
    route = _read_json(_route_path(output_root), label="TiniX route")
    routed = route.get("routed_tables")
    if (
        route.get("dataset_id") != DATASET_ID
        or route.get("revision") != DATASET_REVISION
        or route.get("router") != "hist_gradient_boosting_benefit"
        or route.get("ranking_rule") != "predicted benefit descending, table ID ascending"
        or route.get("budget_fraction") != ROUTE_FRACTION
        or route.get("detected_table_count") != EXPECTED_DETECTED_TABLE_COUNT
        or route.get("call_count") != EXPECTED_CALL_COUNT
        or not isinstance(routed, list)
    ):
        raise TinixError("TiniX route call count changed")
    if len(routed) != EXPECTED_CALL_COUNT:
        raise TinixError("TiniX route table count changed")
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("table_id"), str)
        or isinstance(item.get("route_rank"), bool)
        or not isinstance(item.get("route_rank"), int)
        or isinstance(item.get("predicted_benefit"), bool)
        or not isinstance(item.get("predicted_benefit"), (int, float))
        or not math.isfinite(float(item.get("predicted_benefit")))
        for item in routed
    ):
        raise TinixError("TiniX route record schema changed")
    if len({item["table_id"] for item in routed}) != EXPECTED_CALL_COUNT:
        raise TinixError("TiniX route IDs are not unique")
    return route


def route_tinix(*, output_dir: Path, models_dir: Path, force: bool = False) -> Path:
    """Apply the unchanged development-trained HGB benefit ranking."""

    if _route_path(output_dir).is_file() and not force:
        _load_route(output_dir)
        return _route_path(output_dir)
    records = _load_primary_records(output_dir)
    model_path = models_dir / "router" / "hist_gradient_boosting_benefit.joblib"
    if not model_path.is_file():
        raise TinixError(
            "the development-trained HGB router is missing; place it at "
            f"{model_path}"
        )
    try:
        import joblib

        model = joblib.load(model_path)
    except Exception as exc:
        raise TinixError(f"could not load the HGB router: {model_path}") from exc
    if list(getattr(model, "named_steps", {})) != ["imputer", "model"]:
        raise TinixError("HGB router does not have the specified imputer/model pipeline")
    scored: list[tuple[str, float]] = []
    for record in records:
        features = record.get("features")
        if not isinstance(features, Mapping) or set(features) != set(FEATURE_NAMES):
            raise TinixError(f"Primary feature schema changed: {record.get('table_id')}")
        values = [
            _finite(features.get(name), label=f"{record['table_id']}:{name}")
            for name in FEATURE_NAMES
        ]
        try:
            prediction = model.predict([values])
            if len(prediction) != 1:
                raise ValueError("router returned an unexpected prediction count")
            score = _finite(prediction[0], label=f"router prediction {record['table_id']}")
        except Exception as exc:
            raise TinixError(f"HGB router failed for {record['table_id']}") from exc
        scored.append((str(record["table_id"]), score))
    count = int(round(len(records) * ROUTE_FRACTION))
    if count != EXPECTED_CALL_COUNT:
        raise TinixError("TiniX route fraction does not resolve to 455 calls")
    scored.sort(key=lambda item: (-item[1], item[0]))
    routed = [
        {"route_rank": rank, "table_id": table_id, "predicted_benefit": score}
        for rank, (table_id, score) in enumerate(scored[:count], start=1)
    ]
    route = {
        "study": "tinix",
        "status": "completed",
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "router": "hist_gradient_boosting_benefit",
        "ranking_rule": "predicted benefit descending, table ID ascending",
        "budget_fraction": ROUTE_FRACTION,
        "detected_table_count": len(records),
        "call_count": count,
        "routed_tables": routed,
    }
    _atomic_json(_route_path(output_dir), route)
    return _route_path(output_dir)


def _expert_file(output_root: Path, table_id: str) -> Path:
    return output_root / "tinix" / "expert" / "tables" / (
        hashlib.sha256(table_id.encode("utf-8")).hexdigest() + ".json"
    )


def _load_expert_record(path: Path, table_id: str) -> dict[str, Any]:
    value = _read_json(path, label="TiniX expert record")
    if value.get("table_id") != table_id or value.get("status") not in {
        "success",
        "parse_failed",
        "inference_failed",
        "inference_timeout",
        "out_of_memory",
    }:
        raise TinixError(f"malformed TiniX expert record: {path}")
    if not isinstance(value.get("cells"), list):
        raise TinixError(f"TiniX expert record has no cells list: {path}")
    return value


def _load_expert_summary(output_root: Path) -> dict[str, Any]:
    summary = _read_json(output_root / "tinix" / "expert" / "summary.json", label="TiniX expert summary")
    if summary.get("status") != "completed" or summary.get("recognizer") != "hunyuanocr_1_5":
        raise TinixError("TiniX expert summary identity changed")
    attempted = summary.get("attempted_calls")
    successful = summary.get("successful_canonicalizations")
    parse_failures = summary.get("parse_failures")
    generation_failures = summary.get("generation_failures")
    fallbacks = summary.get("primary_fallbacks")
    values = (attempted, successful, parse_failures, generation_failures, fallbacks)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise TinixError("TiniX expert summary counts are invalid")
    if (
        attempted != EXPECTED_CALL_COUNT
        or successful + parse_failures + generation_failures != EXPECTED_CALL_COUNT
        or fallbacks != EXPECTED_CALL_COUNT - successful
    ):
        raise TinixError("TiniX expert summary counts changed")
    return summary


def expert_tinix(
    *,
    source_root: Path,
    output_dir: Path,
    models_dir: Path,
    protocol_path: Path,
    device: str,
    force: bool = False,
) -> Path:
    """Run Hunyuan on the exact route and retain Primary after any failure."""

    output_root = output_dir / "tinix"
    summary_path = output_root / "expert" / "summary.json"
    route = _load_route(output_dir)
    primary_records = {
        str(item["table_id"]): item for item in _load_primary_records(output_dir)
    }
    route_ids = [str(item["table_id"]) for item in route["routed_tables"]]
    if any(table_id not in primary_records for table_id in route_ids):
        raise TinixError("TiniX route includes a table absent from Primary records")
    protocol = _read_json(protocol_path, label="BEAR protocol")
    result_paths: list[Path] = []
    missing = []
    for table_id in route_ids:
        path = _expert_file(output_dir, table_id)
        if path.is_file() and not force:
            _load_expert_record(path, table_id)
            result_paths.append(path)
        else:
            missing.append((table_id, path))
    recognizer = None
    if missing:
        if device != "cuda":
            raise TinixError("TiniX Hunyuan inference requires --device cuda")
        try:
            from bear.recognition import ExpertRecognizer

            recognizer = ExpertRecognizer(
                "hunyuanocr_1_5",
                protocol,
                models_dir=models_dir,
                device=device,
            )
        except Exception as exc:
            raise TinixError("Hunyuan expert dependencies or model are unavailable") from exc
    for table_id, path in missing:
        source = output_root / str(primary_records[table_id]["crop_file"])
        if not source.is_file():
            raise TinixError(f"routed table crop is missing: {table_id}")
        started = time.perf_counter()
        prediction = recognizer.predict(source)
        status = str(prediction.get("status"))
        if status not in {
            "success",
            "parse_failed",
            "inference_failed",
            "inference_timeout",
            "out_of_memory",
        }:
            raise TinixError(f"expert returned an unknown terminal state: {status}")
        # raw_output is intentionally not written to the stage artifact.
        record = {
            "table_id": table_id,
            "status": status,
            "cells": prediction.get("cells") if status == "success" else [],
            "inference_elapsed_seconds": prediction.get("inference_elapsed_seconds"),
            "parsing_elapsed_seconds": prediction.get("parsing_elapsed_seconds"),
            "elapsed_seconds": time.perf_counter() - started,
            "failure_stage": prediction.get("failure_stage"),
            "error_type": prediction.get("error_type"),
            "error_message": prediction.get("error_message"),
        }
        _atomic_json(path, record)
        result_paths.append(path)
    if len(result_paths) != EXPECTED_CALL_COUNT:
        raise TinixError("TiniX expert stage did not produce 455 terminal call records")
    records = [_load_expert_record(_expert_file(output_dir, table_id), table_id) for table_id in route_ids]
    successful = sum(record["status"] == "success" for record in records)
    parse_failures = sum(record["status"] == "parse_failed" for record in records)
    summary = {
        "study": "tinix",
        "status": "completed",
        "recognizer": "hunyuanocr_1_5",
        "attempted_calls": len(records),
        "successful_canonicalizations": successful,
        "parse_failures": parse_failures,
        "generation_failures": len(records) - successful - parse_failures,
        "primary_fallbacks": len(records) - successful,
        "replacement_rule": (
            "a successful parsed expert output replaces Primary; failed generation "
            "or parse retains Primary"
        ),
    }
    _atomic_json(summary_path, summary)
    _load_expert_summary(output_dir)
    return summary_path


class _TopLevelHTMLTableExtractor(HTMLParser):
    """Collect outermost HTML table blocks without accepting nested tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.top_level_count = 0
        self.table_depth = 0
        self.blocks: list[str] = []
        self.current_parts: list[str] | None = None
        self.nested_table_detected = False
        self.malformed = False

    def _append(self, value: str) -> None:
        if self.current_parts is not None:
            self.current_parts.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        raw = self.get_starttag_text() or f"<{tag}>"
        lowered = tag.casefold()
        if lowered == "table":
            if self.table_depth == 0:
                self.top_level_count += 1
                self.current_parts = []
            else:
                self.nested_table_detected = True
            self.table_depth += 1
        self._append(raw)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        raw = self.get_starttag_text() or f"<{tag}/>"
        if tag.casefold() == "table":
            if self.table_depth == 0:
                self.top_level_count += 1
                self.blocks.append(raw)
            else:
                self.nested_table_detected = True
        self._append(raw)

    def handle_endtag(self, tag: str) -> None:
        raw = f"</{tag}>"
        self._append(raw)
        if tag.casefold() != "table":
            return
        if self.table_depth <= 0:
            self.malformed = True
            return
        if self.table_depth == 1:
            if self.current_parts is None:
                self.malformed = True
            else:
                self.blocks.append("".join(self.current_parts))
            self.current_parts = None
        self.table_depth -= 1

    def handle_data(self, data: str) -> None:
        self._append(data)

    def handle_entityref(self, name: str) -> None:
        self._append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._append(f"<!{decl}>")

    def result(self) -> dict[str, Any]:
        if self.table_depth != 0 or self.current_parts is not None:
            self.malformed = True
        return {
            "top_level_table_count": self.top_level_count,
            "blocks": list(self.blocks),
            "nested_table_detected": self.nested_table_detected,
            "malformed": self.malformed,
        }


def _extract_top_level_tables(page_text: str) -> dict[str, Any]:
    parser = _TopLevelHTMLTableExtractor()
    try:
        parser.feed(page_text)
        parser.close()
    except Exception:
        parser.malformed = True
    return parser.result()


class _StrictHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_count = 0
        self.table_depth = 0
        self.rows: list[list[dict[str, Any]]] = []
        self.current_row: list[dict[str, Any]] | None = None
        self.current_cell: dict[str, Any] | None = None
        self.outside_text: list[str] = []
        self.error: str | None = None

    def _fail(self, message: str) -> None:
        if self.error is None:
            self.error = message

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = {key.casefold(): value for key, value in attrs}
        if tag == "table":
            if self.table_depth:
                self._fail("nested HTML tables are not supported")
            self.table_count += 1
            self.table_depth += 1
            return
        if not self.table_depth:
            return
        if tag == "tr":
            if self.current_row is not None:
                self._fail("nested HTML rows are not supported")
            self.current_row = []
            self.rows.append(self.current_row)
        elif tag in {"td", "th"}:
            if self.current_row is None or self.current_cell is not None:
                self._fail("HTML cell is outside a row or overlaps another cell")
            self.current_cell = {
                "text_parts": [],
                "rowspan": attributes.get("rowspan", "1"),
                "colspan": attributes.get("colspan", "1"),
                "is_column_header": tag == "th",
                "is_row_header": False,
            }

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"td", "th"}:
            if self.current_cell is None or self.current_row is None:
                self._fail("HTML cell closes without an open cell")
            else:
                self.current_row.append(self.current_cell)
                self.current_cell = None
        elif tag == "tr":
            if self.current_row is None or self.current_cell is not None:
                self._fail("HTML row closes before its cells are complete")
            self.current_row = None
        elif tag == "table":
            if self.current_cell is not None or self.current_row is not None:
                self._fail("HTML table closes before its rows are complete")
            self.table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell["text_parts"].append(data)
        elif not self.table_depth:
            self.outside_text.append(data)


def _normalize_text(value: str) -> str:
    from bear.routing import normalize_text_for_cer

    return normalize_text_for_cer(value)


def _canonicalize_silver_html(html: str) -> list[dict[str, Any]]:
    try:
        from grits import TableCell
    except Exception as exc:
        raise TinixError("silver-reference evaluation requires grits-metric") from exc
    parser = _StrictHTMLTableParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        raise TinixError("OCR silver HTML could not be parsed") from exc
    if (
        parser.error
        or parser.table_count != 1
        or parser.table_depth != 0
        or any(part.strip() for part in parser.outside_text)
        or not parser.rows
    ):
        raise TinixError("OCR silver HTML is not one strict complete table")
    occupied: dict[tuple[int, int], int] = {}
    cells: list[dict[str, Any]] = []
    maximum_column = 0
    for row_index, row in enumerate(parser.rows):
        cursor = 0
        for raw_cell in row:
            while (row_index, cursor) in occupied:
                cursor += 1
            try:
                row_span = int(str(raw_cell["rowspan"]).strip())
                column_span = int(str(raw_cell["colspan"]).strip())
            except (TypeError, ValueError) as exc:
                raise TinixError("OCR silver rowspan and colspan must be integers") from exc
            if row_span <= 0 or column_span <= 0 or row_index + row_span > len(parser.rows):
                raise TinixError("OCR silver span is invalid")
            coordinates = [
                (row, column)
                for row in range(row_index, row_index + row_span)
                for column in range(cursor, cursor + column_span)
            ]
            if any(coordinate in occupied for coordinate in coordinates):
                raise TinixError("OCR silver table contains overlapping spans")
            cell_index = len(cells)
            for coordinate in coordinates:
                occupied[coordinate] = cell_index
            cells.append({
                "row_start": row_index,
                "row_end": row_index + row_span,
                "column_start": cursor,
                "column_end": cursor + column_span,
                "text": _normalize_text("".join(raw_cell["text_parts"])),
                "is_column_header": bool(raw_cell["is_column_header"]),
                "is_row_header": bool(raw_cell["is_row_header"]),
                "grits_cell": TableCell(
                    row_nums=list(range(row_index, row_index + row_span)),
                    column_nums=list(range(cursor, cursor + column_span)),
                    cell_text=_normalize_text("".join(raw_cell["text_parts"])),
                    bbox=None,
                    is_column_header=bool(raw_cell["is_column_header"]),
                    is_row_header=bool(raw_cell["is_row_header"]),
                ),
            })
            cursor += column_span
            maximum_column = max(maximum_column, cursor)
    for row_index in range(len(parser.rows)):
        for column_index in range(maximum_column):
            if (row_index, column_index) not in occupied:
                raise TinixError("OCR silver table contains an uncovered grid position")
    return cells


def _score_cells(reference_cells: Sequence[Any], prediction_cells: Sequence[Any]) -> dict[str, float]:
    try:
        from grits import GritsEvaluator
    except Exception as exc:
        raise TinixError("TiniX evaluation requires grits-metric") from exc
    result: dict[str, float] = {}
    for metric in ("con", "top"):
        evaluator = GritsEvaluator(metrics=[metric])
        evaluator.eval_table_cell_lists([list(reference_cells)], [list(prediction_cells)])
        result[f"grits_{metric}"] = float(evaluator.compute_grits()[f"grits_{metric}"])
    return result


def _prediction_grits_cells(cells: Sequence[Mapping[str, Any]], *, allow_overlaps: bool) -> list[Any]:
    try:
        from bear.evaluation import prediction_cells_to_grits

        normalized = []
        for cell in cells:
            if not isinstance(cell, Mapping):
                raise TinixError("TiniX prediction cell is not an object")
            item = dict(cell)
            item["text"] = _normalize_text(item.get("text"))
            normalized.append(item)
        return prediction_cells_to_grits(normalized, allow_overlaps=allow_overlaps)
    except Exception as exc:
        raise TinixError("TiniX prediction cells are not valid for GriTS") from exc


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise TinixError("percentile requires non-empty values")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _report_clustered_bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise TinixError("TiniX evaluation has no aligned table rows")
    by_report: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        report_id = row.get("report_id")
        if not isinstance(report_id, str) or not report_id:
            raise TinixError("TiniX score row has no report ID")
        by_report[report_id].append(row)
    report_ids = sorted(by_report)
    observed_count = len(rows)
    observed_con = statistics.fmean(
        float(row["deployed_grits_con"]) - float(row["primary_grits_con"]) for row in rows
    )
    observed_top = statistics.fmean(
        float(row["deployed_grits_top"]) - float(row["primary_grits_top"]) for row in rows
    )
    totals = {
        report_id: {
            "count": len(report_rows),
            "con": sum(
                float(row["deployed_grits_con"]) - float(row["primary_grits_con"])
                for row in report_rows
            ),
            "top": sum(
                float(row["deployed_grits_top"]) - float(row["primary_grits_top"])
                for row in report_rows
            ),
        }
        for report_id, report_rows in by_report.items()
    }
    rng = random.Random(int(hashlib.sha256(BOOTSTRAP_SEED.encode("utf-8")).hexdigest(), 16))
    con_values: list[float] = []
    top_values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [report_ids[rng.randrange(len(report_ids))] for _ in report_ids]
        count = sum(int(totals[report_id]["count"]) for report_id in sampled)
        con_values.append(sum(float(totals[report_id]["con"]) for report_id in sampled) / count)
        top_values.append(sum(float(totals[report_id]["top"]) for report_id in sampled) / count)
    return {
        "resampling_unit": "report_id",
        "report_count": len(report_ids),
        "aligned_table_count": observed_count,
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "interval": "paired report-clustered percentile 95%",
        "grits_con": {
            "observed": observed_con,
            "lower": _linear_percentile(con_values, 0.025),
            "upper": _linear_percentile(con_values, 0.975),
        },
        "grits_top": {
            "observed": observed_top,
            "lower": _linear_percentile(top_values, 0.025),
            "upper": _linear_percentile(top_values, 0.975),
        },
    }


def _evaluation_paths(output_root: Path) -> tuple[Path, Path]:
    base = output_root / "tinix" / "evaluation"
    return base / "table_scores.jsonl", base / "summary.json"


def _load_evaluation_summary(output_root: Path) -> dict[str, Any]:
    scores_path, summary_path = _evaluation_paths(output_root)
    summary = _read_json(summary_path, label="TiniX evaluation summary")
    rows = _read_jsonl(scores_path, label="TiniX score rows")
    if len(rows) != EXPECTED_SCORED_TABLE_COUNT:
        raise TinixError("TiniX scored-table count changed")
    coverage = summary.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("aligned_table_count") != EXPECTED_SCORED_TABLE_COUNT
        or coverage.get("aligned_report_count") != EXPECTED_ALIGNED_REPORT_COUNT
    ):
        raise TinixError("TiniX evaluation coverage changed")
    return summary


def evaluate_tinix(*, output_dir: Path, force: bool = False) -> Path:
    """Build OCR-silver references and score the deployed Primary/Hunyuan mix."""

    scores_path, summary_path = _evaluation_paths(output_dir)
    if scores_path.is_file() and summary_path.is_file() and not force:
        _load_evaluation_summary(output_dir)
        return summary_path
    output_root = output_dir / "tinix"
    cohort = _load_cohort(output_dir)
    primary_records = _load_primary_records(output_dir)
    primary_by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in primary_records:
        primary_by_page[(str(record["report_id"]), int(record["page_index"]))].append(record)
    route = _load_route(output_dir)
    route_ids = {str(item["table_id"]) for item in route["routed_tables"]}
    expert_records: dict[str, dict[str, Any]] = {}
    for table_id in route_ids:
        expert_records[table_id] = _load_expert_record(_expert_file(output_dir, table_id), table_id)

    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for report in cohort["reports"]:
        ocr_path = output_root / str(report["ocr_file"])
        if not ocr_path.is_file():
            raise TinixError(f"prepared OCR file is missing: {report['report_id']}")
        pages = parse_explicit_page_markers(ocr_path.read_text(encoding="utf-8"))
        counters["reports"] += 1
        counters["pages"] += len(pages)
        for page in pages:
            report_id = str(report["report_id"])
            page_index = int(page["page_index"])
            detected = primary_by_page.get((report_id, page_index), [])
            html_info = _extract_top_level_tables(str(page["text"]))
            top_count = int(html_info["top_level_table_count"])
            if (
                top_count != 1
                or bool(html_info["nested_table_detected"])
                or bool(html_info["malformed"])
                or len(detected) != 1
            ):
                continue
            blocks = html_info["blocks"]
            if not isinstance(blocks, list) or len(blocks) != 1:
                continue
            counters["one_html_one_detection_pages"] += 1
            try:
                silver_cells = _canonicalize_silver_html(str(blocks[0]))
            except TinixError:
                continue
            counters["canonicalizable_tables"] += 1
            primary_record = detected[0]
            table_id = str(primary_record["table_id"])
            silver_grits_cells = [cell["grits_cell"] for cell in silver_cells]
            primary_grits_cells = _prediction_grits_cells(
                primary_record.get("cells") or [],
                allow_overlaps=True,
            )
            primary_scores = _score_cells(silver_grits_cells, primary_grits_cells)
            expert_status = None
            deployed_cells = primary_record.get("cells") or []
            deployed_source = "primary"
            if table_id in route_ids:
                expert_status = expert_records[table_id]["status"]
                if expert_status == "success":
                    deployed_cells = expert_records[table_id]["cells"]
                    deployed_source = "hunyuan"
                    counters["aligned_routed_replaced"] += 1
                else:
                    counters["aligned_routed_fallback"] += 1
                counters["aligned_routed"] += 1
            else:
                counters["aligned_non_routed"] += 1
            if deployed_source == "hunyuan":
                deployed_grits_cells = _prediction_grits_cells(deployed_cells, allow_overlaps=False)
            else:
                deployed_grits_cells = primary_grits_cells
            deployed_scores = _score_cells(silver_grits_cells, deployed_grits_cells)
            rows.append({
                "table_id": table_id,
                "report_id": report_id,
                "page_index": page_index,
                "deployed_source": deployed_source,
                "expert_status": expert_status,
                "primary_grits_con": primary_scores["grits_con"],
                "primary_grits_top": primary_scores["grits_top"],
                "deployed_grits_con": deployed_scores["grits_con"],
                "deployed_grits_top": deployed_scores["grits_top"],
            })
    rows.sort(key=lambda row: str(row["table_id"]))
    if counters["pages"] != EXPECTED_PAGE_COUNT:
        raise TinixError(f"TiniX OCR page count changed: {counters['pages']}")
    if counters["one_html_one_detection_pages"] != EXPECTED_ONE_HTML_ONE_DETECTION_COUNT:
        raise TinixError(
            "TiniX one-HTML/one-detection funnel changed: "
            f"{counters['one_html_one_detection_pages']}"
        )
    if counters["canonicalizable_tables"] != EXPECTED_SCORED_TABLE_COUNT:
        raise TinixError(
            f"TiniX canonicalizable alignment count changed: {counters['canonicalizable_tables']}"
        )
    if len({row["report_id"] for row in rows}) != EXPECTED_ALIGNED_REPORT_COUNT:
        raise TinixError("TiniX aligned report count changed")
    if len(rows) != EXPECTED_SCORED_TABLE_COUNT:
        raise TinixError("TiniX aligned table count changed")
    if counters["aligned_routed"] != 149 or counters["aligned_non_routed"] != 683:
        raise TinixError("TiniX aligned route split changed")
    primary_con = statistics.fmean(float(row["primary_grits_con"]) for row in rows)
    primary_top = statistics.fmean(float(row["primary_grits_top"]) for row in rows)
    deployed_con = statistics.fmean(float(row["deployed_grits_con"]) for row in rows)
    deployed_top = statistics.fmean(float(row["deployed_grits_top"]) for row in rows)
    bootstrap = _report_clustered_bootstrap(rows)
    summary = {
        "study": "tinix",
        "status": "completed",
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "coverage": {
            "report_count": EXPECTED_REPORT_COUNT,
            "page_count": counters["pages"],
            "detected_table_count": EXPECTED_DETECTED_TABLE_COUNT,
            "one_html_one_detection_pages": counters["one_html_one_detection_pages"],
            "canonicalizable_aligned_table_count": counters["canonicalizable_tables"],
            "aligned_table_count": len(rows),
            "aligned_report_count": len({row["report_id"] for row in rows}),
            "aligned_routed_table_count": counters["aligned_routed"],
            "aligned_non_routed_table_count": counters["aligned_non_routed"],
            "aligned_routed_replaced_count": counters["aligned_routed_replaced"],
            "aligned_routed_fallback_count": counters["aligned_routed_fallback"],
        },
        "metrics": {
            "primary": {"grits_con": primary_con, "grits_top": primary_top},
            "bear_hgb_hunyuan": {"grits_con": deployed_con, "grits_top": deployed_top},
        },
        "paired_difference": bootstrap,
        "reference": {
            "type": "OCR-generated silver reference",
            "manual_verification": False,
            "alignment": (
                "explicit page marker, exactly one top-level HTML table, exactly one "
                "detected table, and successful canonicalization"
            ),
            "model_or_score_matching": False,
        },
    }
    _atomic_jsonl(scores_path, rows)
    _atomic_json(summary_path, summary)
    return summary_path


def report_tinix(*, output_dir: Path, force: bool = False) -> Path:
    """Assemble the TiniX result from completed executable stages."""

    output_root = output_dir / "tinix"
    cohort = _load_cohort(output_dir)
    primary_summary = _read_json(output_root / "primary" / "summary.json", label="TiniX Primary summary")
    route = _load_route(output_dir)
    expert = _load_expert_summary(output_dir)
    evaluation = _load_evaluation_summary(output_dir)
    result = {
        "study": "tinix",
        "status": "completed",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "download": DATASET_URL,
        },
        "cohort": {
            "reports": cohort["report_count"],
            "reporting_entities": cohort["reporting_entity_count"],
            "pages": cohort["page_count"],
            "detected_tables": primary_summary["detected_table_count"],
            "reports_with_detected_tables": primary_summary["reports_with_detected_tables"],
            "aligned_reports": evaluation["coverage"]["aligned_report_count"],
            "aligned_scored_tables": evaluation["coverage"]["aligned_table_count"],
        },
        "recognition": {
            "primary": {
                "recognizer": primary_summary["recognizer"],
                "valid_tables": primary_summary["primary_valid_count"],
                "feature_order": primary_summary["feature_order"],
            },
            "hunyuan": {
                "recognizer": expert["recognizer"],
                "attempted_calls": expert["attempted_calls"],
                "successful_canonicalizations": expert["successful_canonicalizations"],
                "parse_failures": expert["parse_failures"],
                "generation_failures": expert["generation_failures"],
                "primary_fallbacks": expert["primary_fallbacks"],
            },
            "replacement_rule": expert["replacement_rule"],
        },
        "routing": {
            "router": "BEAR (HGB)",
            "ranking": route["ranking_rule"],
            "call_count": route["call_count"],
            "call_fraction_of_detected_tables": route["call_count"] / route["detected_table_count"],
            "aligned_routed_tables": evaluation["coverage"]["aligned_routed_table_count"],
            "aligned_non_routed_tables": evaluation["coverage"]["aligned_non_routed_table_count"],
        },
        "metrics": {
            "primary": evaluation["metrics"]["primary"],
            "bear_hgb_hunyuan": evaluation["metrics"]["bear_hgb_hunyuan"],
            "paired_difference": evaluation["paired_difference"],
        },
        "evaluation": evaluation["reference"],
        "claim_boundary": (
            "These metrics measure agreement with OCR-generated silver references. "
            "The references were not manually verified, so the result is not "
            "human-gold accuracy or independent accounting validation."
        ),
    }
    destination = output_root / "results" / "tinix.json"
    if destination.is_file() and not force:
        existing = _read_json(destination, label="TiniX result")
        if existing != result:
            raise TinixError("existing TiniX result is stale; report assembly is deterministic")
        return destination
    _atomic_json(destination, result)
    return destination


def run_tinix_stage(
    stage: str,
    *,
    source_root: str | Path,
    output_dir: str | Path,
    models_dir: str | Path = "models",
    protocol_path: str | Path | None = None,
    device: str = "auto",
    force: bool = False,
) -> Path:
    """Run one resumable TiniX stage."""

    source = Path(source_root)
    output = Path(output_dir)
    models = Path(models_dir)
    if stage == "prepare":
        return prepare_tinix(source_root=source, output_dir=output, force=force)
    if stage == "primary":
        return primary_tinix(
            source_root=source,
            output_dir=output,
            models_dir=models,
            device=device,
            force=force,
        )
    if stage == "route":
        return route_tinix(output_dir=output, models_dir=models, force=force)
    if stage == "expert":
        if protocol_path is None:
            protocol_path = source / "protocol.json"
        return expert_tinix(
            source_root=source,
            output_dir=output,
            models_dir=models,
            protocol_path=Path(protocol_path),
            device=device,
            force=force,
        )
    if stage == "evaluate":
        return evaluate_tinix(output_dir=output, force=force)
    if stage == "report":
        return report_tinix(output_dir=output, force=force)
    raise TinixError(f"unsupported TiniX stage: {stage}")
