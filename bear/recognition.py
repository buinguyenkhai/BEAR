"""The RapidOCR/TableFormer Primary recognition path."""

from __future__ import annotations

import copy
import importlib.metadata
import math
import re
import sys
import time
import unicodedata
from collections.abc import Iterable, Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .routing import (
    FEATURE_NAMES,
    feature_record_from_primary,
    features_from_primary,
    ordered_feature_vector,
)


class RecognitionError(RuntimeError):
    """Raised when the selected Primary path cannot produce a canonical table."""


EXPERT_IDS = ("hunyuanocr_1_5", "glm_ocr")
def _expert_config(protocol: Mapping[str, Any], expert_id: str) -> Mapping[str, Any]:
    experts = protocol.get("experts")
    if not isinstance(experts, Mapping) or set(experts) != set(EXPERT_IDS):
        raise RecognitionError("protocol must define exactly the supported experts")
    if expert_id not in EXPERT_IDS:
        raise RecognitionError(f"unsupported expert: {expert_id}")
    config = experts.get(expert_id)
    if not isinstance(config, Mapping):
        raise RecognitionError(f"protocol expert {expert_id} is not an object")
    required = (
        "configuration_id",
        "checkpoint",
        "backend",
        "model_class",
        "processor_class",
        "prompt",
        "generation",
        "loading",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise RecognitionError(
            f"protocol expert {expert_id} is missing: {', '.join(missing)}"
        )
    if config["configuration_id"] != expert_id:
        raise RecognitionError(f"protocol expert {expert_id} has the wrong configuration ID")
    for field in ("backend", "model_class", "processor_class", "prompt"):
        if not isinstance(config[field], str) or not config[field]:
            raise RecognitionError(f"protocol expert {expert_id}.{field} must be a non-empty string")
    if config["backend"] != "native_transformers":
        raise RecognitionError(f"unsupported backend for {expert_id}")

    checkpoint = config["checkpoint"]
    if not isinstance(checkpoint, Mapping):
        raise RecognitionError(f"protocol expert {expert_id}.checkpoint must be an object")
    for field in ("model_id", "revision"):
        if not isinstance(checkpoint.get(field), str) or not checkpoint[field]:
            raise RecognitionError(f"protocol expert {expert_id}.checkpoint lacks {field}")
    if re.fullmatch(r"[0-9a-f]{40}", checkpoint["revision"]) is None:
        raise RecognitionError(f"protocol expert {expert_id}.checkpoint revision is not immutable")

    generation = config["generation"]
    if not isinstance(generation, Mapping):
        raise RecognitionError(f"protocol expert {expert_id}.generation must be an object")
    max_new_tokens = generation.get("max_new_tokens")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise RecognitionError(f"protocol expert {expert_id}.generation.max_new_tokens is invalid")
    for field in ("do_sample", "skip_special_tokens"):
        if not isinstance(generation.get(field), bool):
            raise RecognitionError(f"protocol expert {expert_id}.generation.{field} must be boolean")

    loading = config["loading"]
    if not isinstance(loading, Mapping):
        raise RecognitionError(f"protocol expert {expert_id}.loading must be an object")
    for field in ("device_map", "required_device"):
        if not isinstance(loading.get(field), str) or not loading[field]:
            raise RecognitionError(f"protocol expert {expert_id}.loading.{field} is invalid")
    if loading["required_device"] != "cuda":
        raise RecognitionError(f"protocol expert {expert_id} requires CUDA")
    if not isinstance(loading.get("quantized"), bool):
        raise RecognitionError(f"protocol expert {expert_id}.loading.quantized must be boolean")
    for field in ("torch_dtype", "torch_dtype_cuda"):
        if field in loading and (not isinstance(loading[field], str) or not loading[field]):
            raise RecognitionError(f"protocol expert {expert_id}.loading.{field} is invalid")
    for field in ("trust_remote_code", "use_fast"):
        if field in loading and not isinstance(loading[field], bool):
            raise RecognitionError(f"protocol expert {expert_id}.loading.{field} must be boolean")
    if expert_id == "hunyuanocr_1_5" and loading.get("torch_dtype_cuda") != "bfloat16":
        raise RecognitionError("Hunyuan must use bfloat16 on CUDA")
    return config


def validate_expert_protocol(
    protocol: Mapping[str, Any],
    expert_id: str | None = None,
) -> None:
    """Validate immutable source identifiers and settings for the two experts."""

    for selected in EXPERT_IDS if expert_id is None else (expert_id,):
        _expert_config(protocol, selected)


def expert_semantic_identity(protocol: Mapping[str, Any], expert_id: str) -> dict[str, Any]:
    """Return the checkpoint identity used by one specified expert configuration."""

    config = _expert_config(protocol, expert_id)
    checkpoint = config["checkpoint"]
    return {
        "expert_id": expert_id,
        "checkpoint_model_id": checkpoint["model_id"],
        "checkpoint_revision": checkpoint["revision"],
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RecognitionError(f"required package is not installed: {name}") from exc


def _validate_expert_environment(expert_id: str, device: str) -> dict[str, Any]:
    if device != "cuda":
        raise RecognitionError("expert inference requires --device cuda")
    try:
        import torch
        import torchvision
        import transformers
        from PIL import Image
    except ImportError as exc:
        raise RecognitionError("the specified expert environment is unavailable") from exc

    versions = {
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "Pillow": _package_version("Pillow"),
    }
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if not cuda_available or cuda_device_count < 1:
        raise RecognitionError("CUDA was requested but is unavailable")
    bfloat16_supported = bool(
        cuda_available and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    if expert_id == "hunyuanocr_1_5" and not bfloat16_supported:
        raise RecognitionError("CUDA bfloat16 support is unavailable for Hunyuan")
    try:
        if expert_id == "hunyuanocr_1_5":
            from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RecognitionError(f"required Transformers class is unavailable for {expert_id}") from exc
    return {
        "python": sys.version.split()[0],
        "torch": versions["torch"],
        "torchvision": versions["torchvision"],
        "transformers": versions["transformers"],
        "accelerate": versions["accelerate"],
        "Pillow": versions["Pillow"],
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "bfloat16_cuda_supported": bfloat16_supported,
        "device_request": device,
    }


def _normalize_expert_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", value, flags=re.UNICODE).strip()


def _validate_expert_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not specs:
        raise RecognitionError("expert output contains no explicit cells")
    occupied: dict[tuple[int, int], int] = {}
    validated: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        rows = spec.get("rows")
        columns = spec.get("columns")
        text = spec.get("text")
        if (
            not isinstance(rows, list)
            or not isinstance(columns, list)
            or not rows
            or not columns
            or any(isinstance(value, bool) or not isinstance(value, int) for value in [*rows, *columns])
            or rows != list(range(rows[0], rows[-1] + 1))
            or columns != list(range(columns[0], columns[-1] + 1))
            or rows[0] < 0
            or columns[0] < 0
        ):
            raise RecognitionError(f"expert cell {index} has an invalid range")
        if not isinstance(text, str):
            raise RecognitionError(f"expert cell {index} text is not a string")
        for row in rows:
            for column in columns:
                coordinate = (row, column)
                if coordinate in occupied:
                    raise RecognitionError(f"expert cells overlap at {coordinate}")
                occupied[coordinate] = index
        validated.append(
            {
                "rows": rows,
                "columns": columns,
                "text": _normalize_expert_text(text),
                "is_column_header": bool(spec.get("is_column_header", False)),
                "is_row_header": bool(spec.get("is_row_header", False)),
            }
        )
    validated.sort(
        key=lambda cell: (
            cell["rows"][0],
            cell["columns"][0],
            cell["rows"][-1],
            cell["columns"][-1],
            cell["text"],
        )
    )
    return validated


class _ExpertHTMLTableParser(HTMLParser):
    """Parse one strict table and record text encountered outside it."""

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
            if self.table_count:
                self._fail("expert output contains multiple HTML tables")
            self.table_count += 1
            self.table_depth = 1
            return
        if not self.table_depth:
            return
        if tag == "tr":
            if self.current_row is not None or self.current_cell is not None:
                self._fail("nested HTML rows are not supported")
                return
            self.current_row = []
            self.rows.append(self.current_row)
        elif tag in {"td", "th"}:
            if self.current_row is None or self.current_cell is not None:
                self._fail("HTML cell is outside a row or overlaps another cell")
                return
            self.current_cell = {
                "tag": tag,
                "text_parts": [],
                "rowspan": attributes.get("rowspan", "1"),
                "colspan": attributes.get("colspan", "1"),
                "is_column_header": tag == "th",
                "is_row_header": False,
            }

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in {"td", "th"}:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)
        elif tag == "table":
            self._fail("self-closing HTML tables are not supported")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"td", "th"}:
            if self.current_cell is None or self.current_row is None:
                self._fail("HTML cell closes without an open cell")
            elif self.current_cell["tag"] != tag:
                self._fail("HTML cell tags are mismatched")
            else:
                self.current_row.append(self.current_cell)
                self.current_cell = None
        elif tag == "tr":
            if self.current_row is None or self.current_cell is not None:
                self._fail("HTML row closes before its cells are complete")
            self.current_row = None
        elif tag == "table":
            if not self.table_depth:
                self._fail("HTML table closes without an open table")
            elif self.current_cell is not None or self.current_row is not None:
                self._fail("HTML table closes before its rows are complete")
            else:
                self.table_depth = 0

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell["text_parts"].append(data)
        elif not self.table_depth:
            self.outside_text.append(data)


def _html_expert_table_to_specs(value: str) -> list[dict[str, Any]]:
    parser = _ExpertHTMLTableParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception as exc:
        raise RecognitionError(f"HTML parser failed: {type(exc).__name__}") from exc
    if parser.error:
        raise RecognitionError(parser.error)
    if parser.table_count != 1 or parser.table_depth != 0:
        raise RecognitionError("expert output must contain exactly one complete HTML table")
    if any(part.strip() for part in parser.outside_text):
        raise RecognitionError("HTML output contains text outside the table")
    if not parser.rows:
        raise RecognitionError("HTML table contains no rows")
    occupied: dict[tuple[int, int], int] = {}
    specs: list[dict[str, Any]] = []
    max_column = 0
    for row_index, row in enumerate(parser.rows):
        cursor = 0
        for cell in row:
            while (row_index, cursor) in occupied:
                cursor += 1
            try:
                row_span = int(str(cell["rowspan"]).strip())
                column_span = int(str(cell["colspan"]).strip())
            except (TypeError, ValueError) as exc:
                raise RecognitionError("HTML rowspan and colspan must be integers") from exc
            if row_span <= 0 or column_span <= 0:
                raise RecognitionError("HTML rowspan and colspan must be positive")
            if row_index + row_span > len(parser.rows):
                raise RecognitionError("HTML rowspan extends beyond the declared rows")
            coordinates = [
                (row, column)
                for row in range(row_index, row_index + row_span)
                for column in range(cursor, cursor + column_span)
            ]
            if any(coordinate in occupied for coordinate in coordinates):
                raise RecognitionError("HTML table contains overlapping spans")
            cell_index = len(specs)
            for coordinate in coordinates:
                occupied[coordinate] = cell_index
            specs.append(
                {
                    "rows": list(range(row_index, row_index + row_span)),
                    "columns": list(range(cursor, cursor + column_span)),
                    "text": "".join(cell["text_parts"]),
                    "is_column_header": bool(cell["is_column_header"]),
                    "is_row_header": bool(cell["is_row_header"]),
                }
            )
            cursor += column_span
            max_column = max(max_column, cursor)
    for row_index in range(len(parser.rows)):
        for column_index in range(max_column):
            if (row_index, column_index) not in occupied:
                raise RecognitionError("HTML table contains an uncovered grid position")
    return _validate_expert_specs(specs)


_EXPERT_MARKDOWN_SEPARATOR = re.compile(r"^:?-{1,}:?$")


def _split_expert_pipe_row(line: str) -> list[str]:
    value = line.strip()
    if "|" not in value:
        raise RecognitionError("Markdown table row has no pipe separator")
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith("\\|"):
        value = value[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in value:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        raise RecognitionError("Markdown table row ends with an escape")
    cells.append("".join(current).strip())
    return cells


def _markdown_expert_table_to_specs(value: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) < 2 or any(line.startswith("```") for line in lines):
        raise RecognitionError("Markdown output is not one strict table")
    rows = [_split_expert_pipe_row(line) for line in lines]
    width = len(rows[0])
    if width == 0 or len(rows[1]) != width:
        raise RecognitionError("Markdown table widths differ")
    if not all(_EXPERT_MARKDOWN_SEPARATOR.fullmatch(cell.replace(" ", "")) for cell in rows[1]):
        raise RecognitionError("Markdown second row is not a separator row")
    if any(len(row) != width for row in rows[2:]):
        raise RecognitionError("Markdown table rows have inconsistent widths")
    actual_rows = [rows[0], *rows[2:]]
    return _validate_expert_specs(
        [
            {
                "rows": [row_index],
                "columns": [column_index],
                "text": cell_text,
                "is_column_header": row_index == 0,
                "is_row_header": False,
            }
            for row_index, row in enumerate(actual_rows)
            for column_index, cell_text in enumerate(row)
        ]
    )


def canonicalize_expert_output(value: str | bytes) -> tuple[list[dict[str, Any]], str]:
    """Strictly canonicalize one Hunyuan or GLM native table output."""

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RecognitionError("expert output bytes are not UTF-8") from exc
    if not isinstance(value, str):
        raise RecognitionError("expert output is not text")
    stripped = re.sub(
        r"(?:<\|(?:user|assistant|endoftext|end_of_text)\|>\s*)+$", "", value.strip()
    ).strip()
    if not stripped:
        raise RecognitionError("expert output is empty")
    if stripped.startswith("<fcel>"):
        raise RecognitionError("Paddle table tokens are not supported by the expert path")
    if "<table" in stripped.casefold():
        return _expert_cells_to_canonical(_html_expert_table_to_specs(stripped)), "html"
    return _expert_cells_to_canonical(_markdown_expert_table_to_specs(stripped)), "markdown"


def _expert_cells_to_canonical(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in specs:
        rows = spec["rows"]
        columns = spec["columns"]
        cell: dict[str, Any] = {
            "row_start": rows[0],
            "row_end": rows[-1] + 1,
            "column_start": columns[0],
            "column_end": columns[-1] + 1,
            "text": spec["text"],
        }
        if spec["is_column_header"]:
            cell["is_column_header"] = True
        if spec["is_row_header"]:
            cell["is_row_header"] = True
        result.append(cell)
    return result


def _expert_native_format(value: Any) -> str | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    stripped = value.strip().casefold()
    if "<table" in stripped:
        return "html"
    if stripped.startswith("<fcel>"):
        return "paddle_table_tokens"
    if "|" in stripped:
        return "markdown"
    return None


def _safe_expert_error(exc: BaseException) -> str:
    value = " ".join(str(exc).split())
    value = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s,;\"']+", "<path>", value)
    return (value or type(exc).__name__)[:500]


def _protocol_artifacts(protocol: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the structurally valid Primary selections in the protocol."""

    primary = protocol.get("primary")
    if not isinstance(primary, Mapping):
        raise RecognitionError("protocol Primary configuration is missing")
    for field in ("name", "configuration_id"):
        if not isinstance(primary.get(field), str) or not primary[field].strip():
            raise RecognitionError(f"protocol Primary {field} is missing")
    rapidocr = primary.get("rapidocr")
    docling = primary.get("docling")
    if not isinstance(rapidocr, Mapping) or not isinstance(docling, Mapping):
        raise RecognitionError("protocol Primary artifact settings are incomplete")
    for field in ("package_version", "backend", "language", "model_family"):
        if not isinstance(rapidocr.get(field), str) or not rapidocr[field].strip():
            raise RecognitionError(f"protocol RapidOCR {field} is missing")
    for stage in ("detector", "classifier", "recognizer"):
        selection = rapidocr.get(stage)
        if (
            not isinstance(selection, Mapping)
            or not isinstance(selection.get("configuration"), str)
            or not selection["configuration"].strip()
            or not isinstance(selection.get("filename"), str)
            or not selection["filename"].strip()
        ):
            raise RecognitionError(f"protocol RapidOCR {stage} selection is incomplete")
    if not isinstance(rapidocr["classifier"].get("active"), bool):
        raise RecognitionError("protocol RapidOCR classifier activity is invalid")
    dictionary = rapidocr.get("dictionary")
    if not isinstance(dictionary, Mapping) or "filename" not in dictionary:
        raise RecognitionError("protocol RapidOCR dictionary settings are incomplete")
    for field in ("package_version", "core_version", "ocr_backend", "ocr_mode", "table_mode"):
        if not isinstance(docling.get(field), str) or not docling[field].strip():
            raise RecognitionError(f"protocol Docling {field} is missing")
    if not isinstance(docling.get("ocr_language"), list) or not docling["ocr_language"]:
        raise RecognitionError("protocol Docling OCR language is invalid")
    if any(not isinstance(value, str) or not value.strip() for value in docling["ocr_language"]):
        raise RecognitionError("protocol Docling OCR language is invalid")
    for field in ("cell_matching", "whole_crop_processing"):
        if not isinstance(docling.get(field), bool):
            raise RecognitionError(f"protocol Docling {field} setting is invalid")
    tableformer = docling.get("tableformer")
    if not isinstance(tableformer, Mapping):
        raise RecognitionError("protocol TableFormer artifact settings are missing")
    for field in ("model_repository", "revision", "path"):
        if not isinstance(tableformer.get(field), str) or not tableformer[field].strip():
            raise RecognitionError(f"protocol TableFormer {field} is missing")
    layout = docling.get("layout")
    if (
        not isinstance(layout, Mapping)
        or not isinstance(layout.get("model_repository"), str)
        or not layout["model_repository"].strip()
        or not isinstance(layout.get("engine"), str)
        or not layout["engine"].strip()
        or not isinstance(layout.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", layout["revision"]) is None
    ):
        raise RecognitionError("protocol Layout artifact must pin a 40-character revision")
    return rapidocr, docling


def primary_semantic_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic semantic identity of the specified Primary path."""

    rapidocr, docling = _protocol_artifacts(protocol)
    primary = protocol["primary"]
    return {
        "configuration_id": primary["configuration_id"],
        "name": primary["name"],
        "rapidocr_package_version": rapidocr["package_version"],
        "rapidocr_backend": rapidocr["backend"],
        "rapidocr_language": rapidocr["language"],
        "rapidocr_model_family": rapidocr["model_family"],
        "detector_configuration": rapidocr["detector"]["configuration"],
        "detector_filename": rapidocr["detector"]["filename"],
        "classifier_configuration": rapidocr["classifier"]["configuration"],
        "classifier_filename": rapidocr["classifier"]["filename"],
        "classifier_active": rapidocr["classifier"]["active"],
        "recognizer_configuration": rapidocr["recognizer"]["configuration"],
        "recognizer_filename": rapidocr["recognizer"]["filename"],
        "docling_package_version": docling["package_version"],
        "docling_core_version": docling["core_version"],
        "ocr_backend": docling["ocr_backend"],
        "ocr_language": docling["ocr_language"],
        "ocr_mode": docling["ocr_mode"],
        "table_mode": docling["table_mode"],
        "cell_matching": docling["cell_matching"],
        "whole_crop_processing": docling["whole_crop_processing"],
        "tableformer_repository": docling["tableformer"]["model_repository"],
        "tableformer_revision": docling["tableformer"]["revision"],
        "layout_repository": docling["layout"]["model_repository"],
        "layout_revision": docling["layout"]["revision"],
        "layout_engine": docling["layout"]["engine"],
    }


def validate_primary_environment(protocol: Mapping[str, Any]) -> None:
    """Check required packages and protocol selections before model initialization."""

    _protocol_artifacts(protocol)
    for package in ("rapidocr", "onnxruntime", "docling", "docling-core"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RecognitionError(f"required package is not installed: {package}") from exc


def _resolve_primary_artifacts(
    models_dir: str | Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the protocol-selected OCR, layout, and TableFormer artifacts into ``models_dir``."""

    rapidocr, docling = _protocol_artifacts(protocol)
    root = Path(models_dir).expanduser()
    if root.exists() and not root.is_dir():
        raise RecognitionError(f"models directory is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    rapid_root = root / "rapidocr"
    rapid_root.mkdir(parents=True, exist_ok=True)
    try:
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

        RapidOcrModel.download_models(
            backend=str(rapidocr["backend"]),
            local_dir=rapid_root,
            progress=False,
            lang=str(rapidocr["language"]),
        )
    except Exception as exc:
        raise RecognitionError("could not resolve the protocol-selected RapidOCR artifacts") from exc

    rapid_paths: dict[str, Path] = {}
    for role in ("detector", "classifier", "recognizer"):
        selection = rapidocr[role]
        filename = str(selection["filename"])
        path = rapid_root / filename
        if not path.is_file():
            raise RecognitionError(f"resolved RapidOCR {role} artifact is missing")
        rapid_paths[role] = path.resolve()

    tableformer_root = root / "docling"
    tableformer_root.mkdir(parents=True, exist_ok=True)
    tableformer = docling["tableformer"]
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=str(tableformer["model_repository"]),
            revision=str(tableformer["revision"]),
            local_dir=str(tableformer_root),
            allow_patterns=["model_artifacts/tableformer/**"],
        )
    except Exception as exc:
        raise RecognitionError("could not resolve the protocol-selected TableFormer artifacts") from exc
    tableformer_path = tableformer_root / str(tableformer["path"])
    if not (tableformer_path / "tm_config.json").is_file():
        raise RecognitionError("resolved accurate TableFormer artifact is missing tm_config.json")

    # Standard Docling image processing also initializes a Heron layout stage.
    # Keep that stage under the same explicit artifact root and resolve the
    # protocol-selected commit so a mutable package default cannot be used.
    layout = docling["layout"]
    layout_root = tableformer_root / str(layout["model_repository"]).replace("/", "--")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=str(layout["model_repository"]),
            revision=str(layout["revision"]),
            local_dir=str(layout_root),
        )
    except Exception as exc:
        raise RecognitionError("could not resolve the protocol-selected Heron layout artifact") from exc
    for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if not (layout_root / filename).is_file():
            raise RecognitionError(f"resolved Heron layout artifact is missing {filename}")

    return {
        "rapidocr": rapid_paths,
        "tableformer_root": tableformer_root.resolve(),
        "tableformer_path": tableformer_path.resolve(),
        "layout_path": layout_root.resolve(),
        "layout_repository": str(layout["model_repository"]),
        "layout_revision": str(layout["revision"]),
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecognitionError(f"{context} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RecognitionError(f"{context} is not finite")
    return result


def _coordinate_origin(bbox: Any) -> str:
    origin = _field(bbox, "coord_origin")
    if origin is None:
        raise RecognitionError("bounding box has no coordinate origin")
    return str(getattr(origin, "value", origin)).casefold().replace("_", "")


def _top_left_bbox(bbox: Any, image_height: int) -> list[float] | None:
    if bbox is None:
        return None
    if hasattr(bbox, "to_bounding_box"):
        bbox = bbox.to_bounding_box()
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = [_number(value, "bounding-box coordinate") for value in bbox]
        left, top, right, bottom = values
        x_min, x_max = sorted((left, right))
        y_min, y_max = sorted((top, bottom))
        if not x_min < x_max or not y_min < y_max:
            raise RecognitionError("bounding box has no positive area")
        return [x_min, y_min, x_max, y_max]
    left = _number(_field(bbox, "l"), "bounding-box left")
    first_y = _number(_field(bbox, "t"), "bounding-box top")
    right = _number(_field(bbox, "r"), "bounding-box right")
    second_y = _number(_field(bbox, "b"), "bounding-box bottom")
    x_min, x_max = sorted((left, right))
    origin = _coordinate_origin(bbox)
    if origin in {"bottomleft", "bottom-left"}:
        y_min, y_max = sorted((float(image_height) - first_y, float(image_height) - second_y))
    elif origin in {"topleft", "top-left"}:
        y_min, y_max = sorted((first_y, second_y))
    else:
        raise RecognitionError(f"unsupported bounding-box origin: {origin}")
    if not x_min < x_max or not y_min < y_max:
        raise RecognitionError("bounding box has no positive area")
    return [x_min, y_min, x_max, y_max]


def _load_docling_api() -> dict[str, Any]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            LayoutObjectDetectionOptions,
            OcrMode,
            PdfPipelineOptions,
            RapidOcrOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.datamodel.stage_model_specs import ObjectDetectionModelSpec
        from docling.document_converter import DocumentConverter, ImageFormatOption
    except ImportError as exc:
        raise RecognitionError("Docling and RapidOCR dependencies are unavailable") from exc
    return {
        "DocumentConverter": DocumentConverter,
        "ImageFormatOption": ImageFormatOption,
        "InputFormat": InputFormat,
        "LayoutObjectDetectionOptions": LayoutObjectDetectionOptions,
        "ObjectDetectionModelSpec": ObjectDetectionModelSpec,
        "OcrMode": OcrMode,
        "PdfPipelineOptions": PdfPipelineOptions,
        "RapidOcrOptions": RapidOcrOptions,
        "TableFormerMode": TableFormerMode,
        "TableStructureOptions": TableStructureOptions,
    }


def _build_converter(
    device: str = "auto",
    *,
    artifacts: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Any:
    """Build the specified full-page Vietnamese RapidOCR/TableFormer setup."""

    if device not in {"auto", "cpu", "cuda"}:
        raise RecognitionError(f"unsupported device: {device}")
    api = _load_docling_api()
    rapid_paths = artifacts.get("rapidocr")
    tableformer_root = artifacts.get("tableformer_root")
    layout_path = artifacts.get("layout_path")
    if (
        not isinstance(rapid_paths, Mapping)
        or not isinstance(tableformer_root, Path)
        or not isinstance(layout_path, Path)
    ):
        raise RecognitionError("resolved Primary artifacts are incomplete")
    layout_repository = str(artifacts.get("layout_repository", ""))
    layout_revision = str(artifacts.get("layout_revision", ""))
    if not layout_repository or not layout_revision:
        raise RecognitionError("resolved Layout Heron selection is unavailable")
    rapidocr, docling = _protocol_artifacts(protocol)
    try:
        ocr_mode = getattr(api["OcrMode"], str(docling["ocr_mode"]).upper())
        table_mode = getattr(api["TableFormerMode"], str(docling["table_mode"]).upper())
    except AttributeError as exc:
        raise RecognitionError("protocol selects an unsupported Primary mode") from exc
    ocr_options = api["RapidOcrOptions"](
        mode=ocr_mode,
        lang=[str(value) for value in docling["ocr_language"]],
        backend=str(rapidocr["backend"]),
        det_model_path=str(rapid_paths["detector"]),
        cls_model_path=str(rapid_paths["classifier"]),
        rec_model_path=str(rapid_paths["recognizer"]),
    )
    pipeline_kwargs: dict[str, Any] = {
        "do_ocr": True,
        "do_table_structure": True,
        "ocr_options": ocr_options,
        "artifacts_path": tableformer_root,
        "enable_remote_services": False,
        "allow_external_plugins": False,
        "table_structure_options": api["TableStructureOptions"](
            mode=table_mode,
            do_cell_matching=bool(docling["cell_matching"]),
        ),
    }
    pipeline_kwargs["layout_options"] = api["LayoutObjectDetectionOptions"](
        model_spec=api["ObjectDetectionModelSpec"](
            name="layout_heron",
            repo_id=layout_repository,
            revision=layout_revision,
        )
    )
    if device != "auto":
        from docling.datamodel.accelerator_options import AcceleratorOptions

        pipeline_kwargs["accelerator_options"] = AcceleratorOptions(device=device)
    pipeline_options = api["PdfPipelineOptions"](**pipeline_kwargs)
    format_option = api["ImageFormatOption"](pipeline_options=pipeline_options)
    converter = api["DocumentConverter"](
        allowed_formats=[api["InputFormat"].IMAGE],
        format_options={api["InputFormat"].IMAGE: format_option},
    )
    converter.initialize_pipeline(api["InputFormat"].IMAGE)
    return converter


def _stable_identity_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        value = vars(value)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _stable_identity_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "index"
        )
    if isinstance(value, (list, tuple)):
        return tuple(_stable_identity_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_stable_identity_value(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _prepare_tableformer_ocr_cells(cells: Iterable[Any]) -> list[Any]:
    retained: list[Any] = []
    seen: set[Any] = set()
    for source in cells:
        identity = _stable_identity_value(source)
        if identity in seen:
            continue
        seen.add(identity)
        if hasattr(source, "model_copy"):
            copied = source.model_copy(deep=True, update={"index": len(retained)})
        else:
            copied = copy.deepcopy(source)
            setattr(copied, "index", len(retained))
        retained.append(copied)
    indexes = [int(_field(cell, "index")) for cell in retained]
    if indexes != list(range(len(retained))):
        raise RecognitionError("prepared OCR cells do not have contiguous indexes")
    return retained


def _whole_image_cluster(width: int, height: int, cells: Iterable[Any]) -> Any:
    if width <= 0 or height <= 0:
        raise RecognitionError("image dimensions must be positive")
    try:
        from docling.datamodel.base_models import Cluster
        from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
    except ImportError as exc:
        raise RecognitionError("Docling whole-image types are unavailable") from exc
    return Cluster(
        id=0,
        label=DocItemLabel.TABLE,
        bbox=BoundingBox(
            l=0.0,
            t=0.0,
            r=float(width),
            b=float(height),
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        confidence=1.0,
        cells=list(cells),
        children=[],
    )


def _table_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _ocr_cells_and_evidence(
    conversion_result: Any,
    image_height: int,
) -> tuple[list[Any], list[list[float] | None], list[float | None]]:
    pages = _field(conversion_result, "pages")
    if not isinstance(pages, list) or len(pages) != 1:
        raise RecognitionError("expected one Docling page for a table crop")
    predictions = _field(pages[0], "predictions")
    layout = _field(predictions, "layout")
    clusters = _field(layout, "clusters")
    if not isinstance(clusters, list):
        raise RecognitionError("Docling did not expose OCR layout cells")
    cells: list[Any] = []
    for cluster in clusters:
        cluster_cells = _field(cluster, "cells", [])
        if isinstance(cluster_cells, list):
            cells.extend(cluster_cells)
    cells = _prepare_tableformer_ocr_cells(cells)
    boxes: list[list[float] | None] = []
    confidences: list[float | None] = []
    for cell in cells:
        bbox = _field(cell, "rect")
        if bbox is None:
            bbox = _field(cell, "bbox")
        if bbox is None:
            boxes.append(None)
        else:
            try:
                boxes.append(_top_left_bbox(bbox, image_height))
            except RecognitionError:
                boxes.append(None)
        confidence = _field(cell, "confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidences.append(None)
        else:
            confidence_value = float(confidence)
            confidences.append(confidence_value if math.isfinite(confidence_value) else None)
    return cells, boxes, confidences


def _ocr_evidence(
    conversion_result: Any,
    image_height: int,
) -> tuple[list[list[float] | None], list[float | None]]:
    """Return only the evidence fields for callers that do not need OCR cells."""

    _cells, boxes, confidences = _ocr_cells_and_evidence(conversion_result, image_height)
    return boxes, confidences


def _run_whole_image_table(
    converter: Any,
    conversion_result: Any,
    image_path: Path,
    width: int,
    height: int,
    *,
    prepared_ocr_cells: list[Any] | None = None,
    ocr_boxes: list[list[float] | None] | None = None,
    ocr_confidences: list[float | None] | None = None,
) -> tuple[Any, list[list[float] | None], list[float | None]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RecognitionError("Pillow is unavailable") from exc
    if prepared_ocr_cells is None:
        prepared_ocr_cells, boxes, confidences = _ocr_cells_and_evidence(
            conversion_result,
            height,
        )
    else:
        boxes = list(ocr_boxes or ())
        confidences = list(ocr_confidences or ())
    pages = _field(conversion_result, "pages")
    pipelines = _field(converter, "initialized_pipelines")
    if not isinstance(pipelines, Mapping) or not pipelines:
        raise RecognitionError("Docling initialized pipelines are unavailable")
    pipeline = next(iter(pipelines.values()))
    table_model = _field(pipeline, "table_model")
    method = _field(table_model, "_do_prediction_on_image_to_table")
    if not callable(method):
        raise RecognitionError("Docling whole-image TableFormer method is unavailable")
    with Image.open(image_path) as opened:
        table_image = opened.convert("RGB")
    table = method(
        table_image=table_image,
        table_cluster=_whole_image_cluster(width, height, prepared_ocr_cells),
        page_no=int(_field(pages[0], "page_no", 1)),
    )
    tables = _table_list(table)
    if len(tables) != 1:
        raise RecognitionError(f"expected exactly one TableFormer result, found {len(tables)}")
    return tables[0], boxes, confidences


def _canonical_cells(table: Any, image_height: int) -> list[dict[str, Any]]:
    raw_cells = _field(table, "table_cells")
    if raw_cells is None:
        raise RecognitionError("TableFormer result has no structured cells")
    result: list[dict[str, Any]] = []
    for index, cell in enumerate(raw_cells):
        row_start = _field(cell, "start_row_offset_idx")
        row_end = _field(cell, "end_row_offset_idx")
        column_start = _field(cell, "start_col_offset_idx")
        column_end = _field(cell, "end_col_offset_idx")
        values = (row_start, row_end, column_start, column_end)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise RecognitionError(f"cell {index} has invalid table offsets")
        if row_start < 0 or column_start < 0 or row_end <= row_start or column_end <= column_start:
            raise RecognitionError(f"cell {index} has invalid table ranges")
        bbox = _top_left_bbox(_field(cell, "bbox"), image_height)
        text = _field(cell, "text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise RecognitionError(f"cell {index} text is not a string")
        record: dict[str, Any] = {
            "row_start": row_start,
            "row_end": row_end,
            "column_start": column_start,
            "column_end": column_end,
            "text": text,
        }
        if bbox is not None:
            record["bbox"] = bbox
        if bool(_field(cell, "column_header", False)):
            record["is_column_header"] = True
        if bool(_field(cell, "row_header", False) or _field(cell, "row_section", False)):
            record["is_row_header"] = True
        result.append(record)
    if not result:
        raise RecognitionError("TableFormer returned no canonical cells")
    result.sort(key=lambda item: (
        item["row_start"],
        item["column_start"],
        item["row_end"],
        item["column_end"],
        item.get("bbox", []),
        item["text"],
    ))
    return result


class PrimaryRecognizer:
    """One persistent Primary converter."""

    def __init__(
        self,
        device: str = "auto",
        *,
        models_dir: str | Path = "models",
        protocol: Mapping[str, Any] | None = None,
    ) -> None:
        if protocol is None:
            from .data import load_protocol

            protocol = load_protocol()
        validate_primary_environment(protocol)
        self.device = device
        self.artifacts = _resolve_primary_artifacts(models_dir, protocol)
        self.converter = _build_converter(device, artifacts=self.artifacts, protocol=protocol)
        rapidocr, docling = _protocol_artifacts(protocol)
        self.artifact_resolution = {
            "runtime_checked": True,
            "rapidocr": {
                "backend": str(rapidocr["backend"]),
                "language": str(rapidocr["language"]),
                "model_family": str(rapidocr["model_family"]),
                "detector": str(rapidocr["detector"]["filename"]),
                "classifier": str(rapidocr["classifier"]["filename"]),
                "classifier_active": bool(rapidocr["classifier"]["active"]),
                "recognizer": str(rapidocr["recognizer"]["filename"]),
            },
            "tableformer": {
                "model_repository": str(docling["tableformer"]["model_repository"]),
                "revision": str(docling["tableformer"]["revision"]),
                "path": str(docling["tableformer"]["path"]),
                "mode": str(docling["table_mode"]),
                "cell_matching": bool(docling["cell_matching"]),
            },
            "layout": {
                "model_repository": str(docling["layout"]["model_repository"]),
                "revision": str(docling["layout"]["revision"]),
                "engine": str(docling["layout"]["engine"]),
                "runtime_path_checked": True,
            },
        }

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """Run one crop and return a JSON-compatible operational record."""

        path = Path(image_path)
        started = time.perf_counter()
        width: int | None = None
        height: int | None = None
        ocr_cells: list[Any] | None = None
        boxes: list[list[float] | None] | None = None
        confidences: list[float | None] | None = None
        cells: list[dict[str, Any]] | None = None
        stage = "image"
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = int(image.width), int(image.height)
            if width <= 0 or height <= 0:
                raise RecognitionError("image dimensions are not positive")

            stage = "conversion"
            conversion = self.converter.convert(str(path), raises_on_error=False)
            stage = "ocr"
            ocr_cells, boxes, confidences = _ocr_cells_and_evidence(conversion, height)
            stage = "tableformer"
            table, boxes, confidences = _run_whole_image_table(
                self.converter,
                conversion,
                path,
                width,
                height,
                prepared_ocr_cells=ocr_cells,
                ocr_boxes=boxes,
                ocr_confidences=confidences,
            )
            stage = "canonicalization"
            cells = _canonical_cells(table, height)
            stage = "features"
            features, vector = features_from_primary(
                ocr_boxes=boxes,
                ocr_confidences=confidences,
                primary_cells=cells,
                image_width=width,
                image_height=height,
            )
            return {
                "status": "success",
                "image_width": width,
                "image_height": height,
                "cells": cells,
                "ocr_box_count": features["ocr_box_count"],
                "ocr_confidence_values": list(confidences),
                "unmatched_ocr_ratio": features["unmatched_ocr_ratio"],
                "features": features,
                "feature_vector": vector,
                "feature_names": list(FEATURE_NAMES),
                "inference_elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            try:
                features = feature_record_from_primary(
                    ocr_boxes=boxes,
                    ocr_confidences=confidences,
                    primary_cells=cells,
                    image_width=width,
                    image_height=height,
                )
                vector = ordered_feature_vector(features)
            except Exception:
                features = {name: None for name in FEATURE_NAMES}
                if width is not None and height is not None and height > 0:
                    features["image_aspect_ratio"] = float(width) / float(height)
                vector = [features[name] for name in FEATURE_NAMES]
            error_message = " ".join(str(exc).split())[:500]
            return {
                "status": "failure",
                "image_width": width,
                "image_height": height,
                "cells": cells if cells is not None else [],
                "ocr_box_count": features["ocr_box_count"],
                "ocr_confidence_values": (
                    list(confidences) if confidences is not None else None
                ),
                "unmatched_ocr_ratio": features["unmatched_ocr_ratio"],
                "features": features,
                "feature_vector": vector,
                "feature_names": list(FEATURE_NAMES),
                "failure_stage": stage,
                "error_type": type(exc).__name__,
                "error_message": error_message or type(exc).__name__,
                "inference_elapsed_seconds": time.perf_counter() - started,
            }


def _expert_required_files(expert_id: str) -> tuple[str, ...]:
    return (
        "config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    )


def _model_path_label(path: Path, expert_id: str) -> str:
    """Return a safe repository-relative model label without exposing local paths."""

    try:
        relative = path.resolve().relative_to(Path.cwd().resolve())
        return relative.as_posix()
    except (OSError, RuntimeError, ValueError):
        return f"models/{expert_id}"


def _resolve_expert_snapshot(
    expert_id: str,
    protocol: Mapping[str, Any],
    models_dir: str | Path,
) -> tuple[Path, str]:
    config = _expert_config(protocol, expert_id)
    checkpoint = config["checkpoint"]
    model_id = str(checkpoint["model_id"])
    revision = str(checkpoint["revision"])
    root = Path(models_dir).expanduser() / expert_id
    if root.exists() and not root.is_dir():
        raise RecognitionError(f"expert model directory is not a directory: {expert_id}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(model_id, revision=revision)
    except Exception as exc:
        raise RecognitionError(f"could not resolve the checkpoint for {expert_id}") from exc
    resolved_sha = str(getattr(info, "sha", ""))
    if resolved_sha != revision:
        raise RecognitionError(f"checkpoint revision drifted for {expert_id}")
    if bool(getattr(info, "private", False)) or bool(getattr(info, "gated", False)):
        raise RecognitionError(f"checkpoint is not anonymously accessible for {expert_id}")
    if bool(getattr(info, "disabled", False)):
        raise RecognitionError(f"checkpoint is disabled for {expert_id}")
    available = {str(item.rfilename) for item in getattr(info, "siblings", ())}
    missing = [filename for filename in _expert_required_files(expert_id) if filename not in available]
    if missing:
        raise RecognitionError(f"checkpoint is missing required files for {expert_id}")
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(root),
            cache_dir=str(root.parent / ".hf-cache"),
            allow_patterns=sorted(filename for filename in available if "/" not in filename),
            max_workers=1,
        )
    except Exception as exc:
        raise RecognitionError(f"could not download the pinned checkpoint for {expert_id}") from exc
    missing_local = [filename for filename in _expert_required_files(expert_id) if not (root / filename).is_file()]
    if missing_local:
        raise RecognitionError(f"downloaded checkpoint is incomplete for {expert_id}")
    return root.resolve(), resolved_sha


def _model_parameter_device(model: Any) -> str:
    try:
        parameter = next(model.parameters())
        return str(parameter.device)
    except (AttributeError, StopIteration):
        return str(getattr(model, "device", "unknown"))


def _model_parameter_dtype(model: Any) -> str:
    try:
        parameter = next(model.parameters())
        return str(parameter.dtype)
    except (AttributeError, StopIteration):
        return "unknown"


def _normalize_cuda_device(value: Any) -> str:
    if isinstance(value, bool):
        raise RecognitionError("expert model placement is not CUDA-only")
    if isinstance(value, int):
        if value < 0:
            raise RecognitionError("expert model placement is not CUDA-only")
        return f"cuda:{value}"
    text = str(value).strip().casefold()
    if text == "cuda":
        return "cuda"
    if re.fullmatch(r"cuda:\d+", text):
        return text
    raise RecognitionError("expert model placement is not CUDA-only")


def _normalize_cuda_device_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RecognitionError("expert model device map is not CUDA-only")
    normalized: dict[str, str] = {}
    for key, placement in value.items():
        normalized[str(key)] = _normalize_cuda_device(placement)
    return normalized


def _validate_cuda_only_model(model: Any, expert_id: str) -> tuple[str, str, dict[str, str]]:
    parameter_device = _model_parameter_device(model)
    if parameter_device == "unknown":
        raise RecognitionError("expert model does not expose a parameter device")
    _normalize_cuda_device(parameter_device)
    normalized_device_map = _normalize_cuda_device_map(getattr(model, "hf_device_map", None))
    parameter_dtype = _model_parameter_dtype(model)
    if expert_id == "hunyuanocr_1_5" and parameter_dtype != "torch.bfloat16":
        raise RecognitionError("Hunyuan must use bfloat16 parameters on CUDA")
    return parameter_device, parameter_dtype, normalized_device_map


def validate_expert_runtime_summary(runtime: Mapping[str, Any], expert_id: str) -> None:
    """Validate the CUDA-only placement recorded for a terminal expert output."""

    required_fields = {
        "required_device",
        "device_request",
        "cuda_available",
        "cuda_device_count",
        "cuda_only_placement",
        "actual_parameter_device",
        "model_device",
        "device_map",
        "model_dtype",
    }
    if not isinstance(runtime, Mapping) or not required_fields.issubset(runtime):
        raise RecognitionError("expert output lacks the tested CUDA-only runtime summary")
    if runtime["required_device"] != "cuda" or runtime["device_request"] != "cuda":
        raise RecognitionError("expert output was not produced with CUDA requested")
    if runtime["cuda_available"] is not True or runtime["cuda_only_placement"] is not True:
        raise RecognitionError("expert output does not record CUDA-only placement")
    device_count = runtime["cuda_device_count"]
    if isinstance(device_count, bool) or not isinstance(device_count, int) or device_count < 1:
        raise RecognitionError("expert output does not record an available CUDA device")
    actual_parameter_device = _normalize_cuda_device(runtime["actual_parameter_device"])
    if runtime["model_device"] != runtime["actual_parameter_device"]:
        raise RecognitionError("expert output has inconsistent CUDA parameter devices")
    _normalize_cuda_device_map(runtime["device_map"])
    if expert_id == "hunyuanocr_1_5" and runtime["model_dtype"] != "torch.bfloat16":
        raise RecognitionError("Hunyuan expert output is not recorded as bfloat16")
    if not actual_parameter_device.startswith("cuda"):
        raise RecognitionError("expert output parameter device is not CUDA")


class ExpertRecognizer:
    """One persistent local Transformers expert selected by the specified protocol."""

    def __init__(
        self,
        expert_id: str,
        protocol: Mapping[str, Any],
        models_dir: str | Path = "models",
        device: str = "cuda",
    ) -> None:
        validate_expert_protocol(protocol, expert_id)
        self.expert_id = expert_id
        self.config = _expert_config(protocol, expert_id)
        self.device = device
        self.environment = _validate_expert_environment(expert_id, device)
        started = time.perf_counter()
        self.checkpoint_path, self.resolved_hub_sha = _resolve_expert_snapshot(
            expert_id,
            protocol,
            models_dir,
        )
        try:
            import torch

            loading = self.config["loading"]
            device_map = str(loading["device_map"])
            if expert_id == "hunyuanocr_1_5":
                from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

                torch_dtype_name = str(loading["torch_dtype_cuda"])
                if torch_dtype_name != "bfloat16":
                    raise RecognitionError("Hunyuan must use bfloat16 on CUDA")
                torch_dtype = getattr(torch, torch_dtype_name, None)
                if torch_dtype is None:
                    raise RecognitionError(f"unsupported Hunyuan CUDA dtype: {torch_dtype_name}")
                self.processor = AutoProcessor.from_pretrained(
                    str(self.checkpoint_path),
                    trust_remote_code=bool(loading.get("trust_remote_code", False)),
                    use_fast=bool(loading.get("use_fast", True)),
                    local_files_only=True,
                )
                self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                    str(self.checkpoint_path),
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=bool(loading.get("trust_remote_code", False)),
                    local_files_only=True,
                ).eval()
            else:
                from transformers import AutoModelForImageTextToText, AutoProcessor

                self.processor = AutoProcessor.from_pretrained(
                    str(self.checkpoint_path),
                    local_files_only=True,
                )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    str(self.checkpoint_path),
                    torch_dtype=str(loading.get("torch_dtype", "auto")),
                    device_map=device_map,
                    local_files_only=True,
                ).eval()
        except Exception as exc:
            raise RecognitionError(f"could not load the pinned {expert_id} model") from exc
        self.initialization_elapsed_seconds = time.perf_counter() - started
        actual_model_name = type(self.model).__name__
        expected_model_name = self.config["model_class"]
        expected_model_types = {
            "hunyuanocr_1_5": "hunyuan_vl",
            "glm_ocr": "glm_ocr",
        }
        actual_model_type = str(
            getattr(getattr(self.model, "config", None), "model_type", "unknown")
        )
        if actual_model_type != expected_model_types[expert_id]:
            raise RecognitionError(f"loaded {expert_id} model configuration does not match the protocol")
        if (
            (expert_id == "hunyuanocr_1_5" and actual_model_name != expected_model_name)
            or (expert_id == "glm_ocr" and actual_model_name != "GlmOcrForConditionalGeneration")
        ):
            raise RecognitionError(f"loaded {expert_id} model class does not match the protocol")
        actual_parameter_device, actual_model_dtype, normalized_device_map = _validate_cuda_only_model(
            self.model, expert_id
        )
        self.runtime_summary = {
            "expert_id": expert_id,
            "model_directory": _model_path_label(self.checkpoint_path, expert_id),
            "checkpoint_model_id": self.config["checkpoint"]["model_id"],
            "checkpoint_revision": self.config["checkpoint"]["revision"],
            "resolved_hub_sha": self.resolved_hub_sha,
            "model_class_requested": expected_model_name,
            "model_class_actual": actual_model_name,
            "processor_class_requested": self.config["processor_class"],
            "processor_class_actual": type(self.processor).__name__,
            "model_type": actual_model_type,
            "model_name_or_path": _model_path_label(self.checkpoint_path, expert_id),
            "device_map": normalized_device_map,
            "required_device": self.config["loading"]["required_device"],
            "actual_parameter_device": actual_parameter_device,
            "model_device": actual_parameter_device,
            "model_dtype": actual_model_dtype,
            "eval_mode": not bool(getattr(self.model, "training", True)),
            "cuda_available": self.environment["cuda_available"],
            "cuda_device_count": self.environment["cuda_device_count"],
            "bfloat16_cuda_supported": self.environment["bfloat16_cuda_supported"],
            "cuda_only_placement": True,
            "torch": self.environment["torch"],
            "torchvision": self.environment["torchvision"],
            "transformers": self.environment["transformers"],
            "accelerate": self.environment["accelerate"],
            "Pillow": self.environment["Pillow"],
            "device_request": device,
            "device_map_policy": self.config["loading"]["device_map"],
            "quantized": self.config["loading"]["quantized"],
            "trust_remote_code": self.config["loading"].get("trust_remote_code", False),
            "use_fast": self.config["loading"].get("use_fast"),
            "initialization_elapsed_seconds": self.initialization_elapsed_seconds,
        }

    def _run_hunyuan(self, image_path: Path) -> str:
        import torch

        generation = self.config["generation"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": self.config["prompt"]},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=bool(generation["do_sample"]),
            )
        return self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=bool(generation["skip_special_tokens"]),
        )[0]

    def _run_glm(self, image_path: Path) -> str:
        import torch

        generation = self.config["generation"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path)},
                    {"type": "text", "text": self.config["prompt"]},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=bool(generation["do_sample"]),
            )
        return self.processor.decode(
            generated[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=bool(generation["skip_special_tokens"]),
        )

    def _memory_summary(self) -> dict[str, int | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "peak_gpu_memory_allocated_bytes": None,
                    "peak_gpu_memory_reserved_bytes": None,
                }
            return {
                "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        except Exception:
            return {
                "peak_gpu_memory_allocated_bytes": None,
                "peak_gpu_memory_reserved_bytes": None,
            }

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """Run one expert call and return a terminal operational record."""

        path = Path(image_path)
        inference_started = time.perf_counter()
        raw_output: str | None = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if self.expert_id == "hunyuanocr_1_5":
                raw_output = self._run_hunyuan(path)
            else:
                raw_output = self._run_glm(path)
        except Exception as exc:
            elapsed = time.perf_counter() - inference_started
            try:
                import torch

                out_of_memory = isinstance(exc, torch.cuda.OutOfMemoryError)
            except (ImportError, AttributeError):
                out_of_memory = False
            message = _safe_expert_error(exc)
            if out_of_memory or "out of memory" in message.casefold():
                status = "out_of_memory"
            elif isinstance(exc, TimeoutError):
                status = "inference_timeout"
            else:
                status = "inference_failed"
            return {
                "status": status,
                "raw_output": raw_output or "",
                "raw_output_byte_count": len((raw_output or "").encode("utf-8")),
                "native_output_format": _expert_native_format(raw_output),
                "inference_elapsed_seconds": elapsed,
                "parsing_elapsed_seconds": 0.0,
                "cells": [],
                "failure_stage": "generation",
                "error_type": type(exc).__name__,
                "error_message": message,
                **self._memory_summary(),
            }

        inference_elapsed = time.perf_counter() - inference_started
        parse_started = time.perf_counter()
        try:
            cells, native_format = canonicalize_expert_output(raw_output or "")
        except Exception as exc:
            parse_elapsed = time.perf_counter() - parse_started
            return {
                "status": "parse_failed",
                "raw_output": raw_output or "",
                "raw_output_byte_count": len((raw_output or "").encode("utf-8")),
                "native_output_format": _expert_native_format(raw_output),
                "inference_elapsed_seconds": inference_elapsed,
                "parsing_elapsed_seconds": parse_elapsed,
                "cells": [],
                "failure_stage": "canonicalization",
                "error_type": type(exc).__name__,
                "error_message": _safe_expert_error(exc),
                **self._memory_summary(),
            }
        parse_elapsed = time.perf_counter() - parse_started
        return {
            "status": "success",
            "raw_output": raw_output,
            "raw_output_byte_count": len(raw_output.encode("utf-8")),
            "native_output_format": native_format,
            "inference_elapsed_seconds": inference_elapsed,
            "parsing_elapsed_seconds": parse_elapsed,
            "cells": cells,
            "failure_stage": None,
            "error_type": None,
            "error_message": None,
            **self._memory_summary(),
        }
