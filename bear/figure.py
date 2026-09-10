"""Dependency-free SVG rendering for the paper's Figure 2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping
from xml.sax.saxutils import escape


class FigureError(RuntimeError):
    """Raised when a reproduced result lacks the Figure 2 inputs."""


EXPECTED_BUDGETS = (0, 30, 60, 90, 120, 180, 300)


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigureError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FigureError(f"{context} must be finite")
    return result


def _read_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FigureError(f"could not read reproduced result: {path}") from exc
    if not isinstance(value, dict):
        raise FigureError("reproduced result must contain an object")
    return value


def _budget_rows(report: Mapping[str, Any]) -> dict[str, dict[int, Mapping[str, Any]]]:
    curves = report.get("hunyuan_routes", {}).get("quality_budget_curves")
    if not isinstance(curves, list):
        raise FigureError("final result lacks Hunyuan quality-budget rows")
    primary_summary = report.get("standalone_results", {}).get("crop_tableformer_rapidocr_v6", {})
    primary_coverage = primary_summary.get("coverage_aware", {}) if isinstance(primary_summary, Mapping) else {}
    primary_con = primary_coverage.get("grits_con") if isinstance(primary_coverage, Mapping) else None
    primary_top = primary_coverage.get("grits_top") if isinstance(primary_coverage, Mapping) else None
    result: dict[str, dict[int, Mapping[str, Any]]] = {}
    method_names = {
        "random_matched_budget": "Random",
        "ridge_benefit": "BEAR (Ridge)",
        "hist_gradient_boosting_benefit": "BEAR (HGB)",
        "oracle_benefit": "Oracle (exact-B)",
    }
    for method, label in method_names.items():
        rows: dict[int, Mapping[str, Any]] = {}
        for row in curves:
            if not isinstance(row, Mapping) or row.get("routing_method") != method:
                continue
            budget_value = row.get("requested_budget", row.get("budget"))
            if budget_value is None:
                budget_value = row.get("requested_expert_call_count")
            if budget_value is None:
                raise FigureError(f"final result has an unlabelled Figure 2 row for {label}")
            normalized = dict(row)
            if "gain_over_always_primary_grits_con" not in normalized:
                hybrid = normalized.get("hybrid_full_set_grits_con")
                primary = normalized.get("primary_full_set_grits_con")
                if isinstance(hybrid, (int, float)) and isinstance(primary, (int, float)):
                    normalized["gain_over_always_primary_grits_con"] = float(hybrid) - float(primary)
                elif isinstance(hybrid, (int, float)) and isinstance(primary_con, (int, float)):
                    normalized["gain_over_always_primary_grits_con"] = float(hybrid) - float(primary_con)
            if "gain_over_always_primary_grits_top" not in normalized:
                hybrid_top = normalized.get("hybrid_full_set_grits_top")
                primary_top_row = normalized.get("primary_full_set_grits_top")
                if isinstance(hybrid_top, (int, float)) and isinstance(primary_top_row, (int, float)):
                    normalized["gain_over_always_primary_grits_top"] = float(hybrid_top) - float(primary_top_row)
                elif isinstance(hybrid_top, (int, float)) and isinstance(primary_top, (int, float)):
                    normalized["gain_over_always_primary_grits_top"] = float(hybrid_top) - float(primary_top)
            rows[int(budget_value)] = normalized
        if not rows:
            raise FigureError(f"final result lacks Figure 2 rows for {label}")
        result[label] = rows
    return result


def _svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    anchor: str = "start",
    weight: str = "normal",
    family: str = "Arial, sans-serif",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" '
        f'font-size="{size}px" text-anchor="{anchor}" font-weight="{weight}">'
        f"{escape(value)}</text>"
    )


def _marker(kind: str, x: float, y: float, *, color: str, open_marker: bool = False) -> str:
    fill = "white" if open_marker else color
    if kind == "square":
        return (
            f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" '
            f'fill="{fill}" stroke="{color}" stroke-width="1"/>'
        )
    if kind == "triangle":
        points = f"{x:.2f},{y - 5:.2f} {x - 5:.2f},{y + 4:.2f} {x + 5:.2f},{y + 4:.2f}"
        return f'<polygon points="{points}" fill="{fill}" stroke="{color}" stroke-width="1"/>'
    if kind == "diamond":
        points = f"{x:.2f},{y - 5:.2f} {x + 5:.2f},{y:.2f} {x:.2f},{y + 5:.2f} {x - 5:.2f},{y:.2f}"
        return f'<polygon points="{points}" fill="{fill}" stroke="{color}" stroke-width="1"/>'
    if kind == "circle":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{fill}" stroke="{color}" stroke-width="1"/>'
    raise FigureError(f"unknown Figure 2 marker: {kind}")


def render_figure2(result_path: str | Path, output_path: str | Path) -> Path:
    """Render the manuscript Figure 2 from one completed VietFinTab result."""

    report = _read_result(Path(result_path))
    figure = report.get("figure2")
    if not isinstance(figure, Mapping):
        raise FigureError("final result lacks the Figure 2 artifact section")
    budgets_value = figure.get("budgets")
    if not isinstance(budgets_value, list):
        raise FigureError("Figure 2 budgets are missing")
    budgets = tuple(int(value) for value in budgets_value)
    if budgets != EXPECTED_BUDGETS:
        raise FigureError(f"Figure 2 budgets changed: expected {EXPECTED_BUDGETS}")
    curves = _budget_rows(report)
    oracle_180 = _number(
        figure.get("oracle_gain_at_180_calls"),
        context="Figure 2 Oracle gain at 180 calls",
    )
    oracle_curve = curves["Oracle (exact-B)"].get(180)
    if oracle_curve is None or not math.isclose(
        _number(
            oracle_curve.get("gain_over_always_primary_grits_con"),
            context="Figure 2 Oracle 180-call curve",
        ),
        oracle_180,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FigureError("Figure 2 Oracle curve does not match its 180-call value")

    intervals_value = figure.get("paired_intervals")
    if not isinstance(intervals_value, Mapping):
        raise FigureError("Figure 2 paired intervals are missing")
    interval_order = (
        ("random", "Random", False),
        ("ocr_confidence", "OCR confidence", False),
        ("combined_risk", "Combined risk", False),
        ("ridge", "BEAR (Ridge)", False),
        ("difficulty", "Difficulty (HGB)†", True),
    )
    intervals: list[tuple[str, str, float, float, float, bool]] = []
    for key, label, open_marker in interval_order:
        value = intervals_value.get(key)
        if not isinstance(value, Mapping):
            raise FigureError(f"Figure 2 interval is missing: {key}")
        observed = _number(value.get("observed"), context=f"Figure 2 {key} observed")
        lower = _number(value.get("lower"), context=f"Figure 2 {key} lower")
        upper = _number(value.get("upper"), context=f"Figure 2 {key} upper")
        if lower > observed or observed > upper:
            raise FigureError(f"Figure 2 interval is not ordered: {key}")
        intervals.append((key, label, observed, lower, upper, open_marker))

    width, height = 1080, 610
    left_x, right_x = 80, 625
    top_y, bottom_y = 92, 465
    plot_width, plot_height = 430, 315
    panel_a_x_min, panel_a_x_max = 0.0, 320.0
    panel_a_y_min, panel_a_y_max = -0.005, 0.082
    panel_b_x_min, panel_b_x_max = -0.012, 0.082

    def sx(value: float) -> float:
        return left_x + plot_width * (value - panel_a_x_min) / (panel_a_x_max - panel_a_x_min)

    def sy(value: float) -> float:
        return bottom_y - plot_height * (value - panel_a_y_min) / (panel_a_y_max - panel_a_y_min)

    def ix(value: float) -> float:
        return right_x + plot_width * (value - panel_b_x_min) / (panel_b_x_max - panel_b_x_min)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(124, 30, "Random", size=12),
        _svg_text(308, 30, "BEAR (Ridge)", size=12),
        _svg_text(124, 52, "BEAR (HGB)", size=12),
        _svg_text(308, 52, "Oracle (exact-B)", size=12),
        '<line x1="96" y1="26" x2="116" y2="26" stroke="#777" stroke-width="1.8" stroke-dasharray="1 5"/>',
        '<line x1="280" y1="26" x2="300" y2="26" stroke="#444" stroke-width="1.8" stroke-dasharray="5 3"/>',
        '<line x1="96" y1="48" x2="116" y2="48" stroke="#000" stroke-width="2.3"/>',
        '<line x1="280" y1="48" x2="300" y2="48" stroke="#999" stroke-width="1.5" stroke-dasharray="7 3 1 3"/>',
        _marker("square", 106, 26, color="#777"),
        _marker("triangle", 290, 26, color="#444"),
        _marker("diamond", 106, 48, color="#000"),
        _marker("circle", 290, 48, color="#999", open_marker=True),
    ]

    # Panel (a): fixed manuscript axes, while the curves use all seven rows.
    svg.extend([
        f'<rect x="{left_x}" y="{top_y}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333"/>',
        f'<line x1="{left_x}" y1="{sy(0):.2f}" x2="{left_x + plot_width}" y2="{sy(0):.2f}" stroke="#777" stroke-dasharray="5 4"/>',
    ])
    for tick in (0, 60, 120, 180, 300):
        x = sx(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{top_y}" x2="{x:.2f}" y2="{bottom_y}" stroke="#e4e4e4"/>')
        svg.append(_svg_text(x, bottom_y + 22, str(tick), size=11, anchor="middle"))
    for tick in (0.00, 0.02, 0.04, 0.06, 0.08):
        y = sy(tick)
        svg.append(f'<line x1="{left_x}" y1="{y:.2f}" x2="{left_x + plot_width}" y2="{y:.2f}" stroke="#e4e4e4"/>')
        svg.append(_svg_text(left_x - 10, y + 4, f"{tick:.2f}", size=11, anchor="end"))
    vertical_x = sx(60)
    svg.append(f'<line x1="{vertical_x:.2f}" y1="{top_y}" x2="{vertical_x:.2f}" y2="{bottom_y}" stroke="#555" stroke-dasharray="6 4"/>')
    svg.append(_svg_text(vertical_x + 8, top_y + 16, "60 calls (20%)", size=11))
    svg.append(_svg_text(left_x + plot_width / 2, bottom_y + 47, "Expert calls", size=12, anchor="middle"))
    svg.extend([
        f'<g transform="translate(28 {top_y + plot_height / 2:.2f}) rotate(-90)">',
        _svg_text(0, 0, "GriTS-Con gain over Primary", size=12, anchor="middle"),
        "</g>",
    ])
    styles = {
        "Random": ("#777", "1 5", "square", 1.8, False),
        "BEAR (Ridge)": ("#444", "5 3", "triangle", 1.8, False),
        "BEAR (HGB)": ("#000", "", "diamond", 2.4, False),
        "Oracle (exact-B)": ("#999", "7 3 1 3", "circle", 1.5, True),
    }
    for label, rows in curves.items():
        points: list[tuple[float, float]] = []
        for budget in budgets:
            row = rows.get(budget)
            if row is None:
                raise FigureError(f"Figure 2 {label} curve lacks budget {budget}")
            points.append((
                sx(budget),
                sy(_number(row.get("gain_over_always_primary_grits_con"), context=f"{label} gain")),
            ))
        color, dash, marker_kind, stroke_width, open_marker = styles[label]
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash_attribute}/>'
        )
        for x, y in points:
            svg.append(_marker(marker_kind, x, y, color=color, open_marker=open_marker))

    # Panel (b): manuscript comparator order and fixed difference axis.
    svg.extend([
        f'<rect x="{right_x}" y="{top_y}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333"/>',
        f'<line x1="{ix(0):.2f}" y1="{top_y}" x2="{ix(0):.2f}" y2="{bottom_y}" stroke="#555" stroke-dasharray="6 4"/>',
    ])
    for tick in (0.00, 0.03, 0.06, 0.08):
        x = ix(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{top_y}" x2="{x:.2f}" y2="{bottom_y}" stroke="#e4e4e4"/>')
        svg.append(_svg_text(x, bottom_y + 22, f"{tick:.2f}", size=11, anchor="middle"))
    row_height = plot_height / len(intervals)
    # Keep the data order aligned with the result schema, but place the rows
    # top-to-bottom as they appear in the manuscript.
    for index, (_key, label, observed, lower, upper, open_marker) in enumerate(reversed(intervals)):
        y = top_y + row_height * (index + 0.5)
        svg.append(_svg_text(right_x - 12, y + 4, label, size=11, anchor="end"))
        low_x, high_x, point_x = ix(lower), ix(upper), ix(observed)
        svg.append(f'<line x1="{low_x:.2f}" y1="{y:.2f}" x2="{high_x:.2f}" y2="{y:.2f}" stroke="#444" stroke-width="1.6"/>')
        svg.append(f'<line x1="{low_x:.2f}" y1="{y - 6:.2f}" x2="{low_x:.2f}" y2="{y + 6:.2f}" stroke="#444"/>')
        svg.append(f'<line x1="{high_x:.2f}" y1="{y - 6:.2f}" x2="{high_x:.2f}" y2="{y + 6:.2f}" stroke="#444"/>')
        svg.append(_marker("circle", point_x, y, color="#444", open_marker=open_marker))
    svg.extend([
        _svg_text(left_x + plot_width / 2, bottom_y + 47, "Expert calls", size=12, anchor="middle"),
        _svg_text(right_x + plot_width / 2, bottom_y + 47, "GriTS-Con difference (BEAR minus comparator)", size=11, anchor="middle"),
        _svg_text(left_x + plot_width / 2, bottom_y + 76, "(a) Quality versus expert calls", size=14, weight="bold", anchor="middle"),
        _svg_text(right_x + plot_width / 2, bottom_y + 76, "(b) Paired differences at 60 calls", size=14, weight="bold", anchor="middle"),
        "</svg>",
    ])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return destination
