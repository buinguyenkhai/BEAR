#!/usr/bin/env python3
"""Recompute the bounded local-repair oracle from per-action evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "analysis" / "local_repair_actions.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "analysis" / "local_repair_oracle.json"

DATASET_ID = "VietFinTabGroup/VietFinTab"
DATASET_REVISION = "6e41a8941ef78fb53c750c622151ce1f2c24c688"
TABLE_COUNT = 540
BENEFIT_TIE_TOLERANCE = 1e-12
WHOLE_TABLE_COST = 1000
BUDGETS = (0, 27000, 54000, 108000, 162000, 270000, 540000)
ACTION_NAMES = (
    "retain_primary",
    "core_read_core_write",
    "halo_read_core_write",
    "halo_read_halo_write",
    "whole_table_hunyuan",
)
LOCAL_ACTION_NAMES = ACTION_NAMES[1:4]
ACTION_RANKS = {name: rank for rank, name in enumerate(ACTION_NAMES)}


class OracleError(RuntimeError):
    """Raised when the action evidence or solver invariant is invalid."""


def _fail(message: str) -> None:
    raise OracleError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing action evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleError(f"could not read action evidence: {path}") from exc
    _require(isinstance(value, dict), "action evidence root is not an object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} is not finite")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} is not an integer")
    return int(value)


def _fraction(action: Mapping[str, Any]) -> Fraction:
    numerator = _integer(action["input_area_fraction_numerator"], "input-area numerator")
    denominator = _integer(action["input_area_fraction_denominator"], "input-area denominator")
    _require(numerator >= 0, "input-area numerator is negative")
    _require(denominator > 0, "input-area denominator is not positive")
    result = Fraction(numerator, denominator)
    _require(Fraction(0, 1) <= result <= Fraction(1, 1), "input-area fraction is outside [0, 1]")
    return result


def _validate_artifact(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(data.get("study") == "vietfintab", "action evidence study differs")
    _require(data.get("analysis") == "local_repair_oracle", "action evidence analysis differs")
    dataset = data.get("dataset")
    _require(isinstance(dataset, Mapping), "dataset metadata is missing")
    _require(dataset.get("id") == DATASET_ID, "dataset identity differs")
    _require(dataset.get("revision") == DATASET_REVISION, "dataset revision differs")
    _require(data.get("quality_metric") == "GriTS-Con", "quality metric differs")

    population = data.get("population")
    _require(isinstance(population, Mapping), "population metadata is missing")
    _require(population.get("table_count") == TABLE_COUNT, "population table count differs")
    records_value = data.get("records")
    _require(isinstance(records_value, list), "action evidence records are missing")
    _require(len(records_value) == TABLE_COUNT, "action evidence table count differs")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, raw_record in enumerate(records_value):
        _require(isinstance(raw_record, Mapping), f"table record {position} is not an object")
        sample_id = raw_record.get("sample_id")
        issuer = raw_record.get("issuer")
        _require(isinstance(sample_id, str) and sample_id, f"table record {position} has no sample ID")
        _require(isinstance(issuer, str) and issuer, f"table record {sample_id} has no issuer")
        _require(sample_id not in seen, f"duplicate table ID: {sample_id}")
        seen.add(sample_id)
        primary_con = _finite(raw_record.get("primary_grits_con"), f"Primary Con {sample_id}")
        primary_top = _finite(raw_record.get("primary_grits_top"), f"Primary Top {sample_id}")
        _require(0.0 <= primary_con <= 1.0, f"Primary Con outside [0, 1]: {sample_id}")
        _require(0.0 <= primary_top <= 1.0, f"Primary Top outside [0, 1]: {sample_id}")

        actions_value = raw_record.get("actions")
        _require(isinstance(actions_value, list), f"actions are missing: {sample_id}")
        _require(len(actions_value) == len(ACTION_NAMES), f"action count differs: {sample_id}")
        actions: dict[str, dict[str, Any]] = {}
        for raw_action in actions_value:
            _require(isinstance(raw_action, Mapping), f"action is not an object: {sample_id}")
            name = raw_action.get("name")
            _require(name in ACTION_RANKS, f"unknown action: {sample_id}/{name}")
            _require(name not in actions, f"duplicate action: {sample_id}/{name}")
            rank = _integer(raw_action.get("rank"), f"action rank {sample_id}/{name}")
            _require(rank == ACTION_RANKS[name], f"action rank differs: {sample_id}/{name}")
            available = raw_action.get("available")
            _require(isinstance(available, bool), f"action availability is not boolean: {sample_id}/{name}")
            terminal_state = raw_action.get("terminal_state")
            _require(
                terminal_state in {"unavailable", "retained", "replacement", "fallback_primary"},
                f"invalid terminal state: {sample_id}/{name}",
            )
            if not available:
                _require(terminal_state == "unavailable", f"unavailable action state differs: {sample_id}/{name}")
                for field in (
                    "grits_con",
                    "grits_top",
                    "benefit_con",
                    "cost_milli_page",
                    "input_area_fraction_numerator",
                    "input_area_fraction_denominator",
                    "input_area_fraction_float",
                    "repair_footprint",
                ):
                    _require(raw_action.get(field) is None, f"unavailable action has data: {sample_id}/{name}/{field}")
                actions[str(name)] = dict(raw_action)
                continue

            grits_con = _finite(raw_action.get("grits_con"), f"action Con {sample_id}/{name}")
            grits_top = _finite(raw_action.get("grits_top"), f"action Top {sample_id}/{name}")
            benefit = _finite(raw_action.get("benefit_con"), f"action benefit {sample_id}/{name}")
            cost = _integer(raw_action.get("cost_milli_page"), f"action cost {sample_id}/{name}")
            footprint = _finite(raw_action.get("repair_footprint"), f"repair footprint {sample_id}/{name}")
            _require(0.0 <= grits_con <= 1.0, f"action Con outside [0, 1]: {sample_id}/{name}")
            _require(0.0 <= grits_top <= 1.0, f"action Top outside [0, 1]: {sample_id}/{name}")
            _require(cost >= 0, f"action cost is negative: {sample_id}/{name}")
            _require(footprint >= 0.0, f"repair footprint is negative: {sample_id}/{name}")
            area = _fraction(raw_action)
            _require(
                benefit == grits_con - primary_con,
                f"action benefit is inconsistent with realized Con: {sample_id}/{name}",
            )
            _require(raw_action.get("expert_return_status") is not None, f"action return status is missing: {sample_id}/{name}")
            if name == "retain_primary":
                _require(terminal_state == "retained", f"retain action state differs: {sample_id}")
                _require(cost == 0, f"retain action cost differs: {sample_id}")
                _require(area == Fraction(0, 1), f"retain action area differs: {sample_id}")
                _require(benefit == 0.0, f"retain action benefit differs: {sample_id}")
                _require(grits_con == primary_con and grits_top == primary_top, f"retain quality differs: {sample_id}")
            if name == "whole_table_hunyuan":
                _require(cost == WHOLE_TABLE_COST, f"whole-table action cost differs: {sample_id}")
                _require(area == Fraction(1, 1), f"whole-table action area differs: {sample_id}")
            action = dict(raw_action)
            action["cost_milli_page"] = cost
            action["repair_footprint"] = footprint
            actions[str(name)] = action

        _require(set(actions) == set(ACTION_NAMES), f"action names differ: {sample_id}")
        local_available = any(actions[name]["available"] for name in LOCAL_ACTION_NAMES)
        _require(raw_record.get("local_proposal_available") is local_available, f"local availability differs: {sample_id}")
        for name in LOCAL_ACTION_NAMES:
            _require(actions[name]["available"] is local_available, f"local action availability differs: {sample_id}/{name}")
        if local_available:
            area_two = _fraction(actions["halo_read_core_write"])
            area_three = _fraction(actions["halo_read_halo_write"])
            _require(
                actions["halo_read_core_write"]["cost_milli_page"] == actions["halo_read_halo_write"]["cost_milli_page"],
                f"halo action costs differ: {sample_id}",
            )
            _require(area_two == area_three, f"halo action areas differ: {sample_id}")
        records.append(
            {
                "sample_id": sample_id,
                "issuer": issuer,
                "primary_grits_con": primary_con,
                "primary_grits_top": primary_top,
                "local_proposal_available": local_available,
                "actions": actions,
            }
        )

    ordered = sorted(records, key=lambda item: item["sample_id"])
    _require([item["sample_id"] for item in records] == [item["sample_id"] for item in ordered], "table records are not in sample ID order")
    local_count = sum(bool(item["local_proposal_available"]) for item in records)
    proposal = data.get("local_proposal")
    if isinstance(proposal, Mapping):
        _require(proposal.get("available_count") == local_count, "local proposal available count is inconsistent")
        _require(proposal.get("unavailable_count") == TABLE_COUNT - local_count, "local proposal unavailable count is inconsistent")
    return records


@dataclass(frozen=True, slots=True)
class Candidate:
    cost: int
    benefit: float
    exact_area: Fraction
    footprint: float
    whole_table_count: int
    ranks: Any


class MergedRanks:
    """Lazy lexicographic view of ranks from two sorted table partitions."""

    def __init__(self, sources: tuple[tuple[int, int], ...], local_ranks: bytes, binary_ranks: bytes) -> None:
        self.sources = sources
        self.local_ranks = local_ranks
        self.binary_ranks = binary_ranks

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> int:
        partition, position = self.sources[index]
        return self.local_ranks[position] if partition == 0 else self.binary_ranks[position]

    def __iter__(self):
        for index in range(len(self.sources)):
            yield self[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MergedRanks):
            return (
                self.sources == other.sources
                and self.local_ranks == other.local_ranks
                and self.binary_ranks == other.binary_ranks
            )
        if isinstance(other, (bytes, bytearray)):
            return bytes(self) == bytes(other)
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, MergedRanks):
            for left, right in zip(self, other, strict=True):
                if left != right:
                    return left < right
            return False
        if isinstance(other, (bytes, bytearray)):
            return bytes(self) < bytes(other)
        return NotImplemented


def _compare_candidates(candidate: Candidate, incumbent: Candidate) -> int:
    if candidate.benefit > incumbent.benefit + BENEFIT_TIE_TOLERANCE:
        return 1
    if incumbent.benefit > candidate.benefit + BENEFIT_TIE_TOLERANCE:
        return -1
    if candidate.cost != incumbent.cost:
        return 1 if candidate.cost < incumbent.cost else -1
    if candidate.exact_area != incumbent.exact_area:
        return 1 if candidate.exact_area < incumbent.exact_area else -1
    if candidate.footprint != incumbent.footprint:
        return 1 if candidate.footprint < incumbent.footprint else -1
    if candidate.whole_table_count != incumbent.whole_table_count:
        return 1 if candidate.whole_table_count < incumbent.whole_table_count else -1
    if candidate.ranks != incumbent.ranks:
        return 1 if candidate.ranks < incumbent.ranks else -1
    return 0


def _candidate_for_action(current: Candidate, action: Mapping[str, Any]) -> Candidate:
    rank = _integer(action["rank"], "action rank")
    _require(0 <= rank <= 4, "action rank is outside the action menu")
    return Candidate(
        cost=current.cost + _integer(action["cost_milli_page"], "action cost"),
        benefit=current.benefit + _finite(action["benefit_con"], "action benefit"),
        exact_area=current.exact_area + _fraction(action),
        footprint=current.footprint + _finite(action["repair_footprint"], "repair footprint"),
        whole_table_count=current.whole_table_count + (1 if action["name"] == "whole_table_hunyuan" else 0),
        ranks=current.ranks + bytes((rank,)),
    )


def _eligible_action(action: Mapping[str, Any]) -> bool:
    if not action["available"]:
        return False
    if action["name"] == "retain_primary":
        return True
    return _finite(action["benefit_con"], "action benefit") > BENEFIT_TIE_TOLERANCE


def _action_names_from_candidate(
    sample_ids: Sequence[str],
    candidate: Candidate,
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[str]:
    _require(len(candidate.ranks) == len(sample_ids), "candidate rank tuple length differs")
    result: list[str] = []
    for sample_id, rank in zip(sample_ids, candidate.ranks, strict=True):
        matches = [
            name
            for name, action in actions_by_sample[sample_id].items()
            if int(action["rank"]) == int(rank) and action["available"]
        ]
        _require(len(matches) == 1, f"candidate rank is ambiguous: {sample_id}/{rank}")
        result.append(matches[0])
    return result


def _solve_exact_cost_states(
    sample_ids: Sequence[str],
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
    action_names_by_sample: Mapping[str, Sequence[str]],
) -> dict[int, Candidate]:
    zero = Candidate(0, 0.0, Fraction(0, 1), 0.0, 0, b"")
    states: dict[int, Candidate] = {0: zero}
    for sample_id in sample_ids:
        next_states: dict[int, Candidate] = {}
        for current in states.values():
            for action_name in action_names_by_sample[sample_id]:
                action = actions_by_sample[sample_id][action_name]
                if not _eligible_action(action):
                    continue
                candidate = _candidate_for_action(current, action)
                if candidate.cost > BUDGETS[-1]:
                    continue
                incumbent = next_states.get(candidate.cost)
                if incumbent is None or _compare_candidates(candidate, incumbent) > 0:
                    next_states[candidate.cost] = candidate
        _require(next_states, f"no feasible state after {sample_id}")
        states = next_states
    return states


def _solve_binary_exact_k(
    sample_ids: Sequence[str],
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[Candidate | None]:
    zero = Candidate(0, 0.0, Fraction(0, 1), 0.0, 0, b"")
    states: list[Candidate | None] = [zero]
    for sample_id in sample_ids:
        next_states: list[Candidate | None] = [None] * (len(states) + 1)
        a0 = actions_by_sample[sample_id]["retain_primary"]
        a4 = actions_by_sample[sample_id]["whole_table_hunyuan"]
        for count, current in enumerate(states):
            if current is None:
                continue
            candidate = _candidate_for_action(current, a0)
            incumbent = next_states[count]
            if incumbent is None or _compare_candidates(candidate, incumbent) > 0:
                next_states[count] = candidate
            if _eligible_action(a4):
                candidate = _candidate_for_action(current, a4)
                incumbent = next_states[count + 1]
                if incumbent is None or _compare_candidates(candidate, incumbent) > 0:
                    next_states[count + 1] = candidate
        states = next_states
    return states


def _solve_whole_table_frontier(
    sample_ids: Sequence[str],
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[int, Candidate]:
    action_names = {sample_id: ("retain_primary", "whole_table_hunyuan") for sample_id in sample_ids}
    states = _solve_exact_cost_states(sample_ids, actions_by_sample, action_names)
    result: dict[int, Candidate] = {}
    for budget in BUDGETS:
        best: Candidate | None = None
        for candidate in states.values():
            if candidate.cost <= budget and (best is None or _compare_candidates(candidate, best) > 0):
                best = candidate
        _require(best is not None, f"no whole-table frontier state at budget {budget}")
        result[budget] = best
    return result


def _combine_candidates(
    local: Candidate,
    binary: Candidate,
    sources: tuple[tuple[int, int], ...],
) -> Candidate:
    return Candidate(
        cost=local.cost + binary.cost,
        benefit=local.benefit + binary.benefit,
        exact_area=local.exact_area + binary.exact_area,
        footprint=local.footprint + binary.footprint,
        whole_table_count=local.whole_table_count + binary.whole_table_count,
        ranks=MergedRanks(sources, local.ranks, binary.ranks),
    )


def _normalize_candidate(
    candidate: Candidate,
    sample_ids: Sequence[str],
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Candidate:
    names = _action_names_from_candidate(sample_ids, candidate, actions_by_sample)
    cost = 0
    benefit = 0.0
    exact_area = Fraction(0, 1)
    footprint = 0.0
    whole_table_count = 0
    for sample_id, action_name in zip(sample_ids, names, strict=True):
        action = actions_by_sample[sample_id][action_name]
        cost += _integer(action["cost_milli_page"], "action cost")
        benefit += _finite(action["benefit_con"], "action benefit")
        exact_area += _fraction(action)
        footprint += _finite(action["repair_footprint"], "repair footprint")
        whole_table_count += action_name == "whole_table_hunyuan"
    return Candidate(cost, benefit, exact_area, footprint, whole_table_count, candidate.ranks)


def _solve_local_frontier(
    sample_ids: Sequence[str],
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[int, Candidate]:
    local_ids = [sample_id for sample_id in sample_ids if actions_by_sample[sample_id]["core_read_core_write"]["available"]]
    binary_ids = [sample_id for sample_id in sample_ids if not actions_by_sample[sample_id]["core_read_core_write"]["available"]]
    _require(local_ids and binary_ids, "local and binary partitions must both be nonempty")
    local_names = {sample_id: ACTION_NAMES for sample_id in local_ids}
    local_states = _solve_exact_cost_states(local_ids, actions_by_sample, local_names)
    binary_exact = _solve_binary_exact_k(binary_ids, actions_by_sample)

    binary_prefix: list[Candidate | None] = []
    best_binary: Candidate | None = None
    for candidate in binary_exact:
        if candidate is not None and (best_binary is None or _compare_candidates(candidate, best_binary) > 0):
            best_binary = candidate
        binary_prefix.append(best_binary)
    _require(all(candidate is not None for candidate in binary_prefix), "binary prefix frontier has a gap")

    local_positions = {sample_id: position for position, sample_id in enumerate(local_ids)}
    binary_positions = {sample_id: position for position, sample_id in enumerate(binary_ids)}
    sources = tuple(
        (0, local_positions[sample_id]) if sample_id in local_positions else (1, binary_positions[sample_id])
        for sample_id in sample_ids
    )
    binary_cost = _integer(actions_by_sample[binary_ids[0]]["whole_table_hunyuan"]["cost_milli_page"], "whole-table cost")
    _require(binary_cost > 0, "whole-table cost is not positive")
    for sample_id in binary_ids:
        _require(
            _integer(actions_by_sample[sample_id]["retain_primary"]["cost_milli_page"], "retain cost") == 0
            and _integer(actions_by_sample[sample_id]["whole_table_hunyuan"]["cost_milli_page"], "whole-table cost") == binary_cost,
            f"binary costs are not constant: {sample_id}",
        )

    result: dict[int, Candidate] = {}
    for budget in BUDGETS:
        best: Candidate | None = None
        for local in local_states.values():
            if local.cost > budget:
                continue
            max_binary_count = min((budget - local.cost) // binary_cost, len(binary_ids))
            binary = binary_prefix[max_binary_count]
            _require(binary is not None, "binary prefix candidate is missing")
            candidate = _combine_candidates(local, binary, sources)
            if candidate.cost <= budget and (best is None or _compare_candidates(candidate, best) > 0):
                best = candidate
        _require(best is not None, f"no local frontier state at budget {budget}")
        result[budget] = _normalize_candidate(best, sample_ids, actions_by_sample)
    return result


def _independent_unconstrained_optimum(
    sample_ids: Sequence[str],
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
    action_names_by_sample: Mapping[str, Sequence[str]],
) -> Candidate:
    result = Candidate(0, 0.0, Fraction(0, 1), 0.0, 0, b"")
    for sample_id in sample_ids:
        best: Candidate | None = None
        for action_name in action_names_by_sample[sample_id]:
            action = actions_by_sample[sample_id][action_name]
            if not _eligible_action(action):
                continue
            candidate = _candidate_for_action(Candidate(0, 0.0, Fraction(0, 1), 0.0, 0, b""), action)
            if best is None or _compare_candidates(candidate, best) > 0:
                best = candidate
        _require(best is not None, f"no independent action for {sample_id}")
        action_name = next(
            name
            for name, action in actions_by_sample[sample_id].items()
            if action["available"] and int(action["rank"]) == best.ranks[0]
        )
        result = _candidate_for_action(result, actions_by_sample[sample_id][action_name])
    return result


def _action_counts(names: Sequence[str]) -> dict[str, int]:
    counts = Counter(names)
    return {name: int(counts.get(name, 0)) for name in ACTION_NAMES}


def _frontier_record(
    candidate: Candidate,
    sample_ids: Sequence[str],
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
    budget: int,
) -> dict[str, Any]:
    names = _action_names_from_candidate(sample_ids, candidate, actions_by_sample)
    selected = [actions_by_sample[sample_id][name] for sample_id, name in zip(sample_ids, names, strict=True)]
    _require(len(selected) == TABLE_COUNT, f"selected table count differs at budget {budget}")
    con_values = [_finite(action["grits_con"], f"selected Con {budget}") for action in selected]
    top_values = [_finite(action["grits_top"], f"selected Top {budget}") for action in selected]
    benefit_values = [_finite(action["benefit_con"], f"selected benefit {budget}") for action in selected]
    exact_area = sum((_fraction(action) for action in selected), Fraction(0, 1))
    total_benefit = sum(benefit_values)
    _require(exact_area == candidate.exact_area, f"exact area reconstruction differs at budget {budget}")
    action_counts = _action_counts(names)
    local_count = sum(action_counts[name] for name in LOCAL_ACTION_NAMES)
    return {
        "budget_milli_page": budget,
        "average_input_area_allowance": budget / (TABLE_COUNT * WHOLE_TABLE_COST),
        "mean_grits_con": float(statistics.fmean(con_values)),
        "mean_grits_top": float(statistics.fmean(top_values)),
        "total_benefit_con": float(total_benefit),
        "used_cost_milli_page": int(candidate.cost),
        "used_exact_input_area_numerator": int(candidate.exact_area.numerator),
        "used_exact_input_area_denominator": int(candidate.exact_area.denominator),
        "used_exact_input_area_float": float(candidate.exact_area),
        "action_counts": action_counts,
        "local_action_count": local_count,
        "whole_table_action_count": action_counts["whole_table_hunyuan"],
        "selected_table_count": len(names),
        "one_action_per_table": len(names) == TABLE_COUNT,
        "within_budget": candidate.cost <= budget,
    }


def _full_budget_integrity(
    actual: Candidate,
    expected: Candidate,
    sample_ids: Sequence[str],
    actions_by_sample: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    actual_names = _action_names_from_candidate(sample_ids, actual, actions_by_sample)
    expected_names = _action_names_from_candidate(sample_ids, expected, actions_by_sample)
    actual_counts = _action_counts(actual_names)
    expected_counts = _action_counts(expected_names)
    checks = {
        "action_tuple_equal": actual_names == expected_names,
        "total_benefit_equal": actual.benefit == expected.benefit,
        "total_cost_equal": actual.cost == expected.cost,
        "exact_total_input_area_equal": actual.exact_area == expected.exact_area,
        "total_repair_footprint_equal": actual.footprint == expected.footprint,
        "whole_table_count_equal": actual.whole_table_count == expected.whole_table_count,
        "action_counts_equal": actual_counts == expected_counts,
    }
    _require(all(checks.values()), "full-budget solver integrity check failed")
    return {
        "enabled": True,
        "budget_milli_page": BUDGETS[-1],
        "expected_construction": "independent_per_table_unconstrained_optimum",
        **checks,
        "actual_action_counts": actual_counts,
        "expected_action_counts": expected_counts,
        "passed": True,
    }


def _run(artifact_path: Path) -> dict[str, Any]:
    records = _validate_artifact(_load_json(artifact_path))
    sample_ids = [record["sample_id"] for record in records]
    actions_by_sample = {record["sample_id"]: record["actions"] for record in records}
    action_names_by_sample = {
        sample_id: tuple(name for name in ACTION_NAMES if actions_by_sample[sample_id][name]["available"])
        for sample_id in sample_ids
    }
    whole_candidates = _solve_whole_table_frontier(sample_ids, actions_by_sample)
    local_candidates = _solve_local_frontier(sample_ids, actions_by_sample)
    whole_records = {
        str(budget): _frontier_record(whole_candidates[budget], sample_ids, actions_by_sample, budget)
        for budget in BUDGETS
    }
    local_records = {
        str(budget): _frontier_record(local_candidates[budget], sample_ids, actions_by_sample, budget)
        for budget in BUDGETS
    }
    full_expected = _independent_unconstrained_optimum(sample_ids, actions_by_sample, action_names_by_sample)
    full_integrity = _full_budget_integrity(local_candidates[BUDGETS[-1]], full_expected, sample_ids, actions_by_sample)

    local_available_count = sum(record["local_proposal_available"] for record in records)
    positive_local_records = [
        record
        for record in records
        if any(
            _eligible_action(record["actions"][name])
            for name in LOCAL_ACTION_NAMES
        )
    ]
    positive_issuers = sorted({record["issuer"] for record in positive_local_records})
    headroom = {
        str(budget): local_records[str(budget)]["mean_grits_con"] - whole_records[str(budget)]["mean_grits_con"]
        for budget in BUDGETS
    }
    return {
        "study": "vietfintab",
        "analysis": "local_repair_oracle",
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "population": {
            "description": "eligible development tables",
            "table_count": TABLE_COUNT,
            "local_proposal_available_count": local_available_count,
            "local_proposal_unavailable_count": TABLE_COUNT - local_available_count,
        },
        "quality_metric": "GriTS-Con",
        "cost": {
            "unit": "normalized expert-input-area proxy",
            "whole_table_milli_page": WHOLE_TABLE_COST,
            "budget_grid_milli_page": list(BUDGETS),
            "average_input_area_allowance": [budget / (TABLE_COUNT * WHOLE_TABLE_COST) for budget in BUDGETS],
        },
        "local_proposal": {
            "description": "bounded region proposed from Primary table structure",
            "available_count": local_available_count,
            "unavailable_count": TABLE_COUNT - local_available_count,
        },
        "whole_table_only_oracle": {
            "action_set": ["retain_primary", "whole_table_hunyuan"],
            "budgets": whole_records,
        },
        "local_enabled_oracle": {
            "action_set": [*ACTION_NAMES],
            "budgets": local_records,
        },
        "headroom": {
            "description": "local-enabled mean GriTS-Con minus whole-table-only mean GriTS-Con",
            "by_budget": headroom,
        },
        "local_opportunity": {
            "definition": "a table with at least one positive-benefit local action",
            "positive_table_count": len(positive_local_records),
            "positive_issuer_count": len(positive_issuers),
        },
        "solver": {
            "method": "exact deterministic multiple-choice knapsack dynamic program",
            "integer_budget_state": "cost_milli_page",
            "benefit_tie_tolerance": BENEFIT_TIE_TOLERANCE,
            "exact_area_tie_break": "exact cumulative Fraction from action numerators and denominators",
            "same_comparator_for_dynamic_program_and_final_selection": True,
            "non_retain_eligibility": "benefit_con > 1e-12",
            "full_budget_integrity": full_integrity,
        },
        "limitations": [
            "development-only evidence",
            "reference-assisted oracle analysis",
            "nondeployable action-space comparison",
            "the action records retain realized qualities; local expert generation is not regenerated",
            "the analysis starts from recorded per-action scores in analysis/local_repair_actions.json and does not rerun local expert inference",
            "normalized input-area proxy is not measured compute",
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the bounded local-repair oracle from per-action evidence."
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="path to the action-evidence JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="path for the computed result JSON")
    args = parser.parse_args(argv)
    try:
        result = _run(args.artifact)
        _write_json(args.output, result)
    except OracleError as exc:
        parser.exit(2, f"local-repair oracle error: {exc}\n")
    main_budget = result["local_enabled_oracle"]["budgets"][str(BUDGETS[3])]
    print(f"local proposal available = {result['population']['local_proposal_available_count']}")
    print(f"positive local opportunity tables = {result['local_opportunity']['positive_table_count']}")
    print(f"main allowance local-enabled GriTS-Con = {main_budget['mean_grits_con']:.16f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
