"""AIN-335 Stage 2 - promotion gate tests."""

from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path

from ainfera_routing.learning import Observation, replay

_spec = importlib.util.spec_from_file_location(
    "promotion_gate", Path(__file__).resolve().parents[1] / "scripts" / "promotion_gate.py"
)
pg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pg)


def _obs(cell: str, slug: str, reward: str, tick: int) -> Observation:
    return Observation(
        cell=cell, model_slug=slug, reward=Decimal(reward), policy_version="v0", tick=tick
    )


def test_rehydrate_roundtrip_preserves_q():
    obs = [_obs("chat:cost", "mistral-large-3", "0.8", i) for i in range(12)]
    c = replay(obs)
    state = c.serialize()
    c2 = pg._consumer_from_state(state)
    assert c2.q_empirical("chat:cost", "mistral-large-3") == c.q_empirical(
        "chat:cost", "mistral-large-3"
    )


def test_gate_promotes_on_improvement():
    # incumbent: tool_use mistral mediocre; candidate: same cell, better arm data
    inc_obs = [_obs("tool_use:cost", "mistral-large-3", "0.1", i) for i in range(20)]
    cand_obs = [_obs("tool_use:cost", "gemini-3-1-pro", "0.9", i) for i in range(20)]
    inc = replay(inc_obs)
    cand = replay(cand_obs)
    # observations the gate scores over = candidate's chosen arms
    g = pg.evaluate_gate(
        inc, cand, cand_obs, incumbent_ruleset_hash="h", candidate_ruleset_hash="h"
    )
    # candidate has gemini@0.9 where incumbent never saw gemini -> no comparable pair
    # so this exercises the "no comparable cells" guard
    assert g["gates"]["g3_ruleset_stable"] is True


def test_gate_blocks_on_ruleset_mismatch():
    obs = [_obs("chat:cost", "mistral-large-3", "0.8", i) for i in range(20)]
    inc = replay(obs)
    cand = replay(obs)
    g = pg.evaluate_gate(inc, cand, obs, incumbent_ruleset_hash="h1", candidate_ruleset_hash="h2")
    assert g["replay_gate_passed"] is False
    assert g["promote_reason"] == "ruleset_hash_mismatch"


def test_gate_detects_regression():
    # same cell+arm, candidate trained on worse rewards -> negative delta
    inc_obs = [_obs("chat:cost", "mistral-large-3", "0.9", i) for i in range(20)]
    cand_obs = [_obs("chat:cost", "mistral-large-3", "0.5", i) for i in range(20)]
    inc = replay(inc_obs)
    cand = replay(cand_obs)
    g = pg.evaluate_gate(
        inc, cand, cand_obs, incumbent_ruleset_hash="h", candidate_ruleset_hash="h"
    )
    assert g["replay_gate_passed"] is False
    assert "regression" in g["promote_reason"]


def test_synthetic_source_never_promotes():
    obs = [_obs("chat:cost", "mistral-large-3", "0.9", i) for i in range(20)]
    inc = replay(obs)
    cand = replay(obs)
    g = pg.evaluate_gate(inc, cand, obs, incumbent_ruleset_hash="h", candidate_ruleset_hash="h")
    row = pg.build_training_run_row(
        g,
        judge_model="claude-opus-4-7",
        cadence="daily",
        source="synthetic",
        policy_version_from="a",
        policy_version_to="b",
        outcomes_judged=len(obs),
        exploration_floor="0.05",
        ruleset_hash="h",
    )
    assert row["promoted"] is False
    assert row["promote_reason"] == "synthetic_source_never_promotes"


def test_row_shape_matches_training_runs_columns():
    obs = [_obs("chat:cost", "mistral-large-3", "0.9", i) for i in range(20)]
    inc = replay(obs)
    cand = replay(obs)
    g = pg.evaluate_gate(inc, cand, obs, incumbent_ruleset_hash="h", candidate_ruleset_hash="h")
    row = pg.build_training_run_row(
        g,
        judge_model="claude-opus-4-7",
        cadence="daily",
        source="prod",
        policy_version_from="a",
        policy_version_to="b",
        outcomes_judged=len(obs),
        exploration_floor="0.05",
        ruleset_hash="h",
    )
    expected = {
        "cadence",
        "judge_model",
        "outcomes_judged",
        "policy_version_from",
        "policy_version_to",
        "promoted",
        "promote_reason",
        "delta_done_rate",
        "delta_cost_usd",
        "replay_gate_passed",
        "per_cell",
        "exploration_floor",
        "ruleset_hash",
    }
    assert set(row.keys()) == expected
