"""AIN-335 - tests for the routing_outcomes -> observations projector.

The load-bearing claim: the stored section-16 coverage cell
(task:model:band) is projected to a model-free bandit cell (task:band) so
the LinUCB learner can compare models within a cell. Plus the INVARIANT-1
synthetic/prod wall and deterministic ticks.
"""

from __future__ import annotations

import pytest

from scripts.export_outcomes import bandit_cell, project_rows


def _row(**kw):
    base = {
        "task_type": "chat",
        "cell": "chat:mistral-large-3:cost",
        "chosen_model_slug": "mistral-large-3",
        "reward": 0.88,
        "policy_version": "cost_first@1.0.0+abcd1234",
        "created_at": "2026-06-01T00:00:00+00:00",
        "judge_status": "labeled",
        "source": "prod",
    }
    base.update(kw)
    return base


# ---- bandit_cell projection (the core fix) --------------------------------

def test_bandit_cell_drops_model_segment():
    assert bandit_cell(stored_cell="chat:mistral-large-3:cost", task_type="chat",
                       policy_version="cost_first@1.0.0+x") == "chat:cost"
    assert bandit_cell(stored_cell="reasoning:gemini-3-1-pro:quality", task_type="reasoning",
                       policy_version="quality_first@1.0.0+x") == "reasoning:quality"


def test_bandit_cell_fallback_when_cell_absent_or_malformed():
    # No stored cell -> derive band from policy preset.
    assert bandit_cell(stored_cell=None, task_type="code",
                       policy_version="cost_first@1.0.0+x") == "code:cost"
    # Malformed (not 3-part) -> fallback path.
    assert bandit_cell(stored_cell="weird", task_type="tool_use",
                       policy_version="balanced@1.0.0+x") == "tool_use:balanced"
    # Unknown preset -> balanced.
    assert bandit_cell(stored_cell=None, task_type="general",
                       policy_version="mystery@9+z") == "general:balanced"


def test_two_models_same_task_band_share_one_bandit_cell():
    # The whole point: different chosen models, same (task, band) -> one cell
    # with two arms to compare.
    rows = [
        _row(cell="chat:mistral-large-3:cost", chosen_model_slug="mistral-large-3", reward=0.6),
        _row(cell="chat:gemini-3-1-pro:cost", chosen_model_slug="gemini-3-1-pro", reward=0.9),
    ]
    obs = project_rows(rows)
    cells = {o["cell"] for o in obs}
    assert cells == {"chat:cost"}
    arms = {o["model_slug"] for o in obs}
    assert arms == {"mistral-large-3", "gemini-3-1-pro"}


# ---- filtering ------------------------------------------------------------

def test_filters_non_prod_source():
    rows = [_row(source="prod"), _row(source="synthetic")]
    # synthetic+prod mixed -> INVARIANT 1 wall (see dedicated test); use
    # allow_mixed=False with only a non-prod, non-synthetic source here:
    rows = [_row(source="prod"), _row(source="staging", chosen_model_slug="gpt-5-5")]
    obs = project_rows(rows, source="prod")
    assert len(obs) == 1
    assert obs[0]["model_slug"] == "mistral-large-3"


def test_drops_unlabeled_and_null_reward():
    rows = [
        _row(),                                   # kept
        _row(reward=None),                        # dropped: no reward
        _row(judge_status="unlabeled"),           # dropped: not labeled
        _row(chosen_model_slug=None),             # dropped: no arm
    ]
    obs = project_rows(rows)
    assert len(obs) == 1


# ---- INVARIANT 1 ----------------------------------------------------------

def test_invariant1_blocks_synthetic_in_prod_export():
    rows = [_row(source="prod"), _row(source="synthetic")]
    with pytest.raises(SystemExit):
        project_rows(rows, source="prod")


def test_invariant1_bypass_with_allow_mixed():
    rows = [_row(source="prod"), _row(source="synthetic")]
    obs = project_rows(rows, source="prod", allow_mixed=True)
    assert len(obs) == 2  # both kept under offline analysis


# ---- shape + determinism --------------------------------------------------

def test_observation_shape_matches_refit_loader():
    obs = project_rows([_row()])
    o = obs[0]
    assert set(o) == {"cell", "model_slug", "reward", "policy_version", "tick"}
    assert isinstance(o["reward"], float)
    assert o["tick"] == 0


def test_ticks_are_deterministic_by_created_at():
    rows = [
        _row(created_at="2026-06-01T03:00:00+00:00", chosen_model_slug="b"),
        _row(created_at="2026-06-01T01:00:00+00:00", chosen_model_slug="a"),
        _row(created_at="2026-06-01T02:00:00+00:00", chosen_model_slug="c"),
    ]
    obs = project_rows(rows)
    assert [o["model_slug"] for o in obs] == ["a", "c", "b"]
    assert [o["tick"] for o in obs] == [0, 1, 2]
