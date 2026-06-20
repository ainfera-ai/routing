"""AIN-335 - tests for the routing_outcomes -> observations projector.

The load-bearing claim: the stored section-16 coverage cell
(task:model:band) is projected to a model-free bandit cell (task:band) so
the LinUCB learner can compare models within a cell. Plus the INVARIANT-1
synthetic/prod wall and deterministic ticks.
"""

from __future__ import annotations

import pytest

from scripts.export_outcomes import (
    _FLEET_TENANT_IDS_DEFAULT,
    _GOLD_DRIVER_AGENT_IDS,
    bandit_cell,
    fleet_downweight,
    project_rows,
)


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
    assert (
        bandit_cell(
            stored_cell="chat:mistral-large-3:cost",
            task_type="chat",
            policy_version="cost_first@1.0.0+x",
        )
        == "chat:cost"
    )
    assert (
        bandit_cell(
            stored_cell="reasoning:gemini-3-1-pro:quality",
            task_type="reasoning",
            policy_version="quality_first@1.0.0+x",
        )
        == "reasoning:quality"
    )


def test_bandit_cell_fallback_when_cell_absent_or_malformed():
    # No stored cell -> derive band from policy preset.
    assert (
        bandit_cell(stored_cell=None, task_type="code", policy_version="cost_first@1.0.0+x")
        == "code:cost"
    )
    # Malformed (not 3-part) -> fallback path.
    assert (
        bandit_cell(stored_cell="weird", task_type="tool_use", policy_version="balanced@1.0.0+x")
        == "tool_use:balanced"
    )
    # Unknown preset -> balanced.
    assert (
        bandit_cell(stored_cell=None, task_type="general", policy_version="mystery@9+z")
        == "general:balanced"
    )


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
        _row(),  # kept
        _row(reward=None),  # dropped: no reward
        _row(judge_status="unlabeled"),  # dropped: not labeled
        _row(chosen_model_slug=None),  # dropped: no arm
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
    assert set(o) == {"cell", "model_slug", "reward", "policy_version", "tick", "weight"}
    assert isinstance(o["reward"], float)
    assert o["tick"] == 0
    # A plain external row (no fleet_agent) gets full weight.
    assert o["weight"] == 1.0


def test_ticks_are_deterministic_by_created_at():
    rows = [
        _row(created_at="2026-06-01T03:00:00+00:00", chosen_model_slug="b"),
        _row(created_at="2026-06-01T01:00:00+00:00", chosen_model_slug="a"),
        _row(created_at="2026-06-01T02:00:00+00:00", chosen_model_slug="c"),
    ]
    obs = project_rows(rows)
    assert [o["model_slug"] for o in obs] == ["a", "c", "b"]
    assert [o["tick"] for o in obs] == [0, 1, 2]


# ---- neutrality rider: down-weight internal-fleet, exclude only degraded ---
# (AIN-388 P0-tail)


def test_fleet_row_is_kept_but_downweighted():
    """An internal-fleet row (fleet_agent set) is KEPT — not dropped — and
    emitted at the down-weight, while an external row keeps full weight.
    """
    rows = [
        _row(chosen_model_slug="m1", fleet_agent="tulkas"),  # internal fleet
        _row(chosen_model_slug="m2", fleet_agent=None),  # external/customer
    ]
    obs = project_rows(rows)
    assert len(obs) == 2, "fleet rows are kept (down-weighted), never dropped"
    by_model = {o["model_slug"]: o["weight"] for o in obs}
    assert by_model["m1"] == fleet_downweight()
    assert by_model["m1"] < 1.0, "internal-fleet must be down-weighted"
    assert by_model["m2"] == 1.0, "external row keeps full weight"


def test_fleet_downweight_env_override(monkeypatch):
    monkeypatch.setenv("AINFERA_FLEET_DOWNWEIGHT", "0.1")
    obs = project_rows([_row(fleet_agent="aule")])
    assert obs[0]["weight"] == 0.1


def test_fleet_downweight_rejects_zero_and_garbage(monkeypatch):
    # 0 / negative / non-numeric must NOT silently erase the seed signal —
    # they fall back to the default (the degraded path is the way to exclude).
    for bad in ("0", "-1", "abc", ""):
        monkeypatch.setenv("AINFERA_FLEET_DOWNWEIGHT", bad)
        assert fleet_downweight() == 0.25
    monkeypatch.setenv("AINFERA_FLEET_DOWNWEIGHT", "2.0")
    assert fleet_downweight() == 1.0  # clamped to (0, 1]


def test_degraded_rows_are_excluded_not_weighted():
    """Degraded/MLX rows are dropped entirely (a degraded backend's reward
    is not a clean signal of the routed model). Several P2-forward shapes.
    """
    rows = [
        _row(chosen_model_slug="clean"),
        _row(chosen_model_slug="d1", degraded=True),
        _row(chosen_model_slug="d2", traffic_origin="degraded"),
        _row(chosen_model_slug="d3", source="prod", traffic_origin="mlx"),
    ]
    obs = project_rows(rows)
    assert {o["model_slug"] for o in obs} == {"clean"}


def test_fleet_and_degraded_combined():
    # A degraded fleet row is excluded (degraded wins over down-weight).
    rows = [
        _row(chosen_model_slug="keep", fleet_agent="namo"),  # fleet → kept, down-weighted
        _row(chosen_model_slug="drop", fleet_agent="namo", degraded=True),  # degraded → excluded
    ]
    obs = project_rows(rows)
    assert {o["model_slug"] for o in obs} == {"keep"}
    assert obs[0]["weight"] == fleet_downweight()


def test_synthetic_probes_excluded_not_downweighted():
    """AIN-424: synthetic health/routing probes are dropped outright, not kept
    at the 0.25 dogfood weight. Authoritative key = traffic_class; fallback =
    the fleet_agent probe label for dumps predating the column. A real fleet
    agent (no probe signal) is still kept + down-weighted."""
    rows = [
        _row(chosen_model_slug="clean"),  # external → full
        _row(chosen_model_slug="p1", traffic_class="internal_probe"),  # authoritative
        _row(chosen_model_slug="p2", fleet_agent="routed-probe"),  # fallback
        _row(chosen_model_slug="p3", fleet_agent="nt1-probe-1779468321"),  # fallback (ts)
        _row(chosen_model_slug="keep", fleet_agent="tulkas"),  # real fleet → dw
    ]
    obs = project_rows(rows)
    by_model = {o["model_slug"]: o["weight"] for o in obs}
    assert set(by_model) == {"clean", "keep"}, "all three probe shapes excluded"
    assert by_model["clean"] == 1.0
    assert by_model["keep"] == fleet_downweight()


def test_internal_probe_traffic_class_wins_over_fleet_tenant():
    # A probe row that ALSO sits on the fleet tenant is still excluded (probe
    # class is stronger than the dogfood down-weight).
    obs = project_rows(
        [
            _row(
                chosen_model_slug="probe",
                tenant_id=_FLEET_TENANT_IDS_DEFAULT,
                fleet_agent="routed-probe",
                traffic_class="internal_probe",
            ),
        ]
    )
    assert obs == []


# ---- AIN-391 §2a: neutrality keyed off tenant_id, NOT the fleet_agent tag ---
# The load-bearing gap fleet_agent-keying left: a fleet row whose per-agent
# tag was never written (NULL fleet_agent) leaked into the moat at FULL weight.
# Keying off tenant_id closes it. Keystone is identical to the api write path
# (services/routing_brain._FLEET_TENANT_IDS_DEFAULT).

_FLEET_TENANT = "280f4469-d318-4ec4-9c63-f3ea83466b03"
_CUSTOMER_TENANT = "11111111-2222-3333-4444-555555555555"


def test_fleet_keyed_off_tenant_id_proof_matrix():
    """A/B/C/D: A(fleet, tagged) and B(fleet, fleet_agent NULL) must BOTH be
    down-weighted to the SAME weight purely on tenant_id; C(customer) full;
    D(degraded) excluded."""
    obs = project_rows(
        [
            # A fleet, tagged
            _row(chosen_model_slug="A", tenant_id=_FLEET_TENANT, fleet_agent="namo"),
            # B fleet, NULL tag (the gap)
            _row(chosen_model_slug="B", tenant_id=_FLEET_TENANT, fleet_agent=None),
            # C customer
            _row(chosen_model_slug="C", tenant_id=_CUSTOMER_TENANT, fleet_agent=None),
            # D degraded
            _row(chosen_model_slug="D", tenant_id=_FLEET_TENANT, traffic_origin="mlx"),
        ]
    )
    w = {o["model_slug"]: o["weight"] for o in obs}
    assert "D" not in w, "degraded fleet row excluded outright"
    assert w["A"] == fleet_downweight()
    assert w["B"] == w["A"], "untagged fleet row (NULL fleet_agent) down-weighted via tenant_id"
    assert w["B"] < 1.0
    assert w["C"] == 1.0, "customer tenant keeps full weight"


def test_fleet_tenant_id_is_case_insensitive():
    obs = project_rows([_row(tenant_id=_FLEET_TENANT.upper(), fleet_agent=None)])
    assert obs[0]["weight"] == fleet_downweight()


def test_fleet_tenant_env_is_additive_never_replaces(monkeypatch):
    extra = "99999999-aaaa-bbbb-cccc-dddddddddddd"
    monkeypatch.setenv("AINFERA_FLEET_TENANT_IDS", extra)
    obs = project_rows(
        [
            _row(chosen_model_slug="extra", tenant_id=extra, fleet_agent=None),
            _row(chosen_model_slug="default", tenant_id=_FLEET_TENANT, fleet_agent=None),
        ]
    )
    w = {o["model_slug"]: o["weight"] for o in obs}
    assert w["extra"] == fleet_downweight(), "env-added tenant treated as fleet"
    assert w["default"] == fleet_downweight(), (
        "default fleet tenant stays fleet (additive, not replace)"
    )


def test_fleet_agent_fallback_for_legacy_dumps_without_tenant_id():
    """Back-compat: a dump row lacking tenant_id still down-weights via the
    fleet_agent fallback (transitional until every dump carries tenant_id)."""
    obs = project_rows([_row(fleet_agent="tulkas")])  # no tenant_id key at all
    assert obs[0]["weight"] == fleet_downweight()


def test_customer_row_without_tenant_or_agent_is_full_weight():
    obs = project_rows([_row(tenant_id=_CUSTOMER_TENANT, fleet_agent=None)])
    assert obs[0]["weight"] == 1.0


def test_routing_fleet_tenant_keystone_matches_api_constant():
    """Cross-repo lock: the projector's fleet tenant MUST equal the literal the
    api write path tags on (api services/routing_brain._FLEET_TENANT_IDS_DEFAULT
    + its own test). A drift in either repo would split fleet detection."""
    assert _FLEET_TENANT_IDS_DEFAULT == "280f4469-d318-4ec4-9c63-f3ea83466b03"


# ---- AIN-544 · verify-gold anchor exclusion (anchor-only, never selection) ----

_VERIFY_GOLD_ID = "ccbf9d9d-5ab6-4f48-b560-58c0b32949aa"


def test_gold_anchor_row_excluded_by_driver_id():
    """A row routed by the dedicated verify-gold driver is dropped outright — it feeds
    the κ anchor, but is synthetic measurement, never a selection signal."""
    obs = project_rows([_row(agent_id=_VERIFY_GOLD_ID)])
    assert obs == []
    # case-insensitive (UUIDs are canonical-lower, but be defensive)
    assert project_rows([_row(agent_id=_VERIFY_GOLD_ID.upper())]) == []


def test_gold_anchor_row_excluded_by_tag():
    """Defense in depth: a row tagged gold (dump LEFT JOIN labs_verify_gold) is dropped
    even if it carries no recognised driver agent_id."""
    assert project_rows([_row(is_gold=True)]) == []
    assert project_rows([_row(gold_id="g-001")]) == []


def test_gold_anchor_keystone_matches_api_constant():
    """Cross-repo lock: the projector's gold driver set MUST contain the api SSOT id
    (api services/training_scope.GOLD_DRIVER_AGENT_IDS). Drift would re-open the leak."""
    assert _VERIFY_GOLD_ID in _GOLD_DRIVER_AGENT_IDS


def test_normal_fleet_row_still_kept():
    """A non-gold fleet row is unaffected (kept, down-weighted as before)."""
    ulmo = "0b382a0b-968b-44f4-b5fe-2919ca51adbc"
    obs = project_rows([_row(tenant_id=_FLEET_TENANT_IDS_DEFAULT, agent_id=ulmo)])
    assert len(obs) == 1
