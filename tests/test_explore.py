"""AIN-335 Part B — counterfactual_explore unit tests (pure, no DB/IO).

Pins the two invariants the founder gate requires:
  1. INERT: shipped defaults (κ=0, cells=∅) ALWAYS return None ⇒ caller is
     byte-for-byte identical to pre-Part-B (decide() is the sole authority).
  2. SAFE: when armed, only enrolled + static-floor-dropped + q_empirical-clears
     arms are ever served — never an unenrolled/vetoed model, never a model the
     data does not vouch for, never a model decide() can already reach.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from ainfera_routing import Candidate, Policy, RoutingRequest, decide
from ainfera_routing.explore import (
    DECISION_RULE,
    CounterfactualPick,
    cells_from_env,
    eligible_arms,
    kappa_from_env,
    select_counterfactual,
)

REASONING_COST = "reasoning:cost"
FLOOR = Decimal("0.88")  # api CELL_MIN_QUALITY[reasoning] (AIN-388)


def _c(mid: str, slug: str, brand: str, q: str, pin: str, pout: str) -> Candidate:
    return Candidate(mid, slug, brand, Decimal(q), Decimal(pin), Decimal(pout), True)


def _anchors() -> list[Candidate]:
    """Enrolled-5 catalog (matches the replay-gate bundle priors)."""
    return [
        _c("id1", "claude-opus-4-7", "anthropic", "0.95", "15", "75"),
        _c("id2", "gpt-5-5", "openai", "0.90", "5", "15"),
        _c("id3", "gemini-3-1-pro", "google", "0.87", "1.25", "10"),
        _c("id4", "grok-4", "xai", "0.86", "5", "15"),
        _c("id5", "mistral-large-3", "mistral", "0.74", "2", "6"),
    ]


# q_empirical for reasoning:cost on the clean corpus: mistral's learned mean
# (~65.75/69 succeeded rows) clears the 0.88 floor its 0.74 prior can't.
QE_REASONING_COST = {"mistral-large-3": Decimal("0.9529")}


def _request() -> RoutingRequest:
    return RoutingRequest("req", "agent", 1000, 1024)


# ── invariant 1: INERT by default ────────────────────────────────────────


def test_default_kappa_zero_is_inert() -> None:
    """κ=0 (shipped default) returns None for any cell / roll / catalog."""
    for roll in (0.0, 0.01, 0.5, 0.999):
        assert (
            select_counterfactual(
                _anchors(),
                Policy(min_quality=FLOOR),
                cell=REASONING_COST,
                q_empirical=QE_REASONING_COST,
                kappa=Decimal("0"),
                cells=frozenset({REASONING_COST}),
                roll=roll,
            )
            is None
        )


def test_empty_cells_is_inert() -> None:
    """cells=∅ (shipped default) returns None even with κ=1 and roll=0."""
    assert (
        select_counterfactual(
            _anchors(),
            Policy(min_quality=FLOOR),
            cell=REASONING_COST,
            q_empirical=QE_REASONING_COST,
            kappa=Decimal("1"),
            cells=frozenset(),
            roll=0.0,
        )
        is None
    )


def test_inert_config_is_byte_identical_to_decide() -> None:
    """With shipped defaults, across a grid of cells x rolls, the helper never
    fires — so the caller's dispatch == decide()'s, byte-for-byte. We assert
    the helper is None everywhere AND that decide() is unaffected by its
    presence (it never mutates inputs)."""
    cands = _anchors()
    pol = Policy(min_quality=FLOOR)
    baseline = decide(_request(), cands, pol)
    for cell in ("reasoning:cost", "chat:cost", "code:balanced", "reasoning:quality"):
        for roll in (0.0, 0.25, 0.5, 0.75, 0.999):
            pick = select_counterfactual(
                cands,
                pol,
                cell=cell,
                q_empirical=QE_REASONING_COST,
                kappa=Decimal("0"),  # default
                cells=frozenset(),  # default
                roll=roll,
            )
            assert pick is None
    # decide() output is unchanged and deterministic alongside the inert helper.
    assert decide(_request(), cands, pol) == baseline


# ── invariant 2: SAFE selection when armed ───────────────────────────────


def test_roll_above_kappa_skips() -> None:
    """roll >= κ → None (the request simply isn't in the κ fraction)."""
    common = dict(
        cell=REASONING_COST,
        q_empirical=QE_REASONING_COST,
        kappa=Decimal("0.10"),
        cells=frozenset({REASONING_COST}),
    )
    assert select_counterfactual(_anchors(), Policy(min_quality=FLOOR), roll=0.5, **common) is None
    pick = select_counterfactual(_anchors(), Policy(min_quality=FLOOR), roll=0.05, **common)
    assert pick is not None and pick.candidate.model_slug == "mistral-large-3"


def test_reasoning_cost_lead_picks_mistral_when_armed() -> None:
    """The real lead: armed on reasoning:cost, the helper serves the
    floor-dropped-but-empirically-clearing arm (mistral) — exactly the arm the
    gate lacks head-to-head support for."""
    pick = select_counterfactual(
        _anchors(),
        Policy(min_quality=FLOOR),
        cell=REASONING_COST,
        q_empirical=QE_REASONING_COST,
        kappa=Decimal("1"),
        cells=frozenset({REASONING_COST}),
        roll=0.0,
    )
    assert isinstance(pick, CounterfactualPick)
    assert pick.candidate.model_slug == "mistral-large-3"
    assert pick.q_prior < FLOOR <= pick.q_empirical_used
    assert pick.cell == REASONING_COST


def test_greedy_decide_still_drops_mistral_below_floor() -> None:
    """Sanity: the greedy pick (q_prior only, no q_empirical) drops mistral
    below the 0.88 floor and picks gpt-5-5 — which is *why* counterfactual
    exploration is needed to ever observe mistral here."""
    d = decide(_request(), _anchors(), Policy(min_quality=FLOOR))
    assert d.chosen is not None and d.chosen.model_slug == "gpt-5-5"


def test_never_serves_floor_clearing_arm() -> None:
    """An arm already at/above the floor is NOT counterfactual-eligible —
    decide()'s greedy pick can already reach it (no exploration needed)."""
    arms = eligible_arms(_anchors(), Policy(min_quality=FLOOR), q_empirical=QE_REASONING_COST)
    slugs = {c.model_slug for c, _ in arms}
    assert "gpt-5-5" not in slugs  # 0.90 >= 0.88, greedy-reachable
    assert "claude-opus-4-7" not in slugs
    assert slugs == {"mistral-large-3"}


def test_requires_empirical_to_clear_floor() -> None:
    """No q_empirical, or one still below the floor → not eligible."""
    pol = Policy(min_quality=FLOOR)
    assert eligible_arms(_anchors(), pol, q_empirical=None) == []
    assert eligible_arms(_anchors(), pol, q_empirical={}) == []
    # mistral learned 0.80 < 0.88 → still not vouched for
    assert eligible_arms(_anchors(), pol, q_empirical={"mistral-large-3": Decimal("0.80")}) == []


def test_never_serves_unenrolled_or_vetoed() -> None:
    """Even with a stellar q_empirical, an unenrolled / vetoed / unpriced model
    is never eligible — the enrolment gates are absolute."""
    pol = Policy(min_quality=FLOOR)
    great = {"mistral-large-3": Decimal("0.99")}
    base = _anchors()
    mistral_idx = next(i for i, c in enumerate(base) if c.model_slug == "mistral-large-3")

    vetoed = list(base)
    vetoed[mistral_idx] = replace(base[mistral_idx], m_allowed=False)
    assert eligible_arms(vetoed, pol, q_empirical=great) == []

    no_verdict = list(base)
    no_verdict[mistral_idx] = replace(base[mistral_idx], m_allowed=None)
    assert eligible_arms(no_verdict, pol, q_empirical=great) == []

    no_prior = list(base)
    no_prior[mistral_idx] = replace(base[mistral_idx], q_prior=None)
    assert eligible_arms(no_prior, pol, q_empirical=great) == []

    unpriced = list(base)
    unpriced[mistral_idx] = replace(base[mistral_idx], price_out_per_mtok_usd=Decimal("0"))
    assert eligible_arms(unpriced, pol, q_empirical=great) == []


def test_selects_cheapest_eligible() -> None:
    """Two floor-dropped, empirically-clearing arms → cheapest combined price."""
    pol = Policy(min_quality=FLOOR)
    cands = _anchors()
    # grok-4 (0.86) also floor-dropped; give it an empirical clear too.
    qe = {"mistral-large-3": Decimal("0.95"), "grok-4": Decimal("0.95")}
    arms = eligible_arms(cands, pol, q_empirical=qe)
    assert [c.model_slug for c, _ in arms] == ["mistral-large-3", "grok-4"]  # $8 < $20
    pick = select_counterfactual(
        cands,
        pol,
        cell=REASONING_COST,
        q_empirical=qe,
        kappa=Decimal("1"),
        cells=frozenset({REASONING_COST}),
        roll=0.0,
    )
    assert pick is not None and pick.candidate.model_slug == "mistral-large-3"


# ── env parsers ──────────────────────────────────────────────────────────


def test_kappa_from_env_defaults_and_clamps() -> None:
    assert kappa_from_env({}) == Decimal("0")
    assert kappa_from_env({"AINFERA_COUNTERFACTUAL_EXPLORE_KAPPA": ""}) == Decimal("0")
    assert kappa_from_env({"AINFERA_COUNTERFACTUAL_EXPLORE_KAPPA": "0.1"}) == Decimal("0.1")
    assert kappa_from_env({"AINFERA_COUNTERFACTUAL_EXPLORE_KAPPA": "-1"}) == Decimal("0")
    assert kappa_from_env({"AINFERA_COUNTERFACTUAL_EXPLORE_KAPPA": "9"}) == Decimal("1")
    assert kappa_from_env({"AINFERA_COUNTERFACTUAL_EXPLORE_KAPPA": "nan?"}) == Decimal("0")


def test_cells_from_env_defaults_and_parses() -> None:
    assert cells_from_env({}) == frozenset()
    assert cells_from_env({"AINFERA_COUNTERFACTUAL_CELLS": ""}) == frozenset()
    assert cells_from_env({"AINFERA_COUNTERFACTUAL_CELLS": "reasoning:cost"}) == frozenset(
        {"reasoning:cost"}
    )
    assert cells_from_env(
        {"AINFERA_COUNTERFACTUAL_CELLS": "reasoning:cost, code:balanced ,"}
    ) == frozenset({"reasoning:cost", "code:balanced"})


def test_decision_rule_marker() -> None:
    assert DECISION_RULE == "counterfactual_explore"
