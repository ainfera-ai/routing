"""Brain unit tests — pure, no DB, no I/O.

Covers:
  · happy path (cheapest survivor wins)
  · determinism (same inputs → identical Decision, byte-for-byte)
  · NT1 — floor reject (all below min_quality → no_candidate_clears_floor)
  · NT2 — M_allowed veto (next-best wins, rule_fired records the veto)
  · NT3 — enrolment gate (q_prior=None / m_allowed=None never enters scoring)
  · tiebreak ordering when prices are equal
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from ainfera_routing import Candidate, Policy, RoutingRequest, decide
from ainfera_routing.decide import project_cost_usd, ruleset_hash
from ainfera_routing.types import DropReason

# ── §C anchor fixture: the 5 frontier models with seeded q_prior ─────────


def _anchors() -> list[Candidate]:
    """Mirrors the migration-0026 seed values (F5 ruling)."""
    return [
        Candidate(
            model_id="00000000-0000-0000-0000-000000000001",
            model_slug="claude-opus-4-7",
            brand_slug="anthropic",
            q_prior=Decimal("0.95"),
            price_in_per_mtok_usd=Decimal("15"),
            price_out_per_mtok_usd=Decimal("75"),
            m_allowed=True,
        ),
        Candidate(
            model_id="00000000-0000-0000-0000-000000000002",
            model_slug="gpt-5-5",
            brand_slug="openai",
            q_prior=Decimal("0.93"),
            price_in_per_mtok_usd=Decimal("5"),
            price_out_per_mtok_usd=Decimal("15"),
            m_allowed=True,
        ),
        Candidate(
            model_id="00000000-0000-0000-0000-000000000003",
            model_slug="gemini-3-1-pro",
            brand_slug="google",
            q_prior=Decimal("0.90"),
            price_in_per_mtok_usd=Decimal("1.25"),
            price_out_per_mtok_usd=Decimal("10"),
            m_allowed=True,
        ),
        Candidate(
            model_id="00000000-0000-0000-0000-000000000004",
            model_slug="grok-4",
            brand_slug="xai",
            q_prior=Decimal("0.86"),
            price_in_per_mtok_usd=Decimal("5"),
            price_out_per_mtok_usd=Decimal("15"),
            m_allowed=True,
        ),
        Candidate(
            model_id="00000000-0000-0000-0000-000000000005",
            model_slug="mistral-large-3",
            brand_slug="mistral",
            q_prior=Decimal("0.80"),
            price_in_per_mtok_usd=Decimal("2"),
            price_out_per_mtok_usd=Decimal("6"),
            m_allowed=True,
        ),
    ]


def _request(estimated_input_tokens: int = 1000, reserved_max_tokens: int = 1024) -> RoutingRequest:
    return RoutingRequest(
        request_id="req-test",
        agent_id="agent-test",
        estimated_input_tokens=estimated_input_tokens,
        reserved_max_tokens=reserved_max_tokens,
    )


# ── happy path ───────────────────────────────────────────────────────────


def test_happy_path_cheapest_clearing_floor_wins() -> None:
    """min_quality=0.80 → all 5 enrol → cheapest combined-price wins.

    mistral-large-3 has the lowest combined price ($8/Mt) and clears 0.80 → wins.
    """
    d = decide(
        _request(),
        _anchors(),
        Policy(min_quality=Decimal("0.80"), policy_name="balanced"),
    )
    assert d.rule_fired == "cheapest_clearing_floor"
    assert d.chosen is not None
    assert d.chosen.model_slug == "mistral-large-3"
    assert d.chosen.rank == 0
    assert d.chosen.drop_reason is None


def test_happy_path_floor_excludes_some_winners() -> None:
    """min_quality=0.91 → only opus and gpt-5-5 clear; gpt-5-5 cheaper → wins."""
    d = decide(
        _request(),
        _anchors(),
        Policy(min_quality=Decimal("0.91"), policy_name="quality_first"),
    )
    assert d.rule_fired == "cheapest_clearing_floor"
    assert d.chosen is not None
    assert d.chosen.model_slug == "gpt-5-5"
    # Below-floor models recorded with BELOW_QUALITY_FLOOR
    below = [c for c in d.candidates if c.drop_reason is DropReason.BELOW_QUALITY_FLOOR]
    assert {c.model_slug for c in below} == {"gemini-3-1-pro", "grok-4", "mistral-large-3"}


# ── determinism ──────────────────────────────────────────────────────────


def test_determinism_identical_decision() -> None:
    """Same (request, candidates, policy) → byte-identical Decision."""
    req = _request()
    cands = _anchors()
    pol = Policy(min_quality=Decimal("0.85"), policy_name="balanced")

    d1 = decide(req, cands, pol)
    d2 = decide(req, cands, pol)
    d3 = decide(req, list(reversed(cands)), pol)  # input order MUST NOT matter

    assert d1 == d2
    assert d1 == d3  # canonicalised ordering inside Decision.candidates


def test_determinism_ruleset_hash_stable() -> None:
    """ruleset_hash is hex of length 8 and changes only when ruleset changes."""
    h = ruleset_hash()
    assert len(h) == 8
    assert all(ch in "0123456789abcdef" for ch in h)
    # Stable across calls
    assert h == ruleset_hash()


# ── NT1 — floor reject ──────────────────────────────────────────────────


def test_nt1_floor_reject_returns_structured_no_candidate_clears_floor() -> None:
    """All 5 < min_quality=0.96 → rule_fired=no_candidate_clears_floor."""
    d = decide(
        _request(),
        _anchors(),
        Policy(min_quality=Decimal("0.96"), policy_name="strict"),
    )
    assert d.rule_fired == "no_candidate_clears_floor"
    assert d.chosen is None
    assert d.is_reject is True
    # Every input must appear in the audit with a drop_reason
    assert len(d.candidates) == 5
    assert all(c.drop_reason is DropReason.BELOW_QUALITY_FLOOR for c in d.candidates)


# ── NT2 — M_allowed veto ─────────────────────────────────────────────────


def test_nt2_m_allowed_veto_excludes_otherwise_winning_candidate() -> None:
    """Mistral vetoed → next-cheapest survivor (gemini) wins; rule names the veto.

    Without the veto, mistral-large-3 would win the 0.80 floor. With it,
    gemini-3-1-pro (next cheapest at $11.25/Mt combined, q=0.90) wins.
    """
    cands = _anchors()
    cands = [replace(c, m_allowed=False) if c.brand_slug == "mistral" else c for c in cands]
    d = decide(
        _request(),
        cands,
        Policy(min_quality=Decimal("0.80"), policy_name="compliance_first"),
    )
    assert d.rule_fired == "m_allowed_veto_applied"
    assert d.chosen is not None
    assert d.chosen.model_slug == "gemini-3-1-pro"
    vetoed = [c for c in d.candidates if c.drop_reason is DropReason.M_ALLOWED_VETO]
    assert {c.model_slug for c in vetoed} == {"mistral-large-3"}


def test_nt2_veto_on_non_winning_candidate_does_not_change_rule_fired() -> None:
    """Vetoing a non-cheapest candidate must not flip rule_fired to veto_applied."""
    cands = _anchors()
    cands = [replace(c, m_allowed=False) if c.brand_slug == "anthropic" else c for c in cands]
    d = decide(
        _request(),
        cands,
        Policy(min_quality=Decimal("0.80"), policy_name="balanced"),
    )
    # mistral still wins; opus was vetoed but never going to win at this floor
    assert d.chosen is not None
    assert d.chosen.model_slug == "mistral-large-3"
    assert d.rule_fired == "cheapest_clearing_floor"


# ── NT3 — enrolment gate (no fabricated priors) ──────────────────────────


def test_nt3_q_prior_none_is_dropped_not_chosen() -> None:
    """A candidate with q_prior=None NEVER scores; emergent gating proven."""
    cands = [
        *_anchors(),
        # claude-sonnet-4-6 is live in prod with NULL q_prior (per F5: not
        # enrolled until AIN-248 backfills it from AA Intelligence Index v4.0).
        Candidate(
            model_id="00000000-0000-0000-0000-000000000099",
            model_slug="claude-sonnet-4-6",
            brand_slug="anthropic",
            q_prior=None,
            price_in_per_mtok_usd=Decimal("3"),
            price_out_per_mtok_usd=Decimal("15"),
            m_allowed=True,
        ),
    ]
    d = decide(
        _request(),
        cands,
        Policy(min_quality=Decimal("0.80"), policy_name="balanced"),
    )
    assert d.chosen is not None
    assert d.chosen.model_slug != "claude-sonnet-4-6"
    not_enrolled = [c for c in d.candidates if c.drop_reason is DropReason.NOT_ENROLLED_NO_Q_PRIOR]
    assert {c.model_slug for c in not_enrolled} == {"claude-sonnet-4-6"}


def test_nt3_m_allowed_none_is_not_enrolled_via_veto_path() -> None:
    """A candidate with m_allowed=None (no verdict, e.g. a gated brand) is dropped."""
    cands = [
        Candidate(
            model_id="00000000-0000-0000-0000-0000000000aa",
            model_slug="qwen-3-max",
            brand_slug="alibaba",
            q_prior=Decimal("0.88"),  # hypothetical post-AIN-248 backfill
            price_in_per_mtok_usd=Decimal("1"),
            price_out_per_mtok_usd=Decimal("3"),
            m_allowed=None,  # no verdict
        ),
        *_anchors(),
    ]
    d = decide(
        _request(),
        cands,
        Policy(min_quality=Decimal("0.80"), policy_name="balanced"),
    )
    assert d.chosen is not None
    # qwen has no verdict → never wins despite being the cheapest, highest-q
    assert d.chosen.model_slug != "qwen-3-max"
    vetoed = [c for c in d.candidates if c.drop_reason is DropReason.M_ALLOWED_VETO]
    assert "qwen-3-max" in {c.model_slug for c in vetoed}


# ── budget cap ───────────────────────────────────────────────────────────


def test_budget_cap_drops_too_expensive_survivors() -> None:
    """budget_cap_usd=0.01 with 1000-in/1024-out tokens excludes opus + gpt-5-5.

    opus projected ≈ 1000*15/1e6 + 1024*75/1e6 = 0.015 + 0.0768 = 0.0918 → excluded
    gpt-5-5 projected ≈ 0.005 + 0.01536 = 0.02036 → excluded at 0.01
    grok-4 same as gpt-5-5 → excluded
    gemini ≈ 0.00125 + 0.01024 = 0.01149 → excluded at 0.01
    mistral ≈ 0.002 + 0.006144 = 0.008144 → survives → wins
    """
    d = decide(
        _request(estimated_input_tokens=1000, reserved_max_tokens=1024),
        _anchors(),
        Policy(
            min_quality=Decimal("0.80"),
            budget_cap_usd=Decimal("0.01"),
            policy_name="cost_first",
        ),
    )
    assert d.chosen is not None
    assert d.chosen.model_slug == "mistral-large-3"
    over_budget = [c for c in d.candidates if c.drop_reason is DropReason.EXCEEDS_BUDGET_CAP]
    assert "claude-opus-4-7" in {c.model_slug for c in over_budget}


def test_budget_cap_too_tight_returns_no_candidate_clears_floor() -> None:
    """Every survivor over budget → reject path."""
    d = decide(
        _request(estimated_input_tokens=1000, reserved_max_tokens=1024),
        _anchors(),
        Policy(
            min_quality=Decimal("0.80"),
            budget_cap_usd=Decimal("0.000001"),
            policy_name="cost_first",
        ),
    )
    assert d.is_reject
    assert d.rule_fired == "no_candidate_clears_floor"


# ── empty + degenerate ──────────────────────────────────────────────────


def test_empty_candidate_set_returns_no_candidate_enrolled() -> None:
    d = decide(_request(), [], Policy(min_quality=Decimal("0.80")))
    assert d.rule_fired == "no_candidate_enrolled"
    assert d.is_reject


def test_tiebreak_q_prior_desc_then_slug_asc_on_equal_price() -> None:
    """gpt-5-5 and grok-4 have identical combined price; higher q (gpt-5-5) wins."""
    cands = [c for c in _anchors() if c.model_slug in ("gpt-5-5", "grok-4")]
    d = decide(
        _request(),
        cands,
        Policy(min_quality=Decimal("0.80"), policy_name="balanced"),
    )
    assert d.chosen is not None
    assert d.chosen.model_slug == "gpt-5-5"


def test_fallback_order_excludes_drops_and_starts_with_winner() -> None:
    d = decide(
        _request(),
        _anchors(),
        Policy(min_quality=Decimal("0.80"), policy_name="balanced"),
    )
    fb = d.fallback_order()
    assert len(fb) == 5  # all 5 cleared the 0.80 floor
    assert fb[0].model_slug == "mistral-large-3"
    # fallback is ranked
    assert [c.rank for c in fb] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "min_q,expected_winner",
    [
        ("0.80", "mistral-large-3"),  # all 5 clear; cheapest wins
        # mistral drops at 0.81; gemini is cheapest of {opus, gpt-5-5, gemini, grok}
        ("0.81", "gemini-3-1-pro"),
        ("0.87", "gemini-3-1-pro"),  # grok drops too
        ("0.91", "gpt-5-5"),  # gemini drops; gpt-5-5 cheaper than opus
        ("0.94", "claude-opus-4-7"),  # only opus
    ],
)
def test_floor_sweep_picks_expected_winner(min_q: str, expected_winner: str) -> None:
    d = decide(
        _request(),
        _anchors(),
        Policy(min_quality=Decimal(min_q), policy_name="balanced"),
    )
    assert d.chosen is not None
    assert d.chosen.model_slug == expected_winner


# ── AIN-246 · q_empirical override (q_prior ⊕ q_empirical) ───────────────


def test_q_empirical_none_is_identical_to_v0() -> None:
    """Omitting q_empirical == passing None == v0: byte-identical Decision."""
    req, cands = _request(), _anchors()
    pol = Policy(min_quality=Decimal("0.85"), policy_name="balanced")
    base = decide(req, cands, pol)
    with_none = decide(req, cands, pol, q_empirical=None)
    with_empty = decide(req, cands, pol, q_empirical={})
    assert base == with_none == with_empty


def test_q_empirical_lifts_below_floor_model_and_wins() -> None:
    """mistral q_prior=0.80 is below a 0.91 floor, but a learned mean of 0.95
    clears it — and it is the cheapest survivor, so it now wins."""
    pol = Policy(min_quality=Decimal("0.91"), policy_name="quality_first")
    # v0: mistral excluded, gpt-5-5 wins
    assert decide(_request(), _anchors(), pol).chosen.model_slug == "gpt-5-5"
    # with empirical override: mistral clears floor and is cheapest ($8/Mt) → wins
    d = decide(
        _request(),
        _anchors(),
        pol,
        q_empirical={"mistral-large-3": Decimal("0.95")},
    )
    assert d.chosen is not None
    assert d.chosen.model_slug == "mistral-large-3"
    assert d.rule_fired == "cheapest_clearing_floor"


def test_q_empirical_drops_high_prior_model_below_floor() -> None:
    """opus q_prior=0.95 clears a 0.91 floor at v0, but a learned mean of 0.50
    pushes it below — it is dropped with BELOW_QUALITY_FLOOR."""
    pol = Policy(min_quality=Decimal("0.91"), policy_name="quality_first")
    d = decide(
        _request(),
        _anchors(),
        pol,
        q_empirical={"claude-opus-4-7": Decimal("0.50")},
    )
    opus = next(c for c in d.candidates if c.model_slug == "claude-opus-4-7")
    assert opus.drop_reason is DropReason.BELOW_QUALITY_FLOOR
    # winner is still the cheapest among the true survivors (gpt-5-5)
    assert d.chosen is not None and d.chosen.model_slug == "gpt-5-5"


def test_q_empirical_override_is_deterministic() -> None:
    """Same inputs incl. q_empirical → byte-identical Decision across runs."""
    req, cands = _request(), _anchors()
    pol = Policy(min_quality=Decimal("0.88"), policy_name="balanced")
    qe = {"grok-4": Decimal("0.99"), "mistral-large-3": Decimal("0.40")}
    assert decide(req, cands, pol, q_empirical=qe) == decide(req, cands, pol, q_empirical=qe)


# ── AIN-446 diversity soft-penalty (ordering-only, default OFF) ───────────


def test_diversity_inert_when_unset_or_zero_is_byte_identical() -> None:
    """No penalty / empty map / all-zero → byte-identical Decision AND the v0
    ruleset_hash (the diversity rule must leave audit/replay untouched off)."""
    req, cands = _request(), _anchors()
    pol = Policy(min_quality=Decimal("0.80"), policy_name="balanced")
    base = decide(req, cands, pol)
    assert decide(req, cands, pol, maker_penalty=None) == base
    assert decide(req, cands, pol, maker_penalty={}) == base
    assert decide(req, cands, pol, maker_penalty={"openai": Decimal("0")}) == base
    assert base.ruleset_hash == ruleset_hash()


def test_diversity_penalty_reorders_winner_on_effective_price() -> None:
    """min_quality=0.80 → mistral ($8) is the v0 winner. A 0.5 markup on the
    mistral maker makes its effective price $12 > gemini's $11.25 → gemini wins.
    The reported cost stays the REAL gemini cost, and the ruleset_hash bumps."""
    req, cands = _request(), _anchors()
    pol = Policy(min_quality=Decimal("0.80"), policy_name="balanced")
    assert decide(req, cands, pol).chosen.model_slug == "mistral-large-3"  # type: ignore[union-attr]

    d = decide(req, cands, pol, maker_penalty={"mistral": Decimal("0.5")})
    assert d.chosen is not None
    assert d.chosen.model_slug == "gemini-3-1-pro"
    # reported cost is the real (un-penalised) projected cost for the winner
    real = project_cost_usd(
        candidate=next(c for c in cands if c.model_slug == "gemini-3-1-pro"),
        estimated_input_tokens=req.estimated_input_tokens,
        reserved_max_tokens=req.reserved_max_tokens,
    )
    assert d.chosen.projected_cost_usd == real
    # audit row is distinguishable from a v0 row
    assert d.ruleset_hash != ruleset_hash()


def test_diversity_active_hash_bumps_even_when_winner_unchanged() -> None:
    """A small penalty that doesn't flip the winner still stamps the diversity
    ruleset_hash — the rule was in force, so the audit row must say so."""
    req, cands = _request(), _anchors()
    pol = Policy(min_quality=Decimal("0.80"), policy_name="balanced")
    d = decide(req, cands, pol, maker_penalty={"mistral": Decimal("0.1")})
    assert d.chosen is not None and d.chosen.model_slug == "mistral-large-3"  # 8*1.1=8.8 < 11.25
    assert d.ruleset_hash != ruleset_hash()


def test_diversity_never_relaxes_budget() -> None:
    """A penalty reorders survivors only — it can never resurrect a candidate
    dropped by the real budget gate. With a cap that leaves only mistral, even a
    huge mistral penalty keeps mistral the winner (nothing else cleared budget)."""
    req, cands = _request(), _anchors()
    pol = Policy(
        min_quality=Decimal("0.80"),
        budget_cap_usd=Decimal("0.009"),  # only mistral's projected cost clears
        policy_name="balanced",
    )
    d = decide(req, cands, pol, maker_penalty={"mistral": Decimal("9")})
    assert d.chosen is not None and d.chosen.model_slug == "mistral-large-3"


def test_diversity_penalty_is_deterministic() -> None:
    """Same inputs incl. maker_penalty → byte-identical Decision across runs."""
    req, cands = _request(), _anchors()
    pol = Policy(min_quality=Decimal("0.80"), policy_name="balanced")
    mp = {"mistral": Decimal("0.5"), "openai": Decimal("0.2")}
    assert decide(req, cands, pol, maker_penalty=mp) == decide(req, cands, pol, maker_penalty=mp)
