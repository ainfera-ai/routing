"""The decision core — pure, deterministic, N-agnostic.

Methodology v1.2 §D, locked 2026-05-22 (AIN-245). Routing rule:

    1. Enrol     — keep candidates with price + q_prior + m_allowed=True
                   (the three "real gates"; never a hardcoded active flag).
    2. Floor     — drop survivors with q_prior < policy.min_quality.
    3. Budget    — drop survivors whose projected cost exceeds the cap.
    4. Pick      — cheapest survivor wins.
                   Tiebreak: higher q_prior (latency dormant; see C1).
    5. Capture   — caller writes Decision to routing_outcomes.

No weighted-λ. No weighted-sum. No fabricated priors.

Pure functions only — no DB, no clock, no RNG. Same `(request, candidates,
policy)` → identical `Decision` byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal

from ainfera_routing.types import (
    Candidate,
    CandidateOutcome,
    Decision,
    DropReason,
    Policy,
    RoutingRequest,
)

POLICY_SEMVER = "1.0.0"


# ── ruleset hash (determinism token) ─────────────────────────────────────


_RULESET_PAYLOAD = {
    "gates": ["price_present", "q_prior_present", "m_allowed_pass"],
    "score_order": [
        "drop_below_quality_floor",
        "drop_above_budget_cap",
        "cheapest_wins",
    ],
    "tiebreak": ["q_prior_desc"],  # latency tiebreak deferred (C1)
    "version": POLICY_SEMVER,
}


# AIN-446 diversity soft-penalty (default OFF). When the caller supplies a
# per-maker order-price penalty (`maker_penalty`), the survivor ORDERING (step
# 4) ranks on a diversity-adjusted effective price; gates / floor / budget and
# the reported projected costs are unchanged. Folded into the ruleset payload
# ONLY when active, so audit rows written under the diversity rule are
# distinguishable and replay stays exact — with no penalty the hash is
# byte-identical to v0.
_DIVERSITY_RULE = {
    "diversity": {
        "mechanism": "maker_share_soft_penalty",
        "applies_to": "ordering_only",
    }
}

# AIN-542 selection layer (default OFF). When the preset carries a latency_cap_ms
# the brain drops survivors slower than the SLO (step 3b). Folded into the ruleset
# payload ONLY when active, so audit rows under the latency rule are distinguishable
# and replay stays exact — with no SLO the hash is byte-identical to v0.
_LATENCY_RULE = {
    "latency": {
        "mechanism": "preset_p95_latency_slo",
        "applies_to": "drop_above_latency_cap",
    }
}


def _ruleset_hash(*, diversity_active: bool, latency_active: bool = False) -> str:
    payload: dict[str, object] = dict(_RULESET_PAYLOAD)
    if diversity_active:
        payload.update(_DIVERSITY_RULE)
    if latency_active:
        payload.update(_LATENCY_RULE)
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def ruleset_hash() -> str:
    """8-char sha256 digest of the canonical-JSON ruleset payload (the v0 /
    diversity-inactive shape).

    Bumps automatically whenever _RULESET_PAYLOAD changes — so any audit row
    written under a different rule shape is trivially distinguishable. When the
    diversity soft-penalty is active for a given decision, `decide` stamps the
    extended (diversity) shape instead; see `_ruleset_hash`.
    """
    return _ruleset_hash(diversity_active=False, latency_active=False)


# ── cost projection ──────────────────────────────────────────────────────


_MILLION = Decimal("1000000")


def project_cost_usd(
    *,
    candidate: Candidate,
    estimated_input_tokens: int,
    reserved_max_tokens: int,
) -> Decimal:
    """Projected upper-bound cost = input * in_rate + max_tokens * out_rate.

    Matches the orchestration layer's cost reservation. The brain only
    uses this for the budget gate; the cheapness ordering uses the simpler
    `total_price_per_mtok` so v0 doesn't depend on input-token estimates
    being consistent across callers.
    """
    return (
        Decimal(estimated_input_tokens) * candidate.price_in_per_mtok_usd / _MILLION
        + Decimal(reserved_max_tokens) * candidate.price_out_per_mtok_usd / _MILLION
    ).quantize(Decimal("0.000001"))


# ── quality source (AIN-246 · q_prior ⊕ q_empirical) ─────────────────────


def _effective_q(
    candidate: Candidate,
    q_empirical: Mapping[str, Decimal] | None,
) -> Decimal | None:
    """Quality used for the floor + q-tiebreak: the learned per-model mean
    (``q_empirical``) when present, else the static ``q_prior``.

    Per ontology v1.3 (``Q = q_prior ⊕ q_empirical``): once labeled outcomes
    accrue, the empirical mean overrides the prior for SCORING. Enrolment is
    unchanged — it still requires a real ``q_prior`` (F5: never fabricate a
    prior), so the override only ever adjusts an already-enrolled candidate.
    ``q_empirical=None`` (steady-state caller) returns ``q_prior`` unchanged,
    so ``decide()`` stays byte-for-byte identical to v0.
    """
    if q_empirical is not None:
        override = q_empirical.get(candidate.model_slug)
        if override is not None:
            return override
    return candidate.q_prior


# ── diversity soft-penalty (AIN-446 · ordering only) ─────────────────────


def _order_price(
    candidate: Candidate,
    maker_penalty: Mapping[str, Decimal] | None,
) -> Decimal:
    """Cheapness key for step 4 — the real combined price scaled up by the
    candidate maker's diversity penalty: ``price * (1 + penalty)``.

    The penalty is a non-negative fractional markup keyed by ``brand_slug``
    (the maker), supplied by the caller from the rolling per-maker share vs the
    ≤25% diversity target (AIN-446). Keeping the share→penalty curve in the
    caller leaves ``decide`` pure. The markup affects ORDERING ONLY — the
    budget gate and the reported ``projected_cost_usd`` always use the real
    price, so a diversity nudge can never relax a real budget. ``None`` /
    absent maker / non-positive penalty → real price unchanged (inert)."""
    if maker_penalty is not None:
        p = maker_penalty.get(candidate.brand_slug)
        if p is not None and p > 0:
            return candidate.total_price_per_mtok() * (Decimal("1") + p)
    return candidate.total_price_per_mtok()


# ── the decision ─────────────────────────────────────────────────────────


def decide(
    request: RoutingRequest,
    candidates: tuple[Candidate, ...] | list[Candidate],
    policy: Policy,
    *,
    q_empirical: Mapping[str, Decimal] | None = None,
    maker_penalty: Mapping[str, Decimal] | None = None,
) -> Decision:
    """Quality-floor-then-min-cost over an N-candidate set.

    ``q_empirical`` (AIN-246): optional ``model_slug -> learned mean`` map. When
    a candidate has an entry, that value replaces ``q_prior`` in the quality
    floor and the q-tiebreak (the enrolment gates and the cheapest-survivor
    pick are unchanged — Disc #12). Omitted/``None`` → identical to v0. The
    map is passed in by the caller (RNG-free, DB-free) so ``decide`` stays a
    pure, deterministic function; exploration (the ≥5% floor) is applied by
    the caller, not here.

    ``maker_penalty`` (AIN-446 diversity soft-penalty, default OFF): optional
    ``brand_slug -> non-negative markup`` map. When supplied, the cheapest-
    survivor ORDERING ranks on a diversity-adjusted effective price
    (``price * (1 + penalty)``) so an over-represented maker is gently
    down-ranked toward the ≤25%-per-maker target. Gates, floor, budget and the
    reported ``projected_cost_usd`` always use the REAL price — the nudge never
    relaxes a budget. ``None`` / empty / all-zero → byte-identical to v0
    (same winner AND same ``ruleset_hash``); the hash only bumps to the
    diversity shape when a positive penalty is actually in force.
    """

    diversity_active = maker_penalty is not None and any(v > 0 for v in maker_penalty.values())
    latency_active = policy.latency_cap_ms is not None
    h = _ruleset_hash(diversity_active=diversity_active, latency_active=latency_active)

    if not candidates:
        return Decision(
            rule_fired="no_candidate_enrolled",
            chosen=None,
            candidates=(),
            policy_name=policy.policy_name,
            policy_semver=POLICY_SEMVER,
            ruleset_hash=h,
        )

    # Walk every input candidate once. Each gets a CandidateOutcome with
    # either a drop_reason or a place in the ranked survivor list.
    outcomes: list[CandidateOutcome] = []
    survivors: list[Candidate] = []

    for c in candidates:
        projected = project_cost_usd(
            candidate=c,
            estimated_input_tokens=request.estimated_input_tokens,
            reserved_max_tokens=request.reserved_max_tokens,
        )

        # 1. Enrolment gate · price
        # ModelORM.input_cost_per_million_usd is NOT NULL in the schema
        # (alembic 0005 CHECK > 0), so this branch is a defense in depth;
        # if a future schema relaxation lets a 0/None through, we drop
        # rather than divide-by-zero downstream.
        if c.price_in_per_mtok_usd <= 0 or c.price_out_per_mtok_usd <= 0:
            outcomes.append(
                CandidateOutcome(
                    model_id=c.model_id,
                    model_slug=c.model_slug,
                    brand_slug=c.brand_slug,
                    q_prior=c.q_prior,
                    price_in_per_mtok_usd=c.price_in_per_mtok_usd,
                    price_out_per_mtok_usd=c.price_out_per_mtok_usd,
                    m_allowed=c.m_allowed,
                    projected_cost_usd=None,
                    drop_reason=DropReason.NOT_ENROLLED_NO_PRICE,
                )
            )
            continue

        # 1. Enrolment gate · q_prior (F5: never fabricate a prior)
        if c.q_prior is None:
            outcomes.append(
                CandidateOutcome(
                    model_id=c.model_id,
                    model_slug=c.model_slug,
                    brand_slug=c.brand_slug,
                    q_prior=None,
                    price_in_per_mtok_usd=c.price_in_per_mtok_usd,
                    price_out_per_mtok_usd=c.price_out_per_mtok_usd,
                    m_allowed=c.m_allowed,
                    projected_cost_usd=projected,
                    drop_reason=DropReason.NOT_ENROLLED_NO_Q_PRIOR,
                )
            )
            continue

        # 1. Enrolment gate · M_allowed compliance veto
        # m_allowed=None and m_allowed=False both fail — no verdict means
        # the brand has not been pre-cleared (the 6 gated brands until
        # AIN-248 lands their verdicts).
        if c.m_allowed is not True:
            outcomes.append(
                CandidateOutcome(
                    model_id=c.model_id,
                    model_slug=c.model_slug,
                    brand_slug=c.brand_slug,
                    q_prior=c.q_prior,
                    price_in_per_mtok_usd=c.price_in_per_mtok_usd,
                    price_out_per_mtok_usd=c.price_out_per_mtok_usd,
                    m_allowed=c.m_allowed,
                    projected_cost_usd=projected,
                    drop_reason=DropReason.M_ALLOWED_VETO,
                )
            )
            continue

        # 2. Quality floor (AIN-246: empirical override when present, else prior).
        # eq is non-None here — enrolment above guarantees c.q_prior is present.
        eq = _effective_q(c, q_empirical)
        if eq is not None and eq < policy.min_quality:
            outcomes.append(
                CandidateOutcome(
                    model_id=c.model_id,
                    model_slug=c.model_slug,
                    brand_slug=c.brand_slug,
                    q_prior=c.q_prior,
                    price_in_per_mtok_usd=c.price_in_per_mtok_usd,
                    price_out_per_mtok_usd=c.price_out_per_mtok_usd,
                    m_allowed=c.m_allowed,
                    projected_cost_usd=projected,
                    drop_reason=DropReason.BELOW_QUALITY_FLOOR,
                )
            )
            continue

        # 3. Budget cap
        if policy.budget_cap_usd is not None and projected > policy.budget_cap_usd:
            outcomes.append(
                CandidateOutcome(
                    model_id=c.model_id,
                    model_slug=c.model_slug,
                    brand_slug=c.brand_slug,
                    q_prior=c.q_prior,
                    price_in_per_mtok_usd=c.price_in_per_mtok_usd,
                    price_out_per_mtok_usd=c.price_out_per_mtok_usd,
                    m_allowed=c.m_allowed,
                    projected_cost_usd=projected,
                    drop_reason=DropReason.EXCEEDS_BUDGET_CAP,
                )
            )
            continue

        # 3b. Latency SLO (AIN-542 selection layer; default OFF). When the preset
        # carries a latency_cap_ms AND this candidate has a known expected latency,
        # drop survivors slower than the SLO — so a cheap-but-slow model can't win
        # on price alone (D6). latency_cap_ms None → inert (v0); unknown latency → kept.
        if (
            policy.latency_cap_ms is not None
            and c.expected_latency_ms is not None
            and c.expected_latency_ms > policy.latency_cap_ms
        ):
            outcomes.append(
                CandidateOutcome(
                    model_id=c.model_id,
                    model_slug=c.model_slug,
                    brand_slug=c.brand_slug,
                    q_prior=c.q_prior,
                    price_in_per_mtok_usd=c.price_in_per_mtok_usd,
                    price_out_per_mtok_usd=c.price_out_per_mtok_usd,
                    m_allowed=c.m_allowed,
                    projected_cost_usd=projected,
                    drop_reason=DropReason.EXCEEDS_LATENCY_CAP,
                )
            )
            continue

        survivors.append(c)

    if not survivors:
        # Distinguish "everyone failed enrolment" from "everyone cleared
        # enrolment but failed the floor/budget" — the latter is the
        # methodology's NT1 case ("no_candidate_clears_floor"). Enrolment-
        # only failures get a separate rule_fired so the audit shows the
        # cause precisely.
        any_enrolled_made_it_to_floor = any(
            o.drop_reason
            in (
                DropReason.BELOW_QUALITY_FLOOR,
                DropReason.EXCEEDS_BUDGET_CAP,
                DropReason.EXCEEDS_LATENCY_CAP,
            )
            for o in outcomes
        )
        return Decision(
            rule_fired=(
                "no_candidate_clears_floor"
                if any_enrolled_made_it_to_floor
                else "no_candidate_enrolled"
            ),
            chosen=None,
            candidates=tuple(outcomes),
            policy_name=policy.policy_name,
            policy_semver=POLICY_SEMVER,
            ruleset_hash=h,
        )

    # 4. Pick — cheapest survivor wins.
    # Sort key is fully deterministic: (effective_price, -q_prior, model_slug).
    # The model_slug tiebreak guarantees no platform-dependent ordering
    # when price *and* q_prior tie (rare but possible at v0; goes from
    # rare to impossible once empirical priors land in v1).
    # `effective_price` = real combined price * (1 + maker diversity penalty);
    # with no penalty it IS the real combined price → v0 ordering unchanged.
    ranked = sorted(
        survivors,
        key=lambda c: (
            _order_price(c, maker_penalty),
            -(_effective_q(c, q_empirical) or Decimal("0")),
            c.model_slug,
        ),
    )

    # Detect whether a veto influenced the winner — used to label rule_fired
    # so NT2's "a compliance-failed candidate that would otherwise win is
    # excluded" assertion is auditable from the Decision alone, without
    # re-running the rule. A veto is influential iff at least one vetoed
    # candidate would have placed cheaper than the actual winner.
    veto_outcomes = [o for o in outcomes if o.drop_reason is DropReason.M_ALLOWED_VETO]
    winner = ranked[0]
    winner_total = winner.total_price_per_mtok()
    veto_changed_winner = any(
        (v.price_in_per_mtok_usd + v.price_out_per_mtok_usd) < winner_total
        # vetoed cheaper rows that would also have cleared the floor
        and (v.q_prior is None or v.q_prior >= policy.min_quality)
        for v in veto_outcomes
    )

    # Build survivor outcomes with their final ranks. `outcomes` already
    # holds every dropped candidate; we append the survivors now.
    #
    # Survivor semantics: drop_reason is None for every model that cleared
    # all gates and the floor + budget. `rank` distinguishes the winner
    # (rank=0) from valid fallback candidates (rank>0). fallback_order()
    # uses `rank is not None` to enumerate the 5xx-retry queue.
    survivor_rank = {c.model_id: i for i, c in enumerate(ranked)}
    ranked_outcomes: list[CandidateOutcome] = list(outcomes)
    for c in survivors:
        rank = survivor_rank[c.model_id]
        projected = project_cost_usd(
            candidate=c,
            estimated_input_tokens=request.estimated_input_tokens,
            reserved_max_tokens=request.reserved_max_tokens,
        )
        ranked_outcomes.append(
            CandidateOutcome(
                model_id=c.model_id,
                model_slug=c.model_slug,
                brand_slug=c.brand_slug,
                q_prior=c.q_prior,
                price_in_per_mtok_usd=c.price_in_per_mtok_usd,
                price_out_per_mtok_usd=c.price_out_per_mtok_usd,
                m_allowed=c.m_allowed,
                projected_cost_usd=projected,
                drop_reason=None,
                rank=rank,
            )
        )

    # Canonicalise candidates order for replay: winner first, then by rank
    # for survivors, then dropped candidates by (drop_reason, model_slug).
    def _sort_key(o: CandidateOutcome) -> tuple[int, int | str, str]:
        if o.drop_reason is None:
            return (0, o.rank if o.rank is not None else 0, o.model_slug)
        return (1, o.drop_reason.value, o.model_slug)

    chosen_outcome = next(o for o in ranked_outcomes if o.drop_reason is None and o.rank == 0)

    return Decision(
        rule_fired="m_allowed_veto_applied" if veto_changed_winner else "cheapest_clearing_floor",
        chosen=chosen_outcome,
        candidates=tuple(sorted(ranked_outcomes, key=_sort_key)),
        policy_name=policy.policy_name,
        policy_semver=POLICY_SEMVER,
        ruleset_hash=h,
    )


__all__ = ["POLICY_SEMVER", "decide", "project_cost_usd", "ruleset_hash"]
