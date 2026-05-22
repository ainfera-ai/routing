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


def ruleset_hash() -> str:
    """8-char sha256 digest of the canonical-JSON ruleset payload.

    Bumps automatically whenever _RULESET_PAYLOAD changes — so any audit row
    written under a different rule shape is trivially distinguishable.
    """
    blob = json.dumps(_RULESET_PAYLOAD, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


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


# ── the decision ─────────────────────────────────────────────────────────


def decide(
    request: RoutingRequest,
    candidates: tuple[Candidate, ...] | list[Candidate],
    policy: Policy,
) -> Decision:
    """Quality-floor-then-min-cost over an N-candidate set."""

    h = ruleset_hash()

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

        # 2. Quality floor
        if c.q_prior < policy.min_quality:
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

        survivors.append(c)

    if not survivors:
        # Distinguish "everyone failed enrolment" from "everyone cleared
        # enrolment but failed the floor/budget" — the latter is the
        # methodology's NT1 case ("no_candidate_clears_floor"). Enrolment-
        # only failures get a separate rule_fired so the audit shows the
        # cause precisely.
        any_enrolled_made_it_to_floor = any(
            o.drop_reason
            in (DropReason.BELOW_QUALITY_FLOOR, DropReason.EXCEEDS_BUDGET_CAP)
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
    # Sort key is fully deterministic: (combined_price, -q_prior, model_slug).
    # The model_slug tiebreak guarantees no platform-dependent ordering
    # when price *and* q_prior tie (rare but possible at v0; goes from
    # rare to impossible once empirical priors land in v1).
    ranked = sorted(
        survivors,
        key=lambda c: (
            c.total_price_per_mtok(),
            -(c.q_prior or Decimal("0")),
            c.model_slug,
        ),
    )

    # Detect whether a veto influenced the winner — used to label rule_fired
    # so NT2's "a compliance-failed candidate that would otherwise win is
    # excluded" assertion is auditable from the Decision alone, without
    # re-running the rule. A veto is influential iff at least one vetoed
    # candidate would have placed cheaper than the actual winner.
    veto_outcomes = [
        o for o in outcomes if o.drop_reason is DropReason.M_ALLOWED_VETO
    ]
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
