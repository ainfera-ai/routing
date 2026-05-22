"""Routing decision types — Candidate / Policy / RoutingRequest / Decision.

All dataclasses are frozen so a Decision is hashable and trivially
round-trippable to JSON for §16 capture + deterministic replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum


class DropReason(StrEnum):
    """Why a candidate did not win.

    The first three are *enrolment* failures — the candidate never entered
    the scoring pool. The last three are *scoring* failures — the candidate
    entered but lost.
    """

    NOT_ENROLLED_NO_PRICE = "not_enrolled_no_price"
    NOT_ENROLLED_NO_Q_PRIOR = "not_enrolled_no_q_prior"
    M_ALLOWED_VETO = "m_allowed_veto"
    BELOW_QUALITY_FLOOR = "below_quality_floor"
    EXCEEDS_BUDGET_CAP = "exceeds_budget_cap"


@dataclass(frozen=True)
class Candidate:
    """One row the brain may pick from.

    `q_prior` and `m_allowed` are Optional so the brain can SEE not-enrolled
    rows and mark them with explicit drop reasons (auditable, not silently
    dropped upstream). Latency is captured in `routing_outcomes` post-call,
    not used in v0 scoring (see C1).
    """

    model_id: str
    model_slug: str
    brand_slug: str
    q_prior: Decimal | None
    price_in_per_mtok_usd: Decimal
    price_out_per_mtok_usd: Decimal
    m_allowed: bool | None = None  # None == no verdict == gated out

    def total_price_per_mtok(self) -> Decimal:
        """Combined in+out price as the cheapness ordering key.

        Equal-weighting in+out is intentional in v0 — we don't know the
        expected input/output ratio per task yet. A future refinement can
        weight by a learned per-task ratio (AIN-246 territory).
        """
        return self.price_in_per_mtok_usd + self.price_out_per_mtok_usd


@dataclass(frozen=True)
class Policy:
    """Per-request policy. Built from request body + agent.spend_policy.

    Per F6: v0 sources policy from request body (routing_hint) and
    agent.spend_policy jsonb. The weighted-λ tenant_routing_policies table
    is NOT read — its weight columns stay unused this session.
    """

    min_quality: Decimal  # 0-1 quality floor on q_prior
    budget_cap_usd: Decimal | None = None  # per-call projected cost cap; None = unlimited
    # C1 caveat: no latency data exists in v0 catalog. Field carried so the
    # signature is forward-compatible; the brain does not act on it today.
    latency_cap_ms: int | None = None
    policy_name: str = "default"


@dataclass(frozen=True)
class RoutingRequest:
    """Inputs derived from the inference request for deterministic replay.

    The brain itself does not estimate cost from `messages`; the caller
    provides `projected_cost_in_mtok` / `projected_cost_out_mtok` *or*
    leaves them None and the brain uses `Candidate.total_price_per_mtok()`
    as the cheapness key directly. Keeping cost projection in the caller
    matches how the api today estimates input tokens via
    `services/routing._estimate_input_tokens` (universal, no per-provider
    tokenizer dep).
    """

    request_id: str
    agent_id: str
    estimated_input_tokens: int
    reserved_max_tokens: int


@dataclass(frozen=True)
class CandidateOutcome:
    """Per-candidate decision audit, included in Decision.candidates."""

    model_id: str
    model_slug: str
    brand_slug: str
    q_prior: Decimal | None
    price_in_per_mtok_usd: Decimal
    price_out_per_mtok_usd: Decimal
    m_allowed: bool | None
    projected_cost_usd: Decimal | None
    drop_reason: DropReason | None  # None == won
    rank: int | None = None  # 0 == winner; None == dropped pre-ranking


@dataclass(frozen=True)
class Decision:
    """Full audit-trail return: who won, who was considered, why.

    The `rule_fired` field names the branch the brain took:
      · cheapest_clearing_floor      — happy path
      · m_allowed_veto_applied       — winner is next-best after a veto
      · no_candidate_clears_floor    — NT1 reject path (no survivor)
      · no_candidate_enrolled        — every input failed the 3 gates
    """

    rule_fired: str
    chosen: CandidateOutcome | None  # None on reject paths
    candidates: tuple[CandidateOutcome, ...]
    policy_name: str
    policy_semver: str
    ruleset_hash: str  # short hex digest of the policy + ordering rules

    @property
    def is_reject(self) -> bool:
        return self.chosen is None

    def fallback_order(self) -> tuple[CandidateOutcome, ...]:
        """Survivors in pick order (chosen first), for caller's 5xx fallback loop.

        Empty on reject paths. Drops are excluded — only candidates that
        cleared every gate and floor appear here.
        """
        if self.is_reject:
            return ()
        ranked = sorted(
            (c for c in self.candidates if c.rank is not None),
            key=lambda c: c.rank if c.rank is not None else 0,
        )
        return tuple(ranked)


__all__ = [
    "Candidate",
    "CandidateOutcome",
    "Decision",
    "DropReason",
    "Policy",
    "RoutingRequest",
]
# field is exported only so dataclass(field=...) callers can reuse it
_ = field  # silence ruff F401 — kept for future field(default_factory=...) usage
