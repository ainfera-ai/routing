"""Ainfera Routing — the brain (importable decision core).

Public API:
    decide(request, candidates, policy) -> Decision         # v0 (q_prior only)
    LinUCBConsumer / Observation / replay                   # AIN-246 mechanics

AIN-246: the consumer reads labeled `routing_outcomes.reward` and
maintains per-(cell, model) statistics. The brain in `decide.py` is
NOT modified — wiring `q_empirical` into the decision rule is a future
PR gated on founder authorization (Disc #12).
"""

from ainfera_routing.decide import decide
from ainfera_routing.learning import (
    CellModelStats,
    LinUCBConsumer,
    Observation,
    replay,
)
from ainfera_routing.types import (
    Candidate,
    Decision,
    DropReason,
    Policy,
    RoutingRequest,
)

__all__ = [
    "Candidate",
    "CellModelStats",
    "Decision",
    "DropReason",
    "LinUCBConsumer",
    "Observation",
    "Policy",
    "RoutingRequest",
    "decide",
    "replay",
]

# Bumped any time the decision rule, gating, or tiebreak changes.
# Patch = pure refactor, minor = additive rule, major = behavior break.
POLICY_NAME = "quality_floor_then_min_cost"
POLICY_SEMVER = "1.0.0"
