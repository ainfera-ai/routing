"""Ainfera Routing — the brain (importable decision core).

Public API:
    decide(request, candidates, policy) -> Decision

The decision is pure, deterministic, and N-agnostic. v0 uses q_prior only;
q_empirical lands in v1 (AIN-246). See methodology v1.2 §D for the rule.
"""

from ainfera_routing.decide import decide
from ainfera_routing.types import (
    Candidate,
    Decision,
    DropReason,
    Policy,
    RoutingRequest,
)

__all__ = [
    "Candidate",
    "Decision",
    "DropReason",
    "Policy",
    "RoutingRequest",
    "decide",
]

# Bumped any time the decision rule, gating, or tiebreak changes.
# Patch = pure refactor, minor = additive rule, major = behavior break.
POLICY_NAME = "quality_floor_then_min_cost"
POLICY_SEMVER = "1.0.0"
