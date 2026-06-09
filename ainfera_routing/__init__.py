"""Ainfera Routing — the brain (importable decision core).

Public API:
    decide(request, candidates, policy) -> Decision         # v0 (q_prior only)
    LinUCBConsumer / Observation / replay                   # AIN-246 mechanics
    select_counterfactual / eligible_arms                   # AIN-335 Part B (INERT)

AIN-246: the consumer reads labeled `routing_outcomes.reward` and
maintains per-(cell, model) statistics. The brain in `decide.py` is
NOT modified — wiring `q_empirical` into the decision rule is a future
PR gated on founder authorization (Disc #12).

AIN-335 Part B: `select_counterfactual` is an INERT, pure exploration helper
(default κ=0, cells=∅ → returns None → behaviour identical to pre-Part-B). It
does not modify `decide.py`. Enabling it (κ>0) is a §17 amendment + founder gate.
"""

from ainfera_routing.decide import decide
from ainfera_routing.explore import (
    CounterfactualPick,
    cells_from_env,
    eligible_arms,
    kappa_from_env,
    select_counterfactual,
)
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
    "CounterfactualPick",
    "Decision",
    "DropReason",
    "LinUCBConsumer",
    "Observation",
    "Policy",
    "RoutingRequest",
    "cells_from_env",
    "decide",
    "eligible_arms",
    "kappa_from_env",
    "replay",
    "select_counterfactual",
]

# Bumped any time the decision rule, gating, or tiebreak changes.
# Patch = pure refactor, minor = additive rule, major = behavior break.
POLICY_NAME = "quality_floor_then_min_cost"
POLICY_SEMVER = "1.0.0"
