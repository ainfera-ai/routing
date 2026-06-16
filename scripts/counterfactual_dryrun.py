#!/usr/bin/env python3
"""AIN-335 Part B · counterfactual_explore DRY-RUN (demonstration, no serving).

Shows — offline, deterministically, serving nothing — that enabling the inert
``counterfactual_explore`` path on ``reasoning:cost`` *would* create the
mistral↔gpt-5-5 head-to-head holdout the replay-gate's positivity guard requires,
and that with the shipped defaults it does nothing at all.

This is a demonstration script; it imports the same pure functions the (future,
§17-gated) live caller would use. Run:

    python3 scripts/counterfactual_dryrun.py
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ainfera_routing import Candidate, Policy, RoutingRequest, decide
from ainfera_routing.explore import (
    DECISION_RULE,
    eligible_arms,
    select_counterfactual,
)

REASONING_COST = "reasoning:cost"
FLOOR = Decimal("0.88")  # api CELL_MIN_QUALITY[reasoning] (AIN-388)


# Enrolled-5 catalog (replay-gate bundle priors).
def _c(mid: str, slug: str, brand: str, q: str, pin: str, pout: str) -> Candidate:
    return Candidate(mid, slug, brand, Decimal(q), Decimal(pin), Decimal(pout), True)


CATALOG = [
    _c("id1", "claude-opus-4-7", "anthropic", "0.95", "15", "75"),
    _c("id2", "gpt-5-5", "openai", "0.90", "5", "15"),
    _c("id3", "gemini-3-1-pro", "google", "0.87", "1.25", "10"),
    _c("id4", "grok-4", "xai", "0.86", "5", "15"),
    _c("id5", "mistral-large-3", "mistral", "0.74", "2", "6"),
]

# reasoning:cost learned mean on the CLEAN corpus (post-AIN-403): mistral's
# 65.75 reward over 69 succeeded rows ≈ 0.9529, clearing the 0.88 floor its
# 0.74 static prior cannot. Source: routing#16 docs/replay-gate-report-2026-06-09-clean.md
Q_EMPIRICAL = {"mistral-large-3": Decimal("0.9529")}


def main() -> int:
    req = RoutingRequest("dryrun", "fleet-agent", 1000, 1024)
    pol = Policy(min_quality=FLOOR, policy_name="reasoning:cost")

    print("AIN-335 Part B · counterfactual_explore DRY-RUN")
    print(f"cell={REASONING_COST}  floor={FLOOR}  (nothing is served — pure demonstration)\n")

    # 1. Greedy decide() — q_prior only, the CURRENT live behaviour.
    greedy = decide(req, CATALOG, pol)
    chosen = greedy.chosen.model_slug if greedy.chosen else "<reject>"
    print(f"1. Greedy decide() (q_prior only, live today): picks {chosen!r}")
    print("   mistral-large-3 is dropped: prior 0.74 < 0.88 floor → never observed here.\n")

    # 2. The arms the gate lacks support for.
    arms = eligible_arms(CATALOG, pol, q_empirical=Q_EMPIRICAL)
    print("2. Counterfactual-eligible (enrolled + floor-dropped + q_emp clears):")
    for c, qe in arms:
        price = c.total_price_per_mtok()
        print(f"   {c.model_slug}: prior {c.q_prior} < {FLOOR} <= q_emp {qe}  (${price}/Mt)")
    print()

    # 3. Inert (shipped defaults) — serves nothing.
    inert = select_counterfactual(
        CATALOG,
        pol,
        cell=REASONING_COST,
        q_empirical=Q_EMPIRICAL,
        kappa=Decimal("0"),
        cells=frozenset(),
        roll=0.0,
    )
    print(f"3. Shipped defaults (kappa=0, cells=empty): -> {inert}  (INERT — greedy stands)\n")

    # 4. Armed (what a founder-gated κ>0 would do).
    armed = select_counterfactual(
        CATALOG,
        pol,
        cell=REASONING_COST,
        q_empirical=Q_EMPIRICAL,
        kappa=Decimal("0.10"),
        cells=frozenset({REASONING_COST}),
        roll=0.05,
    )
    assert armed is not None
    print("4. ARMED (κ=0.10, cells={reasoning:cost}, roll=0.05 — illustrative, NOT enabled):")
    print(
        f"   select_counterfactual → serve {armed.candidate.model_slug!r} "
        f"(decision_rule={DECISION_RULE!r}, exploration=True)"
    )
    print()

    armed_slug = armed.candidate.model_slug
    print("CONCLUSION")
    print(f"  Greedy serves {chosen!r}; armed serves {armed_slug!r} ~kappa of the time.")
    print("  That is exactly the mistral<->gpt-5-5 head-to-head the gate's positivity")
    print("  guard is missing — enabling kappa>0 on this cell would let the next gate")
    print("  re-run CERTIFY or REFUTE the flip on real, current-conditions overlap.")
    print("  Default config changes nothing. κ>0 is a §17 amendment + founder gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
