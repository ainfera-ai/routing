#!/usr/bin/env python3
"""AIN-335 Stage 2 - promotion gate + training_runs row builder.

The missing middle of the cadence. ``refit_policy.py`` mints a versioned
candidate policy and (today) flips ACTIVE unconditionally. That is unsafe
for prod: a refit must only be promoted if it **beats the incumbent on a
replay of the same observations**. This module is that gate.

## What it does (pure, RNG-free, DB-free)

Given the incumbent policy state, a candidate policy state, and the
observation list both were/*would be* trained on, it:

1. Replays each policy's learned q_empirical against every (cell, model)
   present in the observations.
2. Computes per-cell deltas (candidate - incumbent) in mean reward, plus a
   volume-weighted aggregate ``delta_done_rate``.
3. Applies the gate predicate (see GATE below) -> ``replay_gate_passed``.
4. Emits a dict shaped EXACTLY like a ``public.training_runs`` row, ready
   for a thin INSERT step (the cadence/founder owns the actual DB write -
   routing/ carries no DB driver, same wall as export_outcomes.py).

## GATE predicate (Discipline #12 - methodology)

A candidate is promotable iff ALL hold:
  G1  no per-cell regression worse than ``--max-regression`` (default 0.02)
      on any cell with >= ``--min-cell-n`` labeled samples (default 10);
  G2  aggregate volume-weighted ``delta_done_rate`` >= ``--min-delta``
      (default 0.0 - "do no harm": ties allowed so a cell-rescue with
      neutral aggregate still promotes);
  G3  ruleset_hash unchanged (the decision rule itself must not have moved
      under the policy - a rule change is a separate, gated event).

Cells with < min-cell-n samples are reported but excluded from G1/G2 (thin
single-arm cells must not gate - see AIN-335 Stage-0 limit).

INVARIANT 1 (mirrors refit_policy + export_outcomes): a candidate refit
from ``source=synthetic`` must never be promoted to the prod serving slot.
``--source`` is recorded; a synthetic source forces ``promoted=false`` with
reason ``synthetic_source_never_promotes`` regardless of the gate.

## Usage

    promotion_gate.py \
        --incumbent policies/<active>.json \
        --candidate policies/<candidate>.json \
        --observations observations.json \
        --judge-model claude-opus-4-7 \
        --cadence daily --source prod \
        [--out training_run.json]

Prints the training_runs row dict as JSON (stdout) + a human summary
(stderr). Exit 0 = gate evaluated (check ``promoted`` in the row); exit 2 =
bad input. NO ACTIVE flip here - promotion is the caller's explicit step,
performed only when the emitted row has ``promoted=true``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ainfera_routing.learning import LinUCBConsumer, Observation


def _load_state(path: Path) -> dict[str, Any]:
    """Load a refit_policy artifact and return its serialized consumer state."""
    raw = json.loads(path.read_text())
    # refit_policy artifact nests the state under "state"; tolerate a bare
    # serialized state too.
    return raw.get("state", raw)


def _consumer_from_state(state: dict[str, Any]) -> LinUCBConsumer:
    """Rehydrate a LinUCBConsumer from a serialize() dict."""
    from ainfera_routing.learning import CellModelStats  # noqa: PLC0415

    c = LinUCBConsumer(
        alpha=Decimal(str(state.get("alpha", "1.0"))),
        exploration_floor=Decimal(str(state.get("exploration_floor", "0.05"))),
        decay_half_life=state.get("decay_half_life"),
    )
    for cell, models in state.get("cells", {}).items():
        for slug, s in models.items():
            cms = CellModelStats()
            cms.A = Decimal(str(s["A"]))
            cms.b = Decimal(str(s["b"]))
            cms.n = int(s["n"])
            c.state.setdefault(cell, {})[slug] = cms
    return c


def _q(consumer: LinUCBConsumer, cell: str, slug: str) -> Decimal | None:
    try:
        return consumer.q_empirical(cell, slug)
    except Exception:
        return None


def evaluate_gate(
    incumbent: LinUCBConsumer,
    candidate: LinUCBConsumer,
    observations: list[Observation],
    *,
    min_cell_n: int = 10,
    max_regression: Decimal = Decimal("0.02"),
    min_delta: Decimal = Decimal("0.0"),
    incumbent_ruleset_hash: str | None = None,
    candidate_ruleset_hash: str | None = None,
) -> dict[str, Any]:
    """Pure gate evaluation -> {replay_gate_passed, per_cell, deltas, ...}."""
    # Group the chosen-arm observations by (cell, model) and tally n.
    n_by: dict[tuple[str, str], int] = defaultdict(int)
    for o in observations:
        n_by[(o.cell, o.model_slug)] += 1

    per_cell: dict[str, Any] = {}
    weighted_num = Decimal("0")
    weighted_den = Decimal("0")
    worst_regression = Decimal("0")
    regressing_cells: list[str] = []

    for (cell, slug), n in sorted(n_by.items()):
        qi = _q(incumbent, cell, slug)
        qc = _q(candidate, cell, slug)
        delta = None
        if qi is not None and qc is not None:
            delta = qc - qi
        entry = per_cell.setdefault(cell, {"models": {}, "n_cell": 0})
        entry["models"][slug] = {
            "n": n,
            "q_incumbent": str(qi) if qi is not None else None,
            "q_candidate": str(qc) if qc is not None else None,
            "delta": str(delta) if delta is not None else None,
            "counted": n >= min_cell_n,
        }
        entry["n_cell"] += n
        if n >= min_cell_n and delta is not None:
            weighted_num += delta * Decimal(n)
            weighted_den += Decimal(n)
            if delta < -max_regression:
                regressing_cells.append(cell)
                worst_regression = min(worst_regression, delta)

    delta_done_rate = (weighted_num / weighted_den) if weighted_den > 0 else None

    # GATE predicate
    g1 = not regressing_cells  # no counted-cell regression beyond tolerance
    g2 = (delta_done_rate is not None) and (delta_done_rate >= min_delta)
    g3 = (
        incumbent_ruleset_hash is not None
        and candidate_ruleset_hash is not None
        and incumbent_ruleset_hash == candidate_ruleset_hash
    )
    passed = bool(g1 and g2 and g3)

    if delta_done_rate is None:
        reason = "no_comparable_cells_meeting_min_n"
    elif not g3:
        reason = "ruleset_hash_mismatch"
    elif not g1:
        reason = f"regression_in_cells:{','.join(regressing_cells)}"
    elif not g2:
        reason = f"aggregate_delta_below_min:{delta_done_rate}"
    else:
        reason = "passed_all_gates"

    return {
        "replay_gate_passed": passed,
        "promote_reason": reason,
        "delta_done_rate": str(delta_done_rate) if delta_done_rate is not None else None,
        "per_cell": per_cell,
        "gates": {"g1_no_regression": g1, "g2_aggregate_delta": g2, "g3_ruleset_stable": g3},
        "worst_regression": str(worst_regression),
        "counted_volume": int(weighted_den),
    }


def build_training_run_row(
    gate: dict[str, Any],
    *,
    judge_model: str,
    cadence: str,
    source: str,
    policy_version_from: str,
    policy_version_to: str,
    outcomes_judged: int,
    exploration_floor: str | None,
    ruleset_hash: str | None,
) -> dict[str, Any]:
    """Shape the gate result into a public.training_runs row dict."""
    passed = bool(gate["replay_gate_passed"])
    # INVARIANT 1: synthetic source can never reach the prod serving slot.
    if source != "prod":
        promoted = False
        reason = "synthetic_source_never_promotes"
    else:
        promoted = passed
        reason = gate["promote_reason"]
    return {
        "cadence": cadence,
        "judge_model": judge_model,
        "outcomes_judged": outcomes_judged,
        "policy_version_from": policy_version_from,
        "policy_version_to": policy_version_to,
        "promoted": promoted,
        "promote_reason": reason,
        "delta_done_rate": gate["delta_done_rate"],
        # cost delta wired when candidate carries cost; not in q-only replay
        "delta_cost_usd": None,
        "replay_gate_passed": passed,
        "per_cell": gate["per_cell"],
        "exploration_floor": exploration_floor,
        "ruleset_hash": ruleset_hash,
    }


def _load_observations(path: Path) -> list[Observation]:
    raw = json.loads(path.read_text())
    return [
        Observation(
            cell=o["cell"],
            model_slug=o["model_slug"],
            reward=Decimal(str(o["reward"])),
            policy_version=o.get("policy_version", "v0"),
            tick=int(o.get("tick", 0)),
        )
        for o in raw
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AIN-335 Stage 2 promotion gate + training_runs row")
    p.add_argument("--incumbent", required=True, help="incumbent (ACTIVE) policy artifact json")
    p.add_argument("--candidate", required=True, help="candidate refit policy artifact json")
    p.add_argument(
        "--observations", required=True, help="observations.json (export_outcomes output)"
    )
    p.add_argument("--judge-model", required=True)
    p.add_argument("--cadence", default="daily")
    p.add_argument("--source", default="prod")
    p.add_argument("--min-cell-n", type=int, default=10)
    p.add_argument("--max-regression", default="0.02")
    p.add_argument("--min-delta", default="0.0")
    p.add_argument("--out", help="write the training_runs row json here; default stdout")
    args = p.parse_args(argv)

    if args.source not in ("prod", "synthetic"):
        print(f"INVARIANT 1: --source must be prod|synthetic, got {args.source!r}", file=sys.stderr)
        return 2

    inc_raw = json.loads(Path(args.incumbent).read_text())
    cand_raw = json.loads(Path(args.candidate).read_text())
    incumbent = _consumer_from_state(inc_raw.get("state", inc_raw))
    candidate = _consumer_from_state(cand_raw.get("state", cand_raw))
    observations = _load_observations(Path(args.observations))

    gate = evaluate_gate(
        incumbent,
        candidate,
        observations,
        min_cell_n=args.min_cell_n,
        max_regression=Decimal(str(args.max_regression)),
        min_delta=Decimal(str(args.min_delta)),
        incumbent_ruleset_hash=inc_raw.get("ruleset_hash"),
        candidate_ruleset_hash=cand_raw.get("ruleset_hash"),
    )

    row = build_training_run_row(
        gate,
        judge_model=args.judge_model,
        cadence=args.cadence,
        source=args.source,
        policy_version_from=inc_raw.get("version", "unknown"),
        policy_version_to=cand_raw.get("version", "unknown"),
        outcomes_judged=len(observations),
        exploration_floor=cand_raw.get("knobs", {}).get("exploration_floor"),
        ruleset_hash=cand_raw.get("ruleset_hash"),
    )

    payload = json.dumps(row, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(payload)
    else:
        sys.stdout.write(payload)

    summary = [
        f"gate: {'PASS' if gate['replay_gate_passed'] else 'FAIL'} ({gate['promote_reason']})",
        f"promoted: {row['promoted']}",
        f"delta_done_rate: {gate['delta_done_rate']}  counted_volume: {gate['counted_volume']}",
        f"{args.incumbent} -> {args.candidate}",
    ]
    print("\n".join(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
