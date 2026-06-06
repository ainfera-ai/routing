#!/usr/bin/env python3
"""AIN-335 - routing_outcomes -> LinUCB observations (model-free bandit cells).

The missing bridge between section-16 capture and the offline refit. Reads
a raw dump of labeled routing_outcomes rows and emits the observations.json
that refit_policy.py consumes.

## The one job: project the bandit-context cell (AIN-335)

routing_outcomes.cell stores the section-16 COVERAGE cell
"{task_type}:{model_slug}:{constraint_band}" - model baked in, correct for
the coverage dashboard. The LinUCB learner compares MODELS WITHIN a cell,
so it needs a model-free BANDIT-CONTEXT cell "{task_type}:{constraint_band}"
with the chosen model as the ARM.

This projector performs exactly that drop. It is the production-data
analogue of what synthetic_coldstart.py already does when it mints
"{task}|synthetic|balanced" cells with every candidate inside. No prod
capture, dashboard, or live-rule change - purely how training observations
are constructed offline.

## Pure by design

routing/ is a dependency-light decision library (no DB driver). So this
script does NOT talk to the database - it reads a raw-rows JSON dump from a
file or stdin. The dump is produced by a thin, documented query step that
the daily cadence (AIN-298, on Spark) or a one-off run owns: a SELECT over
routing_outcomes (task_type, cell, chosen_model_slug, reward,
policy_version, created_at, judge_status, source, fleet_agent,
traffic_origin) emitted as a JSON list. `fleet_agent` drives the
neutrality-rider down-weight (AIN-388, below) and `traffic_origin` the
P2-forward degraded-exclude — include both columns in the dump query.
Pipe that JSON into this projector, then into refit_policy.py:

    export_outcomes.py --rows dump.json --out observations.json
    refit_policy.py refit --observations observations.json --source prod

## INVARIANT 1 (hard wall, mirrors refit_policy)

A prod export must never silently include source='synthetic' rows - a
synthetic-derived policy must never reach the prod serving slot. Mixed
sources abort unless --allow-mixed is passed (offline analysis only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

# ── AIN-388 P0-tail · neutrality rider (down-weight internal-fleet,
# exclude only degraded) ─────────────────────────────────────────────────
#
# A row whose `fleet_agent` is set originated from the internal Valinor
# fleet (tagged off the shared fleet tenant by the api write path, AIN-388 —
# NOT owner_handle). The fleet must never train on its own dogfood at full
# strength, but the pre-launch fleet dogfood IS the seed proof, so we KEEP
# those rows at a reduced weight rather than excluding them. Degraded/MLX
# fallback rows (P2) are the only ones EXCLUDED outright.
#
# FOUNDER-TUNE: the down-weight constant is moat-methodology. It defaults to
# 0.25 (quarter weight) and is overridable without redeploy via
# AINFERA_FLEET_DOWNWEIGHT (same retune idiom as the api ε / fleet-tenant
# envs). Must be in (0, 1]: 1 disables down-weighting, 0 would silently
# exclude (use the degraded path for exclusion, not a zero weight).
_FLEET_DOWNWEIGHT_ENV = "AINFERA_FLEET_DOWNWEIGHT"
_FLEET_DOWNWEIGHT_DEFAULT = 0.25


def fleet_downweight() -> float:
    """Read the internal-fleet down-weight (FOUNDER-TUNE), clamped to (0, 1]."""
    raw = os.environ.get(_FLEET_DOWNWEIGHT_ENV)
    if not raw:
        return _FLEET_DOWNWEIGHT_DEFAULT
    try:
        w = float(raw)
    except ValueError:
        return _FLEET_DOWNWEIGHT_DEFAULT
    if w <= 0:
        # A zero/negative weight would erase the seed signal — that is the
        # degraded-EXCLUDE path, not down-weight. Refuse to silently drop.
        return _FLEET_DOWNWEIGHT_DEFAULT
    return min(1.0, w)

# Defensive only: the stored section-16 cell already carries the band as
# its 3rd segment (authoritative). This map reconstructs the band only when
# `cell` is malformed/absent, from the policy_version preset prefix.
_BAND_BY_PRESET: dict[str, str] = {
    "balanced": "balanced",
    "cost_first": "cost",
    "quality_first": "quality",
    "latency_first": "latency",
    "strict": "strict",
    "custom": "weighted",
}


def _band_from_policy_version(policy_version: str | None) -> str:
    """`cost_first@1.0.0+hash` -> `cost`. Defensive fallback only."""
    if not policy_version:
        return "balanced"
    preset = policy_version.split("@", 1)[0]
    return _BAND_BY_PRESET.get(preset, "balanced")


def bandit_cell(*, stored_cell: str | None, task_type: str | None, policy_version: str | None) -> str:
    """Project the section-16 coverage cell down to the model-free bandit cell.

    Primary path: stored cell `task:model:band` -> `task:band` (drop the
    middle model segment). Defensive fallback when the stored cell is
    missing or not 3-part: `{task_type}:{band-from-policy}`.
    """
    if stored_cell:
        parts = stored_cell.split(":")
        if len(parts) == 3:
            return f"{parts[0]}:{parts[2]}"
    tt = task_type or "general"
    return f"{tt}:{_band_from_policy_version(policy_version)}"


def _is_degraded(row: dict[str, Any]) -> bool:
    """True iff a row is a degraded/MLX-fallback outcome (EXCLUDE per the
    neutrality rider). P2-forward: the `degraded` signal does not exist in
    the schema yet, so this is falsy for all current rows. Accept a few
    shapes so the projector is ready the moment P2 lands the column without
    a code change: an explicit ``degraded`` truthy flag, or a
    ``traffic_origin``/``source`` of ``degraded``/``mlx``.
    """
    if row.get("degraded"):
        return True
    for key in ("traffic_origin", "source"):
        val = str(row.get(key) or "").lower()
        if val in ("degraded", "mlx", "mlx-degraded"):
            return True
    return False


def project_rows(
    rows: list[dict[str, Any]],
    *,
    source: str = "prod",
    allow_mixed: bool = False,
) -> list[dict[str, Any]]:
    """Raw routing_outcomes rows -> observation dicts (refit_policy shape).

    Keeps only labeled rows with a numeric reward and a chosen model.
    Filters to `source` unless `allow_mixed`. Deterministic tick ordering
    by created_at so a re-export of the same rows yields identical ticks.

    Neutrality rider (AIN-388 P0-tail): internal-fleet rows (``fleet_agent``
    set) are KEPT but emitted with ``weight = fleet_downweight()`` (< 1) so
    the fleet never trains on its own dogfood at full strength; degraded/MLX
    rows are EXCLUDED outright (never emitted). External/customer rows keep
    ``weight = 1``.
    """
    fleet_w = fleet_downweight()
    seen_sources: set[str] = set()
    kept: list[dict[str, Any]] = []
    for r in rows:
        seen_sources.add(str(r.get("source") or "unknown"))
        if not allow_mixed and r.get("source") != source:
            continue
        reward = r.get("reward")
        model = r.get("chosen_model_slug")
        if reward is None or not model:
            continue
        if r.get("judge_status") not in (None, "labeled"):
            continue
        if _is_degraded(r):
            # Degraded/MLX fallback — excluded, not down-weighted. A
            # degraded backend's reward is not a clean signal of the
            # routed model's quality.
            continue
        kept.append(r)

    if not allow_mixed and (seen_sources & {"synthetic"}) and (seen_sources - {source}):
        raise SystemExit(
            f"INVARIANT 1: mixed sources in a --source={source} export "
            f"(saw {sorted(seen_sources)}). Pass --allow-mixed for offline analysis only."
        )

    kept.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("cell") or "")))
    out: list[dict[str, Any]] = []
    for tick, r in enumerate(kept):
        is_fleet = bool(r.get("fleet_agent"))
        out.append(
            {
                "cell": bandit_cell(
                    stored_cell=r.get("cell"),
                    task_type=r.get("task_type"),
                    policy_version=r.get("policy_version"),
                ),
                "model_slug": r["chosen_model_slug"],
                "reward": float(r["reward"]),
                "policy_version": r.get("policy_version", "v0"),
                "tick": tick,
                # Provenance weight the LinUCB consumer honors at ingest.
                # Internal-fleet → down-weighted (kept); else full weight.
                "weight": fleet_w if is_fleet else 1.0,
            }
        )
    return out


def _summary(observations: list[dict[str, Any]]) -> str:
    per_cell: dict[str, set[str]] = defaultdict(set)
    for o in observations:
        per_cell[o["cell"]].add(o["model_slug"])
    n_fleet = sum(1 for o in observations if float(o.get("weight", 1.0)) < 1.0)
    lines = [
        f"observations: {len(observations)}",
        f"internal-fleet (down-weighted, kept): {n_fleet} @ weight={fleet_downweight()}",
        f"distinct bandit cells: {len(per_cell)}",
        "models per cell (arms available to compare):",
    ]
    for cell in sorted(per_cell):
        arms = sorted(per_cell[cell])
        flag = "  <- single-arm (no comparison until exploration)" if len(arms) == 1 else ""
        lines.append(f"  {cell}: {len(arms)} [{', '.join(arms)}]{flag}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AIN-335 routing_outcomes -> LinUCB observations")
    p.add_argument("--rows", help="raw-rows JSON file (list of dicts); default stdin")
    p.add_argument("--out", help="observations.json output path; default stdout")
    p.add_argument("--source", default="prod", help="source filter (default prod)")
    p.add_argument("--allow-mixed", action="store_true", help="offline analysis only; disables INVARIANT 1")
    args = p.parse_args(argv)

    raw_text = open(args.rows).read() if args.rows else sys.stdin.read()
    rows = json.loads(raw_text)
    if not isinstance(rows, list):
        raise SystemExit("expected a JSON list of row objects")

    observations = project_rows(rows, source=args.source, allow_mixed=args.allow_mixed)

    payload = json.dumps(observations, indent=2) + "\n"
    if args.out:
        open(args.out, "w").write(payload)
    else:
        sys.stdout.write(payload)
    print(_summary(observations), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
