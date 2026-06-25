#!/usr/bin/env python3
"""AIN-335 - routing_outcomes -> LinUCB observations (model-free bandit cells).

The missing bridge between section-16 capture and the offline refit. Reads
a raw dump of labeled routing_outcomes rows and emits the observations.json
that refit_policy.py consumes.

## The one job: project the bandit-context cell (AIN-335)

routing_outcomes.cell stores the section-16 COVERAGE cell
"{task_type}:{model_slug}:{constraint_band}" - model baked in, correct for
the coverage dashboard. The LinUCB learner compares MODELS WITHIN a cell,
so it needs a model-free BANDIT-CONTEXT cell
"{task_type}:{tenant_id}:{constraint_band}" (AIN-602/AIN-550 canonical) with
the chosen model as the ARM.

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
policy_version, created_at, judge_status, source, tenant_id, fleet_agent,
traffic_origin, traffic_class) emitted as a JSON list. `tenant_id` drives
the neutrality-rider down-weight (AIN-391 §2a / AIN-388, below — the
authoritative key), `fleet_agent` is now only a transitional fallback,
`traffic_origin` the P2-forward degraded-exclude, and `traffic_class`
(AIN-424) the authoritative synthetic-probe exclude — include all four
columns in the dump query.
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
from pathlib import Path
from typing import Any

# ── AIN-388 P0-tail · neutrality rider (down-weight internal-fleet,
# exclude only degraded) ─────────────────────────────────────────────────
#
# An internal-fleet row is keyed off the row's `tenant_id` matching the
# shared fleet tenant (AIN-391 §2a) — the SAME keystone the api write path
# tags on (services/routing_brain._is_fleet_agent), NOT owner_handle, and no
# longer the `fleet_agent` tag. Keying on the tag left a gap: a fleet row
# whose per-agent tag was never written (NULL fleet_agent) leaked into the
# moat dataset at FULL weight. The fleet must never train on its own dogfood
# at full strength, but the pre-launch fleet dogfood IS the seed proof, so we
# KEEP those rows at a reduced weight rather than excluding them. Degraded/MLX
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


# ── AIN-391 §2a · key the down-weight off `tenant_id` (retire fleet_agent) ──
#
# The authoritative fleet signal is the routing_outcomes row's `tenant_id`
# matching the shared fleet tenant — byte-identical keystone + idiom to the
# api write path (services/routing_brain._FLEET_TENANT_IDS_DEFAULT /
# _csv_id_set / additive AINFERA_FLEET_TENANT_IDS). The default fleet tenant
# is ALWAYS treated as fleet; the env is purely ADDITIVE so a config edit can
# never silently un-tag the live fleet back into moat contamination.
_FLEET_TENANT_IDS_ENV = "AINFERA_FLEET_TENANT_IDS"
_FLEET_TENANT_IDS_DEFAULT = "280f4469-d318-4ec4-9c63-f3ea83466b03"


def _csv_id_set(raw: str | None) -> frozenset[str]:
    """Comma-separated UUID list → lowercased set (UUIDs are canonical-lower)."""
    if not raw:
        return frozenset()
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def _fleet_tenant_ids() -> frozenset[str]:
    """Tenant ids treated as internal fleet: the default constant plus the
    additive env (never a replace — see api idiom)."""
    return _csv_id_set(_FLEET_TENANT_IDS_DEFAULT) | _csv_id_set(
        os.environ.get(_FLEET_TENANT_IDS_ENV)
    )


def is_fleet_row(row: dict[str, Any]) -> bool:
    """True iff a routing_outcomes row is internal-fleet (AIN-391 §2a).

    Authoritative key: the row's ``tenant_id`` is in the fleet-tenant set —
    the SAME signal the api write path tags on, so a fleet row trains
    down-weighted even when its per-agent ``fleet_agent`` tag was never
    written (the gap fleet_agent-keying left open).

    Transitional fallback: a truthy ``fleet_agent`` still marks a row as
    fleet, so dumps emitted before ``tenant_id`` was added to the SELECT keep
    their down-weight. fleet_agent is no longer the primary key; this fallback
    retires once every dump carries ``tenant_id``.
    """
    tid = str(row.get("tenant_id") or "").strip().lower()
    if tid and tid in _fleet_tenant_ids():
        return True
    return bool(row.get("fleet_agent"))


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


def bandit_cell(
    *,
    stored_cell: str | None,
    task_type: str | None,
    policy_version: str | None,
    tenant_id: str | None,
) -> str:
    """Project the section-16 coverage cell down to the model-free bandit cell.

    AIN-602/AIN-550 canonical cell = ``{task_type}:{tenant_id}:{constraint_band}`` (the MODEL
    is the arm, the TENANT is part of the context). Primary path: stored cell
    ``task:model:band`` -> ``task:tenant:band`` (drop the middle MODEL segment, insert the
    tenant). Defensive fallback when the stored cell is missing or not 3-part:
    ``{task_type}:{tenant}:{band-from-policy}``. MUST stay byte-identical to the api consumer
    (``policy_artifact.bandit_cell``) + the refit (``labs.labeled_corpus.model_free_cell``)."""
    tid = str(tenant_id) if tenant_id else "unknown"
    if stored_cell:
        parts = stored_cell.split(":")
        if len(parts) == 3:
            return f"{parts[0]}:{tid}:{parts[2]}"
    tt = task_type or "general"
    return f"{tt}:{tid}:{_band_from_policy_version(policy_version)}"


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


def _is_synthetic_probe(row: dict[str, Any]) -> bool:
    """True iff a row is a synthetic health/routing probe (EXCLUDE, not just
    down-weight). AIN-424: the AIN-285 cron probes are synthetic traffic, not
    real task outcomes — their reward is not a clean signal of model quality
    (same rationale as degraded), so they must never train the moat even at a
    reduced weight. The fleet-dogfood down-weight (0.25) is for *real* internal
    work; probes are a different class and are dropped outright.

    Authoritative key: ``traffic_class == 'internal_probe'`` (the closed-
    vocabulary provenance column added in migration 0057). Transitional
    fallback for dumps emitted before ``traffic_class`` was added to the
    SELECT: a ``fleet_agent`` probe label (``routed-probe`` / ``nt1-probe*`` /
    any ``*-probe``). None of the 7 live fleet agents carry 'probe' in their
    name, so the fallback cannot mis-drop a real agent.
    """
    if str(row.get("traffic_class") or "").lower() == "internal_probe":
        return True
    fa = str(row.get("fleet_agent") or "").lower()
    return fa == "routed-probe" or fa.startswith("nt1-probe") or fa.endswith("-probe")


# ── AIN-615 · judge-quality reward (the LEARNED quality signal for the FLOOR) ──────────────
#
# Distinct from the cost-aware `reward` column (B1 = 1{succeeded} * (1 - norm_cost)) that drives
# RANK. The FLOOR needs a quality signal, not a cost-aware one — using cost-aware q̂ for the floor
# caused the 2026-06-23 :quality 422 (AIN-614). The Opus-4.7 judge (AIN-290) scores a 1-5 Likert
# in `judge_score`; normalize to [0,1]. A quality export replays into a SECOND LinUCB consumer →
# quality_q̂ = b_q/A_q, parallel to the cost q̂. PROMOTION of quality_q̂ to the live floor is gated
# on the κ-HOLD (judge-vs-Council trust) clearing — this only assembles the signal.
def quality_reward(row: dict[str, Any]) -> float | None:
    """Normalized judge quality reward in [0,1]: ``(judge_score - 1) / 4`` for the 1-5 Likert,
    or ``None`` when the row is unlabeled or out of range (so a quality export keeps only
    judge-labeled rows). AIN-615."""
    raw = row.get("judge_score")
    if raw is None:
        return None
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return None
    if not 1.0 <= s <= 5.0:
        return None
    return (s - 1.0) / 4.0


def project_rows(
    rows: list[dict[str, Any]],
    *,
    source: str = "prod",
    allow_mixed: bool = False,
    quality: bool = False,
) -> list[dict[str, Any]]:
    """Raw routing_outcomes rows -> observation dicts (refit_policy shape).

    Keeps only labeled rows with a numeric reward and a chosen model.
    Filters to `source` unless `allow_mixed`. Deterministic tick ordering
    by created_at so a re-export of the same rows yields identical ticks.

    AIN-621 authority ALLOWLIST (cost mode): the cost-aware ``reward`` corpus trains ONLY on
    ``reward_source IN ('council','verify')`` — a 'judge' screening opinion (Gemma/Super) or a
    NULL-provenance reward is excluded by construction. CONTRACT: the dump
    (``scripts/dump_routing_outcomes.sql``) MUST SELECT ``reward_source``; a cost dump without it
    raises INVARIANT 2 (fail-loud, never a silent empty corpus).

    ``quality`` (AIN-615): use the normalized judge-quality reward (``quality_reward``) instead
    of the cost-aware ``reward`` column, keeping ONLY judge-labeled rows — the observation stream
    for the parallel quality_q̂ consumer. That stream is judge-based BY DESIGN (κ-gated
    separately), so the authority allowlist above does NOT apply to it. Other rider/exclusion
    filters are identical.

    Neutrality rider (AIN-391 §2a / AIN-388 P0-tail): internal-fleet rows
    (``tenant_id`` in the fleet-tenant set; ``fleet_agent`` fallback for
    legacy dumps) are KEPT but emitted with ``weight = fleet_downweight()``
    (< 1) so the fleet never trains on its own dogfood at full strength;
    degraded/MLX rows and synthetic ``internal_probe`` rows (AIN-424) are
    EXCLUDED outright (never emitted). External/customer rows keep
    ``weight = 1``.
    """
    fleet_w = fleet_downweight()
    seen_sources: set[str] = set()
    saw_reward_source = False  # AIN-621: did the cost dump carry reward_source provenance?
    n_cost_reward_bearing = 0  # cost-mode rows with a usable reward (pre-allowlist)
    kept: list[tuple[dict[str, Any], float]] = []
    for r in rows:
        seen_sources.add(str(r.get("source") or "unknown"))
        if not allow_mixed and r.get("source") != source:
            continue
        reward = quality_reward(r) if quality else r.get("reward")
        model = r.get("chosen_model_slug")
        if reward is None or not model:
            continue
        # Cost export honors the judge_status gate; a quality export uses the judge_score
        # presence (quality_reward != None, above) AS the label gate.
        if not quality and r.get("judge_status") not in (None, "labeled"):
            continue
        # AIN-621 · authority ALLOWLIST for the COST/reward export (refit_policy): the
        # cost-aware reward-of-record trains ONLY on verifiable + Council provenance. A
        # 'judge' screening opinion (Gemma/Super) or a NULL-provenance reward is excluded
        # by construction (an allowlist, not a NOT IN ('judge') denylist that would still
        # admit the null rows). The quality_q̂ stream (quality=True, AIN-615) is the
        # judge-quality learner BY DESIGN and is governed by the κ gate separately — it is
        # NOT allowlisted here.
        if not quality:
            n_cost_reward_bearing += 1
            reward_source = r.get("reward_source")
            if reward_source is not None:
                saw_reward_source = True
            if reward_source not in ("council", "verify"):
                continue
        if _is_degraded(r):
            # Degraded/MLX fallback — excluded, not down-weighted. A
            # degraded backend's reward is not a clean signal of the
            # routed model's quality.
            continue
        if _is_synthetic_probe(r):
            # AIN-424: synthetic health/routing probes are excluded outright
            # (not down-weighted) — they are not real task outcomes. The
            # 0.25 dogfood down-weight is reserved for real internal work.
            continue
        kept.append((r, float(reward)))

    if not allow_mixed and (seen_sources & {"synthetic"}) and (seen_sources - {source}):
        raise SystemExit(
            f"INVARIANT 1: mixed sources in a --source={source} export "
            f"(saw {sorted(seen_sources)}). Pass --allow-mixed for offline analysis only."
        )

    # AIN-621 INVARIANT 2: a cost export whose dump carries reward-bearing rows but NO
    # reward_source provenance predates the authority-allowlist contract — fail LOUD rather
    # than silently emit an empty (un-provenanced) corpus and quietly stall the refit.
    if not quality and n_cost_reward_bearing and not saw_reward_source:
        raise SystemExit(
            "INVARIANT 2 (AIN-621): cost-export dump carries no reward_source — the authority "
            "allowlist (reward_source IN ('council','verify')) would drop every row. Re-dump "
            "with scripts/dump_routing_outcomes.sql (it now SELECTs reward_source)."
        )

    kept.sort(key=lambda rk: (str(rk[0].get("created_at") or ""), str(rk[0].get("cell") or "")))
    out: list[dict[str, Any]] = []
    for tick, (r, reward) in enumerate(kept):
        is_fleet = is_fleet_row(r)
        out.append(
            {
                "cell": bandit_cell(
                    stored_cell=r.get("cell"),
                    task_type=r.get("task_type"),
                    policy_version=r.get("policy_version"),
                    tenant_id=r.get("tenant_id"),
                ),
                "model_slug": r["chosen_model_slug"],
                "reward": reward,
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
    p.add_argument(
        "--allow-mixed", action="store_true", help="offline analysis only; disables INVARIANT 1"
    )
    p.add_argument(
        "--quality",
        action="store_true",
        help="AIN-615: emit the judge-quality reward (normalized judge_score) for judge-labeled "
        "rows only — the quality_q̂ observation stream (the dump must SELECT judge_score)",
    )
    args = p.parse_args(argv)

    raw_text = Path(args.rows).read_text() if args.rows else sys.stdin.read()
    rows = json.loads(raw_text)
    if not isinstance(rows, list):
        raise SystemExit("expected a JSON list of row objects")

    observations = project_rows(
        rows, source=args.source, allow_mixed=args.allow_mixed, quality=args.quality
    )

    payload = json.dumps(observations, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(payload)
    else:
        sys.stdout.write(payload)
    print(_summary(observations), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
