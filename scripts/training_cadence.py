#!/usr/bin/env python3
"""AIN-335 Stage 4 · daily training cadence orchestrator (offline, DB-free).

Chains the dependency-light routing/ pipeline the Spark cron owns:

    routing_outcomes dump (JSON)
      -> export_outcomes.project_rows
      -> refit_policy (candidate, --no-flip-active)
      -> promotion_gate (+ optional replay_gate bundle)
      -> flip ACTIVE only when gates pass and --apply-promote is set

The judge sweep (step 2) and DB INSERT of ``training_runs`` stay outside
this repo — same wall as export_outcomes.py. This script emits
``training_run.json`` for a thin psql/Supabase insert step.

Example (Spark @ 03:30 WIB after judge labels land):

    python3 scripts/training_cadence.py run \\
        --rows /var/ainfera/dumps/routing_outcomes-$(date -u +%F).json \\
        --workdir /var/ainfera/cadence/$(date -u +%F) \\
        --replay-bundle docs/replay-gate-bundle-2026-06-09-clean.json \\
        --apply-promote

Dry-run (default): evaluates gates, writes artifacts, does NOT flip ACTIVE.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


export_outcomes = _load_module("export_outcomes", _ROOT / "scripts" / "export_outcomes.py")
promotion_gate = _load_module("promotion_gate", _ROOT / "scripts" / "promotion_gate.py")
refit_policy = _load_module("refit_policy", _ROOT / "scripts" / "refit_policy.py")
replay_gate = _load_module("replay_gate", _ROOT / "scripts" / "replay_gate.py")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("rows file must be a JSON list")
    return raw


def _incumbent_artifact(policies_dir: Path) -> Path | None:
    active = policies_dir / "ACTIVE.json"
    if not active.is_file():
        return None
    version = json.loads(active.read_text(encoding="utf-8"))["version"]
    artifact = policies_dir / f"{version}.json"
    if not artifact.is_file():
        # ACTIVE.json points at a deleted/missing version artifact — broken
        # state, not a cold start. Fail closed so the gate cannot be bypassed
        # by clobbering or losing the incumbent file.
        raise RuntimeError(
            f"ACTIVE.json points at missing artifact {artifact.name}; "
            "refusing cold-start fallback. Restore the policy or reset ACTIVE."
        )
    return artifact


def _flip_active(policies_dir: Path, version: str) -> str | None:
    active = policies_dir / "ACTIVE.json"
    prev = json.loads(active.read_text())["version"] if active.is_file() else None
    active.write_text(json.dumps({"version": version}, indent=2) + "\n")
    history = policies_dir / "HISTORY.jsonl"
    with history.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "event": "promote",
                    "from": prev,
                    "to": version,
                    "at": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                },
                sort_keys=True,
            )
            + "\n"
        )
    return prev


def _latest_candidate(policies_dir: Path) -> Path:
    candidates = sorted(policies_dir.glob("policy-*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("refit produced no candidate artifact")
    return candidates[-1]


def _evaluate_promotion(
    *,
    incumbent_path: Path | None,
    candidate_version: str,
    cand_raw: dict[str, Any],
    obs_path: Path,
    observations_count: int,
    source: str,
    judge_model: str,
) -> tuple[dict[str, Any], bool, bool]:
    """Return (gate_row, promotion_passed, cold_start)."""
    cold_start = incumbent_path is None
    if cold_start:
        gate_row = promotion_gate.build_training_run_row(
            {
                "replay_gate_passed": True,
                "promote_reason": "cold_start_no_incumbent",
                "delta_done_rate": None,
                "per_cell": {},
            },
            judge_model=judge_model,
            cadence="daily",
            source=source,
            policy_version_from="none",
            policy_version_to=candidate_version,
            outcomes_judged=observations_count,
            exploration_floor=cand_raw.get("knobs", {}).get("exploration_floor"),
            ruleset_hash=cand_raw.get("ruleset_hash"),
        )
        return gate_row, source == "prod", True

    assert incumbent_path is not None
    obs_for_gate = promotion_gate._load_observations(obs_path)
    inc_raw = json.loads(incumbent_path.read_text(encoding="utf-8"))
    incumbent = promotion_gate._consumer_from_state(inc_raw.get("state", inc_raw))
    candidate = promotion_gate._consumer_from_state(cand_raw.get("state", cand_raw))
    gate = promotion_gate.evaluate_gate(
        incumbent,
        candidate,
        obs_for_gate,
        incumbent_ruleset_hash=inc_raw.get("ruleset_hash"),
        candidate_ruleset_hash=cand_raw.get("ruleset_hash"),
    )
    gate_row = promotion_gate.build_training_run_row(
        gate,
        judge_model=judge_model,
        cadence="daily",
        source=source,
        policy_version_from=inc_raw.get("version", "unknown"),
        policy_version_to=candidate_version,
        outcomes_judged=observations_count,
        exploration_floor=cand_raw.get("knobs", {}).get("exploration_floor"),
        ruleset_hash=cand_raw.get("ruleset_hash"),
    )
    return gate_row, bool(gate_row["promoted"]), False


def run_cadence(
    *,
    rows_path: Path,
    workdir: Path,
    policies_dir: Path,
    source: str = "prod",
    judge_model: str = "claude-opus-4-7",
    replay_bundle_path: Path | None = None,
    apply_promote: bool = False,
    min_observations: int = 1,
) -> dict[str, Any]:
    """Execute export -> refit -> gate -> optional promote. Returns summary dict."""
    workdir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(rows_path)
    observations = export_outcomes.project_rows(rows, source=source)
    if len(observations) < min_observations:
        return {
            "verdict": "HOLD",
            "reason": f"insufficient_observations:{len(observations)}<{min_observations}",
            "outcomes_judged": len(observations),
        }

    obs_path = workdir / "observations.json"
    obs_path.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")

    refit_policy.POLICIES_DIR = policies_dir
    refit_policy.ACTIVE = policies_dir / "ACTIVE.json"
    refit_policy.HISTORY = policies_dir / "HISTORY.jsonl"

    rc = refit_policy.main(
        [
            "refit",
            "--observations",
            str(obs_path),
            "--source",
            source,
            "--no-flip-active",
        ]
    )
    if rc != 0:
        raise RuntimeError(f"refit_policy exited {rc}")

    candidate_path = _latest_candidate(policies_dir)
    candidate_version = candidate_path.stem
    cand_raw = json.loads(candidate_path.read_text(encoding="utf-8"))

    gate_row, promotion_passed, cold_start = _evaluate_promotion(
        incumbent_path=_incumbent_artifact(policies_dir),
        candidate_version=candidate_version,
        cand_raw=cand_raw,
        obs_path=obs_path,
        observations_count=len(observations),
        source=source,
        judge_model=judge_model,
    )

    replay_verdict: str | None = None
    if replay_bundle_path is not None:
        bundle = json.loads(replay_bundle_path.read_text(encoding="utf-8"))
        replay_verdict, _results, _tally = replay_gate.evaluate(bundle)
        (workdir / "replay_gate.json").write_text(
            json.dumps({"overall": replay_verdict}, indent=2) + "\n", encoding="utf-8"
        )

    replay_ok = replay_verdict is None or replay_verdict == "PASS"
    should_promote = promotion_passed and replay_ok and source == "prod"
    promoted = False
    previous_active: str | None = None

    if should_promote and apply_promote:
        previous_active = _flip_active(policies_dir, candidate_version)
        promoted = True
        gate_row["promoted"] = True
        gate_row["promote_reason"] = "cadence_applied_promote"
    elif should_promote:
        # Dry run: gates passed but operator did not pass --apply-promote.
        # Force promoted=false in the emitted DB row so the downstream insert
        # cannot claim a flip that did not happen.
        gate_row["promoted"] = False
        gate_row["promote_reason"] = "gates_passed_apply_promote_not_set"
    elif promotion_passed and not replay_ok:
        # Promotion gate passed but the optional replay bundle held the line.
        # build_training_run_row had already set promoted=true from the
        # promotion gate alone; correct it so the emitted row matches cadence.
        gate_row["promoted"] = False
        gate_row["promote_reason"] = f"replay_gate_failed:{replay_verdict}"

    training_run_path = workdir / "training_run.json"
    training_run_path.write_text(json.dumps(gate_row, indent=2) + "\n", encoding="utf-8")

    if promoted:
        verdict = "PROMOTED"
    elif should_promote:
        verdict = "PROMOTE_READY"
    else:
        verdict = "HOLD"

    summary = {
        "verdict": verdict,
        "promoted": promoted,
        "promotion_gate_passed": promotion_passed,
        "replay_gate_verdict": replay_verdict,
        "candidate_version": candidate_version,
        "previous_active": previous_active,
        "outcomes_judged": len(observations),
        "cold_start": cold_start,
        "training_run_path": str(training_run_path),
        "observations_path": str(obs_path),
        "candidate_path": str(candidate_path),
    }
    (workdir / "cadence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AIN-335 Stage 4 training cadence")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="export -> refit -> gate -> optional promote")
    run.add_argument("--rows", required=True, help="raw routing_outcomes JSON dump")
    run.add_argument("--workdir", required=True, help="per-run artifact directory")
    run.add_argument(
        "--policies-dir",
        default=None,
        help="policy artifact dir (default: $AINFERA_POLICIES_DIR or ./policies)",
    )
    run.add_argument("--source", default="prod", choices=("prod", "synthetic"))
    run.add_argument("--judge-model", default="claude-opus-4-7")
    run.add_argument("--replay-bundle", help="optional replay_gate bundle JSON")
    run.add_argument(
        "--apply-promote",
        action="store_true",
        help="flip ACTIVE when gates pass (founder-gated; default is dry-run)",
    )
    run.add_argument("--min-observations", type=int, default=1)

    args = p.parse_args(argv)
    policies_dir = Path(args.policies_dir or refit_policy.POLICIES_DIR)
    replay_bundle = Path(args.replay_bundle) if args.replay_bundle else None

    try:
        summary = run_cadence(
            rows_path=Path(args.rows),
            workdir=Path(args.workdir),
            policies_dir=policies_dir,
            source=args.source,
            judge_model=args.judge_model,
            replay_bundle_path=replay_bundle,
            apply_promote=args.apply_promote,
            min_observations=args.min_observations,
        )
    except (ValueError, RuntimeError, SystemExit) as exc:
        # export_outcomes.project_rows raises SystemExit for INVARIANT 1
        # (mixed sources). Catch it here so the cron wrapper sees a uniform
        # "cadence failed: ..." + exit code 2 instead of a SystemExit traceback.
        print(f"cadence failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
