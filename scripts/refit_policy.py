#!/usr/bin/env python3
"""AIN-246 · refit → versioned LinUCB policy + one-command rollback.

A *versioned policy* is the serialized `LinUCBConsumer` state (the per-(cell,
model) learned means the brain consults as `q_empirical`) plus provenance
metadata. This tool produces those artifacts deterministically and flips an
`ACTIVE` pointer — it is the offline half of the cadence.

It does **not** touch prod: no DB writes, no live policy mutation. The brain
only ever consults `q_empirical` when a caller explicitly loads a policy and
passes it to `decide()`. Promotion of a refit to the live serving slot is a
separate, founder-gated step (and INVARIANT 1: a policy refit from
`source=synthetic` observations must never become the prod policy).

## Commands

    refit  --observations obs.json [--source prod|synthetic] [--alpha A]
           [--exploration-floor F] [--decay-half-life T]
        → writes policies/policy-<ts>-<hash8>.json, repoints ACTIVE, prints the
          new version. Deterministic: same obs + same knobs → same hash8.

    rollback [--to <version> | --previous]
        → repoints ACTIVE to a prior version. Prints {from, to}. The audit log
          (policies/HISTORY.jsonl) records every refit + rollback.

    show    → prints the ACTIVE version + its metadata.

Observations JSON: a list of {cell, model_slug, reward, policy_version, tick}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Allow running as a plain script (no install) by adding the package root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ainfera_routing.decide import ruleset_hash
from ainfera_routing.learning import Observation, replay

# Overridable so the cadence can target a SHADOW directory and tests can
# isolate. Defaults to <routing>/policies.
POLICIES_DIR = Path(
    os.environ.get("AINFERA_POLICIES_DIR", str(Path(__file__).resolve().parents[1] / "policies"))
)
ACTIVE = POLICIES_DIR / "ACTIVE.json"
HISTORY = POLICIES_DIR / "HISTORY.jsonl"


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


def _audit(event: dict[str, object]) -> None:
    POLICIES_DIR.mkdir(exist_ok=True)
    with HISTORY.open("a") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def _set_active(version: str) -> str | None:
    prev = json.loads(ACTIVE.read_text())["version"] if ACTIVE.exists() else None
    ACTIVE.write_text(json.dumps({"version": version}, indent=2) + "\n")
    return prev


def cmd_refit(args: argparse.Namespace) -> int:
    if args.source not in ("prod", "synthetic"):
        print(f"🔴 --source must be prod|synthetic, got {args.source!r}", file=sys.stderr)
        return 2
    obs = _load_observations(Path(args.observations))
    consumer = replay(
        obs,
        alpha=Decimal(str(args.alpha)),
        exploration_floor=Decimal(str(args.exploration_floor)),
        decay_half_life=args.decay_half_life,
    )
    state_json = consumer.to_json()
    hash8 = hashlib.sha256(state_json.encode()).hexdigest()[:8]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    version = f"policy-{ts}-{hash8}"
    artifact = {
        "version": version,
        "source": args.source,  # INVARIANT 1: prod-promote requires source==prod
        "n_observations": len(obs),
        "ruleset_hash": ruleset_hash(),
        "knobs": {
            "alpha": str(args.alpha),
            "exploration_floor": str(args.exploration_floor),
            "decay_half_life": args.decay_half_life,
        },
        "state": consumer.serialize(),
        "state_hash8": hash8,
        "created_at": ts,
    }
    POLICIES_DIR.mkdir(exist_ok=True)
    (POLICIES_DIR / f"{version}.json").write_text(json.dumps(artifact, indent=2) + "\n")
    prev = _set_active(version)
    _audit({"event": "refit", "version": version, "source": args.source,
            "n_observations": len(obs), "from": prev, "at": ts})
    print(version)
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    versions = sorted(p.stem for p in POLICIES_DIR.glob("policy-*.json"))
    if not versions:
        print("🔴 no policy versions to roll back to", file=sys.stderr)
        return 1
    current = json.loads(ACTIVE.read_text())["version"] if ACTIVE.exists() else None
    if args.to:
        target = args.to
        if target not in versions:
            print(f"🔴 unknown version {target!r}", file=sys.stderr)
            return 1
    else:  # --previous: the newest version that is not the current ACTIVE
        candidates = [v for v in versions if v != current]
        if not candidates:
            print("🔴 no prior version to roll back to", file=sys.stderr)
            return 1
        target = candidates[-1]
    prev = _set_active(target)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    _audit({"event": "rollback", "from": prev, "to": target, "at": ts})
    print(json.dumps({"from": prev, "to": target}))
    return 0


def cmd_show(_: argparse.Namespace) -> int:
    if not ACTIVE.exists():
        print("(no ACTIVE policy)")
        return 0
    version = json.loads(ACTIVE.read_text())["version"]
    meta = json.loads((POLICIES_DIR / f"{version}.json").read_text())
    print(json.dumps({k: meta[k] for k in ("version", "source", "n_observations",
                                            "ruleset_hash", "created_at")}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AIN-246 versioned-policy refit/rollback")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refit", help="refit a versioned policy from observations")
    r.add_argument("--observations", required=True)
    r.add_argument("--source", default="prod")
    r.add_argument("--alpha", default="1.0")
    r.add_argument("--exploration-floor", default="0.05")
    r.add_argument("--decay-half-life", type=int, default=None)
    r.set_defaults(fn=cmd_refit)

    b = sub.add_parser("rollback", help="repoint ACTIVE to a prior version")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--to")
    g.add_argument("--previous", action="store_true")
    b.set_defaults(fn=cmd_rollback)

    s = sub.add_parser("show", help="print the ACTIVE policy metadata")
    s.set_defaults(fn=cmd_show)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
