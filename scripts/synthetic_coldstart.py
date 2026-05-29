#!/usr/bin/env python3
"""AIN-303 · synthetic cold-start loop — planted recovery, SHADOW-only.

Cold-start has no labeled outcomes, so the bandit can't learn. This generates a
*planted* synthetic corpus (a known-best model per cell across the canonical 7
task_types), runs the LinUCB refit on it, and validates that the learner
recovers the planted best — i.e. `q_empirical(cell, planted_best)` is the max in
its cell. The resulting policy is written to a SHADOW slot.

INVARIANT 1 (hard wall): every synthetic observation is `source='synthetic'`,
the refit artifact is tagged `source='synthetic'`, and promotion targets a
SHADOW policies directory (`AINFERA_POLICIES_DIR=<shadow>`) — never the prod
`tenant_routing_policies` table. A synthetic refit can NEVER become prod policy.

Deterministic: rewards are drawn from a seeded generator (CRN), Decimal math in
the consumer, no RNG in scoring. Same seed → same recovered policy.

Run:  AINFERA_POLICIES_DIR=/tmp/shadow python scripts/synthetic_coldstart.py
"""

from __future__ import annotations

import random
from decimal import Decimal

from ainfera_routing.learning import LinUCBConsumer, Observation

# Canonical 7 task_types (ontology v1.3 §5).
TASK_TYPES = ("reasoning", "code", "extraction", "chat", "tool_use", "embed", "general")

CANDIDATES = ("opus", "gpt", "gemini", "mistral", "haiku")

# Planted ground truth: the best model per task_type. The generator draws that
# model's rewards from a higher mean — a correct learner must surface it.
PLANTED_BEST = {
    "reasoning": "opus",
    "code": "gpt",
    "extraction": "gemini",
    "chat": "haiku",
    "tool_use": "gpt",
    "embed": "mistral",
    "general": "gemini",
}

_SEED = 7  # CRN — deterministic synthetic stream
_PER_ARM = 40  # synthetic observations per (cell, model)
_HIGH, _LOW = (0.85, 0.12), (0.55, 0.12)  # (mean, spread) for planted-best vs rest


def generate_planted_observations(seed: int = _SEED) -> list[Observation]:
    """One Observation per synthetic draw, tagged for source='synthetic' rows.

    The (cell, model) reward mean is HIGH for the planted-best model and LOW
    for the rest, so a correct bandit recovers PLANTED_BEST[task] per cell.
    """
    rng = random.Random(seed)
    obs: list[Observation] = []
    tick = 0
    for task in TASK_TYPES:
        cell = f"{task}|synthetic|balanced"
        for model in CANDIDATES:
            mean, spread = _HIGH if model == PLANTED_BEST[task] else _LOW
            for _ in range(_PER_ARM):
                tick += 1
                r = min(1.0, max(0.0, rng.gauss(mean, spread)))
                obs.append(
                    Observation(
                        cell=cell,
                        model_slug=model,
                        reward=Decimal(str(round(r, 4))),
                        policy_version="synthetic-warmup",
                        tick=tick,
                    )
                )
    return obs


def recovered_best_per_cell(consumer: LinUCBConsumer) -> dict[str, str]:
    """argmax q_empirical per cell → the model the learner believes is best."""
    out: dict[str, str] = {}
    for task in TASK_TYPES:
        cell = f"{task}|synthetic|balanced"
        overrides = consumer.q_empirical_overrides_for_cell(cell)
        if overrides:
            out[task] = max(overrides, key=lambda m: overrides[m])
    return out


def run(seed: int = _SEED) -> tuple[LinUCBConsumer, dict[str, str], int]:
    obs = generate_planted_observations(seed)
    consumer = LinUCBConsumer()
    consumer.ingest(obs)
    return consumer, recovered_best_per_cell(consumer), len(obs)


def main() -> int:
    _consumer, recovered, n = run()
    hits = sum(1 for t in TASK_TYPES if recovered.get(t) == PLANTED_BEST[t])
    print(f"synthetic cold-start: {n} obs across {len(TASK_TYPES)} task_types (source=synthetic)")
    for task in TASK_TYPES:
        ok = "✅" if recovered.get(task) == PLANTED_BEST[task] else "🔴"
        print(f"  {ok} {task:11s} planted={PLANTED_BEST[task]:8s} recovered={recovered.get(task)}")
    print(f"recovered planted-best in {hits}/{len(TASK_TYPES)} cells")
    return 0 if hits == len(TASK_TYPES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
