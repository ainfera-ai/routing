"""AIN-246 · LinUCB read-back consumer (mechanics-only, brain-untouched).

Reads labeled `routing_outcomes` rows (per AIN-285 capture + AIN-290 judge
labels), maintains per-(cell, model) statistics, and exposes them as a
`q_empirical(cell, model)` override that the brain may consult later.

**The brain in `decide.py` is NOT modified by this module.** Disc #12 forbids
retuning the objective / weights / candidate-set logic; only mechanics live
here. A future PR — gated on founder authorization — wires the brain to
prefer `q_empirical` over `q_prior` when both are present, but that wire-in
is out of scope.

## Algorithm

LinUCB with a degenerate identity context (feature dim d = 1, context vector
constant = [1.0]). With d = 1 the LinUCB update equations collapse to a
running mean + count, plus an upper-confidence-bound bonus:

    mean(cell, model)  = sum_rewards / max(n, 1)
    ucb(cell, model)   = mean + alpha * sqrt(2 * ln(N_cell) / max(n, 1))

The `alpha` knob (default 1.0) is the standard UCB exploration weight.
N_cell is the total count of observations in the cell.

The full LinUCB matrix form is in place (`A`, `b`) so a future PR can add
real context features (task length, latency floor, cost budget, etc.)
without re-architecting the consumer state.

## Exploration floor (A4)

A hard parameter (default 0.05 = 5%) reserves a minimum fraction of
selection probability for the least-explored model in each cell. This
ensures new catalog entries get sampled regardless of how well existing
models score. Hard means no `q_empirical` value, however high, can push
this below the floor. The brain enforces this at selection time by
consulting `exploration_quota(cell)`.

## Decay (A3)

Per-observation weighting via `decay_half_life` (in observation ticks).
An observation t ticks old contributes `0.5 ** (t / half_life)` weight,
applied at ingest time so the running state remains a simple weighted
sum. `decay_half_life = None` disables decay (pure-mean accumulation).

A `catalog_change` event resets the per-cell counts for the affected
model, NOT the mean — so a re-listed model keeps its prior signal but
re-enters the exploration window.

## Provenance weight (AIN-388 P0-tail · neutrality rider)

Each `Observation` carries a `weight` (default 1) that the ingest path
multiplies into the decay weight. Internal-fleet dogfood rows are
DOWN-WEIGHTED (weight < 1) — kept as seed signal, never trained on at full
strength — while external/customer rows keep weight 1. The fleet/customer
distinction and the down-weight constant live in the projector
(`scripts/export_outcomes.py`, `AINFERA_FLEET_DOWNWEIGHT`), not here: this
library stays policy-free and just honors the number. Degraded/MLX rows are
EXCLUDED at projection time (never emitted as observations), not weighted —
"down-weight internal-fleet, exclude only degraded." The default weight of 1
keeps pre-AIN-388 replay state byte-identical.

## Deterministic replay

`Decimal` arithmetic throughout (no float drift), sorted iteration in
`serialize()`, no RNG in the consumer. Same observation stream + same
config = same bytes out. The `replay()` helper rehydrates a consumer
from an observation list.

## Brain hookup contract (NOT in this PR)

When the founder authorizes the brain wire-in, `decide.py` would:

    overrides = consumer.q_empirical_overrides_for_cell(cell)
    effective_q = overrides.get(model_slug) or candidate.q_prior

The brain stays Disc-#12-compliant: same enrolment gates, same floor
rule, same cheapest-survivor pick — only the `q_prior` source changes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Any

# Lock decimal precision so a replay across hosts gives byte-identical
# state. 28 is the Python default but pinning here protects against an
# operator who sets it elsewhere.
getcontext().prec = 28

# ── public types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Observation:
    """One labeled outcome that the consumer ingests.

    `policy_version` is carried for replay determinism — a consumer
    SHOULD refuse to mix observations from incompatible policy versions
    (caller decides; the consumer just records).
    """

    cell: str
    model_slug: str
    reward: Decimal  # normalized [0, 1] from judge or implicit signal
    policy_version: str
    # Monotonic tick used by the decay rule. Caller assigns; the consumer
    # only requires that ticks be non-decreasing within a single stream.
    tick: int = 0
    # Provenance weight (AIN-388 P0-tail · neutrality rider). The caller
    # multiplies the decay weight by this to DOWN-WEIGHT internal-fleet
    # dogfood rows — kept as seed signal, not dropped — so the fleet never
    # trains on its own traffic at full strength. `1` = full weight (an
    # external/customer row). The projector (export_outcomes.py) sets it;
    # the consumer just honors it (no fleet policy lives in this library).
    # Degraded/MLX rows are EXCLUDED upstream (never emitted), not weighted.
    weight: Decimal = Decimal("1")


@dataclass
class CellModelStats:
    """Per-(cell, model) running state. Decimal everywhere for replay
    determinism.

    `A` and `b` are the LinUCB matrix-form scalars at d=1 (`A` is the
    weighted observation count, `b` is the weighted sum of rewards).
    They satisfy `mean = b / A` when A > 0. Keeping the matrix-form
    names makes the eventual d>1 expansion mechanical.
    """

    A: Decimal = field(default_factory=lambda: Decimal("0"))
    b: Decimal = field(default_factory=lambda: Decimal("0"))
    n: int = 0  # raw observation count (pre-decay); for UCB log term

    def update(self, reward: Decimal, weight: Decimal) -> None:
        self.A += weight
        self.b += weight * reward
        self.n += 1

    def mean(self) -> Decimal:
        if self.A <= 0:
            return Decimal("0")
        return self.b / self.A


# ── consumer ────────────────────────────────────────────────────────────


_DEFAULT_ALPHA = Decimal("1.0")
_DEFAULT_FLOOR = Decimal("0.05")  # 5% hard min per arm (A4 floor)


@dataclass
class LinUCBConsumer:
    """Read-back consumer over labeled routing_outcomes rows.

    State is per (cell, model_slug). Floor + alpha + decay are constants
    of the consumer instance; constructing a new instance with different
    knobs makes deterministic-replay comparisons explicit (the knob
    changes are visible in the constructor args, not silently buried in
    a mutated singleton).

    Per Disc #12: this module does not change ANY routing decision on
    its own. The brain in decide.py reads from q_empirical_overrides
    only when the founder explicitly wires it in (future PR).
    """

    alpha: Decimal = _DEFAULT_ALPHA
    exploration_floor: Decimal = _DEFAULT_FLOOR
    decay_half_life: int | None = None  # in observation ticks; None == no decay
    # cell → model_slug → CellModelStats
    state: dict[str, dict[str, CellModelStats]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (Decimal("0") <= self.exploration_floor <= Decimal("1")):
            raise ValueError(f"exploration_floor must be in [0, 1]; got {self.exploration_floor}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0; got {self.alpha}")
        if self.decay_half_life is not None and self.decay_half_life <= 0:
            raise ValueError(
                "decay_half_life must be positive integer ticks or None; "
                f"got {self.decay_half_life}"
            )

    # ── ingest ──────────────────────────────────────────────────────────

    def ingest(self, obs: Iterable[Observation], now_tick: int | None = None) -> None:
        """Apply a batch of observations to the running state.

        Each observation is weighted by `decay_factor(tick - now_tick)`.
        With decay disabled, every observation contributes weight 1.

        Observations within a single batch must have non-decreasing
        ticks — the caller controls ordering for replay determinism.
        """
        materialized = list(obs)
        if not materialized:
            return
        effective_now = now_tick if now_tick is not None else max(o.tick for o in materialized)
        # Sort by tick so the streaming order is canonical (a caller that
        # passes shuffled batches gets the same q_empirical out).
        for o in sorted(materialized, key=lambda x: (x.tick, x.cell, x.model_slug)):
            if not (Decimal("0") <= o.reward <= Decimal("1")):
                raise ValueError(
                    f"reward out of [0, 1] for cell={o.cell!r} model={o.model_slug!r}: {o.reward}"
                )
            if o.weight < 0:
                raise ValueError(
                    f"weight must be >= 0 for cell={o.cell!r} model={o.model_slug!r}: {o.weight}"
                )
            # Total weight = time-decay * provenance weight. With the default
            # provenance weight of 1 this is byte-identical to the pre-AIN-388
            # behaviour, so existing replay/golden state is unchanged; a
            # down-weighted fleet row contributes proportionally less to the
            # mean while still counting as one observation (n) — kept, not
            # dropped (the neutrality rider's "down-weight, don't exclude").
            weight = self._decay_weight(effective_now - o.tick) * o.weight
            cell_state = self.state.setdefault(o.cell, {})
            stats = cell_state.setdefault(o.model_slug, CellModelStats())
            stats.update(o.reward, weight)

    def _decay_weight(self, age_ticks: int) -> Decimal:
        if self.decay_half_life is None or age_ticks <= 0:
            return Decimal("1")
        # 0.5 ** (age / half_life) computed in Decimal-space; the math.pow
        # path is fine here because the result is folded into Decimal at
        # multiply time. For full byte-determinism across architectures
        # the caller can pin a different decay scheme.
        return Decimal(str(0.5 ** (age_ticks / self.decay_half_life)))

    # ── read APIs (the brain's eventual hookup) ─────────────────────────

    def q_empirical(self, cell: str, model_slug: str) -> Decimal | None:
        """Mean reward for (cell, model). None when the bucket has zero
        weighted mass (caller should fall back to q_prior).
        """
        stats = self.state.get(cell, {}).get(model_slug)
        if stats is None or stats.A <= 0:
            return None
        return stats.mean()

    def ucb_score(self, cell: str, model_slug: str) -> Decimal | None:
        """Mean + alpha * sqrt(2 ln N_cell / n). None for empty buckets.

        The standard UCB1 bonus, expressed in LinUCB form (which
        coincides at d=1). Higher = more confident or under-explored.
        """
        stats = self.state.get(cell, {}).get(model_slug)
        if stats is None or stats.A <= 0:
            return None
        n_cell = self._n_cell(cell)
        if n_cell <= 1:
            # Single observation in this cell — UCB bonus is undefined;
            # return the mean so the brain falls back to "trust the
            # prior" rather than "this arm is infinitely uncertain".
            return stats.mean()
        bonus = self.alpha * Decimal(str(math.sqrt(2.0 * math.log(n_cell) / max(stats.n, 1))))
        return stats.mean() + bonus

    def q_empirical_overrides_for_cell(self, cell: str) -> dict[str, Decimal]:
        """All non-None q_empirical values in a cell, by model_slug."""
        bucket = self.state.get(cell, {})
        out: dict[str, Decimal] = {}
        for slug, stats in bucket.items():
            if stats.A > 0:
                out[slug] = stats.mean()
        return out

    def exploration_quota(self, cell: str, candidate_slugs: list[str]) -> dict[str, Decimal]:
        """Per-model selection-probability lower bound for the brain to honor.

        Splits `self.exploration_floor` across the least-explored arms
        in the cell. If `floor == 0.10` and three arms are tied at the
        minimum count, each gets `0.10 / 3` reserved. The remainder
        `(1 - floor)` flows to whatever selection rule the brain uses.

        Returns floors keyed by every candidate_slug — slugs not in the
        floor get `Decimal('0')` so the caller can sum the dict and
        verify the budget.
        """
        if not candidate_slugs:
            return {}
        bucket = self.state.get(cell, {})
        counts = {slug: bucket.get(slug, CellModelStats()).n for slug in candidate_slugs}
        # Least-explored = smallest n. With a hard floor of 0, no
        # exploration is enforced.
        if self.exploration_floor == 0:
            return {slug: Decimal("0") for slug in candidate_slugs}
        min_n = min(counts.values())
        least = [slug for slug, n in counts.items() if n == min_n]
        per_arm = self.exploration_floor / Decimal(len(least))
        out = {slug: Decimal("0") for slug in candidate_slugs}
        for slug in least:
            out[slug] = per_arm
        return out

    # ── housekeeping ────────────────────────────────────────────────────

    def reset_arm(self, cell: str, model_slug: str) -> None:
        """Catalog-change trigger: a re-listed model re-enters the
        exploration window. The mean is preserved (so prior signal isn't
        thrown away), the count is reset to zero so the floor sees the
        arm as fresh.
        """
        stats = self.state.get(cell, {}).get(model_slug)
        if stats is not None:
            stats.n = 0

    def drop_arm(self, cell: str, model_slug: str) -> None:
        """Catalog-change trigger: a retired model is removed. The
        brain will stop offering it as a candidate; the consumer drops
        its stats so future ingestion of stale rows doesn't repopulate.
        """
        self.state.get(cell, {}).pop(model_slug, None)

    def _n_cell(self, cell: str) -> int:
        return sum(stats.n for stats in self.state.get(cell, {}).values())

    # ── serialization (deterministic, for replay + checkpoint) ──────────

    def serialize(self) -> dict[str, Any]:
        """Sorted, canonical-JSON-shaped state for byte-identical replay."""
        cells: dict[str, dict[str, dict[str, str | int]]] = {}
        for cell in sorted(self.state.keys()):
            cell_out: dict[str, dict[str, str | int]] = {}
            for slug in sorted(self.state[cell].keys()):
                s = self.state[cell][slug]
                cell_out[slug] = {"A": str(s.A), "b": str(s.b), "n": s.n}
            cells[cell] = cell_out
        return {
            "alpha": str(self.alpha),
            "exploration_floor": str(self.exploration_floor),
            "decay_half_life": self.decay_half_life,
            "cells": cells,
        }

    def to_json(self) -> str:
        return json.dumps(self.serialize(), separators=(",", ":"), sort_keys=True)


# ── replay (deterministic check over an observation list) ───────────────


def replay(
    observations: Iterable[Observation],
    *,
    alpha: Decimal = _DEFAULT_ALPHA,
    exploration_floor: Decimal = _DEFAULT_FLOOR,
    decay_half_life: int | None = None,
    now_tick: int | None = None,
) -> LinUCBConsumer:
    """Pure-function replay: same observations + same config = same state.

    Used by Spark replay / CI golden tests. Caller can hash `c.to_json()`
    of the returned consumer to detect drift across consumer versions.
    """
    c = LinUCBConsumer(
        alpha=alpha,
        exploration_floor=exploration_floor,
        decay_half_life=decay_half_life,
    )
    c.ingest(observations, now_tick=now_tick)
    return c


__all__ = [
    "CellModelStats",
    "LinUCBConsumer",
    "Observation",
    "replay",
]
