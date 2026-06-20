"""exploration_thompson.py — online Thompson exploration weights (D7 online, AIN-542).

The OFFLINE twin lives in the research repo (`labs/thompson.py`) and allocates the
nightly refit's exploration. THIS is its ONLINE counterpart: it samples the LinUCB
consumer's own continuously-updated per-arm posterior, so live exploration is never
stale (no daily-snapshot bridge).

At d=1 the consumer's `(b, A)` sufficient statistics ARE a Beta posterior:
    alpha = b            (weighted reward-sum)
    beta = A - b        (weighted "failure" mass);  mean = b/A = alpha/(alpha+beta)
so this is exactly the same Beta-Thompson as the offline tool — applied to live state.

`thompson_weights` returns a per-arm weight distribution (summing to 1) = the posterior
probability each arm is best, with a min-sample floor so an under-sampled arm is never
starved below `floor_pct` before it has had `min_samples` observations. The caller hands
these to `LinUCBConsumer.exploration_quota`, which feeds BOTH the exploration draw and
the propensity μ from the SAME dict — so importance weights stay unbiased.

Deterministic given a seed (pure stdlib, no numpy). Weights are scale-invariant for the
sampler/propensity, but normalised here for a clean distribution.
"""

from __future__ import annotations

import hashlib
import random

_EPS = 1e-9


def seed_for_cell(cell: str) -> int:
    """Stable per-cell seed so the Monte-Carlo weights replay identically."""
    return int.from_bytes(hashlib.sha256(cell.encode("utf-8")).digest()[:8], "big")


def beta_params(b: float, a_total: float, *, eps: float = _EPS) -> tuple[float, float]:
    """Beta(alpha, beta) from the consumer's (b, A): alpha = b, beta = A - b. Clamped > 0 so an
    unseen arm (A=0) becomes Beta(eps, eps) ≈ uniform → maximally explorable."""
    alpha = max(b, eps)
    beta = max(a_total - b, eps)
    return alpha, beta


def thompson_weights(
    arms: list[tuple[str, float, float, int]],
    *,
    min_samples: int,
    floor_pct: float,
    draws: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """P[arm is best] over each arm's Beta posterior, with a min-sample floor.

    `arms` = list of (slug, b, A, n). Returns a distribution (sums to 1). Under-sampled
    arms (n < min_samples) are guaranteed ≥ floor_pct; the rest of the budget is split
    by Thompson probability. Deterministic given (arms, draws, seed).
    """
    if not arms:
        return {}
    if len(arms) == 1:
        return {arms[0][0]: 1.0}

    rng = random.Random(seed)
    params = [(slug, *beta_params(b, a_total), n) for (slug, b, a_total, n) in arms]
    wins = {slug: 0 for slug, *_ in params}
    for _ in range(draws):
        best_slug: str | None = None
        best_sample = -1.0
        for slug, alpha, beta, _n in params:
            sample = rng.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample, best_slug = sample, slug
        assert best_slug is not None
        wins[best_slug] += 1
    probs = {slug: w / draws for slug, w in wins.items()}

    n_by = {slug: n for slug, _a, _b, n in params}
    return _allocate_with_floor(probs, n_by, min_samples=min_samples, floor_pct=floor_pct)


def _allocate_with_floor(
    probs: dict[str, float],
    n_by: dict[str, int],
    *,
    min_samples: int,
    floor_pct: float,
) -> dict[str, float]:
    """Guarantee each under-sampled arm ≥ floor_pct; split the rest by Thompson prob.
    Mirrors labs.thompson.allocate_with_floor (kept in sync deliberately)."""
    cands = list(probs)
    if floor_pct <= 0:
        return dict(probs)
    under = {c for c in cands if n_by.get(c, 0) < min_samples}
    reserved = len(under) * floor_pct
    if under and reserved >= 1.0:
        share = 1.0 / len(under)
        return {c: (share if c in under else 0.0) for c in cands}

    alloc = {c: (floor_pct if c in under else 0.0) for c in cands}
    remaining = 1.0 - reserved
    pool = [c for c in cands if c not in under] or cands
    pool_mass = sum(probs[c] for c in pool)
    for c in pool:
        if pool_mass > _EPS:
            alloc[c] += remaining * probs[c] / pool_mass
        else:
            alloc[c] += remaining / len(pool)
    return alloc


__all__ = ["beta_params", "seed_for_cell", "thompson_weights"]
