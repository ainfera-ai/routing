"""AIN-542 — online Thompson exploration weights + the consumer's thompson quota path."""

from __future__ import annotations

from decimal import Decimal

from ainfera_routing import LinUCBConsumer, Observation
from ainfera_routing.exploration_thompson import (
    beta_params,
    seed_for_cell,
    thompson_weights,
)


def _obs(cell: str, slug: str, reward: float) -> Observation:
    return Observation(
        cell=cell, model_slug=slug, reward=Decimal(str(reward)), policy_version="1.0.0"
    )


# ── pure module ────────────────────────────────────────────────────────────


def test_beta_params_from_b_and_a() -> None:
    a, b = beta_params(8.0, 10.0)  # b=8, A=10 → alpha=8, beta=2 (mean 0.8)
    assert a == 8.0 and b == 2.0


def test_beta_params_unseen_arm_is_proper() -> None:
    a, b = beta_params(0.0, 0.0)  # never sampled → Beta(eps, eps), still proper
    assert a > 0 and b > 0


def test_weights_sum_to_one_and_deterministic() -> None:
    arms = [("a", 8.0, 10.0, 10), ("b", 2.0, 10.0, 10), ("c", 5.0, 10.0, 10)]
    w1 = thompson_weights(arms, min_samples=0, floor_pct=0.0, draws=2000, seed=1)
    w2 = thompson_weights(arms, min_samples=0, floor_pct=0.0, draws=2000, seed=1)
    assert w1 == w2
    assert abs(sum(w1.values()) - 1.0) < 1e-9


def test_better_arm_gets_more_weight() -> None:
    arms = [("good", 18.0, 20.0, 20), ("bad", 2.0, 20.0, 20)]  # means 0.9 vs 0.1
    w = thompson_weights(arms, min_samples=0, floor_pct=0.0, draws=4000, seed=1)
    assert w["good"] > 0.95


def test_min_sample_floor_rescues_starved_arm() -> None:
    # "starved" has a low mean but only n=1 (< min_samples) → guaranteed ≥ floor
    arms = [("good", 18.0, 20.0, 50), ("starved", 0.1, 1.0, 1)]
    w = thompson_weights(arms, min_samples=30, floor_pct=0.10, draws=2000, seed=1)
    assert w["starved"] >= 0.10
    assert w["good"] > w["starved"]


def test_seed_for_cell_is_stable() -> None:
    assert seed_for_cell("code|balanced") == seed_for_cell("code|balanced")


# ── consumer.exploration_quota ─────────────────────────────────────────────


def _consumer_two_arms() -> LinUCBConsumer:
    c = LinUCBConsumer(exploration_floor=Decimal("0.05"))
    c.ingest([_obs("cell", "good", 1.0) for _ in range(40)])
    c.ingest([_obs("cell", "good", 1.0) for _ in range(0)])
    c.ingest([_obs("cell", "bad", 0.0) for _ in range(40)])
    return c


def test_quota_default_is_unchanged_v0() -> None:
    c = _consumer_two_arms()
    # v0: least-explored split of the floor. Both arms have n=40 → tie → 0.025 each.
    q = c.exploration_quota("cell", ["good", "bad"])
    assert q["good"] == Decimal("0.05") / Decimal("2")
    assert q["bad"] == Decimal("0.05") / Decimal("2")


def test_quota_thompson_prefers_the_better_arm() -> None:
    c = _consumer_two_arms()
    q = c.exploration_quota("cell", ["good", "bad"], thompson=True, thompson_draws=2000)
    assert q["good"] > q["bad"]  # exploration now leans to the arm likely-best
    assert sum(q.values()) > Decimal("0.99")  # a full distribution, not a floor


def test_quota_thompson_floors_a_young_arm() -> None:
    c = LinUCBConsumer(exploration_floor=Decimal("0.05"))
    c.ingest([_obs("cell", "good", 1.0) for _ in range(50)])
    c.ingest([_obs("cell", "fresh", 0.0) for _ in range(2)])  # n=2 < min_samples
    q = c.exploration_quota(
        "cell", ["good", "fresh"], thompson=True, thompson_min_samples=30, thompson_draws=2000
    )
    assert q["fresh"] >= Decimal("0.05")  # min-sample floor honored
