"""AIN-246 · LinUCB consumer tests — math, decay, floor, replay determinism.

Pure functions, no DB. The consumer is mechanics-only; the brain in
decide.py is not touched, so these tests don't exercise the decision
rule at all — they lock the q_empirical / UCB / floor math against
textbook values and prove same-stream → same-bytes replay.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from ainfera_routing import LinUCBConsumer, Observation, replay
from ainfera_routing.learning import CellModelStats


def _obs(cell: str, slug: str, reward: float, tick: int = 0) -> Observation:
    return Observation(
        cell=cell,
        model_slug=slug,
        reward=Decimal(str(reward)),
        policy_version="1.0.0",
        tick=tick,
    )


# ── constructor guards ──────────────────────────────────────────────────


@pytest.mark.parametrize("floor", [Decimal("-0.01"), Decimal("1.01")])
def test_exploration_floor_must_be_in_unit_interval(floor: Decimal) -> None:
    with pytest.raises(ValueError, match="exploration_floor must be in"):
        LinUCBConsumer(exploration_floor=floor)


def test_alpha_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        LinUCBConsumer(alpha=Decimal("-0.1"))


@pytest.mark.parametrize("half", [0, -1])
def test_decay_half_life_must_be_positive_or_none(half: int) -> None:
    with pytest.raises(ValueError, match="decay_half_life must be positive"):
        LinUCBConsumer(decay_half_life=half)


# ── q_empirical math ────────────────────────────────────────────────────


def test_q_empirical_is_simple_mean_without_decay() -> None:
    c = LinUCBConsumer()
    c.ingest([_obs("c1", "m1", 1.0), _obs("c1", "m1", 0.5), _obs("c1", "m1", 0.0)])
    # mean(1.0, 0.5, 0.0) = 0.5
    assert c.q_empirical("c1", "m1") == Decimal("0.5")


def test_q_empirical_returns_none_for_empty_bucket() -> None:
    c = LinUCBConsumer()
    assert c.q_empirical("c1", "m1") is None
    c.ingest([_obs("c1", "m1", 0.7)])
    assert c.q_empirical("c1", "m2") is None  # bucket missing for m2
    assert c.q_empirical("c-other", "m1") is None  # bucket missing for cell


def test_reward_out_of_range_rejected_on_ingest() -> None:
    c = LinUCBConsumer()
    with pytest.raises(ValueError, match="reward out of"):
        c.ingest([_obs("c1", "m1", 1.5)])
    with pytest.raises(ValueError, match="reward out of"):
        c.ingest([_obs("c1", "m1", -0.01)])


def test_q_empirical_isolates_per_cell_and_per_model() -> None:
    """Two cells with identical model slugs must not bleed signal. A
    routing decision in cell A informed by cell B's reward distribution
    is a moat-contamination bug.
    """
    c = LinUCBConsumer()
    c.ingest(
        [
            _obs("cell-a", "m1", 0.8),
            _obs("cell-a", "m1", 0.8),
            _obs("cell-b", "m1", 0.2),
            _obs("cell-b", "m1", 0.2),
        ]
    )
    assert c.q_empirical("cell-a", "m1") == Decimal("0.8")
    assert c.q_empirical("cell-b", "m1") == Decimal("0.2")


# ── UCB bonus ───────────────────────────────────────────────────────────


def test_ucb_bonus_decreases_with_observation_count() -> None:
    """Same mean, more observations → tighter confidence → smaller UCB
    score. This is the property the brain will exploit: under-sampled
    arms get a quality boost that fades as evidence accrues.
    """
    rare = LinUCBConsumer()
    plenty = LinUCBConsumer()
    rare.ingest([_obs("c1", "m1", 0.5), _obs("c1", "m2", 0.5)])
    plenty.ingest([_obs("c1", "m1", 0.5)] * 20 + [_obs("c1", "m2", 0.5)] * 20)
    # With the same mean, the consumer that has seen 2 obs/arm must
    # have a larger UCB than the one with 20 obs/arm.
    rare_score = rare.ucb_score("c1", "m1")
    plenty_score = plenty.ucb_score("c1", "m1")
    assert rare_score is not None
    assert plenty_score is not None
    assert rare_score > plenty_score


def test_ucb_falls_back_to_mean_on_single_observation() -> None:
    """A single observation gives ln(1)=0 → divide-by-meaningless. We
    return the mean (= "trust the prior; we have no signal to bonus").
    """
    c = LinUCBConsumer()
    c.ingest([_obs("c1", "m1", 0.7)])
    assert c.ucb_score("c1", "m1") == Decimal("0.7")


def test_ucb_returns_none_for_empty_bucket() -> None:
    c = LinUCBConsumer()
    assert c.ucb_score("c1", "missing") is None


# ── exploration floor ──────────────────────────────────────────────────


def test_exploration_floor_reserves_quota_for_least_explored_arm() -> None:
    """Floor 0.10 with one under-explored arm out of three should
    reserve 0.10 for the under-explored arm and 0 for the others.
    """
    c = LinUCBConsumer(exploration_floor=Decimal("0.10"))
    # Three arms: m1 + m2 each seen 5x, m3 never seen.
    obs = [_obs("c1", "m1", 0.5)] * 5 + [_obs("c1", "m2", 0.5)] * 5
    c.ingest(obs)
    q = c.exploration_quota("c1", ["m1", "m2", "m3"])
    assert q["m1"] == Decimal("0")
    assert q["m2"] == Decimal("0")
    assert q["m3"] == Decimal("0.10")


def test_exploration_floor_split_evenly_across_tied_least_explored_arms() -> None:
    """Two arms tied at the minimum → split the floor evenly so neither
    gets starved. With floor 0.10 + two tied newcomers, each reserves 0.05.
    """
    c = LinUCBConsumer(exploration_floor=Decimal("0.10"))
    c.ingest([_obs("c1", "veteran", 0.5)] * 10)
    q = c.exploration_quota("c1", ["veteran", "rookie1", "rookie2"])
    assert q["veteran"] == Decimal("0")
    assert q["rookie1"] == Decimal("0.05")
    assert q["rookie2"] == Decimal("0.05")


def test_exploration_floor_of_zero_disables_floor_entirely() -> None:
    c = LinUCBConsumer(exploration_floor=Decimal("0"))
    q = c.exploration_quota("c1", ["m1", "m2"])
    assert q == {"m1": Decimal("0"), "m2": Decimal("0")}


def test_exploration_quota_empty_when_no_candidates() -> None:
    c = LinUCBConsumer()
    assert c.exploration_quota("c1", []) == {}


# ── decay ──────────────────────────────────────────────────────────────


def test_decay_weights_older_observations_less() -> None:
    """With a half-life of 1 tick, an observation 1 tick old contributes
    half the weight of a fresh one. The resulting mean shifts toward
    the recent value.
    """
    c = LinUCBConsumer(decay_half_life=1)
    # Old obs (tick=0) = 0.0; fresh obs (tick=1) = 1.0. With half-life=1
    # and now_tick=1: weights 0.5 and 1.0. mean = (0.5*0 + 1.0*1) / 1.5 = 2/3.
    c.ingest([_obs("c1", "m1", 0.0, tick=0), _obs("c1", "m1", 1.0, tick=1)])
    mean = c.q_empirical("c1", "m1")
    assert mean is not None
    # Decimal-Decimal: tolerate the float-to-Decimal half-life conversion.
    assert abs(mean - Decimal("0.6666666666666666666666666667")) < Decimal("1e-9")


def test_no_decay_when_disabled() -> None:
    """Decay = None means every observation has weight 1, regardless of tick."""
    c = LinUCBConsumer(decay_half_life=None)
    c.ingest([_obs("c1", "m1", 0.0, tick=0), _obs("c1", "m1", 1.0, tick=1000)])
    assert c.q_empirical("c1", "m1") == Decimal("0.5")


# ── catalog change ─────────────────────────────────────────────────────


def test_reset_arm_keeps_mean_but_clears_count() -> None:
    """A re-listed model should keep its prior quality signal but enter
    the exploration window — count zeroed so the floor sees it as fresh.
    """
    c = LinUCBConsumer()
    c.ingest([_obs("c1", "m1", 0.9)] * 5)
    assert c.q_empirical("c1", "m1") == Decimal("0.9")
    c.reset_arm("c1", "m1")
    # Mean preserved (b/A unchanged), n cleared.
    assert c.q_empirical("c1", "m1") == Decimal("0.9")
    assert c.state["c1"]["m1"].n == 0


def test_drop_arm_removes_state_entirely() -> None:
    c = LinUCBConsumer()
    c.ingest([_obs("c1", "m1", 0.5)])
    c.drop_arm("c1", "m1")
    assert c.q_empirical("c1", "m1") is None
    # Idempotent — re-dropping a missing arm is harmless.
    c.drop_arm("c1", "m1")
    c.drop_arm("never-seen-cell", "never-seen-model")


# ── deterministic replay ───────────────────────────────────────────────


def test_replay_is_deterministic_byte_for_byte() -> None:
    """Same observation stream + same config → same serialized state.
    Spark replay (and CI drift detection) depend on this invariant.
    """
    obs = [
        _obs("c1", "m1", 0.8, tick=1),
        _obs("c1", "m2", 0.4, tick=2),
        _obs("c2", "m1", 0.6, tick=3),
        _obs("c1", "m1", 0.7, tick=4),
    ]
    a = replay(obs)
    b = replay(obs)
    assert a.to_json() == b.to_json()


def test_replay_is_order_invariant_within_same_tick() -> None:
    """Within a single tick, observation order shouldn't change the
    resulting state (the consumer sorts internally before applying).
    """
    obs_forward = [
        _obs("c1", "m1", 0.8, tick=1),
        _obs("c1", "m2", 0.4, tick=1),
        _obs("c1", "m3", 0.6, tick=1),
    ]
    obs_reverse = list(reversed(obs_forward))
    assert replay(obs_forward).to_json() == replay(obs_reverse).to_json()


def test_replay_with_decay_uses_inferred_now_tick() -> None:
    """When now_tick is not passed, the consumer uses the max tick in
    the batch. This makes the decay reference point a property of the
    data, not the wall clock — safe for offline replay.
    """
    obs = [_obs("c1", "m1", 0.0, tick=0), _obs("c1", "m1", 1.0, tick=10)]
    state_a = replay(obs, decay_half_life=5).to_json()
    state_b = replay(obs, decay_half_life=5).to_json()
    assert state_a == state_b


# ── matrix-form scalars ────────────────────────────────────────────────


def test_a_b_scalars_track_weighted_observation_count_and_sum() -> None:
    """At d=1, A is the weighted count, b is the weighted sum of
    rewards, mean = b / A. These are the LinUCB matrix elements; a
    future PR can expand them to d>1 with context features and the
    rest of the algorithm stays unchanged.
    """
    c = LinUCBConsumer()
    c.ingest([_obs("c1", "m1", 0.3), _obs("c1", "m1", 0.7)])
    stats = c.state["c1"]["m1"]
    assert Decimal("2") == stats.A
    assert stats.b == Decimal("1.0")
    assert stats.mean() == Decimal("0.5")
    assert stats.n == 2


def test_cell_model_stats_mean_returns_zero_on_empty() -> None:
    s = CellModelStats()
    assert s.mean() == Decimal("0")


# ── provenance weight (AIN-388 P0-tail · down-weight, don't drop) ─────────


def _wobs(cell: str, slug: str, reward: float, weight: str, tick: int = 0) -> Observation:
    return Observation(
        cell=cell,
        model_slug=slug,
        reward=Decimal(str(reward)),
        policy_version="1.0.0",
        tick=tick,
        weight=Decimal(weight),
    )


def test_provenance_weight_downweights_mean_but_keeps_count() -> None:
    """A down-weighted (fleet) observation is KEPT — it still counts as one
    observation (n) — but contributes proportionally less to the mean.
    """
    c = LinUCBConsumer()
    c.ingest([
        _wobs("c1", "m1", 1.0, "1"),     # external, full weight
        _wobs("c1", "m1", 0.0, "0.25"),  # internal-fleet, down-weighted
    ])
    stats = c.state["c1"]["m1"]
    # mean = (1*1.0 + 0.25*0.0) / (1 + 0.25) = 0.8  (vs 0.5 at equal weight)
    assert stats.mean() == Decimal("0.8")
    assert Decimal("1.25") == stats.A   # weighted count
    assert stats.n == 2                  # kept, NOT dropped


def test_default_weight_is_one_and_matches_unweighted() -> None:
    """An explicit weight=1 observation produces byte-identical state to a
    default-weight observation — proves the field is backward-compatible.
    """
    a = LinUCBConsumer()
    a.ingest([_obs("c1", "m1", 0.3), _obs("c1", "m1", 0.7)])
    b = LinUCBConsumer()
    b.ingest([_wobs("c1", "m1", 0.3, "1"), _wobs("c1", "m1", 0.7, "1")])
    assert a.to_json() == b.to_json()


def test_negative_weight_rejected() -> None:
    c = LinUCBConsumer()
    with pytest.raises(ValueError, match="weight must be >= 0"):
        c.ingest([_wobs("c1", "m1", 0.5, "-0.1")])


def test_zero_weight_is_inert_but_counts() -> None:
    """weight=0 contributes no mass to the mean (the projector EXCLUDES
    degraded rows upstream rather than passing 0) yet still increments n —
    documents the boundary so a future change is a conscious one.
    """
    c = LinUCBConsumer()
    c.ingest([_wobs("c1", "m1", 0.9, "0"), _wobs("c1", "m1", 0.4, "1")])
    stats = c.state["c1"]["m1"]
    assert stats.mean() == Decimal("0.4")  # the weight-0 reward adds nothing
    assert stats.n == 2


def test_from_serialized_roundtrip() -> None:
    c = LinUCBConsumer()
    c.ingest([_obs("reasoning:cost", "mistral-large-3", 0.9)])
    restored = LinUCBConsumer.from_serialized(c.serialize())
    assert restored.to_json() == c.to_json()
    assert restored.q_empirical("reasoning:cost", "mistral-large-3") == Decimal("0.9")


def test_from_serialized_accepts_refit_policy_envelope() -> None:
    """A serve-time loader passes the full `policy-*.json` artifact (as written
    by `scripts/refit_policy.py`, with state nested under `state`) straight into
    `from_serialized`; that envelope shape must rehydrate identically.
    """
    c = LinUCBConsumer()
    c.ingest([_obs("reasoning:cost", "mistral-large-3", 0.9)])
    envelope = {
        "version": "policy-test-deadbeef",
        "source": "synthetic",
        "n_observations": 1,
        "knobs": {"alpha": "1.0", "exploration_floor": "0.05", "decay_half_life": None},
        "state": c.serialize(),
    }
    restored = LinUCBConsumer.from_serialized(envelope)
    assert restored.to_json() == c.to_json()
