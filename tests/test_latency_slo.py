"""AIN-542 selection layer — latency SLO (the deferred C1, now wired).

The brain drops survivors slower than the preset's latency_cap_ms, so a
cheap-but-slow model can't win on price alone (D6: a 41 s model). Default OFF
(latency_cap_ms None) → byte-identical to v0.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from ainfera_routing import Candidate, Policy, decide
from ainfera_routing.decide import ruleset_hash
from ainfera_routing.types import DropReason

from .test_decide import _anchors, _request

# combined prices (in+out): mistral $8 (cheapest) < gemini $11.25 < gpt-5-5 $20
# = grok $20 < claude $90.


def _with_latency(
    latency_by_slug: dict[str, int | None], default: int | None = 5000
) -> list[Candidate]:
    return [
        replace(c, expected_latency_ms=latency_by_slug.get(c.model_slug, default))
        for c in _anchors()
    ]


def test_latency_cap_none_is_inert() -> None:
    # expected latencies present but NO cap → cheapest still wins, v0 hash byte-identical
    cands = _with_latency({"mistral-large-3": 99999})
    d = decide(_request(), cands, Policy(min_quality=Decimal("0.80"), policy_name="balanced"))
    assert d.chosen.model_slug == "mistral-large-3"
    assert d.ruleset_hash == ruleset_hash()


def test_latency_cap_drops_the_slow_cheapest() -> None:
    # mistral is cheapest ($8) but 41 s; cap 30 s → dropped; next-cheapest fast wins (gemini)
    cands = _with_latency({"mistral-large-3": 41000})
    d = decide(_request(), cands, Policy(min_quality=Decimal("0.80"), latency_cap_ms=30000))
    drops = {o.model_slug: o.drop_reason for o in d.candidates}
    assert drops["mistral-large-3"] == DropReason.EXCEEDS_LATENCY_CAP
    assert d.chosen.model_slug == "gemini-3-1-pro"  # $11.25, fast → the new winner
    assert d.ruleset_hash != ruleset_hash()  # latency shape stamped


def test_unknown_latency_is_kept() -> None:
    # cheapest with UNKNOWN latency (None) + a tight cap → still wins (conservative)
    cands = _with_latency({"mistral-large-3": None}, default=999)
    d = decide(_request(), cands, Policy(min_quality=Decimal("0.80"), latency_cap_ms=1000))
    assert d.chosen.model_slug == "mistral-large-3"


def test_all_too_slow_is_no_candidate_clears_floor() -> None:
    cands = _with_latency({}, default=99999)
    d = decide(_request(), cands, Policy(min_quality=Decimal("0.80"), latency_cap_ms=1000))
    assert d.chosen is None
    assert d.rule_fired == "no_candidate_clears_floor"


def test_latency_winner_is_deterministic() -> None:
    cands = _with_latency({"mistral-large-3": 41000})
    pol = Policy(min_quality=Decimal("0.80"), latency_cap_ms=30000)
    a = decide(_request(), cands, pol)
    b = decide(_request(), cands, pol)
    assert a == b  # byte-identical
