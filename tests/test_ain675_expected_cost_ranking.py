"""AIN-675 - expected-cost ranking: selection by cost per unit of success.

When ``expected_cost_ranking=True``, the survivor ORDERING ranks on
``effective_price / q_effective`` (expected cost per success) instead of
raw price. This lets a slightly more expensive model with much higher
quality win over a cheaper model with lower quality.
"""

from __future__ import annotations

from decimal import Decimal

from ainfera_routing import Candidate, Policy, RoutingRequest, decide
from ainfera_routing.decide import ruleset_hash


def _request() -> RoutingRequest:
    return RoutingRequest(
        request_id="req-test",
        agent_id="agent-test",
        estimated_input_tokens=1000,
        reserved_max_tokens=1024,
    )


def _policy() -> Policy:
    return Policy(
        min_quality=Decimal("0.70"),
        budget_cap_usd=None,
        policy_name="test",
    )


# Two candidates with SAME price but different q_prior.
# Under v0 (raw price, q is tiebreak): higher q wins (tiebreak).
# Under AIN-675 (cost/success): higher q wins (lower CPST).
# Both should pick the same winner, but for different reasons.
_CHEAP_LOW_Q = Candidate(
    model_id="00000000-0000-0000-0000-000000000001",
    model_slug="cheap-low-q",
    brand_slug="brand-a",
    q_prior=Decimal("0.75"),
    price_in_per_mtok_usd=Decimal("1"),
    price_out_per_mtok_usd=Decimal("3"),  # total = 4
    m_allowed=True,
)

_BETTER_Q_SAME_PRICE = Candidate(
    model_id="00000000-0000-0000-0000-000000000002",
    model_slug="better-q-same-price",
    brand_slug="brand-b",
    q_prior=Decimal("0.99"),
    price_in_per_mtok_usd=Decimal("1"),
    price_out_per_mtok_usd=Decimal("3"),  # total = 4, same price
    m_allowed=True,
)

# A model that is MORE expensive but has much higher q.
# CPST: cheap_low_q = 4/0.75 = 5.33, expensive_high_q = 8/0.99 = 8.08
# Under v0: cheap wins (lower price).
# Under AIN-675: cheap STILL wins (5.33 < 8.08) -- price gap too large.
_EXPENSIVE_HIGH_Q = Candidate(
    model_id="00000000-0000-0000-0000-000000000003",
    model_slug="expensive-high-q",
    brand_slug="brand-b",
    q_prior=Decimal("0.99"),
    price_in_per_mtok_usd=Decimal("2"),
    price_out_per_mtok_usd=Decimal("6"),  # total = 8
    m_allowed=True,
)

# A model that is slightly more expensive but has dramatically higher q.
# CPST: cheap_low_q = 4/0.75 = 5.33, slightly_expensive = 5/0.99 = 5.05
# Under v0: cheap wins (price 4 < 5).
# Under AIN-675: slightly_expensive wins (5.05 < 5.33)!
_SLIGHTLY_EXPENSIVE_HIGH_Q = Candidate(
    model_id="00000000-0000-0000-0000-000000000004",
    model_slug="slightly-expensive-high-q",
    brand_slug="brand-b",
    q_prior=Decimal("0.99"),
    price_in_per_mtok_usd=Decimal("1.25"),
    price_out_per_mtok_usd=Decimal("3.75"),  # total = 5
    m_allowed=True,
)


class TestExpectedCostRankingOff:
    """Flag OFF -> byte-identical to v0."""

    def test_off_picks_cheapest(self) -> None:
        cands = (_SLIGHTLY_EXPENSIVE_HIGH_Q, _CHEAP_LOW_Q)
        d = decide(_request(), cands, _policy(), expected_cost_ranking=False)
        assert d.chosen is not None
        # v0: raw price sort -> cheap (4) beats slightly_expensive (5)
        assert d.chosen.model_slug == "cheap-low-q"

    def test_off_hash_matches_v0(self) -> None:
        cands = (_SLIGHTLY_EXPENSIVE_HIGH_Q, _CHEAP_LOW_Q)
        d = decide(_request(), cands, _policy(), expected_cost_ranking=False)
        assert d.ruleset_hash == ruleset_hash()


class TestExpectedCostRankingOn:
    """Flag ON -> selection by expected cost per success."""

    def test_on_picks_lower_cpst(self) -> None:
        cands = (_SLIGHTLY_EXPENSIVE_HIGH_Q, _CHEAP_LOW_Q)
        d = decide(_request(), cands, _policy(), expected_cost_ranking=True)
        assert d.chosen is not None
        # AIN-675: CPST slightly_expensive = 5/0.99 = 5.05 < cheap = 4/0.75 = 5.33
        assert d.chosen.model_slug == "slightly-expensive-high-q"

    def test_on_hash_differs_from_v0(self) -> None:
        cands = (_SLIGHTLY_EXPENSIVE_HIGH_Q, _CHEAP_LOW_Q)
        d = decide(_request(), cands, _policy(), expected_cost_ranking=True)
        assert d.ruleset_hash != ruleset_hash()

    def test_on_same_price_higher_q_wins(self) -> None:
        cands = (_CHEAP_LOW_Q, _BETTER_Q_SAME_PRICE)
        d = decide(_request(), cands, _policy(), expected_cost_ranking=True)
        assert d.chosen is not None
        # Same price -> lower CPST = higher q
        assert d.chosen.model_slug == "better-q-same-price"

    def test_on_price_gap_too_large_cheaper_still_wins(self) -> None:
        cands = (_EXPENSIVE_HIGH_Q, _CHEAP_LOW_Q)
        d = decide(_request(), cands, _policy(), expected_cost_ranking=True)
        assert d.chosen is not None
        # CPST: cheap = 4/0.75 = 5.33, expensive = 8/0.99 = 8.08
        # Even with CPST ranking, cheap wins because the price gap is too large
        assert d.chosen.model_slug == "cheap-low-q"

    def test_on_does_not_change_gates_or_floor(self) -> None:
        """The flag only reorders survivors; it never changes who passes gates."""
        # A candidate below the quality floor should still be dropped
        below_floor = Candidate(
            model_id="00000000-0000-0000-0000-000000000010",
            model_slug="below-floor",
            brand_slug="brand-c",
            q_prior=Decimal("0.50"),  # below min_quality 0.70
            price_in_per_mtok_usd=Decimal("0.01"),
            price_out_per_mtok_usd=Decimal("0.01"),
            m_allowed=True,
        )
        cands = (_CHEAP_LOW_Q, below_floor)
        d = decide(_request(), cands, _policy(), expected_cost_ranking=True)
        # below-floor should be dropped, not picked despite being ultra-cheap
        assert d.chosen is not None
        assert d.chosen.model_slug == "cheap-low-q"
        # Verify the below-floor candidate was actually dropped
        dropped = [c for c in d.candidates if c.model_slug == "below-floor"]
        assert len(dropped) == 1
        assert dropped[0].drop_reason is not None
        assert "quality_floor" in dropped[0].drop_reason.value

    def test_on_composes_with_q_empirical(self) -> None:
        """q_empirical overrides feed into the CPST calculation."""
        # Give cheap-low-q a HIGH q_empirical (learned it's actually great)
        # and slightly_expensive a LOW q_empirical (learned it's worse than prior).
        # CPST with empirical: cheap = 4/0.95 = 4.21, slightly = 5/0.80 = 6.25
        # -> cheap wins even under expected-cost ranking
        cands = (_SLIGHTLY_EXPENSIVE_HIGH_Q, _CHEAP_LOW_Q)
        q_emp = {
            "cheap-low-q": Decimal("0.95"),
            "slightly-expensive-high-q": Decimal("0.80"),
        }
        d = decide(
            _request(),
            cands,
            _policy(),
            q_empirical=q_emp,
            expected_cost_ranking=True,
        )
        assert d.chosen is not None
        assert d.chosen.model_slug == "cheap-low-q"


class TestExpectedCostRankingDeterminism:
    """Same inputs -> identical decision, byte-for-byte."""

    def test_same_inputs_same_output(self) -> None:
        cands = (_SLIGHTLY_EXPENSIVE_HIGH_Q, _CHEAP_LOW_Q)
        d1 = decide(_request(), cands, _policy(), expected_cost_ranking=True)
        d2 = decide(_request(), cands, _policy(), expected_cost_ranking=True)
        assert d1.chosen is not None
        assert d2.chosen is not None
        assert d1.chosen.model_slug == d2.chosen.model_slug
        assert d1.ruleset_hash == d2.ruleset_hash
