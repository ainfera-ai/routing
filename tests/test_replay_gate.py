"""AIN-335 · tests for the LinUCB offline replay-gate.

The gate must (a) PASS only on a supported, positive, overlapping win, and
(b) FAIL_CLOSED on every degenerate shape — no flip, positivity violation,
regression, or coverage collapse. These synthetic bundles pin each branch so a
future change to the gate's thresholds is a visible, deliberate diff.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# Load the script module directly (scripts/ is not an importable package).
# Register in sys.modules before exec so @dataclass can resolve cls.__module__.
_SPEC = importlib.util.spec_from_file_location(
    "replay_gate", Path(__file__).resolve().parents[1] / "scripts" / "replay_gate.py"
)
replay_gate = importlib.util.module_from_spec(_SPEC)
sys.modules["replay_gate"] = replay_gate
_SPEC.loader.exec_module(replay_gate)  # type: ignore[union-attr]


# Two enrolled arms: A is cheap (clears a 0.70 floor on q_prior), B is dearer.
_CATALOG = [
    {"model_slug": "A", "brand_slug": "x", "q_prior": 0.80,
     "price_in": 1.0, "price_out": 1.0, "m_allowed": True},
    {"model_slug": "B", "brand_slug": "y", "q_prior": 0.85,
     "price_in": 5.0, "price_out": 5.0, "m_allowed": True},
]
_FLOORS = {"t": 0.70}


def _agg(arm, split, oc, n, reward_sum, cell="t:band", task="t"):
    return {"bandit_cell": cell, "task": task, "arm": arm, "split": split,
            "outcome_class": oc, "n": n, "reward_sum": reward_sum}


def _bundle(aggregates, catalog=None, floors=None):
    return {"generated_at": "test", "source": "synthetic",
            "catalog": catalog or _CATALOG, "floors": floors or _FLOORS,
            "default_floor": 0.50, "aggregates": aggregates}


def _cell(results, name="t:band"):
    return next(r for r in results if r.cell == name)


def test_clean_win_passes():
    # A is cheap so coverage picks A; but A's learned mean (0.40) is below the
    # 0.70 floor → LinUCB demotes A and picks B; B beats A on held-out by 0.50,
    # both with ≥5 holdout support → certified LINUCB_WIN → overall PASS.
    aggs = [
        _agg("A", "train", "succeeded", 10, 4.0),    # mean 0.40 < floor
        _agg("B", "train", "succeeded", 10, 9.0),    # mean 0.90 ≥ floor
        _agg("A", "holdout", "succeeded", 10, 4.0),  # 0.40, n=10
        _agg("B", "holdout", "succeeded", 10, 9.0),  # 0.90, n=10
    ]
    overall, results, _tally = replay_gate.evaluate(_bundle(aggs))
    r = _cell(results)
    assert r.coverage_pick == "A" and r.linucb_pick == "B" and r.flipped
    assert r.verdict == "LINUCB_WIN"
    assert overall == "PASS"


def test_regression_fails_closed():
    # Same flip A→B but B is WORSE on held-out (0.30 < A's 0.40) → regression.
    aggs = [
        _agg("A", "train", "succeeded", 10, 4.0),
        _agg("B", "train", "succeeded", 10, 9.0),
        _agg("A", "holdout", "succeeded", 10, 4.0),  # 0.40
        _agg("B", "holdout", "succeeded", 10, 3.0),  # 0.30
    ]
    overall, results, _ = replay_gate.evaluate(_bundle(aggs))
    assert _cell(results).verdict == "LINUCB_REGRESSION"
    assert overall == "FAIL_CLOSED"


def test_positivity_violation_uncertifiable():
    # Flip A→B but B has only 2 holdout rows (< MIN_SUPPORT) → cannot certify.
    aggs = [
        _agg("A", "train", "succeeded", 10, 4.0),
        _agg("B", "train", "succeeded", 10, 9.0),
        _agg("A", "holdout", "succeeded", 10, 4.0),
        _agg("B", "holdout", "succeeded", 2, 1.8),   # n=2 < 5
    ]
    overall, results, _ = replay_gate.evaluate(_bundle(aggs))
    assert _cell(results).verdict == "UNCERTIFIABLE_POSITIVITY"
    assert overall == "FAIL_CLOSED"


def test_no_change_when_learned_mean_clears_floor():
    # A's learned mean (0.90) stays above the floor → LinUCB keeps the cheap A.
    aggs = [
        _agg("A", "train", "succeeded", 10, 9.0),    # 0.90 ≥ floor
        _agg("B", "train", "succeeded", 10, 9.5),
        _agg("A", "holdout", "succeeded", 10, 9.0),
        _agg("B", "holdout", "succeeded", 10, 9.5),
    ]
    overall, results, _ = replay_gate.evaluate(_bundle(aggs))
    r = _cell(results)
    assert r.coverage_pick == "A" and r.linucb_pick == "A" and not r.flipped
    assert r.verdict == "NO_CHANGE"
    assert overall == "FAIL_CLOSED"  # no certified win → still fail-closed


def test_coverage_collapse_when_all_demoted():
    # Single enrolled arm A, learned mean below floor → q_empirical demotes the
    # only survivor → decide() returns no winner → COVERAGE_COLLAPSE.
    catalog = [_CATALOG[0]]  # just A
    aggs = [
        _agg("A", "train", "succeeded", 10, 1.0),    # 0.10 < floor 0.70
        _agg("A", "holdout", "succeeded", 10, 1.0),
    ]
    overall, results, _ = replay_gate.evaluate(_bundle(aggs, catalog=catalog))
    assert _cell(results).verdict == "COVERAGE_COLLAPSE"
    assert overall == "FAIL_CLOSED"


def test_below_min_train_falls_back_to_prior():
    # A has only 3 succeeded train rows (< MIN_TRAIN) → no learned mean → A
    # keeps its q_prior (0.80, clears floor) → LinUCB == coverage (no flip).
    aggs = [
        _agg("A", "train", "succeeded", 3, 0.0),     # mean would be 0 but n<MIN_TRAIN
        _agg("B", "train", "succeeded", 10, 9.0),
        _agg("A", "holdout", "succeeded", 10, 0.0),
        _agg("B", "holdout", "succeeded", 10, 9.0),
    ]
    _, results, _ = replay_gate.evaluate(_bundle(aggs))
    r = _cell(results)
    assert r.linucb_pick == "A" and not r.flipped and r.verdict == "NO_CHANGE"


def test_reliability_and_judge_mono_flags():
    # failed rows raise a reliability flag; ≥10 succeeded all-zero raises the
    # judge_mono_zero bias flag.
    aggs = [
        _agg("A", "train", "succeeded", 12, 0.0),    # 12 succeeded all 0 → judge_mono
        _agg("A", "train", "failed", 4, 0.0),        # reliability
        _agg("A", "holdout", "succeeded", 6, 0.0),
        _agg("B", "train", "succeeded", 10, 9.0),
        _agg("B", "holdout", "succeeded", 10, 9.0),
    ]
    _, results, _ = replay_gate.evaluate(_bundle(aggs))
    flags = " ".join(_cell(results).flags)
    assert "reliability:A" in flags
    assert "judge_mono_zero:A" in flags


def test_real_prod_bundle_fails_closed_if_present():
    # If the committed prod bundle is reachable, the gate must FAIL_CLOSED on it
    # (no certified win exists on the 2026-06-08 corpus). Skip if not present.
    bundle_path = (
        Path(__file__).resolve().parents[1] / "docs" / "replay-gate-bundle-2026-06-08.json"
    )
    if not bundle_path.exists():
        return
    overall, _results, tally = replay_gate.evaluate(json.loads(bundle_path.read_text()))
    assert overall == "FAIL_CLOSED"
    assert tally.get("LINUCB_WIN", 0) == 0
