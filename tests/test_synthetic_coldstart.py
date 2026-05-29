"""AIN-303 · synthetic cold-start — planted recovery, determinism, SHADOW-only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


sc = _load("synthetic_coldstart", "scripts/synthetic_coldstart.py")
refit_policy = _load("refit_policy", "scripts/refit_policy.py")


def test_bandit_recovers_planted_best_in_all_7_cells():
    _, recovered, n = sc.run()
    assert n == len(sc.TASK_TYPES) * len(sc.CANDIDATES) * sc._PER_ARM
    for task in sc.TASK_TYPES:
        assert recovered[task] == sc.PLANTED_BEST[task], f"cell {task}: {recovered[task]}"


def test_covers_canonical_7_task_types():
    assert set(sc.TASK_TYPES) == {
        "reasoning", "code", "extraction", "chat", "tool_use", "embed", "general"
    }


def test_deterministic_same_seed_same_recovery():
    _, r1, _ = sc.run(seed=7)
    _, r2, _ = sc.run(seed=7)
    assert r1 == r2


def test_promotion_targets_shadow_only_never_prod(tmp_path, monkeypatch):
    """Refit the synthetic corpus to a SHADOW policies dir tagged source=synthetic.
    INVARIANT 1: the artifact is source='synthetic' and lives in the shadow dir —
    there is no path here that writes tenant_routing_policies (prod)."""
    shadow = tmp_path / "shadow_policies"
    monkeypatch.setattr(refit_policy, "POLICIES_DIR", shadow)
    monkeypatch.setattr(refit_policy, "ACTIVE", shadow / "ACTIVE.json")
    monkeypatch.setattr(refit_policy, "HISTORY", shadow / "HISTORY.jsonl")

    obs = sc.generate_planted_observations()
    obs_file = tmp_path / "synthetic_obs.json"
    obs_file.write_text(json.dumps([
        {"cell": o.cell, "model_slug": o.model_slug, "reward": str(o.reward),
         "policy_version": o.policy_version, "tick": o.tick} for o in obs
    ]))
    rc = refit_policy.main(["refit", "--observations", str(obs_file), "--source", "synthetic"])
    assert rc == 0
    version = json.loads((shadow / "ACTIVE.json").read_text())["version"]
    artifact = json.loads((shadow / f"{version}.json").read_text())
    assert artifact["source"] == "synthetic"   # HARD WALL: never promotable to prod
    assert shadow.exists()  # shadow slot, isolated from any prod policy store
