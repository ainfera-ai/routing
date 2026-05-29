"""AIN-246 · refit → versioned policy + rollback (determinism + rollback verified)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "refit_policy", Path(__file__).resolve().parents[1] / "scripts" / "refit_policy.py"
)
refit_policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refit_policy)  # type: ignore[union-attr]


@pytest.fixture
def policies_dir(tmp_path, monkeypatch):
    d = tmp_path / "policies"
    monkeypatch.setattr(refit_policy, "POLICIES_DIR", d)
    monkeypatch.setattr(refit_policy, "ACTIVE", d / "ACTIVE.json")
    monkeypatch.setattr(refit_policy, "HISTORY", d / "HISTORY.jsonl")
    return d


def _obs_file(tmp_path, rows):
    p = tmp_path / "obs.json"
    p.write_text(json.dumps(rows))
    return str(p)


_OBS_A = [
    {"cell": "code|t|balanced", "model_slug": "gpt-5-5", "reward": 0.9, "tick": 1},
    {"cell": "code|t|balanced", "model_slug": "mistral-large-3", "reward": 0.4, "tick": 2},
]
_OBS_B = [
    {"cell": "chat|t|balanced", "model_slug": "gemini-3-1-pro", "reward": 0.7, "tick": 1},
]


def test_refit_is_deterministic(policies_dir, tmp_path):
    """Same observations + knobs → identical state hash (version differs only by ts)."""
    obs = _obs_file(tmp_path, _OBS_A)
    rc1 = refit_policy.main(["refit", "--observations", obs, "--source", "synthetic"])
    v1 = json.loads((policies_dir / "ACTIVE.json").read_text())["version"]
    h1 = json.loads((policies_dir / f"{v1}.json").read_text())["state_hash8"]
    rc2 = refit_policy.main(["refit", "--observations", obs, "--source", "synthetic"])
    v2 = json.loads((policies_dir / "ACTIVE.json").read_text())["version"]
    h2 = json.loads((policies_dir / f"{v2}.json").read_text())["state_hash8"]
    assert rc1 == 0 and rc2 == 0
    assert h1 == h2  # deterministic state


def test_source_is_recorded_for_invariant1(policies_dir, tmp_path):
    refit_policy.main(
        ["refit", "--observations", _obs_file(tmp_path, _OBS_A), "--source", "synthetic"]
    )
    v = json.loads((policies_dir / "ACTIVE.json").read_text())["version"]
    assert json.loads((policies_dir / f"{v}.json").read_text())["source"] == "synthetic"


def test_rollback_to_previous_flips_active(policies_dir, tmp_path):
    """refit A, refit B (ACTIVE=B), rollback --previous → ACTIVE=A. Rollback verified."""
    refit_policy.main(["refit", "--observations", _obs_file(tmp_path, _OBS_A), "--source", "prod"])
    va = json.loads((policies_dir / "ACTIVE.json").read_text())["version"]
    refit_policy.main(["refit", "--observations", _obs_file(tmp_path, _OBS_B), "--source", "prod"])
    vb = json.loads((policies_dir / "ACTIVE.json").read_text())["version"]
    assert va != vb
    rc = refit_policy.main(["rollback", "--previous"])
    active_after = json.loads((policies_dir / "ACTIVE.json").read_text())["version"]
    assert rc == 0
    assert active_after == va  # rolled back off B onto A
    # audit log recorded the rollback
    hist = [json.loads(line) for line in (policies_dir / "HISTORY.jsonl").read_text().splitlines()]
    assert any(e["event"] == "rollback" and e["to"] == va and e["from"] == vb for e in hist)
