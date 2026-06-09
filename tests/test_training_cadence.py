"""AIN-335 Stage 4 - training cadence orchestrator tests."""

from __future__ import annotations

import importlib.util
import json
import sys as _sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "training_cadence",
    Path(__file__).resolve().parents[1] / "scripts" / "training_cadence.py",
)
tc = importlib.util.module_from_spec(_SPEC)
_sys.modules["training_cadence"] = tc
_SPEC.loader.exec_module(tc)  # type: ignore[union-attr]


def _sample_rows(n: int = 15) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "source": "prod",
                "cell": "chat:model-a:cost",
                "task_type": "chat",
                "chosen_model_slug": "mistral-large-3",
                "reward": 0.85,
                "policy_version": "v0",
                "created_at": f"2026-06-0{(i % 9) + 1}T12:00:00Z",
                "judge_status": "labeled",
                "tenant_id": "00000000-0000-0000-0000-000000000099",
            }
        )
    return rows


def test_cadence_cold_start_dry_run(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(_sample_rows()))
    workdir = tmp_path / "work"
    policies = tmp_path / "policies"

    summary = tc.run_cadence(
        rows_path=rows_path,
        workdir=workdir,
        policies_dir=policies,
        source="prod",
        apply_promote=False,
    )

    assert summary["verdict"] == "PROMOTE_READY"
    assert summary["cold_start"] is True
    assert summary["promoted"] is False
    assert (workdir / "training_run.json").is_file()
    assert (workdir / "observations.json").is_file()
    assert not (policies / "ACTIVE.json").exists()


def test_cadence_apply_promote_flips_active(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(_sample_rows()))
    workdir = tmp_path / "work"
    policies = tmp_path / "policies"

    summary = tc.run_cadence(
        rows_path=rows_path,
        workdir=workdir,
        policies_dir=policies,
        source="prod",
        apply_promote=True,
    )

    assert summary["verdict"] == "PROMOTED"
    assert summary["promoted"] is True
    active = json.loads((policies / "ACTIVE.json").read_text())
    assert active["version"] == summary["candidate_version"]


def test_refit_no_flip_active(tmp_path, monkeypatch) -> None:
    refit = tc.refit_policy
    d = tmp_path / "policies"
    monkeypatch.setattr(refit, "POLICIES_DIR", d)
    monkeypatch.setattr(refit, "ACTIVE", d / "ACTIVE.json")
    monkeypatch.setattr(refit, "HISTORY", d / "HISTORY.jsonl")
    obs = tmp_path / "obs.json"
    obs.write_text(
        json.dumps(
            [
                {
                    "cell": "chat:cost",
                    "model_slug": "mistral-large-3",
                    "reward": "0.8",
                    "tick": 0,
                }
            ]
        )
    )
    rc = refit.main(["refit", "--observations", str(obs), "--source", "prod", "--no-flip-active"])
    assert rc == 0
    assert not (d / "ACTIVE.json").exists()
    assert list(d.glob("policy-*.json"))
