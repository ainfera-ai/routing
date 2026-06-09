#!/usr/bin/env python3
"""AIN-335 · LinUCB offline replay-gate — certify (or refuse) the q_empirical
wire-in BEFORE any live cutover.

The brain in ``ainfera_routing.decide`` already accepts a ``q_empirical`` map
(AIN-246); the live caller passes ``None`` so routing is byte-identical to v0.
Turning q_empirical ON live is a **founder-gated** step (Disc#12). This gate is
the evidence the founder reviews first: a held-out counterfactual that answers
"if we fed the learned per-cell means into ``decide()`` today, would routing get
*better* outcomes than the current coverage rule — provably, on data we did not
train on?"

It is **fail-closed**: it certifies a live cutover only when a real, supported,
positive margin exists. On thin/biased/non-overlapping data it refuses and says
exactly why. Refusing is the correct, safe default — a bandit promoted on bad
offline evidence entrenches whatever bias the offline data carried.

## Method (direct-method OPE with an explicit support/positivity guard)

Inputs (a JSON bundle; see ``--help`` and the committed report for how the
prod bundle is produced — a read-only ``routing_outcomes`` query + a per-cell
temporal split, kept OUT of this dependency-light library):

  * ``catalog``  — the enrolled routable models (those with a real ``q_prior``;
                   F5: never fabricate a prior). Price + q_prior + m_allowed.
  * ``floors``   — task → ``min_quality`` (the api ``CELL_MIN_QUALITY`` map,
                   AIN-388). The *forward* floor we would actually deploy.
  * ``aggregates`` — per ``(bandit_cell, arm, split, outcome_class)``: row count
                   + reward sum. ``split`` ∈ {train, holdout}. ``outcome_class``
                   ∈ {succeeded, failed, other}.

Per ``bandit_cell`` (= ``task:band``):

  1. ``q_empirical[arm]`` = mean reward over the arm's **succeeded** *train*
     rows (failed calls are a reliability signal, not a quality label — they
     never set quality means). Only arms with ≥ ``min_train`` succeeded train
     rows get a learned mean.
  2. ``coverage_pick`` = ``decide(catalog, q_empirical=None, floor)`` — today's
     live rule.
  3. ``linucb_pick``   = ``decide(catalog, q_empirical=q_empirical, floor)`` —
     the candidate rule. (Same enrolment + cheapest-survivor logic; only the
     quality source changes — Disc#12-clean.)
  4. Each pick's **held-out value** = the arm's mean reward over its *holdout*
     succeeded rows, with the holdout count as its *support*.

A cell certifies a LinUCB win only if: the pick FLIPS, **both** the baseline and
the chosen arm have ≥ ``min_support`` held-out rows (no positivity violation),
the new arm beats the baseline by ≥ ``win_margin``, and ``decide`` does not
collapse to "no candidate clears the floor" under q_empirical.

## Overall verdict

``PASS`` (cutover may be founder-ratified) iff ≥1 cell is a certified
``LINUCB_WIN`` AND zero certified ``LINUCB_REGRESSION`` AND zero
``COVERAGE_COLLAPSE``. Otherwise ``FAIL_CLOSED`` with the blocking reasons
enumerated. This script never promotes anything; it prints a verdict + report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ainfera_routing.decide import decide
from ainfera_routing.types import Candidate, Policy, RoutingRequest

# ── gate constants (FOUNDER-TUNE — conservative defaults) ─────────────────
# These are the bars a cell must clear to *certify* a cutover. Raising them
# makes the gate stricter (refuses more); they are deliberately conservative
# because the downside of a false PASS (promote a biased policy live) dwarfs
# the downside of a false FAIL (gather more exploration data first).
MIN_TRAIN = 5  # succeeded train rows before an arm earns a learned mean
MIN_SUPPORT = 5  # succeeded holdout rows before a pick's value is trustworthy
WIN_MARGIN = Decimal("0.05")  # held-out mean-reward margin to call a win
JUDGE_MONO_N = 10  # ≥ this many succeeded rows all scoring exactly 0 → bias flag

# A request shape that makes the cheapness key == the catalog total price
# (1 input tok, 1 reserved out tok would distort; use the library's
# total_price_per_mtok ordering by giving equal token weights). The budget
# gate is unused here (no cap), so token counts only need to be consistent.
_REQ = RoutingRequest(
    request_id="replay", agent_id="replay",
    estimated_input_tokens=1_000_000, reserved_max_tokens=1_000_000,
)


@dataclass
class ArmStats:
    train_succ_n: int = 0
    train_succ_sum: Decimal = Decimal("0")
    train_failed_n: int = 0
    holdout_succ_n: int = 0
    holdout_succ_sum: Decimal = Decimal("0")
    holdout_failed_n: int = 0

    def train_mean(self) -> Decimal | None:
        if self.train_succ_n < MIN_TRAIN:
            return None
        return self.train_succ_sum / self.train_succ_n

    def holdout_mean(self) -> Decimal | None:
        if self.holdout_succ_n == 0:
            return None
        return self.holdout_succ_sum / self.holdout_succ_n


@dataclass
class CellResult:
    cell: str
    task: str
    floor: str
    coverage_pick: str | None
    linucb_pick: str | None
    flipped: bool
    coverage_value: str | None
    coverage_support: int
    linucb_value: str | None
    linucb_support: int
    verdict: str
    flags: list[str] = field(default_factory=list)
    note: str = ""


def _candidates(catalog: list[dict[str, Any]]) -> list[Candidate]:
    out = []
    for m in catalog:
        out.append(
            Candidate(
                model_id=m["model_slug"],
                model_slug=m["model_slug"],
                brand_slug=m.get("brand_slug", "?"),
                q_prior=Decimal(str(m["q_prior"])) if m.get("q_prior") is not None else None,
                price_in_per_mtok_usd=Decimal(str(m["price_in"])),
                price_out_per_mtok_usd=Decimal(str(m["price_out"])),
                m_allowed=bool(m.get("m_allowed", True)),
            )
        )
    return out


def _pick(cands: list[Candidate], policy: Policy, q_emp: dict[str, Decimal] | None) -> str | None:
    d = decide(_REQ, cands, policy, q_empirical=q_emp)
    return d.chosen.model_slug if d.chosen is not None else None


def _arm_flags(arms: dict[str, ArmStats]) -> list[str]:
    """Data-integrity flags: reliability (failed calls) + judge-mono-zero."""
    flags: list[str] = []
    for a, st in arms.items():
        failed = st.train_failed_n + st.holdout_failed_n
        if failed:
            flags.append(f"reliability:{a}({failed} failed)")
        tot_succ = st.train_succ_n + st.holdout_succ_n
        tot_sum = st.train_succ_sum + st.holdout_succ_sum
        if tot_succ >= JUDGE_MONO_N and tot_sum == 0:
            flags.append(f"judge_mono_zero:{a}(n={tot_succ})")
    return flags


def _classify(
    cov: str | None, lin: str | None,
    cov_val: Decimal | None, cov_sup: int,
    lin_val: Decimal | None, lin_sup: int,
) -> tuple[str, str]:
    """The per-cell verdict + a one-line note. Pure decision table."""
    if lin is None:
        return "COVERAGE_COLLAPSE", "q_empirical demotes every arm below the floor → no survivor."
    if cov == lin:
        return "NO_CHANGE", ""
    if cov_sup < MIN_SUPPORT or lin_sup < MIN_SUPPORT:
        cov_s = (f"baseline {cov}={cov_val:.3f}(n={cov_sup})" if cov_val is not None
                 and cov_sup >= MIN_SUPPORT else f"baseline {cov} UNSUPPORTED(n={cov_sup})")
        lin_s = (f"linucb {lin}={lin_val:.3f}(n={lin_sup})" if lin_val is not None
                 and lin_sup >= MIN_SUPPORT else f"linucb {lin} UNSUPPORTED(n={lin_sup})")
        return "UNCERTIFIABLE_POSITIVITY", f"flip but no head-to-head overlap; {cov_s}, {lin_s}"
    delta = (lin_val or Decimal("0")) - (cov_val or Decimal("0"))
    if delta >= WIN_MARGIN:
        verdict = "LINUCB_WIN"
    elif -delta >= WIN_MARGIN:
        verdict = "LINUCB_REGRESSION"
    else:
        verdict = "TIE"
    return verdict, f"delta={delta:+.3f} ({lin}={lin_val:.3f} vs {cov}={cov_val:.3f})"


def evaluate(bundle: dict[str, Any]) -> tuple[str, list[CellResult], dict[str, int]]:
    catalog = _candidates(bundle["catalog"])
    floors = {k: Decimal(str(v)) for k, v in bundle["floors"].items()}
    default_floor = Decimal(str(bundle.get("default_floor", "0.50")))

    # group aggregates -> cell -> arm -> ArmStats
    cells: dict[str, dict[str, ArmStats]] = {}
    cell_task: dict[str, str] = {}
    for r in bundle["aggregates"]:
        cell = r["bandit_cell"]
        cell_task[cell] = r["task"]
        arm = cells.setdefault(cell, {}).setdefault(r["arm"], ArmStats())
        n, s = int(r["n"]), Decimal(str(r["reward_sum"]))
        oc, split = r["outcome_class"], r["split"]
        if split == "train" and oc == "succeeded":
            arm.train_succ_n += n
            arm.train_succ_sum += s
        elif split == "train" and oc == "failed":
            arm.train_failed_n += n
        elif split == "holdout" and oc == "succeeded":
            arm.holdout_succ_n += n
            arm.holdout_succ_sum += s
        elif split == "holdout" and oc == "failed":
            arm.holdout_failed_n += n

    results: list[CellResult] = []
    for cell in sorted(cells):
        task = cell_task[cell]
        floor = floors.get(task, default_floor)
        policy = Policy(min_quality=floor, policy_name=f"replay:{cell}")
        arms = cells[cell]

        q_emp = {a: m for a, st in arms.items() if (m := st.train_mean()) is not None}
        cov = _pick(catalog, policy, None)
        lin = _pick(catalog, policy, q_emp or None)

        cov_st = arms.get(cov) if cov else None
        lin_st = arms.get(lin) if lin else None
        cov_val = cov_st.holdout_mean() if cov_st else None
        lin_val = lin_st.holdout_mean() if lin_st else None
        cov_sup = cov_st.holdout_succ_n if cov_st else 0
        lin_sup = lin_st.holdout_succ_n if lin_st else 0

        flags = _arm_flags(arms)
        flipped = cov != lin
        verdict, note = _classify(cov, lin, cov_val, cov_sup, lin_val, lin_sup)

        results.append(
            CellResult(
                cell=cell, task=task, floor=str(floor),
                coverage_pick=cov, linucb_pick=lin, flipped=flipped,
                coverage_value=(f"{cov_val:.4f}" if cov_val is not None else None),
                coverage_support=cov_sup,
                linucb_value=(f"{lin_val:.4f}" if lin_val is not None else None),
                linucb_support=lin_sup,
                verdict=verdict, flags=flags, note=note,
            )
        )

    tally: dict[str, int] = {}
    for r in results:
        tally[r.verdict] = tally.get(r.verdict, 0) + 1

    certified_win = tally.get("LINUCB_WIN", 0) > 0
    blocking = tally.get("LINUCB_REGRESSION", 0) > 0 or tally.get("COVERAGE_COLLAPSE", 0) > 0
    overall = "PASS" if (certified_win and not blocking) else "FAIL_CLOSED"
    return overall, results, tally


_PASS_BLURB = (
    "At least one cell certifies a supported LinUCB win with no regression or "
    "coverage collapse. Founder may ratify the cutover (W5 gate)."
)
_FAIL_BLURB = (
    "LinUCB is **NOT** certified for live cutover. The held-out evidence does "
    "not show a supported, positive improvement over the current coverage rule. "
    "Do not flip the wire-in. The blocking reasons + the one promising "
    "direction are below."
)
_HDR = (
    "| cell | floor | coverage→ | linucb→ | flip | cov(holdout,n) | "
    "lin(holdout,n) | verdict |"
)


def render_markdown(
    overall: str, results: list[CellResult], tally: dict[str, int], meta: dict[str, Any]
) -> str:
    lines = [
        "# AIN-335 · LinUCB offline replay-gate report",
        "",
        f"- generated: {meta.get('generated_at', '?')}",
        f"- source: {meta.get('source', '?')}",
        f"- gate constants: MIN_TRAIN={MIN_TRAIN}, MIN_SUPPORT={MIN_SUPPORT}, "
        f"WIN_MARGIN={WIN_MARGIN}",
        "",
        f"## VERDICT: **{overall}**",
        "",
        "tally: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())),
        "",
        _PASS_BLURB if overall == "PASS" else _FAIL_BLURB,
        "",
        _HDR,
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cov = (f"{r.coverage_value}(n={r.coverage_support})" if r.coverage_value
               else f"—(n={r.coverage_support})")
        lin = (f"{r.linucb_value}(n={r.linucb_support})" if r.linucb_value
               else f"—(n={r.linucb_support})")
        flip = "YES" if r.flipped else "·"
        lines.append(
            f"| {r.cell} | {r.floor} | {r.coverage_pick} | {r.linucb_pick} | "
            f"{flip} | {cov} | {lin} | {r.verdict} |"
        )
    lines += ["", "### Per-cell notes"]
    for r in results:
        if r.note or r.flags:
            tail = f"  _flags_: {', '.join(r.flags)}" if r.flags else ""
            lines.append(f"- **{r.cell}**: {r.note}{tail}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AIN-335 LinUCB offline replay-gate")
    p.add_argument("--bundle", required=True, help="input JSON bundle (catalog+floors+aggregates)")
    p.add_argument("--json-out", help="write machine-readable result JSON here")
    p.add_argument("--md-out", help="write the markdown report here")
    args = p.parse_args(argv)

    bundle = json.loads(Path(args.bundle).read_text())
    overall, results, tally = evaluate(bundle)
    meta = {k: bundle.get(k) for k in ("generated_at", "source")}

    md = render_markdown(overall, results, tally, meta)
    if args.md_out:
        Path(args.md_out).write_text(md)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"overall": overall, "tally": tally,
             "cells": [r.__dict__ for r in results], "meta": meta},
            indent=2) + "\n")
    sys.stdout.write(md)
    # exit 0 always — the verdict is the payload, not the process status (the
    # cadence reads the verdict; a non-zero would mask FAIL_CLOSED as a crash).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
