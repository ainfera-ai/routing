"""Counterfactual exploration — AIN-335 Part B (INERT by default).

The replay-gate's only remaining blocker is **positivity**: every uncertifiable
flip routes to a floor-dropped-but-empirically-clearing arm that has no
head-to-head holdout support, so off-policy evaluation cannot certify it (see
``docs/replay-gate-report-2026-06-09-clean.md`` in routing#16 — FAIL_CLOSED is
now a *coverage* verdict, not a *label* verdict).

The lead is ``reasoning:cost -> mistral-large-3``: static prior 0.74 < the 0.88
floor, but its learned mean (~0.95) clears it, held-out 1.0 over n=21. The live
ε-floor (api ``routing_brain``, AIN-388) can never serve mistral there — it
explores only among floor-*clearing* survivors, and mistral isn't one. This path
serves such an arm with small probability ``κ`` to collect the missing overlap,
**without lowering the floor for the greedy pick**.

Inert + pure by construction
-----------------------------
* **Pure** — the caller supplies the random ``roll`` (∈ [0,1)) and the κ/cells
  config (read from env), exactly like the AIN-388 ε-floor. No RNG / DB / clock
  here, so the decision is deterministic and replay-safe.
* **Inert** — with the shipped defaults (``κ=0``, ``cells=∅``)
  :func:`select_counterfactual` ALWAYS returns ``None`` ⇒ the caller falls
  through to :func:`ainfera_routing.decide.decide` and behaviour is
  **byte-for-byte identical** to pre-Part-B. That is the contract the tests pin.

Why this is not "serve bad models"
-----------------------------------
* Only **enrolled** candidates are eligible (price > 0, ``q_prior`` present,
  ``m_allowed is True``) — a compliance-vetoed or unpriced model can never be
  served; the enrolment gates from ``decide()`` are mirrored exactly.
* Only arms whose ``q_empirical`` **clears the floor** are eligible — the *data*
  vouches for the arm, not a lowered bar. A model with no learned mean, or one
  whose learned mean is still below the floor, is never picked.
* The caller scopes this to a **cell allowlist** (``reasoning:cost`` first) and
  to **fleet/dogfood traffic only** (down-weighted, attributed) — customer
  routing quality is untouched.

Capture
-------
A served counterfactual row is written with ``decision_rule=DECISION_RULE`` and
``exploration=True`` so the neutrality down-weight and the gate's existing
filters treat it as exploration, never as a greedy outcome.

Live wire-in
------------
The live wire-in — api ``routing_brain`` reading
:data:`ENV_KAPPA` / :data:`ENV_CELLS`, performing the RNG roll, and serving the
pick — is a **§17 routing-methodology amendment + founder gate (Disc #12)**. It
is NOT enabled here. Shipping this module changes no behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ainfera_routing.types import Candidate, Policy

# Env knobs the live caller (api) would read. Named here so the contract is
# discoverable from the brain repo; the brain itself never reads the environment.
ENV_KAPPA = "AINFERA_COUNTERFACTUAL_EXPLORE_KAPPA"  # float in [0,1]; default "0" (off)
ENV_CELLS = "AINFERA_COUNTERFACTUAL_CELLS"  # csv of cells; default "" (none)

# Marker written to routing_outcomes.decision_rule for a served counterfactual
# row, so it is attributed + down-weighted as exploration (never a greedy pick).
DECISION_RULE = "counterfactual_explore"


@dataclass(frozen=True)
class CounterfactualPick:
    """An arm chosen by the counterfactual path for ONE request.

    The caller serves ``candidate`` and writes the outcome with
    ``decision_rule=DECISION_RULE`` + ``exploration=True``. The remaining fields
    are carried for the audit trail (why this arm was eligible).
    """

    candidate: Candidate
    cell: str
    q_empirical_used: Decimal
    q_prior: Decimal
    floor: Decimal


def _is_enrolled(c: Candidate) -> bool:
    """The three real enrolment gates from ``decide()``, mirrored so the
    counterfactual path can NEVER serve a model ``decide()`` would refuse to
    enrol (unpriced, no prior, or compliance-vetoed)."""
    return (
        c.price_in_per_mtok_usd > 0
        and c.price_out_per_mtok_usd > 0
        and c.q_prior is not None
        and c.m_allowed is True
    )


def eligible_arms(
    candidates: tuple[Candidate, ...] | list[Candidate],
    policy: Policy,
    *,
    q_empirical: Mapping[str, Decimal] | None,
) -> list[tuple[Candidate, Decimal]]:
    """Arms that are **enrolled**, **dropped by the static-prior floor**, yet
    whose ``q_empirical`` **clears the floor** — the exact set the gate needs
    head-to-head support for.

    Pure and config-free (κ / cells / roll are NOT consulted). Returned
    cheapest-first with a fully deterministic tiebreak
    ``(total_price, -q_empirical, model_slug)`` — the order the caller prefers.
    """
    floor = policy.min_quality
    out: list[tuple[Candidate, Decimal]] = []
    for c in candidates:
        if not _is_enrolled(c):
            continue
        # _is_enrolled guarantees q_prior is not None.
        assert c.q_prior is not None
        if c.q_prior >= floor:
            continue  # not floor-dropped — decide()'s greedy pick can already reach it
        if q_empirical is None:
            continue
        qe = q_empirical.get(c.model_slug)
        if qe is None or qe < floor:
            continue  # the data does not vouch for it -> never serve
        out.append((c, qe))
    out.sort(key=lambda t: (t[0].total_price_per_mtok(), -t[1], t[0].model_slug))
    return out


def select_counterfactual(
    candidates: tuple[Candidate, ...] | list[Candidate],
    policy: Policy,
    *,
    cell: str,
    q_empirical: Mapping[str, Decimal] | None,
    kappa: Decimal,
    cells: frozenset[str] | set[str],
    roll: float,
) -> CounterfactualPick | None:
    """INERT gate + arm selection. Returns a :class:`CounterfactualPick` or
    ``None``.

    Returns ``None`` (caller falls through to ``decide()``) unless ALL hold:

    * ``kappa > 0``                 — default 0 ⇒ always ``None``
    * ``cell in cells``             — default ∅ ⇒ always ``None``
    * ``0 <= roll < kappa``         — the κ-probability draw (caller-supplied)
    * at least one eligible arm     — enrolled + floor-dropped + ``q_emp`` clears

    Deterministic given ``(inputs, roll)``: no RNG / DB / clock.
    """
    if kappa <= 0:
        return None
    if not cells or cell not in cells:
        return None
    if not (0.0 <= roll < float(kappa)):
        return None
    arms = eligible_arms(candidates, policy, q_empirical=q_empirical)
    if not arms:
        return None
    candidate, qe = arms[0]
    # eligible_arms only returns enrolled arms, so q_prior is not None.
    assert candidate.q_prior is not None
    return CounterfactualPick(
        candidate=candidate,
        cell=cell,
        q_empirical_used=qe,
        q_prior=candidate.q_prior,
        floor=policy.min_quality,
    )


def kappa_from_env(env: Mapping[str, str]) -> Decimal:
    """Parse κ from the environment (default 0 = off), clamped to ``[0, 1]``.

    A single canonical parser so the live caller and tests agree; the brain
    never calls this itself.
    """
    raw = (env.get(ENV_KAPPA) or "").strip()
    if not raw:
        return Decimal("0")
    try:
        k = Decimal(raw)
    except (InvalidOperation, ValueError):
        return Decimal("0")
    if k < 0:
        return Decimal("0")
    if k > 1:
        return Decimal("1")
    return k


def cells_from_env(env: Mapping[str, str]) -> frozenset[str]:
    """Parse the cell allowlist (csv) from the environment (default ∅ = off)."""
    raw = (env.get(ENV_CELLS) or "").strip()
    if not raw:
        return frozenset()
    return frozenset(p.strip() for p in raw.split(",") if p.strip())
