# Exploration-policy PROPOSAL · close the `reasoning:cost` positivity gap (INERT)

**Status: PROPOSAL ONLY. Nothing here is enabled. Live cutover + any exploration
change is a founder gate (Disc#12).** This is the offline scope the AIN-335
replay-gate's one promising-but-uncertifiable flip points to. It does **not**
touch the gate's verdict (still `FAIL_CLOSED`) and does **not** wire LinUCB live.

## The lead (from the AIN-335 replay-gate, routing#16)

`reasoning:cost` is the single cell where q_empirical would do something useful:
mistral-large-3's learned mean **0.9309** lifts it above the 0.88 reasoning
floor its static q_prior (0.74) can't clear → LinUCB would route reasoning:cost
to mistral (held-out **1.0 over n=22**, ~2.5× cheaper than the current gpt-5-5
pick). The gate can't *certify* it because the baseline arm has no overlapping
held-out support. The binding constraint is **exploration coverage / positivity,
not the model.**

## Correction to the naive framing ("just ε-boost reasoning:cost")

The live ε-floor (`api … routing_brain._choose_dispatch_order`, AIN-388) explores
**only among floor-clearing survivors**. Under today's reasoning floor (0.88) the
survivors are gpt-5-5 (0.90) and opus (0.95); **mistral (0.74) is not a survivor,
so a plain ε-boost can never serve it in reasoning:cost.** Raising global ε would
just sample gpt-5-5 ↔ opus — not the head-to-head we need. So the lead needs a
*different* mechanism than the existing ε-floor, scoped below.

## What actually closes the gap (two parts, sequenced)

**Part A — gpt-5-5's side fixes itself (no exploration needed).** Going forward,
reasoning:cost's greedy pick *is* gpt-5-5 (cheapest clearing 0.88), so its
held-out reward accumulates naturally — **once AIN-403 (the Responses-API
envelope extraction bug) is fixed.** Until then every gpt-5-5 reasoning row scores
a parsing-artifact 0.0 (proven: 41 graded rows, 0 positive, real text stranded in
`output[].content[].text`). **Hard precondition: AIN-403 fixed + the contaminated
gpt-5-5 rows re-judged BEFORE any of this runs** — else the "head-to-head" is
built on the artifact and would falsely confirm mistral.

**Part B — mistral's side needs a floor-aware counterfactual serve.** To compare
under *current* conditions (not just stale pre-0.88-floor data), serve mistral in
reasoning:cost with small probability even though it's below the static floor —
*because* its empirical mean (0.93) clears it. Mechanism, INERT proposal:

- A new, separately-gated **`counterfactual_explore`** path (NOT the ε-floor):
  with probability `κ` (default **0**, i.e. off), in a **cell allowlist**
  (`reasoning:cost` only to start), dispatch one enrolled-but-floor-dropped
  candidate whose `q_empirical ≥ floor` (here mistral) to collect held-out reward.
- Gated behind a dedicated env, default-off, additive allowlist (same idiom as
  `AINFERA_ROUTING_EXPLORATION_EPSILON` / fleet-tenant envs):
  `AINFERA_COUNTERFACTUAL_EXPLORE_KAPPA=0`, `AINFERA_COUNTERFACTUAL_CELLS=` (empty).
- Rows tagged `exploration=true` + a `counterfactual` marker so the neutrality
  down-weight + the gate's existing filters treat them correctly; never written
  as a greedy outcome.
- **Bounded quality risk** (deliberately serving a below-static-floor model) is
  why this is founder-gated, not the default ε-floor.

## Sizing (so the founder sees the cost)

mistral already has 22 held-out reasoning:cost rows @1.0 (supported, ≥ the gate's
`MIN_SUPPORT=5`). The missing side is **fresh** gpt-5-5 reasoning:cost reward,
which Part A produces from greedy traffic post-AIN-403 — no extra serves, just the
fix. So **Part A alone may close the gate** (if post-fix gpt-5-5 scores well,
there's no win and we keep gpt-5-5; if it scores poorly, mistral's existing 1.0
becomes a certifiable win). **Part B (κ) is only needed if mistral's signal must
be refreshed under current conditions** — recommend leaving κ=0 until the gate is
re-run on post-AIN-403 data and explicitly asks for it.

## Sequence (the only safe order)

1. **AIN-403** — fix the envelope extractor + re-judge the contaminated rows.
2. Let reasoning:cost accumulate **real** gpt-5-5 greedy reward (Part A, no change).
3. **Re-run the AIN-335 replay-gate.** If it now certifies (or refutes) the
   mistral flip → done, no exploration change shipped.
4. **Only if** the gate still reports positivity on mistral's side → founder
   ratifies enabling Part B (κ small, reasoning:cost only), re-run, re-evaluate.

**Stop point: this proposal. No env is set, no code path enabled. Founder gates
every step.**

---

## STATUS UPDATE 2026-06-09 — Part B BUILT (inert); sequence steps 1–3 done

- **Step 1 (AIN-403)** — DONE, merged + deployed, contaminated rows re-judged.
- **Step 2 (real gpt-5-5 reward)** — accumulated: reasoning:quality gpt-5-5 holdout
  recovered 0.0000(10) → 0.6172(32); the `judge_mono_zero` flag cleared.
- **Step 3 (re-run the gate)** — DONE on the clean corpus
  (`routing#16 docs/replay-gate-report-2026-06-09-clean.md`). Verdict **still
  `FAIL_CLOSED`**, now `NO_CHANGE=8 / UNCERTIFIABLE=3`. The surviving blockers are
  **positivity/overlap only** — incl. `reasoning:cost`, where mistral has held-out
  support but gpt-5-5's *current-conditions* side does not (`cov(holdout,n=0)`).
  So the gate did **not** self-certify the flip; Part B's mechanism is the
  remaining lever.

**Part B is now BUILT and INERT** (this PR):
- `ainfera_routing/explore.py` — `select_counterfactual()` (pure, RNG-free; caller
  supplies the roll) + `eligible_arms()` + `kappa_from_env()` / `cells_from_env()`.
- Defaults `κ=0`, `cells=∅` → **always returns `None`** → caller byte-identical to
  pre-Part-B. `decide.py` is untouched.
- `tests/test_explore.py` (36 tests) pins the inert contract and the safety
  invariants (only enrolled + static-floor-dropped + `q_empirical ≥ floor` arms are
  ever served; never an unenrolled/vetoed model; never a greedy-reachable arm).
- `scripts/counterfactual_dryrun.py` demonstrates — serving nothing — that arming
  `reasoning:cost` would serve mistral-large-3 (prior 0.74 < 0.88 ≤ q_emp 0.9529),
  creating the mistral↔gpt-5-5 head-to-head the gate's positivity guard needs.

**Still NOT done (founder gates, unchanged):**
- The **live wire-in** (api `routing_brain` reading the κ/cells env, doing the roll,
  serving the pick) — a **§17 routing-methodology amendment + founder approval**.
- Setting `κ>0` on any cell. Scope when enabled: **fleet/dogfood traffic only**,
  `reasoning:cost` first; customer routing quality untouched.
- The eventual LinUCB cutover (its own founder gate; the gate must certify first).
