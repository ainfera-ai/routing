# AIN-335 · LinUCB offline replay-gate — report (2026-06-08)

**Verdict: `FAIL_CLOSED` — do NOT cut LinUCB over to live.**
Tally: `NO_CHANGE=7`, `UNCERTIFIABLE_POSITIVITY=4`, `LINUCB_WIN=0`, `LINUCB_REGRESSION=0`, `COVERAGE_COLLAPSE=0`.

This is the evidence the founder reviews **before** authorizing the q_empirical
wire-in (Disc#12 / W5 gate). The gate ran on the real prod corpus
(`routing_outcomes`, Supabase `dftfpwzqxoebwzepygzl`) and refuses to certify a
cutover. Refusing is the correct, safe outcome on this corpus — and the gate
names exactly what's blocking and what's promising.

Reproduce:
```
python3 scripts/replay_gate.py --bundle docs/replay-gate-bundle-2026-06-08.json
```
Inputs are committed (`replay-gate-bundle-2026-06-08.json`) so the verdict is
independently re-runnable; machine result in `replay-gate-result-2026-06-08.json`.

---

## What the gate does

The brain in `ainfera_routing.decide` already accepts a `q_empirical` map
(AIN-246); the live caller passes `None`, so routing is byte-identical to v0.
Turning q_empirical **on** is the founder-gated step. This gate is a held-out
counterfactual: build the learned per-cell means from a **train** split, then
ask whether feeding them to `decide()` would beat today's coverage rule on a
**held-out** split we did not train on — using today's catalog + today's floors
(`CELL_MIN_QUALITY`, AIN-388), so both policies are compared apples-to-apples
and only the quality source differs (Disc#12-clean).

Method: direct-method off-policy evaluation with an explicit **support /
positivity guard**. A cell certifies a win only if the pick flips, **both** the
baseline and the new arm have ≥5 held-out rows (real overlap), the new arm
beats the baseline by ≥0.05 mean reward, and `decide()` does not collapse to
"no candidate clears the floor."

`reward = (judge_score − 1) / 4` (a 1–5 Likert → [0,1]; 1 = worst). Confirmed
consistent across the whole corpus — there is **no** reward/judge_score bug.

---

## Per-cell result

| cell | floor | coverage→ | linucb→ | flip | cov(holdout,n) | lin(holdout,n) | verdict |
|---|---|---|---|---|---|---|---|
| chat:cost | 0.58 | mistral-large-3 | mistral-large-3 | · | 0.7188(24) | 0.7188(24) | NO_CHANGE |
| code:balanced | 0.78 | gemini-3-1-pro | gpt-5-5 | YES | 0.0000(2) | 0.1875(4) | UNCERTIFIABLE_POSITIVITY |
| code:cost | 0.78 | gemini-3-1-pro | gemini-3-1-pro | · | —(0) | —(0) | NO_CHANGE |
| embed:cost | 0.52 | mistral-large-3 | mistral-large-3 | · | 0.0000(3) | 0.0000(3) | NO_CHANGE |
| extraction:cost | 0.58 | mistral-large-3 | mistral-large-3 | · | 0.9861(18) | 0.9861(18) | NO_CHANGE |
| general:cost | 0.68 | mistral-large-3 | mistral-large-3 | · | 0.7955(22) | 0.7955(22) | NO_CHANGE |
| reasoning:balanced | 0.88 | gpt-5-5 | gpt-5-5 | · | —(0) | —(0) | NO_CHANGE |
| reasoning:cost | 0.88 | gpt-5-5 | **mistral-large-3** | YES | —(0) | **1.0000(22)** | UNCERTIFIABLE_POSITIVITY |
| reasoning:quality | 0.88 | gpt-5-5 | claude-opus-4-7 | YES | 0.0000(10) | —(0) | UNCERTIFIABLE_POSITIVITY |
| tool_use:balanced | 0.78 | gemini-3-1-pro | gpt-5-5 | YES | 0.1875(4) | —(0) | UNCERTIFIABLE_POSITIVITY |
| tool_use:cost | 0.78 | gemini-3-1-pro | gemini-3-1-pro | · | —(0) | —(0) | NO_CHANGE |

---

## Why FAIL_CLOSED (the blocking reasons)

1. **Positivity / overlap violation (the dominant blocker).** Every q_empirical
   flip routes to (or away from) an arm that was **never served** in that cell
   under the historical near-monoculture policy, so there is no held-out reward
   to compare against. The data was collected almost entirely on
   `mistral-large-3` (and stale, looser floors), so the arms today's rules would
   explore have ~0 production track record. Off-policy evaluation simply cannot
   certify a policy whose chosen arms have no support — and it correctly refuses.

2. **Reliability ≠ quality contamination.** Several 0.0-reward arms are
   `failed_other` calls (the call errored), not quality judgments —
   `gemini-3-1-flash` in chat/extraction/general is **100% failed calls**.
   Training a quality bandit on "the call errored" would demote models for
   infrastructure flakiness. The gate excludes failed calls from the learned
   means and flags them (`reliability:*`); the projector
   (`export_outcomes.py`) does **not** yet exclude them — fix before any cutover.

3. **Judge-bias risk on frontier arms.** `gpt-5-5` scores a uniform **1/5**
   across 25 *succeeded* reasoning:quality calls (`judge_mono_zero` flag);
   `gemini-3-1-pro` likewise on 12 succeeded code:balanced calls. A frontier
   model scoring worst-possible with zero variance on its *successful* outputs
   is a red flag for judge self-preference, not settled quality truth. This must
   be cross-checked by the independent / self-output judge (W2 **AIN-396**) and
   the L8 self-preference firewall (AIN-385) **before** these become training
   labels — otherwise LinUCB would entrench a biased judge.

4. **Thin per-cell volume.** The multi-arm cells with genuine signal collapse to
   single-digit holdout support once failed calls are removed. No cell has the
   ≥5/≥5 head-to-head overlap needed to certify.

## The one promising direction (worth gathering data on)

**`reasoning:cost` → `mistral-large-3`.** Under q_empirical, mistral's learned
mean (0.931 train) lifts it **above** the 0.88 reasoning floor that its static
q_prior (0.74) can't clear — so LinUCB would route reasoning:cost to mistral,
which scores **1.000 over 22 held-out rows** and is ~2.5× cheaper than the
current pick (gpt-5-5, $8 vs $20 total/mtok). This is the LinUCB mechanism
working as designed (empirical evidence unlocking a cheaper, proven arm). It is
**uncertifiable today** only because the baseline (gpt-5-5) was never served in
reasoning:cost → no head-to-head. This is the highest-value cell to target with
the ε-exploration floor (AIN-388, already live) so a future replay-gate run can
certify it.

---

## What must happen before a re-run can PASS

- **Exploration data.** Let the live ε-exploration floor (AIN-388) accrue
  served observations for the non-mistral arms in the cost cells (esp.
  reasoning:cost gpt-5-5 vs mistral) so positivity holds.
- **Failed-call exclusion in the projector.** `export_outcomes.py` should drop
  (or separately model) `outcome_status='failed%'` rows — reliability is not
  quality. (Gate already does this for its own means; the training projector
  should match.)
- **Judge integrity (W2 AIN-396 + L8).** Resolve whether the uniform-1/5
  frontier scores are real or judge bias before they train the bandit.
- Then re-run this gate. PASS requires ≥1 certified `LINUCB_WIN`, zero
  regressions, zero collapses.

**The live cutover remains a founder gate (W5#4) regardless of a future PASS —
this report informs that decision; it does not make it.**
