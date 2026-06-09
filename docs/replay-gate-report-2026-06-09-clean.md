# AIN-335 · LinUCB offline replay-gate report

- generated: 2026-06-09
- source: routing_outcomes (prod, Supabase dftfpwzqxoebwzepygzl) · CLEAN corpus post AIN-403 re-judge · labeled+reward · cannot_evaluate-excluded · degraded-excluded · per-cell temporal split 0.7 · catalog=enrolled-5 · floors=api CELL_MIN_QUALITY (AIN-388)
- gate constants: MIN_TRAIN=5, MIN_SUPPORT=5, WIN_MARGIN=0.05

## Point of record — the verdict changed *meaning*, not *value* (2026-06-08 → 2026-06-09)

This is the re-run on the **clean** corpus, after AIN-403 (the Responses-API
envelope-extraction bug) was fixed, deployed, and the contaminated rows
re-judged. It supersedes `replay-gate-report-2026-06-08.md` as the current
evidence. The verdict is still `FAIL_CLOSED` — but **for a different reason**, and
that distinction is the whole point.

| | 06-08 (poisoned corpus) | 06-09 (clean corpus) |
|---|---|---|
| Verdict | FAIL_CLOSED | FAIL_CLOSED |
| Tally | NO_CHANGE=7, UNCERTIFIABLE=4 | **NO_CHANGE=8, UNCERTIFIABLE=3** |
| `reasoning:quality` | UNCERTIFIABLE — gpt-5-5 holdout **0.0000(10)**, `judge_mono_zero` flagged | **NO_CHANGE** — gpt-5-5 holdout **0.6172(32)**, flag **cleared** |
| Blocking class | label integrity **+** positivity | **positivity only** |

- **AIN-403 dividend, gate-proven.** The poisoned `reasoning:quality` cell is now
  clean and stable: gpt-5-5's held-out reward recovered from a parsing-artifact
  0.0 to 0.617, the greedy pick is unchanged, and the `judge_mono_zero` flag on
  gpt-5-5 is gone. The fix did exactly what it should and nothing more.
- **The remaining 3 UNCERTIFIABLE verdicts are all positivity/overlap** — every
  flip routes to (or away from) an arm with **no head-to-head holdout support**.
  This is not a labels problem and not poison; it is a *coverage* problem.
- **FAIL_CLOSED is now a coverage verdict, not a label verdict.** Closing it
  needs held-out overlap on the floor-dropped-but-empirically-clearing arms
  (the `reasoning:cost → mistral-large-3` lead), which is what the inert
  `counterfactual_explore` path (PR #17 Part B) is built to create — *not* a
  re-judge or a floor change. LinUCB cutover remains founder-gated regardless.
- The surviving `judge_mono_zero:gemini-3-1-pro(n=24)` flag in `code:balanced`
  is **AIN-416** (the routed-capture probe truncates code outputs at
  `max_tokens=80`), a harness artifact — **not** an AIN-403 parsing sibling
  (those rows are `labeled` with real content + content-specific rationale; the
  genuinely-empty ones correctly went to `cannot_evaluate`).

Reproduce: `python3 scripts/replay_gate.py --bundle docs/replay-gate-bundle-2026-06-09-clean.json`

## VERDICT: **FAIL_CLOSED**

tally: NO_CHANGE=8, UNCERTIFIABLE_POSITIVITY=3

LinUCB is **NOT** certified for live cutover. The held-out evidence does not show a supported, positive improvement over the current coverage rule. Do not flip the wire-in. The blocking reasons + the one promising direction are below.

| cell | floor | coverage→ | linucb→ | flip | cov(holdout,n) | lin(holdout,n) | verdict |
|---|---|---|---|---|---|---|---|
| chat:cost | 0.58 | mistral-large-3 | mistral-large-3 | · | 0.7500(n=32) | 0.7500(n=32) | NO_CHANGE |
| code:balanced | 0.78 | gemini-3-1-pro | gpt-5-5 | YES | 0.0000(n=8) | 0.5000(n=1) | UNCERTIFIABLE_POSITIVITY |
| code:cost | 0.78 | gemini-3-1-pro | gemini-3-1-pro | · | —(n=0) | —(n=0) | NO_CHANGE |
| embed:cost | 0.52 | mistral-large-3 | mistral-large-3 | · | 0.0000(n=1) | 0.0000(n=1) | NO_CHANGE |
| extraction:cost | 0.58 | mistral-large-3 | mistral-large-3 | · | 0.9712(n=26) | 0.9712(n=26) | NO_CHANGE |
| general:cost | 0.68 | mistral-large-3 | mistral-large-3 | · | 0.9464(n=28) | 0.9464(n=28) | NO_CHANGE |
| reasoning:balanced | 0.88 | gpt-5-5 | gpt-5-5 | · | —(n=0) | —(n=0) | NO_CHANGE |
| reasoning:cost | 0.88 | gpt-5-5 | mistral-large-3 | YES | —(n=0) | 1.0000(n=21) | UNCERTIFIABLE_POSITIVITY |
| reasoning:quality | 0.88 | gpt-5-5 | gpt-5-5 | · | 0.6172(n=32) | 0.6172(n=32) | NO_CHANGE |
| tool_use:balanced | 0.78 | gemini-3-1-pro | gpt-5-5 | YES | 0.3571(n=7) | —(n=0) | UNCERTIFIABLE_POSITIVITY |
| tool_use:cost | 0.78 | gemini-3-1-pro | gemini-3-1-pro | · | —(n=0) | —(n=0) | NO_CHANGE |

### Per-cell notes
- **code:balanced**: flip but no head-to-head overlap; baseline gemini-3-1-pro=0.000(n=8), linucb gpt-5-5 UNSUPPORTED(n=1)  _flags_: judge_mono_zero:gemini-3-1-pro(n=24)
- **reasoning:cost**: flip but no head-to-head overlap; baseline gpt-5-5 UNSUPPORTED(n=0), linucb mistral-large-3=1.000(n=21)
- **tool_use:balanced**: flip but no head-to-head overlap; baseline gemini-3-1-pro=0.357(n=7), linucb gpt-5-5 UNSUPPORTED(n=0)
