# AGENTS.md — operating contract for routing

Audience: the Ainfera fleet (Aulë and peers) and any agent operating this repo. Read before touching code.

## Identity
- **Core** — agent-native **inference routing**: the outcome-aware routing runtime (`ainfera_routing`) + policy templates + methodology for production agent workloads.
- Source of truth for names: the Naming law (`hizrianraz/obsidian/_ontology/Naming.md`, v1.3).

## Naming (law v1.3) — use these exactly
- Canonical wire model string: **`ainfera-inference`**. `ainfera-mithril` / `ainfera-auto` are **silent legacy aliases** — public docs/examples should show the canonical string and mention the aliases as legacy.
- **ATS (Ainfera Trust Score) is retired** — quality is now Routing **`q_empirical`** (implicit outcome signals + judge). Use `q_empirical` / "quality floor" in new policy templates, not "ATS".

## §0 Premise verification (mandatory)
Open every change with an explicit PASS/FAIL probe (clean tree? correct remote? tests green?) **before** editing. A failed premise → halt and surface; never fix-forward.

## Definition of done — verified, not PR proof
```bash
uv run pytest          # router resolver, learning/q_empirical, policy-template validation
```
Done = tests green and the policy templates validate. PR opened ≠ shipped.

## Secrets — hard rules
- `.env` is gitignored. No service keys belong in this repo. Never commit or echo a secret value.

## License
Apache-2.0. © Ainfera Inc. 2026.
