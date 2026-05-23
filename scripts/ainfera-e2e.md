---
canonical: true
version: 1.0
written: 2026-05-17 Sun 12:00 WIB
audience: AI agents AND humans (dual-mode prompt)
target: full production ainfera.ai (L1–L5 + discovery + MCP + dashboard)
mode: read + write (creates one anonymous test agent in production)
cost: ~$0.001 USD (free 100 inferences cover all paths)
runtime: ~3 minutes for full E2E
exit_codes: 0 all-pass · 1 critical-fail · 2 partial-pass
license: MIT (do whatever)
spec: https://github.com/ainfera-ai/specs
---

# Ainfera Production E2E — One Super Prompt for AI Agents AND Humans

**The Hello World of Ainfera.** Hand this single document to any AI agent or human, and in under 3 minutes they will exercise every L1–L5 surface of production `ainfera.ai`, leaving a hash-chained audit trail anyone can verify offline.

Same script. Same output format. Same proof. **Dual-mode.**

---

## Who is reading this?

| You are... | Jump to |
|---|---|
| an **AI agent** running autonomously | [§5 Bash script](#5-the-bash-script) → set `MODE=json` → parse JSON report |
| a **human developer** evaluating Ainfera | [§5 Bash script](#5-the-bash-script) → set `MODE=pretty` (default) → read along |
| a **VC / auditor** verifying claims | [§4 What this proves](#4-what-this-test-proves), then run the script yourself |
| a **prospective design partner** seeing a demo | [§8 60-second demo mode](#8-60-second-demo-mode) |
| **Varda or another dogfood agent** smoke-testing | [§5 Bash script](#5-the-bash-script) → run nightly cron |
| a **Claude Code / Cursor session** asked to test Ainfera | [§9 Agent execution contract](#9-agent-execution-contract) |

---

## 1. Why this exists

Every production AI infra company makes claims. Most of those claims aren't independently verifiable. Ainfera's design thesis is the opposite: **trust no one, verify the chain yourself**. This script is that thesis as a single bash file.

The script is also Ainfera's **Hello World**. It is the shortest path from "never heard of this" to "I have a production-signed AgentCard, a drain-proof wallet, a hash-chained audit trail, and an MCP server I can point Claude Desktop at."

Memory lock 2026-05-16 PM: **Done = curl-200 AND fresh-incognito browser-render-200**. This script enforces that for the entire L1–L5 stack in one shot.

---

## 2. What this test will do to your account / the production database

**Honest disclosure** — this script is **not a dry-run**. It exercises production. Specifically it will:

| Action | Production effect | Reversible? |
|---|---|---|
| Anonymous signup with a unique `e2e-{timestamp}-{rand}` handle | Creates one agent row in `agents` table + 100 free inferences | No (agent persists; identifies as test by handle prefix) |
| ~10 inference calls across 4 model families | Spends ~$0.001 of margin from the 100 free inferences | No |
| 1 spend-policy POST (set $0.01 per-call cap) | Adds 1 row to spend-policy table | No |
| 1 deliberately-too-expensive inference call (the drain-proof demo) | Emits 1 `inference.rejected_cap_violation` audit event | No |
| ~15 `AuditEvent` rows | Hash-chained into the public audit chain at `api.ainfera.ai/v1/audit/public` | No (audit chains are immutable by design) |
| 0 wallet topups | No real money | n/a |
| 0 settlement transactions | No x402, no Stripe (memory lock #17: deferred 2 weeks) | n/a |

**You can skip the signup phase by pre-setting `AINFERA_API_KEY=ai_infera_...` in your environment.** The script will reuse your existing agent and only emit audit events under your handle.

**`ainfera-e2e` is now a soft-reserved handle prefix** to make AIN-118 segregation easier when Week-1 lands. Test agents created by this script use that prefix.

---

## 3. Prerequisites

```bash
# Required (must all return paths):
command -v curl
command -v jq
command -v python3

# Optional but nice (richer output):
command -v openssl     # cryptographic JWS verification
command -v base64      # JWS payload decode
```

Minimum: bash 4+, curl 7.70+, jq 1.6+, python3 3.10+. macOS, Linux, WSL all work.

---

## 4. What this test proves

When the script exits 0, you have empirically verified the following 12 claims about production `ainfera.ai`:

| # | Claim | Verified by |
|---|-------|-------------|
| 1 | Discovery surfaces match live reality | `mcp.json.status` = "live" AND `mcp.ainfera.ai/health` = 200 |
| 2 | API is reachable + descriptors served | `api.ainfera.ai/` returns service JSON + `/health` = ok |
| 3 | Model catalog has ≥ 11 entries | `GET /v1/models` returns ≥ 11 |
| 4 | Provider list has ≥ 10 active adapters | `GET /v1/providers` returns ≥ 10 with `active=true` |
| 5 | Anonymous self-signup works (L1 identity) | `POST /v1/agents/signup` returns 200 + `ai_infera_*` key + agent_id |
| 6 | Issued AgentCard is JWS-signed (RFC 7515) | `GET /v1/agents/{id}/card` returns 3-segment JWS |
| 7 | Wallet starts with 100 free inferences (L3 settlement) | `GET /v1/wallets/{id}` shows `free_remaining = 100` |
| 8 | Inference works across ≥ 4 model families (L2 routing) | 4 different `provider` values in successful responses |
| 9 | Drain-proof wallet rejects cap violations | After spend-policy + over-cap call → `inference.rejected_cap_violation` event |
| 10 | Audit chain is hash-linked + verifiable (L4) | Each event's `prev_hash` = SHA-256 of prior event's canonical form |
| 11 | Public audit mirror shows our events | Our events appear in `GET /v1/audit/public?limit=20` |
| 12 | MCP server speaks streamable-http (L5) | `mcp.ainfera.ai/health` = 200 AND `POST /mcp` with `Accept: application/json, text/event-stream` returns 200 |

If all 12 pass: the script prints `🎉 12/12 PASS — Ainfera production verified end-to-end.`

If anything fails: the script prints exactly which claim broke, with the failing curl command, so you can reproduce.

---

## 5. The bash script

The runnable script lives next to this doc as [`ainfera-e2e.sh`](./ainfera-e2e.sh). Save it, `chmod +x`, run.

See that file for the full implementation; this doc describes the contract and expected behavior.

---

## 6. Expected output (human mode)

```
━━━ 1. Discovery surfaces (read-only) ━━━
  ✓ 1.1 mcp.json — status=live available=true
  ✓ 1.2 agent-card.json — mcp_url=https://mcp.ainfera.ai · 5 layers · issued 2026-05-17T00:00:00Z
  ✓ 1.3 llms.txt — 1282 bytes · mentions mcp.ainfera.ai
  ✓ 1.4 audit-ticker — status=ok · chain_height=801

━━━ 2. API health (public, read-only) ━━━
  ✓ 2.1 /health — status=ok
  ✓ 2.2 api root — service descriptor JSON present
  ✓ 2.3 /v1/models — 11 models in catalog
  ✓ 2.4 /v1/providers — 10 active providers

━━━ 3. L1 Identity — anonymous self-signup ━━━
  ✓ 3.1 signup — received ai_infera_* key + agent_id=a3f7c182...

━━━ 4. L1 Identity — JWS-signed AgentCard ━━━
  ✓ 4.1 agent-card.jws — 3-segment JWS (RFC 7515) for agent a3f7c182...

━━━ 5. L3 Settlement — wallet starts with free tier ━━━
  ✓ 5.1 wallet — 100 free inferences remaining (anonymous tier)

━━━ 6. L2 Routing — first inference (claude-haiku-4-5) ━━━
  ✓ 6.1 inference — provider=anthropic · reply: "alive" · receipt=18e4a290...

━━━ 7. L2 Routing — same key, 4 model families ━━━
  ✓ 7.1 routing[gpt-5-5] — provider=openai
  ✓ 7.2 routing[gemini-3-1-pro] — provider=gemini
  ✓ 7.3 routing[mistral-medium-3] — provider=mistral

━━━ 8. L3 Settlement — drain-proof per-call cap ━━━
  ✓ 8.1 spend-policy — per_call_cap=$0.0001 set on agent
  ✓ 8.2 drain-proof — HTTP 402 · reason: per_call_cap_exceeded

━━━ 9. L4 Audit — hash-chained AuditEvents ━━━
  ✓ 9.1 audit/{agent_id} — 14 events emitted by this agent
  ✓ 9.2 cap-violation event — inference.rejected_cap_violation event present in chain
  ✓ 9.3 hash chain — prev_hash linkage intact across 14 events

━━━ 10. L4 Audit — public mirror ━━━
  ✓ 10.1 public mirror — 7 of our events visible at /v1/audit/public
  ✓ 10.2 chain height — public chain at seq 815

━━━ 11. L5 Orchestration — MCP server ━━━
  ✓ 11.1 mcp.ainfera.ai/health — ok
  ✓ 11.2 mcp initialize — protocol=2024-11-05 · server=ainfera-mcp

━━━ 12. Dashboard reachability (humans see this; agents skip) ━━━
  ✓ 12.1 app.ainfera.ai — HTTP 307 (login redirect or rendered)

━━━ Report card ━━━

  Test run:        ainfera-e2e-20260517-115530-9201
  Agent created:   a3f7c182b9d44e8c...
  Audit events:    14 emitted under this agent
  Public ticker:   chain at seq 815

  Pass: 18
  Fail: 0
  Skip: 0

🎉 PASS — Ainfera production verified end-to-end.

  Your agent: https://app.ainfera.ai/agents/ainfera-e2e-20260517-115530-9201
  Your audit: https://api.ainfera.ai/v1/audit/a3f7c182b9d44e8c...
  Public mirror: https://api.ainfera.ai/v1/audit/public
```

---

## 7. Expected output (agent mode, `MODE=json`)

Pipeline-friendly JSON; parse with `jq` or any agent runtime. Example:

```json
{
  "test_run_id": "ainfera-e2e-20260517-115530-9201",
  "started_at": "2026-05-17T11:55:30Z",
  "target": {
    "api": "https://api.ainfera.ai",
    "web": "https://ainfera.ai",
    "mcp": "https://mcp.ainfera.ai",
    "app": "https://app.ainfera.ai"
  },
  "agent": {
    "agent_id": "a3f7c182-b9d4-4e8c-...",
    "handle": "ainfera-e2e-20260517-115530-9201",
    "test_name": "ainfera-e2e-20260517-115530-9201"
  },
  "summary": {
    "pass": 18,
    "fail": 0,
    "skip": 0,
    "audit_events_seen": 14
  },
  "phases": [
    {"phase": "1.1 mcp.json", "status": "pass", "detail": "status=live available=true"},
    {"phase": "1.2 agent-card.json", "status": "pass", "detail": "mcp_url=https://mcp.ainfera.ai · 5 layers · issued 2026-05-17T00:00:00Z"}
  ],
  "verdict": "PASS"
}
```

Agents should branch on `.verdict == "PASS"`. On `FAIL`, inspect `.phases[] | select(.status=="fail")` for actionable detail.

---

## 8. 60-second demo mode

For Derek, prospective design partners, or video shoots. Strip to 4 commands. Pure proof, no narration.

```bash
# Demo Mode — 60 seconds, the essential drain-proof story.
# Costs: 0 cents on the free tier.

API=https://api.ainfera.ai
NAME=demo-$(date +%s)

# 1. Sign up (5 seconds)
SIGNUP=$(curl -sS -X POST "$API/v1/agents/signup" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$NAME\",\"use_case\":\"demo\"}")
KEY=$(echo "$SIGNUP" | jq -r '.api_key')
AID=$(echo "$SIGNUP" | jq -r '.agent_id')
echo "→ Got Agent Card + drain-proof wallet. Key: ${KEY:0:20}..."

# 2. First inference (15 seconds)
curl -sS -X POST "$API/v1/inference" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"one word reply: alive"}]}' \
  | jq '.content, .provider, .receipt_id'

# 3. Set $0.0001 cap, attempt expensive call → 402 PAYMENT REQUIRED (the moment)
curl -sS -X POST "$API/v1/agents/$AID/spend-policy" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"per_call_cap_usd":0.0001}' >/dev/null

echo "→ Now attempting a $0.50 call against a $0.0001 cap..."
curl -sS -X POST "$API/v1/inference" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"claude-opus-4-7","messages":[{"role":"user","content":"500 word essay"}]}' \
  | jq '.detail, .reason'
echo "→ DRAIN PROOF. Wallet doesn't read prompts. It just refuses."

# 4. The audit chain proves it (10 seconds)
echo "→ Hash-chained audit trail anyone can verify:"
curl -sS "$API/v1/audit/$AID" \
  | jq '.events | sort_by(.seq) | .[-5:] | .[] | {seq, type: .event_type, hash: .event_hash[:16]}'

echo ""
echo "https://app.ainfera.ai/agents/$NAME"
```

That's the conversion artifact. **3 minutes from `curl signup` to "wallet rejected a drain attempt, here's the hash."** Record once, attach to every cold DM.

---

## 9. Agent execution contract

If you are an AI agent (Claude, GPT, Gemini, Grok, Mistral, or any other) executing this prompt autonomously, follow this contract:

1. **Do not embellish.** Run the script as-is. Do not "improve" it without explicit founder permission.
2. **Set `MODE=json`** if your output is being parsed by another agent. Set `MODE=pretty` if your output is being read by a human.
3. **On failure**, output the failing phase's curl reproduction command so a human can debug. Do not retry silently more than 1x per phase.
4. **Do not delete the test agent** at the end. It is intentional production state. The handle prefix `ainfera-e2e-*` marks it as test data for AIN-118 segregation.
5. **Do not exceed 100 inferences** in one run. The free tier covers ~20 inferences this script will make. If you re-run, you will eventually hit the free-tier cap, which is a feature, not a bug.
6. **Honor `AINFERA_FAST=1`** if set — skip multi-provider Phase 7.
7. **Treat any HTTP 5xx response from `api.ainfera.ai` as an Ainfera-side issue**, not a script issue. Report it; do not blame your prompt.
8. **Per memory lock #11**: do not output the founder's name or PII in any logs you produce. The test agent is anonymous by design.

---

## 10. What "pass" means + what "fail" means

| Verdict | Meaning | Action |
|---|---|---|
| `🎉 12/12 PASS` | Every L1–L5 surface verified live. Production is shippable. | Continue with Monday launch sequence (AIN-115). |
| `≥ 1 FAIL on phase 1.x or 2.x` | Discovery or API surface broken. **Stop**. | Investigate Cloudflare DNS, Railway deploy, Vercel build status. |
| `≥ 1 FAIL on phase 3-9` | L1–L4 broken. **Stop**. | Investigate API logs on Railway, Supabase audit chain integrity. |
| `≥ 1 FAIL on phase 11` | MCP server broken. | Lower-priority; defer to Week-1 if launch is imminent, since MCP is L5 nice-to-have. |
| `≥ 1 SKIP on phase 7 or 11` | Test-mode bypass or protocol-only behavior. | OK; not a failure. |

**The strict launch gate**: zero failures on phases 1.x, 2.x, 3.x, 6.x, 8.x, 9.x. Everything else is recoverable post-launch.

---

## 11. Memory + Linear cross-refs

This script reflects these locks:

- **API key prefix `ai_infera_*`** (memory #16, locked 2026-05-16 PM)
- **Settlement deferred 2 weeks** (memory #17) — script doesn't test Stripe / x402 settlement; only the free-tier prepaid ledger
- **No Stripe/x402 code paths exercised Sunday** — exactly per #17
- **6 dogfood agents + the 5 canonical routing backends** are separate from this script — this script creates its own anonymous test agent (the AAMC "voter" framing was retired by AIN-243; the 5-model lock survives as routing backends)
- **Done = curl + browser verify** (memory #28) — this script IS the curl half; humans/agents can pair with `app.ainfera.ai` browser walk for the second half

Linear:
- Subsumes most of **AIN-114** (smoke script) — the same checks plus more
- Companion to **AIN-115** (Mon launch) — run this at 04:00 SGT before the Derek email
- Feeds **AIN-117/AIN-123** verification pattern — explicit curl proof per phase

---

## 12. Run it

### Anonymous (creates a new test agent)

```bash
chmod +x ainfera-e2e.sh
./ainfera-e2e.sh
```

### With your existing key

```bash
export AINFERA_API_KEY=ai_infera_your_key_here
./ainfera-e2e.sh
```

### JSON output for agent pipelines

```bash
MODE=json ./ainfera-e2e.sh | jq '.verdict'
# → "PASS"
```

### Fast mode (skip multi-provider, ~90 seconds total)

```bash
AINFERA_FAST=1 ./ainfera-e2e.sh
```

### Verbose (see every curl command)

```bash
AINFERA_VERBOSE=1 ./ainfera-e2e.sh
```

### Nightly cron for Varda

```cron
# /var/spool/cron/varda
0 4 * * * cd /home/varda && AINFERA_API_KEY=$(op read 'op://Ainfera/Varda Agent Key/credential') /home/varda/ainfera-e2e.sh > /var/log/ainfera-e2e-$(date +\%Y\%m\%d).log 2>&1
```

---

## 13. What this is NOT

- ❌ **Not a load test.** This is a correctness test. For load, use k6 or Locust against `/v1/inference` with concurrency.
- ❌ **Not a security audit.** This verifies functionality, not threat model. For security audits, fuzz signup, inject malformed JWS, test rate limits.
- ❌ **Not a billing test.** Free tier only. Stripe + x402 deferred 2 weeks per memory #17.
- ❌ **Not a dashboard E2E.** Phase 12 only checks reachability. For full dashboard UX, use Playwright against `app.ainfera.ai`.
- ❌ **Not a regression suite.** No baseline diffs, no historic comparison. For regression, archive each run's JSON output and diff seq counts week-over-week.

For all five of the above, file as Week-1 P2 tickets if needed.

---

## 14. Endgame

Hand this document — the whole markdown including the bash script — to:

- **Derek Goh** in the Monday 09:00 SGT email body. "Run this. Or copy-paste it to ChatGPT and ask it to run it. Same proof either way."
- **Cocoon / Wavemaker / Sequoia Arc Asia / Antler / Hustle Fund** in the warm-intro forward.
- **HN/X/LinkedIn launch posts**. "Here is the Hello World of Ainfera. Run it yourself."
- **Cold DM #6 in the 60-target list**. "30 seconds: `curl signup`. 60 seconds: drain-proof demo."
- **Varda's nightly cron**. Production smoke runs continuously.
- **Future CC Cursor sessions**. "Verify production before any deploy that touches L1–L5."

**One artifact. Two audiences. Twelve verifiable claims. Three minutes.**

End of super prompt.
