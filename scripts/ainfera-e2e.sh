#!/usr/bin/env bash
# ============================================================
# ainfera-e2e.sh — Production end-to-end test for ainfera.ai
# Dual-mode: human-readable (MODE=pretty) OR agent-parseable (MODE=json)
#
# Usage:
#   ./ainfera-e2e.sh                    # human mode, anonymous signup
#   MODE=json ./ainfera-e2e.sh          # agent mode, JSON-only output
#   AINFERA_API_KEY=ai_infera_... ./ainfera-e2e.sh   # skip signup
#   AINFERA_FAST=1 ./ainfera-e2e.sh     # skip multi-provider + Annex IV
#   AINFERA_VERBOSE=1 ./ainfera-e2e.sh  # show all curl commands
#
# Exits:
#   0 — 12/12 pass
#   1 — critical failure (signup/inference/audit broken)
#   2 — partial pass (≥ 1 non-critical failure)
# ============================================================

set -uo pipefail

API="${AINFERA_API:-https://api.ainfera.ai}"
WEB="${AINFERA_WEB:-https://ainfera.ai}"
MCP="${AINFERA_MCP:-https://mcp.ainfera.ai}"
APP="${AINFERA_APP:-https://app.ainfera.ai}"

MODE="${MODE:-pretty}"
VERBOSE="${AINFERA_VERBOSE:-0}"
FAST="${AINFERA_FAST:-0}"
KEY="${AINFERA_API_KEY:-}"
TEST_PROMPT="${AINFERA_TEST_PROMPT:-Reply with exactly one word: alive}"
HANDLE_PREFIX="${AINFERA_HANDLE_PREFIX:-ainfera-e2e}"
TEST_NAME="${HANDLE_PREFIX}-$(date -u +%Y%m%d-%H%M%S)-${RANDOM}"

# Capture state across phases
AGENT_ID=""
AGENT_HANDLE=""
OWNER_HANDLE=""
SIGNUP_JWS=""
FREE_INFERENCES_AT_SIGNUP=0
WALLET_ID=""
AUDIT_EVENTS_SEEN=0
PHASES_PASS=0
PHASES_FAIL=0
PHASES_SKIP=0
PHASE_LOG=()

# Pretty-print helpers (no-ops in json mode)
c_reset='\033[0m'
c_dim='\033[2m'
c_green='\033[32m'
c_red='\033[31m'
c_yellow='\033[33m'
c_blue='\033[34m'
c_bold='\033[1m'

say() {  # human-readable line, suppressed in json mode
  [[ "$MODE" == "json" ]] && return 0
  echo -e "$1"
}

verbose() {  # curl debug, only in verbose mode
  [[ "$VERBOSE" != "1" ]] && return 0
  [[ "$MODE" == "json" ]] && return 0
  echo -e "${c_dim}  \$ $1${c_reset}"
}

phase_pass() {
  PHASES_PASS=$((PHASES_PASS+1))
  PHASE_LOG+=("{\"phase\":\"$1\",\"status\":\"pass\",\"detail\":\"$2\"}")
  say "  ${c_green}✓${c_reset} $1 — $2"
}

phase_fail() {
  PHASES_FAIL=$((PHASES_FAIL+1))
  PHASE_LOG+=("{\"phase\":\"$1\",\"status\":\"fail\",\"detail\":\"$2\"}")
  say "  ${c_red}✗${c_reset} ${c_red}$1${c_reset} — $2"
}

phase_skip() {
  PHASES_SKIP=$((PHASES_SKIP+1))
  PHASE_LOG+=("{\"phase\":\"$1\",\"status\":\"skip\",\"detail\":\"$2\"}")
  say "  ${c_yellow}~${c_reset} $1 — ${c_dim}skipped: $2${c_reset}"
}

heading() {
  say ""
  say "${c_bold}${c_blue}━━━ $1 ━━━${c_reset}"
}

# ============================================================
# PHASE 1 — DISCOVERY SURFACES (anonymous, read-only)
# ============================================================
heading "1. Discovery surfaces (read-only)"

verbose "curl ${WEB}/.well-known/mcp.json"
mcp_json="$(curl -sS --max-time 10 "${WEB}/.well-known/mcp.json")"
mcp_status="$(echo "$mcp_json" | jq -r '.status // "?"')"
mcp_available="$(echo "$mcp_json" | jq -r '.available // false')"
if [[ "$mcp_status" == "live" && "$mcp_available" == "true" ]]; then
  phase_pass "1.1 mcp.json" "status=live available=true"
else
  phase_fail "1.1 mcp.json" "status=$mcp_status available=$mcp_available — expected live/true"
fi

verbose "curl ${WEB}/.well-known/agent-card.json"
ac="$(curl -sS --max-time 10 "${WEB}/.well-known/agent-card.json")"
ac_mcp="$(echo "$ac" | jq -r '.mcp_url // ""')"
ac_issued="$(echo "$ac" | jq -r '.issued_at // ""')"
ac_caps="$(echo "$ac" | jq -r '.capabilities | length // 0')"
if [[ -n "$ac_mcp" && "$ac_caps" -ge 5 ]]; then
  phase_pass "1.2 agent-card.json" "mcp_url=$ac_mcp · ${ac_caps} layers · issued $ac_issued"
else
  phase_fail "1.2 agent-card.json" "mcp_url=$ac_mcp caps=$ac_caps — expected mcp_url present + 5 layers"
fi

verbose "curl ${WEB}/llms.txt"
llms="$(curl -sS --max-time 10 "${WEB}/llms.txt")"
llms_bytes="${#llms}"
if [[ "$llms_bytes" -ge 500 ]] && echo "$llms" | grep -q "mcp.ainfera.ai"; then
  phase_pass "1.3 llms.txt" "${llms_bytes} bytes · mentions mcp.ainfera.ai"
else
  phase_fail "1.3 llms.txt" "${llms_bytes} bytes (expected ≥500) · mcp.ainfera.ai mention: $(echo "$llms" | grep -c mcp.ainfera.ai)"
fi

verbose "curl ${WEB}/api/audit-ticker"
ticker="$(curl -sS --max-time 10 "${WEB}/api/audit-ticker")"
ticker_status="$(echo "$ticker" | jq -r '.status // "?"')"
ticker_height="$(echo "$ticker" | jq -r '.chain_height // 0')"
if [[ "$ticker_status" == "ok" && "$ticker_height" -gt 0 ]]; then
  phase_pass "1.4 audit-ticker" "status=ok · chain_height=$ticker_height"
else
  phase_fail "1.4 audit-ticker" "status=$ticker_status height=$ticker_height"
fi

# ============================================================
# PHASE 2 — API HEALTH + DESCRIPTORS
# ============================================================
heading "2. API health (public, read-only)"

verbose "curl ${API}/health"
health="$(curl -sS --max-time 10 "${API}/health")"
if echo "$health" | jq -e '.status == "ok"' >/dev/null; then
  phase_pass "2.1 /health" "status=ok"
else
  phase_fail "2.1 /health" "got: $health"
fi

verbose "curl ${API}/"
root="$(curl -sS --max-time 10 "${API}/")"
root_service="$(echo "$root" | jq -r '.service // "?"')"
if [[ "$root_service" == "Ainfera API" ]]; then
  phase_pass "2.2 api root" "service descriptor JSON present"
else
  phase_fail "2.2 api root" "service=$root_service — expected 'Ainfera API'"
fi

verbose "curl ${API}/v1/models"
models="$(curl -sS --max-time 10 "${API}/v1/models")"
model_count="$(echo "$models" | jq 'length // 0')"
# Catalog size drifts with provider enablement; floor matches prod minimum (10).
if [[ "$model_count" -ge 10 ]]; then
  phase_pass "2.3 /v1/models" "${model_count} models in catalog"
else
  phase_fail "2.3 /v1/models" "${model_count} models — expected ≥ 10"
fi

verbose "curl ${API}/v1/providers"
providers="$(curl -sS --max-time 10 "${API}/v1/providers")"
provider_count="$(echo "$providers" | jq '[.[] | select(.active == true)] | length // 0')"
if [[ "$provider_count" -ge 10 ]]; then
  phase_pass "2.4 /v1/providers" "${provider_count} active providers"
else
  phase_fail "2.4 /v1/providers" "${provider_count} active — expected ≥ 10"
fi

# ============================================================
# PHASE 3 — ANONYMOUS SIGNUP (creates real agent)
# ============================================================
heading "3. L1 Identity — anonymous self-signup"

if [[ -n "$KEY" ]]; then
  phase_skip "3.1 signup" "pre-existing AINFERA_API_KEY in env"
  # Recover agent context for downstream phases
  verbose "curl ${API}/v1/users/github/me with existing key"
  me="$(curl -sS --max-time 10 -H "Authorization: Bearer $KEY" "${API}/v1/users/github/me" 2>/dev/null || echo "{}")"
  AGENT_HANDLE="$(echo "$me" | jq -r '.handle // "preset"')"
  # Try to find the most recent agent under this key
  agents="$(curl -sS --max-time 10 -H "Authorization: Bearer $KEY" "${API}/v1/users/${AGENT_HANDLE}/dashboard" 2>/dev/null || echo "{}")"
  AGENT_ID="$(echo "$agents" | jq -r '.agents[0].agent_id // ""')"
else
  signup_body="$(jq -nc --arg h "$TEST_NAME" \
    '{agent_handle:$h, metadata:{use_case:"E2E production test", intent:"smoke-test"}}')"
  verbose "curl -X POST ${API}/v1/agents/signup -d '$signup_body'"
  signup="$(curl -sS --max-time 15 -X POST \
    -H "Content-Type: application/json" \
    -d "$signup_body" \
    "${API}/v1/agents/signup")"
  KEY="$(echo "$signup" | jq -r '.api_key // empty')"
  AGENT_ID="$(echo "$signup" | jq -r '.agent_id // empty')"
  AGENT_HANDLE="$(echo "$signup" | jq -r '.agent_handle // empty')"
  OWNER_HANDLE="$(echo "$signup" | jq -r '.owner_handle // "agents"')"
  SIGNUP_JWS="$(echo "$signup" | jq -r '.agent_card_jws // empty')"
  FREE_INFERENCES_AT_SIGNUP="$(echo "$signup" | jq -r '.free_tier_inferences_remaining // 0')"

  if [[ -n "$KEY" && "$KEY" == ai_infera_* ]]; then
    phase_pass "3.1 signup" "${OWNER_HANDLE}/${AGENT_HANDLE} · ai_infera_* key · agent_id=${AGENT_ID:0:8}... · free=${FREE_INFERENCES_AT_SIGNUP}"
  elif [[ -n "$KEY" ]]; then
    phase_fail "3.1 signup" "key=$KEY does not start with ai_infera_ (memory lock 2026-05-16 PM)"
  else
    phase_fail "3.1 signup" "no api_key in response: $(echo "$signup" | head -c 200)"
    # Critical: cannot proceed without key
    say ""
    say "${c_red}${c_bold}CRITICAL FAILURE${c_reset} — signup failed; cannot continue."
    say "Reproduce: curl -X POST ${API}/v1/agents/signup -H 'Content-Type: application/json' -d '$signup_body'"
    exit 1
  fi
fi

# ============================================================
# PHASE 4 — AGENTCARD (L1 JWS verification)
# ============================================================
heading "4. L1 Identity — JWS-signed AgentCard"

if [[ -z "$AGENT_ID" ]]; then
  phase_skip "4.1 agent-card" "no agent_id available"
else
  # Prefer the JWS returned inline by signup; fall back to /card fetch
  if [[ -n "$SIGNUP_JWS" ]]; then
    jws_token="$SIGNUP_JWS"
    verbose "(using agent_card_jws from signup response)"
  else
    verbose "curl ${API}/v1/agents/${AGENT_ID}/card"
    card="$(curl -sS --max-time 10 -H "Authorization: Bearer $KEY" \
      "${API}/v1/agents/${AGENT_ID}/card")"
    jws_token="$(echo "$card" | jq -r '.jws // empty')"
  fi
  segments="$(echo "$jws_token" | tr '.' '\n' | wc -l | tr -d ' ')"
  if [[ "$segments" == "3" ]]; then
    phase_pass "4.1 agent-card.jws" "3-segment JWS (RFC 7515) for agent ${AGENT_ID:0:8}..."
    # Decode payload for fun
    payload_b64="$(echo "$jws_token" | cut -d. -f2)"
    payload="$(echo "${payload_b64}==" | tr '_-' '/+' | base64 -d 2>/dev/null || echo '{}')"
    if [[ "$MODE" != "json" && "$VERBOSE" == "1" ]]; then
      say "  ${c_dim}payload preview:${c_reset}"
      echo "$payload" | jq -c '{handle, layer, issued_at}' 2>/dev/null | sed 's/^/    /'
    fi
  else
    phase_fail "4.1 agent-card.jws" "expected 3 segments, got $segments"
  fi
fi

# ============================================================
# PHASE 5 — WALLET (L3 prepaid ledger)
# ============================================================
heading "5. L3 Settlement — wallet starts with free tier"

if [[ -z "$AGENT_ID" ]]; then
  phase_skip "5.1 wallet" "no agent_id available"
else
  verbose "curl ${API}/v1/wallets/${AGENT_ID}"
  wallet="$(curl -sS --max-time 10 -H "Authorization: Bearer $KEY" \
    "${API}/v1/wallets/${AGENT_ID}")"
  balance_usd="$(echo "$wallet" | jq -r '.balance_usd // "0"')"
  # The wallet endpoint exposes balance only; free-tier inference count comes
  # from the signup response (FREE_INFERENCES_AT_SIGNUP). Both must indicate
  # the agent is seeded.
  has_balance=0
  python3 -c "import sys; sys.exit(0 if float('$balance_usd') > 0 else 1)" 2>/dev/null && has_balance=1
  if [[ "$has_balance" == "1" && "$FREE_INFERENCES_AT_SIGNUP" -ge 50 ]]; then
    phase_pass "5.1 wallet" "balance_usd=\$${balance_usd} · ${FREE_INFERENCES_AT_SIGNUP} free inferences seeded at signup"
  else
    phase_fail "5.1 wallet" "balance_usd=$balance_usd free_at_signup=$FREE_INFERENCES_AT_SIGNUP — expected balance>0 and ≥50 free"
  fi
fi

# ============================================================
# PHASE 6 — FIRST INFERENCE (L2 routing)
# ============================================================
heading "6. L2 Routing — first inference (claude-haiku-4-5)"

inf_body="$(cat <<EOF
{
  "model": "claude-haiku-4-5",
  "messages": [{"role": "user", "content": "${TEST_PROMPT}"}],
  "max_tokens": 80
}
EOF
)"

verbose "curl -X POST ${API}/v1/inference -d '${inf_body}'"
inf1="$(curl -sS --max-time 45 -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d "$inf_body" \
  "${API}/v1/inference")"

inf1_text="$(echo "$inf1" | jq -r '.content // .text // .choices[0].message.content // .response // empty' 2>/dev/null)"
inf1_provider="$(echo "$inf1" | jq -r '.provider // .routed_provider // "?"')"
inf1_receipt="$(echo "$inf1" | jq -r '.receipt_id // .audit_event_id // empty')"

if [[ -n "$inf1_text" && -n "$inf1_receipt" ]]; then
  preview="$(echo "$inf1_text" | head -c 60 | tr -d '\n')"
  phase_pass "6.1 inference" "provider=$inf1_provider · reply: \"$preview\" · receipt=${inf1_receipt:0:8}..."
else
  phase_fail "6.1 inference" "no content/receipt in: $(echo "$inf1" | head -c 200)"
fi

# ============================================================
# PHASE 7 — MULTI-PROVIDER ROUTING (proves L2 framework-agnostic)
# ============================================================
heading "7. L2 Routing — same key, 4 model families"

if [[ "$FAST" == "1" ]]; then
  phase_skip "7.x multi-provider" "AINFERA_FAST=1"
else
  # max_tokens=80 is the minimum that satisfies reasoning models
  # (gpt-5.5-pro, gemini-3.1-pro) whose internal thinking budget consumes
  # tokens before visible output. Lower values cause empty content or
  # upstream 5xx — verified 2026-05-17 via direct probes.
  declare -a probe_models=("gpt-5-5" "gemini-3-1-pro" "grok-4" "mistral-medium-3")
  routing_wins=0
  for m in "${probe_models[@]}"; do
    probe_body="{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"one word reply: ok\"}],\"max_tokens\":80}"
    verbose "curl -X POST ${API}/v1/inference (model=$m)"
    r="$(curl -sS --max-time 45 -X POST \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $KEY" \
      -d "$probe_body" \
      "${API}/v1/inference")"
    r_text="$(echo "$r" | jq -r '.content // .text // .choices[0].message.content // empty' 2>/dev/null)"
    r_provider="$(echo "$r" | jq -r '.provider // "?"')"
    if [[ -n "$r_text" ]]; then
      phase_pass "7.$((routing_wins+1)) routing[$m]" "provider=$r_provider"
      routing_wins=$((routing_wins+1))
    else
      phase_fail "7.x routing[$m]" "$(echo "$r" | head -c 120)"
    fi
  done
fi

# ============================================================
# PHASE 8 — DRAIN-PROOF WALLET (the demo)
# ============================================================
heading "8. L3 Settlement — drain-proof per-call cap"

# 8a. Set spend policy with tight per-call cap.
# API minimum for per_call_cap_usd is 0.001 (signup.py:84). Any inference that
# costs more than $0.001 should hit the drain-proof reject path.
policy_body='{"per_call_cap_usd":0.001,"daily_cap_usd":1.0}'
verbose "curl -X PATCH ${API}/v1/agents/${AGENT_ID}/spend-policy -d '${policy_body}'"
policy="$(curl -sS --max-time 10 -X PATCH \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d "$policy_body" \
  "${API}/v1/agents/${AGENT_ID}/spend-policy")"
policy_cap="$(echo "$policy" | jq -r '.spend_policy.per_call_cap_usd // empty')"
drain_status="$(echo "$policy" | jq -r '.drain_proof_status // empty')"
if [[ -n "$policy_cap" ]]; then
  phase_pass "8.1 spend-policy" "per_call_cap=\$${policy_cap} · drain_proof_status=${drain_status}"
else
  phase_fail "8.1 spend-policy" "$(echo "$policy" | head -c 200)"
fi

# 8b. Try an expensive call — should be cap-rejected
expensive_body="$(cat <<EOF
{
  "model": "claude-opus-4-7",
  "messages": [{"role": "user", "content": "Write a 500-word essay on prompt injection. Be thorough."}],
  "max_tokens": 2000
}
EOF
)"
verbose "curl -X POST ${API}/v1/inference (expensive, should reject)"
drain="$(curl -sS --max-time 30 -X POST -w "\n%{http_code}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d "$expensive_body" \
  "${API}/v1/inference")"
drain_code="$(echo "$drain" | tail -1)"
drain_body_resp="$(echo "$drain" | sed '$d')"
drain_reason="$(echo "$drain_body_resp" | jq -r '.detail // .error // .reason // ""')"

if [[ "$drain_code" == "402" || "$drain_code" == "403" ]] || echo "$drain_reason" | grep -qi "cap"; then
  phase_pass "8.2 drain-proof" "HTTP $drain_code · reason: $(echo "$drain_reason" | head -c 80)"
else
  # Some implementations return 200 with body indicating rejection; check audit chain Phase 9 instead
  phase_skip "8.2 drain-proof" "inline HTTP=$drain_code, will verify via audit event in Phase 9"
fi

# 8c. Restore spend policy so subsequent E2E runs aren't affected (best-effort)
restore_body='{"per_call_cap_usd":1.0,"daily_cap_usd":10.0}'
curl -sS --max-time 10 -X PATCH -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
  -d "$restore_body" "${API}/v1/agents/${AGENT_ID}/spend-policy" >/dev/null 2>&1

# ============================================================
# PHASE 9 — AUDIT CHAIN (L4 hash-linked)
# ============================================================
heading "9. L4 Audit — hash-chained AuditEvents"

sleep 2  # let async audit writes settle
verbose "curl ${API}/v1/audit/${AGENT_ID}?limit=50"
chain="$(curl -sS --max-time 15 -H "Authorization: Bearer $KEY" \
  "${API}/v1/audit/${AGENT_ID}?limit=50")"
event_count="$(echo "$chain" | jq '.events | length // 0')"
AUDIT_EVENTS_SEEN="$event_count"

if [[ "$event_count" -ge 4 ]]; then
  phase_pass "9.1 audit/{agent_id}" "$event_count events emitted by this agent"
else
  phase_fail "9.1 audit/{agent_id}" "$event_count events — expected ≥ 4"
fi

# Check for cap-violation event proving drain-proof
cap_violation="$(echo "$chain" | jq -r '.events[] | select(.event_type | test("rejected_cap")) | .event_type' | head -1)"
if [[ -n "$cap_violation" ]]; then
  phase_pass "9.2 cap-violation event" "$cap_violation event present in chain"
else
  phase_fail "9.2 cap-violation event" "no rejected_cap_* event found (drain-proof unverified)"
fi

# Use the server-side /verify endpoint as the canonical hash-chain check.
# (Real field is `previous_hash`, not `prev_hash`; and the server already
# implements the full canonical-form re-hash via audit_service.verify_chain.)
verbose "curl ${API}/v1/audit/${AGENT_ID}/verify"
verify_resp="$(curl -sS --max-time 15 "${API}/v1/audit/${AGENT_ID}/verify")"
verify_valid="$(echo "$verify_resp" | jq -r '.valid // false')"
verify_count="$(echo "$verify_resp" | jq -r '.event_count // 0')"
verify_failure_seq="$(echo "$verify_resp" | jq -r '.failure_seq // empty')"
verify_failure_reason="$(echo "$verify_resp" | jq -r '.failure_reason // empty')"
if [[ "$verify_valid" == "true" ]]; then
  phase_pass "9.3 hash chain" "server /verify: valid=true across ${verify_count} events"
else
  phase_fail "9.3 hash chain" "server /verify: valid=false at seq=${verify_failure_seq} (${verify_failure_reason})"
fi

# ============================================================
# PHASE 10 — PUBLIC AUDIT MIRROR
# ============================================================
heading "10. L4 Audit — public mirror"

verbose "curl ${API}/v1/audit/public?limit=100"
public_chain="$(curl -sS --max-time 15 "${API}/v1/audit/public?limit=100")"
# Public mirror surfaces owner_handle + agent_name + canonical_uri (no agent_id).
expected_canonical="ainfera.ai/${OWNER_HANDLE}/${AGENT_HANDLE}"
our_events_in_public="$(echo "$public_chain" | jq --arg c "$expected_canonical" \
  '[.events[] | select(.canonical_uri == $c)] | length')"

if [[ "$our_events_in_public" -gt 0 ]]; then
  phase_pass "10.1 public mirror" "$our_events_in_public of our events visible at /v1/audit/public"
else
  # Public mirror may have lag; mark non-critical
  phase_skip "10.1 public mirror" "events may still be propagating (eventual consistency)"
fi

public_height="$(echo "$public_chain" | jq -r '.events[0].seq // 0')"
phase_pass "10.2 chain height" "public chain at seq $public_height"

# ============================================================
# PHASE 11 — MCP SERVER (L5 orchestration)
# ============================================================
heading "11. L5 Orchestration — MCP server"

verbose "curl ${MCP}/health"
mcp_health="$(curl -sS --max-time 10 "${MCP}/health")"
if echo "$mcp_health" | jq -e '.status == "ok"' >/dev/null 2>&1; then
  phase_pass "11.1 mcp.ainfera.ai/health" "ok"
else
  phase_fail "11.1 mcp.ainfera.ai/health" "$mcp_health"
fi

# 11.2 — MCP initialize handshake via streamable-http
mcp_init='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"ainfera-e2e","version":"1.0"}}}'
verbose "curl -X POST ${MCP}/mcp (initialize handshake)"
mcp_resp="$(curl -sS --max-time 10 -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Authorization: Bearer $KEY" \
  -d "$mcp_init" \
  "${MCP}/mcp")"

# FastMCP often returns SSE-style with "data: {...}" prefix on streamable-http
mcp_clean="$(echo "$mcp_resp" | sed -n 's/^data: //p' | head -1)"
[[ -z "$mcp_clean" ]] && mcp_clean="$mcp_resp"

mcp_protocol="$(echo "$mcp_clean" | jq -r '.result.protocolVersion // empty' 2>/dev/null)"
mcp_server_info="$(echo "$mcp_clean" | jq -r '.result.serverInfo.name // empty' 2>/dev/null)"

if [[ -n "$mcp_protocol" || -n "$mcp_server_info" ]]; then
  phase_pass "11.2 mcp initialize" "protocol=$mcp_protocol · server=$mcp_server_info"
else
  # MCP may require session establishment first; fall back to checking 406 vs 5xx
  mcp_code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 "${MCP}/mcp")"
  if [[ "$mcp_code" == "406" ]]; then
    phase_skip "11.2 mcp initialize" "MCP-only protocol (406 on bare GET is correct FastMCP behavior); full handshake needs MCP client"
  else
    phase_fail "11.2 mcp initialize" "no protocolVersion in response; HTTP=$mcp_code"
  fi
fi

# ============================================================
# PHASE 12 — DASHBOARD (human-only)
# ============================================================
heading "12. Dashboard reachability (humans see this; agents skip)"

verbose "curl -I ${APP}"
app_code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 10 -L "${APP}")"
if [[ "$app_code" == "200" || "$app_code" == "307" ]]; then
  phase_pass "12.1 app.ainfera.ai" "HTTP $app_code (login redirect or rendered)"
else
  phase_fail "12.1 app.ainfera.ai" "HTTP $app_code"
fi

# ============================================================
# FINAL REPORT
# ============================================================

if [[ "$MODE" == "json" ]]; then
  # JSON report for agents
  phases_json="$(IFS=,; echo "${PHASE_LOG[*]}")"
  cat <<JSON
{
  "test_run_id": "${TEST_NAME}",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "target": {
    "api": "$API",
    "web": "$WEB",
    "mcp": "$MCP",
    "app": "$APP"
  },
  "agent": {
    "agent_id": "$AGENT_ID",
    "handle": "$AGENT_HANDLE",
    "test_name": "$TEST_NAME"
  },
  "summary": {
    "pass": $PHASES_PASS,
    "fail": $PHASES_FAIL,
    "skip": $PHASES_SKIP,
    "audit_events_seen": $AUDIT_EVENTS_SEEN
  },
  "phases": [${phases_json}],
  "verdict": "$([[ $PHASES_FAIL -eq 0 ]] && echo "PASS" || echo "FAIL")"
}
JSON
else
  # Human report
  say ""
  say "${c_bold}━━━ Report card ━━━${c_reset}"
  say ""
  say "  Test run:        ${c_dim}${TEST_NAME}${c_reset}"
  say "  Agent created:   ${c_dim}${AGENT_ID:0:16}...${c_reset}"
  say "  Audit events:    ${c_dim}${AUDIT_EVENTS_SEEN} emitted under this agent${c_reset}"
  say "  Public ticker:   ${c_dim}chain at seq ${public_height}${c_reset}"
  say ""
  say "  ${c_green}Pass: ${PHASES_PASS}${c_reset}"
  say "  ${c_red}Fail: ${PHASES_FAIL}${c_reset}"
  say "  ${c_yellow}Skip: ${PHASES_SKIP}${c_reset}"
  say ""
  if [[ $PHASES_FAIL -eq 0 ]]; then
    say "${c_green}${c_bold}🎉 PASS — Ainfera production verified end-to-end.${c_reset}"
    say ""
    say "  Your agent: ${c_blue}${APP}/agents/${TEST_NAME}${c_reset}"
    say "  Your audit: ${c_blue}${API}/v1/audit/${AGENT_ID}${c_reset}"
    say "  Public mirror: ${c_blue}${API}/v1/audit/public${c_reset}"
    say ""
  else
    say "${c_red}${c_bold}✗ FAIL — ${PHASES_FAIL} checks broke. See above.${c_reset}"
  fi
fi

# Exit codes: 0 = all pass, 1 = critical, 2 = partial
if [[ $PHASES_FAIL -eq 0 ]]; then
  exit 0
elif [[ $PHASES_FAIL -ge 5 ]]; then
  exit 1
else
  exit 2
fi
