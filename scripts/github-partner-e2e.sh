#!/usr/bin/env bash
# ============================================================
# github-partner-e2e.sh — the GitHub design-partner GOLDEN PATH gate.
#
# Charter: "GitHub design-partner golden path · E2E · payment deferred".
# PASS = a design partner goes  signup → inference → audit  on the FREE tier,
#        with ZERO payment calls anywhere in the path.
#
# The ONE interactive leg (GitHub OAuth in a browser) cannot be scripted, so this
# harness starts from the artifacts that leg produces and proves steps 2–6 live.
#
# Capture the key first (once, in a browser):
#   1. open  ${API}/v1/users/github/connect   (or app.ainfera.ai "Continue with GitHub")
#   2. complete GitHub OAuth → you land on app.ainfera.ai with a session cookie
#   3. the onboarding page shows the key ONCE; or grab it yourself:
#        curl -s --cookie "ainfera_session=<cookie>" ${API}/v1/users/github/api-key
#
# Then run:
#   AINFERA_API_KEY=ainfera_... HANDLE=<your-github-login> ./github-partner-e2e.sh
#   MODE=json AINFERA_API_KEY=ainfera_... HANDLE=... ./github-partner-e2e.sh
#
# Optionally pass the session cookie to let the script reveal the key for you:
#   AINFERA_SESSION="<cookie>" HANDLE=<login> ./github-partner-e2e.sh
#
# Exits: 0 — all pass · 1 — critical (no key / inference / audit broken) · 2 — partial
# ============================================================

set -uo pipefail

API="${AINFERA_API:-https://api.ainfera.ai}"
MODE="${MODE:-pretty}"
VERBOSE="${AINFERA_VERBOSE:-0}"
KEY="${AINFERA_API_KEY:-}"
HANDLE="${HANDLE:-}"
SESSION="${AINFERA_SESSION:-}"
WIRE_MODEL="${AINFERA_WIRE_MODEL:-ainfera-inference}"  # Naming law v1.3 — NEVER a vendor pin

AGENT_ID=""
PHASES_PASS=0
PHASES_FAIL=0
PHASES_SKIP=0
PHASE_LOG=()

c_reset='\033[0m'; c_dim='\033[2m'; c_green='\033[32m'; c_red='\033[31m'
c_yellow='\033[33m'; c_blue='\033[34m'; c_bold='\033[1m'

say() { [[ "$MODE" == "json" ]] && return 0; echo -e "$1"; }
verbose() { [[ "$VERBOSE" != "1" || "$MODE" == "json" ]] && return 0; echo -e "${c_dim}  \$ $1${c_reset}"; }
phase_pass() { PHASES_PASS=$((PHASES_PASS+1)); PHASE_LOG+=("{\"phase\":\"$1\",\"status\":\"pass\",\"detail\":\"$2\"}"); say "  ${c_green}✓${c_reset} $1 — $2"; }
phase_fail() { PHASES_FAIL=$((PHASES_FAIL+1)); PHASE_LOG+=("{\"phase\":\"$1\",\"status\":\"fail\",\"detail\":\"$2\"}"); say "  ${c_red}✗${c_reset} ${c_red}$1${c_reset} — $2"; }
phase_skip() { PHASES_SKIP=$((PHASES_SKIP+1)); PHASE_LOG+=("{\"phase\":\"$1\",\"status\":\"skip\",\"detail\":\"$2\"}"); say "  ${c_yellow}~${c_reset} $1 — ${c_dim}skipped: $2${c_reset}"; }
heading() { say ""; say "${c_bold}${c_blue}━━━ $1 ━━━${c_reset}"; }
crit_exit() { say ""; say "${c_red}${c_bold}CRITICAL${c_reset} — $1"; [[ "$MODE" == "json" ]] && echo "{\"verdict\":\"FAIL\",\"critical\":\"$1\"}"; exit 1; }

# ============================================================
# PHASE 0 — preflight: the routed model is LISTED (W1 deployed)
# ============================================================
heading "0. Preflight — golden-path surfaces are live"

verbose "curl ${API}/v1/models"
models="$(curl -sS --max-time 10 "${API}/v1/models")"
listed="$(echo "$models" | jq -r --arg m "$WIRE_MODEL" '[.[] | select(.slug == $m)] | length')"
listed_type="$(echo "$models" | jq -r --arg m "$WIRE_MODEL" 'first(.[] | select(.slug == $m) | .type) // ""')"
if [[ "$listed" -ge 1 && "$listed_type" == "router" ]]; then
  phase_pass "0.1 /v1/models lists ${WIRE_MODEL}" "type=router (W1 deployed)"
else
  # Not fatal to the golden path — the router accepts the string regardless —
  # but the listing is the W1 DoD, so flag it loudly.
  phase_fail "0.1 /v1/models lists ${WIRE_MODEL}" "not listed as type=router (W1 not deployed?)"
fi

# ============================================================
# PHASE 1 — capture the key from the OAuth leg
# ============================================================
heading "1. Identity — GitHub OAuth artifact (key captured)"

if [[ -z "$KEY" && -n "$SESSION" ]]; then
  verbose "curl --cookie ainfera_session=… ${API}/v1/users/github/api-key"
  reveal="$(curl -sS --max-time 10 --cookie "ainfera_session=${SESSION}" "${API}/v1/users/github/api-key")"
  KEY="$(echo "$reveal" | jq -r '.api_key // empty')"
fi
[[ -z "$KEY" ]] && crit_exit "no AINFERA_API_KEY (complete the browser OAuth leg + capture the key — see header)"
[[ -z "$HANDLE" ]] && crit_exit "no HANDLE (your GitHub login) — needed for the wallet lookup"

if [[ "$KEY" == ainfera_* ]]; then
  phase_pass "1.1 key shape" "ainfera_* key captured for handle '${HANDLE}'"
else
  phase_fail "1.1 key shape" "key does not start with ainfera_ (AIN-152 / AIN-368)"
fi

# Optional: confirm the session resolves to a user (W3 /me).
if [[ -n "$SESSION" ]]; then
  me="$(curl -sS --max-time 10 --cookie "ainfera_session=${SESSION}" "${API}/v1/users/github/me")"
  me_handle="$(echo "$me" | jq -r '.github_handle // empty')"
  [[ -n "$me_handle" ]] && phase_pass "1.2 /users/github/me" "session → ${me_handle}" \
    || phase_skip "1.2 /users/github/me" "session cookie absent/expired"
else
  phase_skip "1.2 /users/github/me" "no session cookie passed"
fi

# ============================================================
# PHASE 2 — free wallet seeded (preview credit > 0)
# ============================================================
heading "2. Free wallet — preview credit seeded ( > 0 )"

verbose "curl -H 'Authorization: Bearer …' ${API}/v1/users/${HANDLE}/wallets"
wallets="$(curl -sS --max-time 15 -H "Authorization: Bearer ${KEY}" "${API}/v1/users/${HANDLE}/wallets")"
total="$(echo "$wallets" | jq -r '.total_balance_usd // "0"')"
expires="$(echo "$wallets" | jq -r 'first(.wallets[].credit_expires_at) // ""')"
has_balance=0
python3 -c "import sys; sys.exit(0 if float('${total}') > 0 else 1)" 2>/dev/null && has_balance=1
if [[ "$has_balance" == "1" ]]; then
  phase_pass "2.1 wallet balance" "\$${total} free credit · expires ${expires:-n/a} (90-day grant)"
else
  phase_fail "2.1 wallet balance" "total_balance_usd=${total} — expected > 0 (provisioning ran?)"
fi

# ============================================================
# PHASE 3 — /v1/chat/completions on the routed string
# ============================================================
heading "3. Routed inference — /v1/chat/completions (model=${WIRE_MODEL})"

cc_body="{\"model\":\"${WIRE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: alive\"}],\"max_tokens\":80}"
verbose "curl -X POST ${API}/v1/chat/completions -d '${cc_body}'"
cc="$(curl -sS --max-time 45 -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${KEY}" -d "$cc_body" "${API}/v1/chat/completions")"
cc_text="$(echo "$cc" | jq -r '.choices[0].message.content // empty')"
cc_model="$(echo "$cc" | jq -r '.model // "?"')"
cc_receipt="$(echo "$cc" | jq -r '.ainfera.receipt_id // empty')"
AGENT_ID="$(echo "$cc" | jq -r '.ainfera.agent_id // empty')"
if [[ -n "$cc_text" && -n "$cc_receipt" ]]; then
  phase_pass "3.1 chat/completions" "routed→${cc_model} · receipt=${cc_receipt:0:8}… · \"$(echo "$cc_text" | head -c 24)\""
else
  phase_fail "3.1 chat/completions" "no content/receipt: $(echo "$cc" | head -c 160)"
fi

# ============================================================
# PHASE 4 — /v1/inference on the routed string
# ============================================================
heading "4. Routed inference — /v1/inference (model=${WIRE_MODEL})"

inf_body="{\"model\":\"${WIRE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"one word: ok\"}],\"max_tokens\":80}"
verbose "curl -X POST ${API}/v1/inference -d '${inf_body}'"
inf="$(curl -sS --max-time 45 -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${KEY}" -d "$inf_body" "${API}/v1/inference")"
inf_text="$(echo "$inf" | jq -r '.content // .text // .choices[0].message.content // empty')"
inf_provider="$(echo "$inf" | jq -r '.provider // .ainfera.provider // "?"')"
inf_receipt="$(echo "$inf" | jq -r '.receipt_id // .ainfera.receipt_id // empty')"
[[ -z "$AGENT_ID" ]] && AGENT_ID="$(echo "$inf" | jq -r '.ainfera.agent_id // .agent_id // empty')"
if [[ -n "$inf_text" && -n "$inf_receipt" ]]; then
  phase_pass "4.1 inference" "provider=${inf_provider} · receipt=${inf_receipt:0:8}…"
else
  phase_fail "4.1 inference" "no content/receipt: $(echo "$inf" | head -c 160)"
fi

# ============================================================
# PHASE 5 — audit chain verifies
# ============================================================
heading "5. Audit — hash chain verifies (public, no auth)"

if [[ -z "$AGENT_ID" ]]; then
  phase_fail "5.1 audit verify" "no agent_id surfaced from the inference receipts"
else
  sleep 2  # let async audit writes settle
  verbose "curl ${API}/v1/audit/${AGENT_ID}/verify"
  verify="$(curl -sS --max-time 15 "${API}/v1/audit/${AGENT_ID}/verify")"
  valid="$(echo "$verify" | jq -r '.valid // false')"
  vcount="$(echo "$verify" | jq -r '.event_count // 0')"
  if [[ "$valid" == "true" ]]; then
    phase_pass "5.1 audit verify" "valid=true across ${vcount} events (agent ${AGENT_ID:0:8}…)"
  else
    phase_fail "5.1 audit verify" "valid=false ($(echo "$verify" | jq -r '.failure_reason // "?"'))"
  fi
fi

# ============================================================
# PHASE 6 — payment-free invariant
# ============================================================
heading "6. Free tier — NO payment rail in the path"

# By construction this harness calls ZERO payment endpoints. We also assert the
# platform agrees the rails are dormant, so the partner truly reached inference on
# free credit only (settlement rails OUT OF SCOPE for this charter).
verbose "curl ${API}/v1/payments/status"
pay="$(curl -sS --max-time 10 "${API}/v1/payments/status" 2>/dev/null || echo '{}')"
pay_live="$(echo "$pay" | jq -r '.live // .payments_live // false' 2>/dev/null)"
if [[ "$pay_live" != "true" ]]; then
  phase_pass "6.1 payment-free" "0 payment calls in path · rails dormant (payments_live=${pay_live})"
else
  # Rails being live doesn't break the golden path, but the charter is "payment
  # deferred" — surface it so the founder knows the posture changed.
  phase_skip "6.1 payment-free" "payments_live=true — path still used free credit, but rails are no longer dormant"
fi

# ============================================================
# FINAL REPORT
# ============================================================
if [[ "$MODE" == "json" ]]; then
  phases_json="$(IFS=,; echo "${PHASE_LOG[*]}")"
  cat <<JSON
{
  "gate": "github-partner-golden-path",
  "target": {"api": "${API}", "handle": "${HANDLE}", "wire_model": "${WIRE_MODEL}"},
  "agent_id": "${AGENT_ID}",
  "summary": {"pass": ${PHASES_PASS}, "fail": ${PHASES_FAIL}, "skip": ${PHASES_SKIP}},
  "phases": [${phases_json}],
  "verdict": "$([[ $PHASES_FAIL -eq 0 ]] && echo "PASS" || echo "FAIL")"
}
JSON
else
  say ""
  say "${c_bold}━━━ Golden-path gate ━━━${c_reset}"
  say "  ${c_green}Pass: ${PHASES_PASS}${c_reset}  ${c_red}Fail: ${PHASES_FAIL}${c_reset}  ${c_yellow}Skip: ${PHASES_SKIP}${c_reset}"
  say ""
  if [[ $PHASES_FAIL -eq 0 ]]; then
    say "${c_green}${c_bold}🎉 PASS — signup → inference → audit on free credit, zero payment.${c_reset}"
  else
    say "${c_red}${c_bold}✗ FAIL — ${PHASES_FAIL} checks broke. The golden path is not green.${c_reset}"
  fi
fi

[[ $PHASES_FAIL -eq 0 ]] && exit 0
[[ $PHASES_FAIL -ge 3 ]] && exit 1
exit 2
