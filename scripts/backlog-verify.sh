#!/usr/bin/env bash
# Backlog verification — run all production smokes (post v1.8).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAIL=0

run() {
  echo ""
  echo "━━━ $1 ━━━"
  if (cd "$2" && bash "$3"); then
    echo "OK: $1"
  else
    echo "FAIL: $1"
    FAIL=$((FAIL + 1))
  fi
}

run "E2E (routing)" "$ROOT/routing/scripts" "./ainfera-e2e.sh"
run "MCP tools" "$ROOT/mcp-server/cloudflare" "./smoke-mcp.sh"
run "MCP keyed inference" "$ROOT/mcp-server/cloudflare" "./smoke-mcp-keyed.sh"

for repo in ainfera-hermes ainfera-openclaw ainfera-langchain ainfera-crewai \
  ainfera-google-adk ainfera-letta ainfera-langgraph ainfera-llamaindex \
  ainfera-openai-compatible; do
  script="$ROOT/$repo/curl-example.sh"
  if [[ -x "$script" ]]; then
    run "adapter:$repo" "$ROOT/$repo" "./curl-example.sh"
  fi
done

echo ""
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS — backlog verification complete"
  exit 0
fi
echo "FAIL — $FAIL check(s) failed"
exit 1
