#!/usr/bin/env bash
# AIN-335 Stage 4 · Spark cron wrapper for the daily training cadence.
#
# Expects env (Doppler spark_prd):
#   AINFERA_POLICIES_DIR   — versioned policy artifacts + ACTIVE.json
#   AINFERA_CADENCE_ROOT   — per-run workdirs (default /var/ainfera/cadence)
#   AINFERA_ROWS_DUMP      — today's routing_outcomes JSON dump path
#
# Usage (03:30 WIB anchor, after judge sweep):
#   ./scripts/training_cadence.sh
#   ./scripts/training_cadence.sh --apply-promote   # founder-gated live flip
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE_STAMP="$(date -u +%F)"
: "${AINFERA_POLICIES_DIR:?AINFERA_POLICIES_DIR must be set (versioned policy artifacts + ACTIVE.json)}"
POLICIES_DIR="${AINFERA_POLICIES_DIR}"
WORKROOT="${AINFERA_CADENCE_ROOT:-/var/ainfera/cadence}"
WORKDIR="${WORKROOT}/${DATE_STAMP}"
ROWS="${AINFERA_ROWS_DUMP:-/var/ainfera/dumps/routing_outcomes-${DATE_STAMP}.json}"
BUNDLE="${AINFERA_REPLAY_BUNDLE:-}"
APPLY=()

if [[ "${1:-}" == "--apply-promote" ]]; then
  APPLY=(--apply-promote)
fi

if [[ ! -f "$ROWS" ]]; then
  echo "cadence: missing rows dump: $ROWS" >&2
  exit 2
fi

CMD=(python3 "$ROOT/scripts/training_cadence.py" run
  --rows "$ROWS"
  --workdir "$WORKDIR"
  --policies-dir "$POLICIES_DIR"
  --source prod
  --judge-model "${AINFERA_JUDGE_MODEL:-claude-opus-4-7}"
  "${APPLY[@]}"
)

if [[ -n "$BUNDLE" && -f "$BUNDLE" ]]; then
  CMD+=(--replay-bundle "$BUNDLE")
fi

echo "cadence: rows=$ROWS workdir=$WORKDIR" >&2
"${CMD[@]}"
