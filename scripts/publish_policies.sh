#!/usr/bin/env bash
# AIN-550 · publish the ACTIVE LinUCB policy artifact to object storage so the serving
# tier (Railway, ephemeral FS) can fetch it (the api consumer reads it via
# AINFERA_POLICIES_URL + AINFERA_POLICIES_TOKEN). Uploads the referenced {version}.json
# FIRST, then ACTIVE.json, so a concurrent reader never sees ACTIVE point at a missing
# version. Idempotent (x-upsert). This is the go-live publish step — founder-gated.
#
# Env:
#   AINFERA_POLICIES_DIR          local dir with ACTIVE.json + {version}.json (default /var/ainfera/policies)
#   AINFERA_POLICIES_STORAGE      object base URL, e.g.
#                                 https://<ref>.supabase.co/storage/v1/object/policy-artifacts
#   AINFERA_POLICIES_STORAGE_KEY  service-role key (sent as bearer + apikey) for the private bucket
#
# Wiring: run after a promoting cadence (training_cadence.sh --apply-promote), or as an
# explicit publish step once you've inspected the candidate.
set -euo pipefail

DIR="${AINFERA_POLICIES_DIR:-/var/ainfera/policies}"
: "${AINFERA_POLICIES_STORAGE:?set AINFERA_POLICIES_STORAGE (object base URL)}"
: "${AINFERA_POLICIES_STORAGE_KEY:?set AINFERA_POLICIES_STORAGE_KEY (service-role key)}"

ACTIVE="$DIR/ACTIVE.json"
[[ -f "$ACTIVE" ]] || { echo "publish_policies: no ACTIVE.json at $DIR" >&2; exit 2; }
VERSION="$(python3 -c "import json;print(json.load(open('$ACTIVE'))['version'])")"
ARTIFACT="$DIR/${VERSION}.json"
[[ -f "$ARTIFACT" ]] || { echo "publish_policies: missing artifact $ARTIFACT" >&2; exit 2; }
BASE="${AINFERA_POLICIES_STORAGE%/}"

_upload() {  # <local-path> <object-name>
  curl -fsS -X POST "$BASE/$2" \
    -H "Authorization: Bearer ${AINFERA_POLICIES_STORAGE_KEY}" \
    -H "apikey: ${AINFERA_POLICIES_STORAGE_KEY}" \
    -H "x-upsert: true" -H "Content-Type: application/json" \
    --data-binary "@$1" >/dev/null
  echo "publish_policies: uploaded $2"
}

_upload "$ARTIFACT" "${VERSION}.json"   # version artifact first
_upload "$ACTIVE"   "ACTIVE.json"       # then flip the pointer
echo "publish_policies: published ${VERSION} -> ${BASE}"
