#!/usr/bin/env bash
# =============================================================================
# vast_destroy.sh — final sync + destroy the active vast.ai instance
#
# Usage:
#   bash scripts/vast_destroy.sh          # sync results, then destroy
#   bash scripts/vast_destroy.sh --now    # destroy immediately (no sync)
#
# IMPORTANT: destroying an instance is IRREVERSIBLE. All data on the instance
# is lost. This script syncs runs/ and videos/ first by default.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ID_FILE="$ROOT_DIR/.vast_instance_id"
ENV_FILE="$ROOT_DIR/.env"

# ── Load secrets -----------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found"; exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

if [[ -z "${VAST_AI_API_KEY:-}" ]]; then
  echo "ERROR: VAST_AI_API_KEY not set in .env"; exit 1
fi

vastai set api-key "$VAST_AI_API_KEY" 2>/dev/null

# ── Read instance ID -------------------------------------------------------
if [[ ! -f "$ID_FILE" ]]; then
  echo "ERROR: .vast_instance_id not found."
  echo "Destroy manually:  vastai destroy instance <id>"
  exit 1
fi
# shellcheck source=/dev/null
source "$ID_FILE"

# ── Parse args ------------------------------------------------------------
SKIP_SYNC=0
for arg in "$@"; do
  [[ "$arg" == "--now" ]] && SKIP_SYNC=1
done

echo "================================================================"
echo " About to DESTROY instance $INSTANCE_ID"
echo " This is IRREVERSIBLE — the instance will be deleted permanently."
echo "================================================================"
echo ""

# ── Optional final sync ---------------------------------------------------
if [[ $SKIP_SYNC -eq 0 ]]; then
  echo "Step 1/2: Final sync of results ..."
  bash "$SCRIPT_DIR/vast_sync.sh" || {
    echo ""
    echo "WARNING: Sync failed (instance may already be stopped)."
    read -r -p "Continue with destroy anyway? [y/N] " CONFIRM
    [[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
  }
else
  echo "Skipping sync (--now flag set)."
fi

# ── Confirm before destroy ------------------------------------------------
echo ""
read -r -p "Destroy instance $INSTANCE_ID? [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "Aborted. Instance is still running (costs will continue)."
  exit 0
fi

# ── Destroy ---------------------------------------------------------------
echo ""
echo "Step 2/2: Destroying instance $INSTANCE_ID ..."
vastai destroy instance "$INSTANCE_ID"

# Clean up local state
rm -f "$ID_FILE"

echo ""
echo "================================================================"
echo " Instance $INSTANCE_ID destroyed."
echo " Local results are in runs/ and videos/"
echo " TensorBoard: tensorboard --logdir runs/"
echo "================================================================"
