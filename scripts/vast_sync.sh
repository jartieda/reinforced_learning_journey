#!/usr/bin/env bash
# =============================================================================
# vast_sync.sh — rsync training results from the active vast.ai instance
#
# Usage:
#   bash scripts/vast_sync.sh            # sync runs/ and videos/ to local
#   bash scripts/vast_sync.sh --log      # also tail the live training log
#
# Reads instance ID from .vast_instance_id (written by vast_launch.sh).
# Safe to run multiple times while training is still in progress.
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
  echo "ERROR: .vast_instance_id not found. Did you run vast_launch.sh?"
  exit 1
fi
# shellcheck source=/dev/null
source "$ID_FILE"
echo "Syncing from instance $INSTANCE_ID ($SSH_HOST:$SSH_PORT) ..."

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# ── Show live training log (optional) -------------------------------------
TAIL_LOG=0
for arg in "$@"; do
  [[ "$arg" == "--log" ]] && TAIL_LOG=1
done

if [[ $TAIL_LOG -eq 1 ]]; then
  echo "Tailing live log (Ctrl+C to stop, results will still sync) ..."
  ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "tail -f /workspace/train.log" || true
fi

# ── Rsync results back to local machine -----------------------------------
echo ""
echo "Syncing runs/ ..."
rsync -az --progress \
  -e "ssh $SSH_OPTS -p $SSH_PORT" \
  "root@$SSH_HOST:/workspace/reinforced/runs/" \
  "$ROOT_DIR/runs/"

if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "[ -d /workspace/reinforced/videos ]" 2>/dev/null; then
  echo ""
  echo "Syncing videos/ ..."
  rsync -az --progress \
    -e "ssh $SSH_OPTS -p $SSH_PORT" \
    "root@$SSH_HOST:/workspace/reinforced/videos/" \
    "$ROOT_DIR/videos/"
fi

echo ""
echo "Sync complete. Local runs/ is up to date."
echo "  TensorBoard: tensorboard --logdir runs/"
echo ""
echo "Training still running? Check with:"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'tmux attach -t train'"
