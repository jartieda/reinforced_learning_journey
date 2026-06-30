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

# ── Resolve remote project root -------------------------------------------
# New launcher path: /workspace/robot_nav/reinforced
# Legacy path:       /workspace/reinforced
REMOTE_ROOT=$(ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "
  if [ -d /workspace/robot_nav/reinforced ]; then
    echo /workspace/robot_nav/reinforced
  elif [ -d /workspace/reinforced ]; then
    echo /workspace/reinforced
  else
    echo ''
  fi
" 2>/dev/null || true)

if [[ -z "$REMOTE_ROOT" ]]; then
  echo "WARNING: Could not find remote reinforced directory."
  echo "Checked: /workspace/robot_nav/reinforced and /workspace/reinforced"
  echo "Skipping sync."
  exit 0
fi
echo "Remote root detected: $REMOTE_ROOT"

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
if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "[ -d $REMOTE_ROOT/runs ]" 2>/dev/null; then
  rsync -az --progress \
    -e "ssh $SSH_OPTS -p $SSH_PORT" \
    "root@$SSH_HOST:$REMOTE_ROOT/runs/" \
    "$ROOT_DIR/runs/"
else
  echo "No remote runs/ directory found at $REMOTE_ROOT/runs (nothing to sync yet)."
fi

if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "[ -d $REMOTE_ROOT/videos ]" 2>/dev/null; then
  echo ""
  echo "Syncing videos/ ..."
  rsync -az --progress \
    -e "ssh $SSH_OPTS -p $SSH_PORT" \
    "root@$SSH_HOST:$REMOTE_ROOT/videos/" \
    "$ROOT_DIR/videos/"
fi

echo ""
echo "Sync complete. Local runs/ is up to date."
echo "  TensorBoard: tensorboard --logdir runs/"
echo ""
echo "Training still running? Check with:"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'tmux attach -t train'"
