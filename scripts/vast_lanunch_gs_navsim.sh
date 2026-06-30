#!/usr/bin/env bash
# =============================================================================
# vast_lanunch_gs_navsim.sh
#
# Provision (or reuse) a vast.ai GPU instance and run Example 10 fully remote:
#   - gs_navsim backend (Node.js)
#   - headless browser renderer (Google Chrome + Xvfb)
#   - PPO training (examples.10_gs_navsim_nomad.train)
#
# Usage:
#   bash scripts/vast_lanunch_gs_navsim.sh \
#     --scene-id my_scene \
#     --goal /workspace/robot_nav/reinforced/examples/10_gs_navsim_nomad/goal.png
#
# Required on local machine:
#   - reinforced/.env with VAST_AI_API_KEY
#   - vastai CLI installed (or available in reinforced/env_isaaclab/bin)
#
# Required on remote (this script validates):
#   - /workspace/data/<scene_id>/3dgs_uncompressed.ply OR 3dgs_compressed.ply
#   - /workspace/data/<scene_id>/occupancy.json
#   - /workspace/data/<scene_id>/occupancy.png
#   - /workspace/robot_nav/visualnav-transformer/model_weights/nomad.pth
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REINFORCED_DIR="$(dirname "$SCRIPT_DIR")"
ROBOT_NAV_DIR="$(dirname "$REINFORCED_DIR")"
ENV_FILE="$REINFORCED_DIR/.env"
ID_FILE="$REINFORCED_DIR/.vast_instance_id"

# ── Defaults ────────────────────────────────────────────────────────────────
GPU_FILTER="RTX_4090"
DISK_GB=80
IMAGE="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"
MAX_DPH=0.80
MIN_DPH=0.00
DATA_DIR_REMOTE="/workspace/data"
GOAL_REMOTE="/workspace/robot_nav/reinforced/examples/10_gs_navsim_nomad/goal.png"
TIMESTEPS=200000
MAX_STEPS=500
SEED=42
EXPERIMENT_NAME="10_gs_navsim_nomad"
CHECKPOINT=""
WS_URL="ws://127.0.0.1:8081"
SCENE_ID=""
SKIP_SYNC=0
SKIP_INSTALL=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/vast_lanunch_gs_navsim.sh [options]

Options:
  --scene-id <id>        Required. Scene folder name under remote data dir.
  --goal <path>          Goal image path inside remote instance.
                         Default: /workspace/robot_nav/reinforced/examples/10_gs_navsim_nomad/goal.png
  --data-dir <path>      Remote scene data directory root.
                         Default: /workspace/data
  --timesteps <int>      PPO total timesteps. Default: 200000
  --max-steps <int>      Max env steps per episode. Default: 500
  --seed <int>           RNG seed. Default: 42
  --experiment-name <s>  TensorBoard run label. Default: 10_gs_navsim_nomad
  --checkpoint <path>    Optional remote checkpoint to resume.
  --ws-url <url>         WebSocket URL from remote trainer. Default: ws://127.0.0.1:8081

  --gpu <name>           vast.ai GPU filter. Default: RTX_4090
  --disk <gb>            Instance disk size (GB). Default: 80
  --image <image>        Docker image. Default: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
  --max-dph <float>      Max $/hour. Default: 0.80
  --min-dph <float>      Min $/hour. Default: 0.00

  --skip-sync            Do not rsync local robot_nav to remote.
  --skip-install         Do not run apt/pip/npm setup on remote.

Examples:
  bash scripts/vast_lanunch_gs_navsim.sh \
    --scene-id office_01 \
    --goal /workspace/robot_nav/reinforced/examples/10_gs_navsim_nomad/goal.png

  bash scripts/vast_lanunch_gs_navsim.sh \
    --scene-id office_01 --timesteps 400000 --gpu RTX_5090 --max-dph 1.20
USAGE
}

# ── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-id)         SCENE_ID="$2"; shift 2 ;;
    --goal)             GOAL_REMOTE="$2"; shift 2 ;;
    --data-dir)         DATA_DIR_REMOTE="$2"; shift 2 ;;
    --timesteps)        TIMESTEPS="$2"; shift 2 ;;
    --max-steps)        MAX_STEPS="$2"; shift 2 ;;
    --seed)             SEED="$2"; shift 2 ;;
    --experiment-name)  EXPERIMENT_NAME="$2"; shift 2 ;;
    --checkpoint)       CHECKPOINT="$2"; shift 2 ;;
    --ws-url)           WS_URL="$2"; shift 2 ;;

    --gpu)              GPU_FILTER="$2"; shift 2 ;;
    --disk)             DISK_GB="$2"; shift 2 ;;
    --image)            IMAGE="$2"; shift 2 ;;
    --max-dph)          MAX_DPH="$2"; shift 2 ;;
    --min-dph)          MIN_DPH="$2"; shift 2 ;;

    --skip-sync)        SKIP_SYNC=1; shift ;;
    --skip-install)     SKIP_INSTALL=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SCENE_ID" ]]; then
  echo "ERROR: --scene-id is required"
  usage
  exit 1
fi

# ── Load secrets from .env ──────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found"
  echo "Create it with:"
  echo "  echo 'VAST_AI_API_KEY=your_key_here' > $ENV_FILE"
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

if [[ -z "${VAST_AI_API_KEY:-}" ]]; then
  echo "ERROR: VAST_AI_API_KEY is not set in $ENV_FILE"
  exit 1
fi

# ── Check vastai CLI ────────────────────────────────────────────────────────
if ! command -v vastai &>/dev/null; then
  VENV_VASTAI="$REINFORCED_DIR/env_isaaclab/bin/vastai"
  if [[ -x "$VENV_VASTAI" ]]; then
    export PATH="$REINFORCED_DIR/env_isaaclab/bin:$PATH"
    echo "Using vastai from env_isaaclab venv"
  else
    echo "ERROR: vastai CLI not found. Install with: pip install vastai"
    exit 1
  fi
fi

vastai set api-key "$VAST_AI_API_KEY" 2>/dev/null

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes"

# ── Reuse existing instance if possible ─────────────────────────────────────
SKIP_CREATE=0
if [[ -f "$ID_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ID_FILE"
  echo "Found existing instance: id=$INSTANCE_ID ($SSH_HOST:$SSH_PORT)"
  if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" &>/dev/null; then
    echo "Reusing existing instance"
    SKIP_CREATE=1
  else
    echo "Stored instance unreachable. Will create a new one."
    rm -f "$ID_FILE"
  fi
fi

if [[ $SKIP_CREATE -eq 0 ]]; then
  echo "Searching offers: gpu=$GPU_FILTER, dph=${MIN_DPH}-${MAX_DPH}, disk>=${DISK_GB}GB"
  OFFER_JSON=$(vastai search offers \
    --raw \
    --order dph_total \
    "gpu_name=$GPU_FILTER num_gpus=1 reliability>0.95 inet_down>100 disk_space>=$DISK_GB dph_total<=$MAX_DPH dph_total>=$MIN_DPH" \
    2>/dev/null)

  if [[ -z "$OFFER_JSON" ]]; then
    echo "ERROR: no offers found. Try another GPU or higher --max-dph"
    exit 1
  fi

  BANNED_MACHINES="9020"
  OFFER_ID=$(echo "$OFFER_JSON" | python3 -c "
import sys, json
banned = {int(x) for x in '$BANNED_MACHINES'.split() if x}
min_dph = float('$MIN_DPH')
offers = [o for o in json.load(sys.stdin) if o.get('machine_id') not in banned and o.get('dph_total', 0) >= min_dph]
if not offers:
    print('ERROR: no offers left after filters', file=sys.stderr)
    sys.exit(1)
print(offers[0]['ask_contract_id'])
")
  OFFER_DPH=$(echo "$OFFER_JSON" | python3 -c "
import sys, json
banned = {int(x) for x in '$BANNED_MACHINES'.split() if x}
min_dph = float('$MIN_DPH')
offers = [o for o in json.load(sys.stdin) if o.get('machine_id') not in banned and o.get('dph_total', 0) >= min_dph]
print(round(offers[0]['dph_total'], 4)) if offers else print('?')
" 2>/dev/null || echo "?")

  echo "Best offer: id=$OFFER_ID (\$$OFFER_DPH/hr)"
  echo "Creating instance (image=$IMAGE, disk=${DISK_GB}GB)..."

  CREATE_OUT=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    --direct \
    --env "-e PYTHONUNBUFFERED=1" \
    --raw 2>&1)

  INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])" 2>/dev/null \
               || echo "$CREATE_OUT" | grep -oP '"new_contract":\s*\K[0-9]+' | head -1)

  if [[ -z "$INSTANCE_ID" ]]; then
    echo "ERROR: failed to create instance"
    echo "$CREATE_OUT"
    exit 1
  fi

  echo "Instance created: id=$INSTANCE_ID"

  # Wait for SSH readiness
  echo "Waiting for SSH..."
  MAX_WAIT=600
  ELAPSED=0
  SSH_URL=""
  while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))

    SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
    if [[ -z "$SSH_URL" ]]; then
      continue
    fi

    SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://[^@]*@||' | cut -d: -f1)
    SSH_PORT=$(echo "$SSH_URL" | sed 's|.*:||')

    if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" &>/dev/null; then
      break
    fi
  done

  if [[ -z "$SSH_URL" ]] || ! ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" &>/dev/null; then
    echo "ERROR: timeout waiting for SSH"
    exit 1
  fi

  cat > "$ID_FILE" <<EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
EOF
  echo "Saved connection details to $ID_FILE"
fi

# shellcheck source=/dev/null
source "$ID_FILE"

echo ""
echo "Using instance: $INSTANCE_ID ($SSH_HOST:$SSH_PORT)"

# ── Sync project to remote ──────────────────────────────────────────────────
if [[ $SKIP_SYNC -eq 0 ]]; then
  echo "Syncing robot_nav to remote /workspace/robot_nav ..."
  ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "mkdir -p /workspace/robot_nav"

  rsync -az --progress \
    --exclude=".git" \
    --exclude=".venv" \
    --exclude="env_isaaclab" \
    --exclude="runs" \
    --exclude="videos" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".env" \
    -e "ssh $SSH_OPTS -p $SSH_PORT" \
    "$ROBOT_NAV_DIR/" \
    "root@$SSH_HOST:/workspace/robot_nav/"
else
  echo "Skipping sync (--skip-sync)"
fi

# ── Remote setup and launch ────────────────────────────────────────────────
REMOTE_CHECKPOINT_ARG=""
if [[ -n "$CHECKPOINT" ]]; then
  REMOTE_CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
fi

if [[ $SKIP_INSTALL -eq 0 ]]; then
  echo "Installing remote dependencies (apt, pip)..."
  # shellcheck disable=SC2087
  ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# Base system deps
apt-get update
apt-get install -y --no-install-recommends \
  python3-pip python3-dev build-essential \
  tmux xvfb curl ca-certificates gnupg

# Install Node.js 20 from NodeSource (required for modern JS syntax in server.js)
mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
  | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
chmod a+r /etc/apt/keyrings/nodesource.gpg
cat > /etc/apt/sources.list.d/nodesource.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main
EOF
apt-get update
apt-get install -y --no-install-recommends nodejs

# Install Google Chrome Stable (works reliably in Docker + Xvfb)
curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
  | gpg --dearmor -o /etc/apt/keyrings/google-chrome.gpg
chmod a+r /etc/apt/keyrings/google-chrome.gpg
cat > /etc/apt/sources.list.d/google-chrome.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main
EOF
apt-get update
apt-get install -y --no-install-recommends google-chrome-stable

# If Ubuntu's old node packages exist, remove them and fix deps.
apt-get remove -y libnode-dev libnode72 nodejs-doc 2>/dev/null || true
apt-get -f install -y || true

# Re-assert desired Node.js version after cleanup
apt-get install -y --no-install-recommends nodejs

python3 -m pip install --break-system-packages --no-cache-dir -r /workspace/robot_nav/reinforced/requirements.txt
cd /workspace/robot_nav/gs_navsim/backend
npm install
REMOTE
else
  echo "Skipping remote install (--skip-install)"
fi

echo "Validating remote assets..."
# shellcheck disable=SC2087
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<REMOTE
set -euo pipefail
SCENE_DIR="$DATA_DIR_REMOTE/$SCENE_ID"
if [[ ! -d "\$SCENE_DIR" ]]; then
  echo "ERROR: scene dir not found: \$SCENE_DIR"
  exit 1
fi
if [[ ! -f "\$SCENE_DIR/occupancy.json" ]]; then
  echo "ERROR: missing \$SCENE_DIR/occupancy.json"
  exit 1
fi
if [[ ! -f "\$SCENE_DIR/occupancy.png" ]]; then
  echo "ERROR: missing \$SCENE_DIR/occupancy.png"
  exit 1
fi
if [[ ! -f "\$SCENE_DIR/3dgs_uncompressed.ply" && ! -f "\$SCENE_DIR/3dgs_compressed.ply" ]]; then
  echo "ERROR: missing 3dgs_uncompressed.ply or 3dgs_compressed.ply in \$SCENE_DIR"
  exit 1
fi
if [[ ! -f "/workspace/robot_nav/visualnav-transformer/model_weights/nomad.pth" ]]; then
  echo "ERROR: missing NOMAD weights at /workspace/robot_nav/visualnav-transformer/model_weights/nomad.pth"
  exit 1
fi
if [[ ! -f "$GOAL_REMOTE" ]]; then
  echo "ERROR: missing goal image at $GOAL_REMOTE"
  exit 1
fi
REMOTE

echo "Starting remote gs_navsim backend (tmux: gs_backend)..."
# shellcheck disable=SC2087
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<REMOTE
set -euo pipefail
tmux has-session -t gs_backend 2>/dev/null && tmux kill-session -t gs_backend || true
tmux new-session -d -s gs_backend "
  export GS_DATA_DIR=$DATA_DIR_REMOTE
  cd /workspace/robot_nav/gs_navsim/backend
  node server.js 2>&1 | tee /workspace/gs_backend.log
"
REMOTE

echo "Starting remote headless browser (tmux: gs_browser)..."
# shellcheck disable=SC2087
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<'REMOTE'
set -euo pipefail
tmux has-session -t gs_browser 2>/dev/null && tmux kill-session -t gs_browser || true
tmux new-session -d -s gs_browser "
  export DISPLAY=:99
  # Reuse existing Xvfb if available; otherwise start a clean :99 display.
  if [[ -S /tmp/.X11-unix/X99 ]]; then
    :
  else
    rm -f /tmp/.X99-lock
    Xvfb :99 -screen 0 1280x720x24 > /tmp/xvfb.log 2>&1 &
    sleep 1
  fi
  BROWSER_BIN=/usr/bin/google-chrome-stable
  if [[ ! -x "\$BROWSER_BIN" ]]; then
    echo 'ERROR: /usr/bin/google-chrome-stable not found. Re-run without --skip-install.' >&2
    exit 1
  fi
  "\$BROWSER_BIN" \
    --no-sandbox \
    --disable-dev-shm-usage \
    --disable-gpu \
    --ignore-gpu-blocklist \
    --enable-webgl \
    --use-gl=swiftshader \
    --enable-unsafe-swiftshader \
    --no-first-run \
    --disable-default-apps \
    --user-data-dir=/tmp/chrome-profile \
    --app=http://127.0.0.1:3000 \
    > /tmp/chromium.log 2>&1
"
REMOTE

echo "Waiting for remote WebSocket on 127.0.0.1:8081 ..."
# shellcheck disable=SC2087
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<'REMOTE'
set -euo pipefail
for i in {1..30}; do
  if python3 - <<'PY'
import socket
s = socket.socket()
ok = s.connect_ex(("127.0.0.1", 8081)) == 0
s.close()
raise SystemExit(0 if ok else 1)
PY
  then
    echo "WebSocket ready"
    exit 0
  fi
  sleep 2
done
echo "ERROR: WebSocket server did not come up on time"
exit 1
REMOTE

echo "Starting remote training (tmux: train)..."
# shellcheck disable=SC2087
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<REMOTE
set -euo pipefail
tmux has-session -t train 2>/dev/null && tmux kill-session -t train || true
tmux new-session -d -s train "
  export PYTHONPATH=/workspace/robot_nav/reinforced
  cd /workspace/robot_nav/reinforced
  python3 -m examples.10_gs_navsim_nomad.train \
    --goal $GOAL_REMOTE \
    --scene-id $SCENE_ID \
    --ws-url $WS_URL \
    --timesteps $TIMESTEPS \
    --max-steps $MAX_STEPS \
    --seed $SEED \
    --experiment-name $EXPERIMENT_NAME \
    $REMOTE_CHECKPOINT_ARG \
    2>&1 | tee /workspace/train.log
"
REMOTE

echo "Starting remote TensorBoard on 6006 ..."
# shellcheck disable=SC2087
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<'REMOTE'
set -euo pipefail
pkill -f 'tensorboard' 2>/dev/null || true
python3 -m pip install --break-system-packages -q tensorboard || true
nohup python3 -m tensorboard.main \
  --logdir /workspace/robot_nav/reinforced/runs \
  --host 127.0.0.1 --port 6006 \
  > /tmp/tb.log 2>&1 &
REMOTE

echo ""
echo "================================================================"
echo " Remote gs_navsim + NOMAD training is running"
echo "----------------------------------------------------------------"
echo "Instance      : $INSTANCE_ID"
echo "SSH           : ssh -p $SSH_PORT root@$SSH_HOST"
echo "Scene ID      : $SCENE_ID"
echo "Data dir      : $DATA_DIR_REMOTE"
echo "Goal image    : $GOAL_REMOTE"
echo ""
echo "Logs"
echo "  Backend     : ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/gs_backend.log'"
echo "  Training    : ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/train.log'"
echo "  Browser     : ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /tmp/chromium.log'"
echo ""
echo "tmux sessions"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'tmux ls'"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'tmux attach -t train'"
echo ""
echo "TensorBoard tunnel"
echo "  ssh -o StrictHostKeyChecking=no -p $SSH_PORT -L 6006:127.0.0.1:6006 -N root@$SSH_HOST"
echo "  open http://localhost:6006"
echo ""
echo "Sync / destroy (local)"
echo "  cd $REINFORCED_DIR"
echo "  bash scripts/vast_sync.sh"
echo "  bash scripts/vast_destroy.sh"
echo "================================================================"
