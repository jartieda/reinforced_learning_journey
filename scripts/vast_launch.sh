#!/usr/bin/env bash
# =============================================================================
# vast_launch.sh — provision a vast.ai GPU instance and start training
#
# Usage:
#   bash scripts/vast_launch.sh [options] -- <python -m module> [module-args]
#
# Options:
#   --gpu         GPU model filter (default: RTX_3090)
#   --disk        Disk size in GB  (default: 40)
#   --image       Docker image     (default: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel)
#   --max-dph     Max $/hour       (default: 0.50)
#   --min-dph     Min $/hour       (default: 0.00, skip suspiciously cheap offers)
#   --isaac        Use Isaac Sim image instead (overrides --image)
#   --livestream  Streaming mode (default: 0 = disabled):
#                   2 = WebRTC  ← the only working mode for vast.ai
#                       The streaming client hardcodes port 49100; it cannot use the
#                       dynamically-mapped vast.ai host port.  Use --livestream 2 so
#                       Isaac Sim binds 49100/tcp and the SSH tunnel forwards it to
#                       localhost:49100.  Enter "localhost" in the streaming client.
#                   1 = WebRTC public-network flag (Isaac Sim side only, same behaviour
#                       in practice — client still needs the SSH tunnel)
#
# TensorBoard is always started on the instance (port 6006).
# An SSH tunnel command is printed at the end; run it in a separate terminal.
#
# Examples:
#   # Train example 05 on a 3090
#   bash scripts/vast_launch.sh -- python -m examples.05_ppo_cnn.train --timesteps 250000
#
#   # Train example 07 (Isaac Lab) on a 5090 with WebRTC streaming
#   bash scripts/vast_launch.sh --gpu RTX_5090 --isaac --max-dph 1.00 --min-dph 0.8 \
#       --livestream 2 -- python -m examples.07_isaac_transfer.train --timesteps 200000
#   # Then run the printed SSH tunnel and enter "localhost" in the streaming client
#
#   # Quick smoke-test, cheap GPU, short run
#   bash scripts/vast_launch.sh --gpu RTX_3080 --max-dph 0.30 \
#       -- python -m examples.02_ppo_classic_control.train --timesteps 20000
# =============================================================================
set -euo pipefail

# ── Load secrets from .env -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$ROOT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found. Create it with:"
  echo "  echo 'VAST_AI_API_KEY=your_key_here' > .env"
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

if [[ -z "${VAST_AI_API_KEY:-}" ]]; then
  echo "ERROR: VAST_AI_API_KEY is not set in $ENV_FILE"
  exit 1
fi

# ── Defaults ---------------------------------------------------------------
GPU_FILTER="RTX_3090"
DISK_GB=40
IMAGE="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"
MAX_DPH=0.50
MIN_DPH=0.00
ISAAC=0
LIVESTREAM=0
TRAIN_CMD=()

# ── Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)        GPU_FILTER="$2"; shift 2 ;;
    --disk)        DISK_GB="$2";    shift 2 ;;
    --image)       IMAGE="$2";      shift 2 ;;
    --max-dph)     MAX_DPH="$2";    shift 2 ;;
    --min-dph)     MIN_DPH="$2";    shift 2 ;;
    --isaac)       ISAAC=1;          shift   ;;
    --livestream)  LIVESTREAM="$2"; shift 2 ;;
    --)            shift; TRAIN_CMD=("$@"); break ;;
    *)             echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ${#TRAIN_CMD[@]} -eq 0 ]]; then
  echo "ERROR: No training command provided after --"
  echo "Usage: bash scripts/vast_launch.sh [options] -- python -m examples.XX.train ..."
  exit 1
fi

if [[ $ISAAC -eq 1 ]]; then
  IMAGE="nvcr.io/nvidia/isaac-sim:5.1.0"
  DISK_GB=80   # Isaac Sim image is ~50 GB
fi

# ── Check vastai CLI -------------------------------------------------------
# vastai is installed in the env_isaaclab venv; auto-activate it if needed.
if ! command -v vastai &>/dev/null; then
  VENV_VASTAI="$ROOT_DIR/env_isaaclab/bin/vastai"
  if [[ -x "$VENV_VASTAI" ]]; then
    # Prepend venv bin to PATH for this script's lifetime
    export PATH="$ROOT_DIR/env_isaaclab/bin:$PATH"
    echo "  → Using vastai from env_isaaclab venv"
  else
    echo "ERROR: vastai CLI not found. Run once:"
    echo "  source env_isaaclab/bin/activate && pip install vastai"
    exit 1
  fi
fi

vastai set api-key "$VAST_AI_API_KEY" 2>/dev/null

# ── Reuse existing instance if available ----------------------------------
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes"

if [[ -f "$ROOT_DIR/.vast_instance_id" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.vast_instance_id"
  echo ""
  echo "Found existing instance: id=$INSTANCE_ID  ($SSH_HOST:$SSH_PORT)"
  if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" &>/dev/null; then
    echo "  → Instance is reachable. Reusing it (skipping create)."
    echo "  → To force a new instance: rm .vast_instance_id"
    SKIP_CREATE=1
  else
    echo "  → Instance not reachable (may be stopped). Creating a new one."
    rm -f "$ROOT_DIR/.vast_instance_id"
    SKIP_CREATE=0
  fi
else
  SKIP_CREATE=0
fi

if [[ $SKIP_CREATE -eq 0 ]]; then
# ── Find cheapest matching offer ------------------------------------------
echo ""
echo "Searching for offers: gpu=$GPU_FILTER, dph=${MIN_DPH}–${MAX_DPH} ..."
OFFER_JSON=$(vastai search offers \
  --raw \
  --order dph_total \
  "gpu_name=$GPU_FILTER num_gpus=1 reliability>0.95 inet_down>100 disk_space>=$DISK_GB dph_total<=$MAX_DPH dph_total>=$MIN_DPH" \
  2>/dev/null)

if [[ -z "$OFFER_JSON" ]]; then
  echo "No offers found. Try --gpu RTX_4090 or raise --max-dph."
  echo "Browse manually:  vastai search offers \"gpu_name=RTX_3090\""
  exit 1
fi

# Machines known to be broken (never boot / SSH unreachable).
# Add machine_id values here to permanently skip them.
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

echo "  \u2192 Best offer: id=$OFFER_ID  (\$$OFFER_DPH/hr)"
echo ""

# ── Create instance --------------------------------------------------------
# Isaac Sim 5.x WebRTC signaling (49100/tcp) is tunnelled over SSH so the
# client can always connect to localhost:49100 regardless of vast.ai port
# mapping.  The UDP media stream (47998) typically works once the signaling
# channel is established through the same host.
# Native Omniverse client (livestream=1) still uses a direct port exposure.
STREAM_PORTS=""
if [[ $LIVESTREAM -eq 1 ]]; then
  # WebRTC over public/internet networks — vast.ai exposes 49100/tcp + 47998/udp
  # Isaac Sim also needs PUBLIC_IP set so ICE candidates include the external address.
  echo "  → Livestream 1 (WebRTC, public network) — ports 49100/tcp + 47998/udp exposed"
  echo "     Download client: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html"
elif [[ $LIVESTREAM -eq 2 ]]; then
  # WebRTC over local/private network — signaling tunnelled via SSH to localhost:49100
  echo "  → Livestream 2 (WebRTC, local/private network) — tunnel via SSH to localhost:49100"
  echo "     Download client: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html"
fi

echo "Creating instance (image: $IMAGE, disk: ${DISK_GB}GB) ..."
CREATE_OUT=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --ssh \
  --direct \
  --env "-e PYTHONUNBUFFERED=1 $STREAM_PORTS" \
  --raw 2>&1)

INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])" 2>/dev/null \
             || echo "$CREATE_OUT" | grep -oP '"new_contract":\s*\K[0-9]+' | head -1)

if [[ -z "$INSTANCE_ID" ]]; then
  echo "ERROR: Failed to create instance."
  echo "$CREATE_OUT"
  exit 1
fi

echo "  \u2192 Instance created: id=$INSTANCE_ID"
echo "  \u2192 Monitor at: https://cloud.vast.ai/instances/"
echo ""

# ── Wait for SSH to become available --------------------------------------
echo "Waiting for instance to boot (this takes 1\u20133 minutes) ..."
MAX_WAIT=600
ELAPSED=0
SSH_URL=""

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
  sleep 10
  ELAPSED=$((ELAPSED + 10))

  SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
  if [[ -z "$SSH_URL" ]]; then
    printf "  [%3ds] waiting for IP...\n" "$ELAPSED"
    continue
  fi

  SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://[^@]*@||' | cut -d: -f1)
  SSH_PORT=$(echo "$SSH_URL" | sed 's|.*:||')

  if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" &>/dev/null; then
    echo "  \u2192 SSH ready after ${ELAPSED}s"
    break
  fi
  printf "  [%3ds] SSH not ready yet...\n" "$ELAPSED"
done

if [[ -z "$SSH_URL" ]] || ! ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" &>/dev/null; then
  echo "ERROR: Timed out waiting for SSH."
  echo "Check instance status:  vastai show instance $INSTANCE_ID"
  exit 1
fi

# Save all connection details as sourceable shell variables
cat > "$ROOT_DIR/.vast_instance_id" <<EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
EOF
echo "  \u2192 Connection details saved to .vast_instance_id"

fi  # end SKIP_CREATE==0

# ── Upload project code ----------------------------------------------------
echo ""
echo "Uploading project files ..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "mkdir -p /workspace/reinforced"
rsync -az --progress \
  --exclude=".git" \
  --exclude="env_isaaclab" \
  --exclude="runs" \
  --exclude="videos" \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude=".env" \
  -e "ssh $SSH_OPTS -p $SSH_PORT" \
  "$ROOT_DIR/" \
  "root@$SSH_HOST:/workspace/reinforced/"

# ── Install Python dependencies on the instance ---------------------------
echo ""
echo "Installing dependencies ..."
# NOTE: heredoc with unquoted REMOTE expands local vars ($ISAAC) before
# sending — that is intentional so the remote sees the literal 0 or 1.
# shellcheck disable=SC2087
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<REMOTE
set -e
if [[ $ISAAC -eq 1 ]]; then
  # ── Isaac Sim path ──────────────────────────────────────────────────────
  # /isaac-sim/python.sh bootstraps Omniverse; NEVER use system python here.
  ISAAC_PY=/isaac-sim/python.sh

  # 1. Clone Isaac Lab (skip if already present from a previous run)
  if [[ ! -d /workspace/isaaclab_src/.git ]]; then
    echo "Cloning Isaac Lab ..."
    apt-get install -y git -q 2>/dev/null || true
    git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git /workspace/isaaclab_src
  else
    echo "Isaac Lab already cloned, skipping."
  fi

  # 2. Install Isaac Lab core into Isaac Sim's Python
  echo "Installing isaaclab into Isaac Sim Python ..."
  \$ISAAC_PY -m pip install --no-cache-dir -e /workspace/isaaclab_src/source/isaaclab 

  # 3. Install skrl, gymnasium, h5py, and video recording deps into same Python
  #    h5py is required by isaaclab_tasks at extension startup (recorder_manager)
  #    imageio + imageio-ffmpeg are used by examples/07_isaac_transfer/record.py
  \$ISAAC_PY -m pip install --no-cache-dir "skrl>=2.0.0" "gymnasium>=1.0.0" h5py imageio imageio-ffmpeg 
else
  # ── Regular PyTorch image ───────────────────────────────────────────────
  # --break-system-packages is safe: we own the whole container.
  apt-get install -y python3-pip -q 2>/dev/null || true
  pip3 install --break-system-packages --no-cache-dir -r /workspace/reinforced/requirements.txt 2>&1 | tail -5
fi
REMOTE

# ── Launch training (detached via tmux so SSH disconnect is safe) ----------
echo ""
echo "Starting training (detached session 'train') ..."
REMOTE_CMD="${TRAIN_CMD[*]}"

# Isaac examples must run under Isaac Sim's Python; others use python3.
if [[ $ISAAC -eq 1 ]]; then
  PYTHON_BIN="/isaac-sim/python.sh"
else
  PYTHON_BIN="python3"
fi

# Rewrite 'python' or 'python3' at the start of the command to the right binary
REMOTE_CMD=$(echo "$REMOTE_CMD" | sed "s|^python[0-9.]*|$PYTHON_BIN|")

# Forward --livestream to Isaac Sim's AppLauncher (it opens the WebRTC/native server)
if [[ $LIVESTREAM -gt 0 ]]; then
  REMOTE_CMD="$REMOTE_CMD --livestream $LIVESTREAM"
fi

# shellcheck disable=SC2087
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" bash <<REMOTE
set -e
# PYTHONPATH lets 'python -m examples.XX.train' find the package
export PYTHONPATH=/workspace/reinforced
cd /workspace/reinforced
tmux new-session -d -s train "
  export PYTHONPATH=/workspace/reinforced
  export PUBLIC_IP=\$(curl -s --max-time 5 ifconfig.me 2>/dev/null || hostname -I | awk '{print \$1}')
  cd /workspace/reinforced
  ${REMOTE_CMD} 2>&1 | tee /workspace/train.log
  echo '=== TRAINING FINISHED ===' >> /workspace/train.log
"
echo "  tmux session 'train' started"
echo "  Tail log:  ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/train.log'"

# ── TensorBoard (detached, logs to /tmp/tb.log) ----------------------------
pkill -f 'tensorboard' 2>/dev/null || true
if [[ $ISAAC -eq 1 ]]; then
  TB_CMD="/isaac-sim/python.sh -m tensorboard.main"
  # Ensure tensorboard is installed in Isaac Sim's Python
  /isaac-sim/python.sh -m pip install -q tensorboard 2>/dev/null || true
else
  TB_CMD="python3 -m tensorboard.main"
fi
nohup \$TB_CMD --logdir /workspace/reinforced/runs --host 127.0.0.1 --port 6006 \
  > /tmp/tb.log 2>&1 &
# Wait until TensorBoard is actually listening
for _i in {1..15}; do
  sleep 2
  if python3 -c "import socket; s=socket.socket(); r=s.connect_ex(('127.0.0.1',6006)); s.close(); exit(0 if r==0 else 1)" 2>/dev/null; then
    echo "  TensorBoard ready on remote port 6006"
    break
  fi
done
REMOTE

# ── Start remote UDP→TCP socat bridge (for WebRTC media on 47998/udp) -----
# SSH can only forward TCP. socat relays UDP 47998 ↔ TCP 47999 so the SSH
# tunnel can carry WebRTC media.
if [[ $LIVESTREAM -gt 0 ]]; then
  ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
    "apt-get install -y -qq socat 2>/dev/null; \
     pkill -f 'socat.*47998' 2>/dev/null || true; \
     nohup socat TCP4-LISTEN:47999,reuseaddr,fork UDP4:127.0.0.1:47998 \
       > /tmp/socat_udp.log 2>&1 &"
  echo "  Remote socat UDP bridge started (47999/tcp → 47998/udp)"
fi

# ── Build tunnel command --------------------------------------------------
# Streaming client hardcodes port 49100 — always tunnel it regardless of mode.
# Port 47999 carries the socat-bridged WebRTC media (UDP 47998 on remote).
TUNNEL_SPEC="-L 6006:127.0.0.1:6006"
if [[ $LIVESTREAM -gt 0 ]]; then
  TUNNEL_SPEC="$TUNNEL_SPEC -L 49100:127.0.0.1:49100 -L 47999:127.0.0.1:47999"
fi

# ── Print summary ----------------------------------------------------------
echo ""
echo "================================================================"
echo " Instance ready"
echo "----------------------------------------------------------------"
echo " ID        : $INSTANCE_ID"
echo " SSH       : ssh -p $SSH_PORT root@$SSH_HOST"
echo " Tail log  : ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/train.log'"
echo " TensorBoard: http://localhost:6006"
echo " Dashboard : https://cloud.vast.ai/instances/"
echo ""
echo " Sync results  : bash scripts/vast_sync.sh"
echo " Destroy when done: bash scripts/vast_destroy.sh"
echo ""
echo " ── Tunnels (run in a separate terminal) ────────────"
echo "   ssh $SSH_OPTS -p $SSH_PORT $TUNNEL_SPEC -N root@$SSH_HOST"
if [[ $LIVESTREAM -gt 0 ]]; then
  echo ""
  echo " ── Livestream ──────────────────────────────────────"
  echo " 1) Run the tunnel above in a separate terminal."
  echo " 2) In ANOTHER terminal run the local UDP bridge:"
  echo "      socat UDP4-LISTEN:47998,reuseaddr,fork TCP4:127.0.0.1:47999"
  echo " 3) Open the Isaac Sim WebRTC Streaming Client and enter: localhost"
  echo "    (client hardcodes port 49100)"
fi
echo "================================================================"
