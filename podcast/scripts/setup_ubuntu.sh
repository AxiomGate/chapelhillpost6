#!/usr/bin/env bash
# Base system setup for the podcast box: Ubuntu 24.04 LTS with NVIDIA cards.
# Idempotent -- safe to re-run.
set -euo pipefail

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m !! %s\033[0m\n' "$*"; }

if [[ $EUID -eq 0 ]]; then
  echo "Run as your normal user, not root. It will sudo where needed." >&2
  exit 1
fi

log "System packages"
sudo apt-get update
sudo apt-get install -y \
  build-essential git curl wget pkg-config \
  python3.11 python3.11-venv python3.11-dev python3-pip \
  ffmpeg libsndfile1 libgl1 libglib2.0-0 \
  fonts-dejavu-core fonts-inter-variable \
  sqlite3 htop nvtop tmux

log "NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "No driver found. Installing the recommended one; a reboot will be needed."
  sudo ubuntu-drivers install
  warn "Reboot, then re-run this script."
  exit 0
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

log "Checking NVENC in ffmpeg"
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
  echo "h264_nvenc available"
else
  warn "ffmpeg has no NVENC support. Ubuntu's build usually does; if not, install"
  warn "a full build (e.g. from the ffmpeg PPA) or set video.encoder: libx264."
fi

log "Persistence mode and power limits"
sudo nvidia-smi -pm 1 || warn "could not enable persistence mode"
# 280W instead of 350W costs 5-8% performance and drops peak draw by 140W across
# two 3090s. Worth it for thermals even on a large PSU. Comment out if unwanted.
for index in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  name=$(nvidia-smi -i "$index" --query-gpu=name --format=csv,noheader)
  if [[ "$name" == *3090* ]]; then
    sudo nvidia-smi -i "$index" -pl 280 || warn "could not set power limit on GPU$index"
  fi
done

log "Orchestrator environment"
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e '.[review,captions,publish,dev]'

log "Directories"
mkdir -p work output assets/{voice,avatar,brand,broll}

if [[ ! -f .env ]]; then
  cp config/env.example .env
  warn "Created .env from the example. Fill it in before running."
fi

log "Done"
cat <<'EOF'

Next:
  1. scripts/install_models.sh      # TTS and avatar environments (large download)
  2. Record assets/voice/reference.wav      -- 60-120s of clean speech
  3. Record assets/avatar/base_loop.mp4     -- 3-5 min on camera, listening
  4. Add assets/brand/background.png        -- any 1920x1080 image to start
  5. source .venv/bin/activate && podcastpipe doctor

EOF
