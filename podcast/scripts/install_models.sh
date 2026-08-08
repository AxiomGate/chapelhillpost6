#!/usr/bin/env bash
# Install the ML tools into separate virtualenvs.
#
# They are separate on purpose: MuseTalk, Chatterbox, VibeVoice and LatentSync
# pin mutually incompatible versions of torch, diffusers and transformers.
# Installing them together does not work. The orchestrator drives each one as a
# subprocess, so isolation costs nothing.
#
# Expect 40-60 GB of downloads and a while to run.
set -euo pipefail

ENVS="${ENVS:-$HOME/envs}"
SRC="${SRC:-$HOME/src}"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m !! %s\033[0m\n' "$*"; }

mkdir -p "$ENVS" "$SRC"

# ---------------------------------------------------------------- Chatterbox
log "Chatterbox TTS -> $ENVS/chatterbox"
if [[ ! -d "$ENVS/chatterbox" ]]; then
  python3.11 -m venv "$ENVS/chatterbox"
fi
"$ENVS/chatterbox/bin/pip" install --upgrade pip
"$ENVS/chatterbox/bin/pip" install torch torchaudio --index-url "$TORCH_INDEX"
"$ENVS/chatterbox/bin/pip" install chatterbox-tts

# ------------------------------------------------------------------ MuseTalk
log "MuseTalk -> $ENVS/musetalk, repo in $SRC/MuseTalk"
if [[ ! -d "$SRC/MuseTalk" ]]; then
  git clone https://github.com/TMElyralab/MuseTalk "$SRC/MuseTalk"
fi
if [[ ! -d "$ENVS/musetalk" ]]; then
  python3.11 -m venv "$ENVS/musetalk"
fi
"$ENVS/musetalk/bin/pip" install --upgrade pip
"$ENVS/musetalk/bin/pip" install torch torchvision torchaudio --index-url "$TORCH_INDEX"
"$ENVS/musetalk/bin/pip" install -r "$SRC/MuseTalk/requirements.txt"

# MuseTalk needs mmpose/mmdet for face landmarks; these are picky about install
# order and must come after torch.
"$ENVS/musetalk/bin/pip" install --no-cache-dir openmim
"$ENVS/musetalk/bin/mim" install "mmengine" "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"

log "MuseTalk weights"
if [[ -f "$SRC/MuseTalk/download_weights.sh" ]]; then
  (cd "$SRC/MuseTalk" && bash download_weights.sh)
else
  warn "download_weights.sh not found. Follow the repo README to fetch weights"
  warn "into $SRC/MuseTalk/models before the first render."
fi

# --------------------------------------------------------- LatentSync (opt.)
if [[ "${INSTALL_LATENTSYNC:-0}" == "1" ]]; then
  log "LatentSync -> $ENVS/latentsync"
  if [[ ! -d "$SRC/LatentSync" ]]; then
    git clone https://github.com/bytedance/LatentSync "$SRC/LatentSync"
  fi
  python3.11 -m venv "$ENVS/latentsync"
  "$ENVS/latentsync/bin/pip" install --upgrade pip
  "$ENVS/latentsync/bin/pip" install torch torchvision torchaudio --index-url "$TORCH_INDEX"
  "$ENVS/latentsync/bin/pip" install -r "$SRC/LatentSync/requirements.txt"
else
  echo
  echo "Skipping LatentSync. INSTALL_LATENTSYNC=1 to add it (hero shots only)."
fi

# ---------------------------------------------------------- VibeVoice (opt.)
if [[ "${INSTALL_VIBEVOICE:-0}" == "1" ]]; then
  log "VibeVoice -> $ENVS/vibevoice"
  python3.11 -m venv "$ENVS/vibevoice"
  "$ENVS/vibevoice/bin/pip" install --upgrade pip
  "$ENVS/vibevoice/bin/pip" install torch torchaudio --index-url "$TORCH_INDEX"
  "$ENVS/vibevoice/bin/pip" install soundfile \
    "git+https://github.com/vibevoice-community/VibeVoice.git"
else
  echo "Skipping VibeVoice. INSTALL_VIBEVOICE=1 to add it (multi-host shows)."
fi

log "Done"
echo "Verify with: podcastpipe doctor"
