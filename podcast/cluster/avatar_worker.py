#!/usr/bin/env python
"""Avatar worker — MuseTalk, warm. Runs on node-b and node-c, one RTX 3090 each.

This is the slowest stage and the one that decides how long an episode takes, so
it is also where the 1 GbE budget matters most.

**Nothing large crosses the network.** The driving video for a 25-minute episode
is several gigabytes. Rather than build it centrally and ship it, each worker
keeps a node-local copy of the base loop on fast local disk, builds its own
ping-pong clip once, and slices whatever window it is asked for locally. Over the
wire a job sends only an audio window (a few MB) and returns a rendered clip.

    POST /run  {"id": "w000", "audio": "/pipeline/.../window_000.wav",
                "out_path": "/pipeline/.../avatar_000.mp4",
                "base_loop": "/pipeline/assets/avatar/base_loop.mp4"}
    -> {"ok": true, "out_path": "...", "frames": 3000}
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from worker_common import build_app, serve  # noqa: E402

DEFAULT_PORT = 8081
REQUIRED_VRAM_MB = int(os.environ.get("REQUIRED_VRAM_MB", "8000"))
# Node-local scratch. Must NOT point at the shared export -- the whole point is
# to keep this traffic off the network.
LOCAL_CACHE = Path(os.environ.get("LOCAL_CACHE", "/scratch"))
MUSETALK_HOME = Path(os.environ.get("MUSETALK_HOME", "/opt/MuseTalk"))

# Every path scripts.inference needs, stated explicitly rather than left to its
# defaults -- which are internally inconsistent upstream. As of the 2026-08
# clone, --version defaults to v15 and --unet_model_path to musetalkV15/unet.pth,
# but --unet_config defaults to ./models/musetalk/config.json, a file that does
# not exist in TMElyralab/MuseTalk at all. The repo publishes exactly four
# weights files and both configs are named musetalk.json:
#
#   musetalk/musetalk.json      musetalk/pytorch_model.bin
#   musetalkV15/musetalk.json   musetalkV15/unet.pth
#
# Passing these ourselves also means a MuseTalk update that moves its defaults
# again cannot silently change which weights render the show.
MUSETALK_VERSION = os.environ.get("MUSETALK_VERSION", "v15")
_MODELS = MUSETALK_HOME / "models"
_VARIANT = "musetalkV15" if MUSETALK_VERSION == "v15" else "musetalk"

UNET_CONFIG = Path(
    os.environ.get("MUSETALK_UNET_CONFIG") or _MODELS / _VARIANT / "musetalk.json"
)
UNET_WEIGHTS = Path(
    os.environ.get("MUSETALK_UNET_WEIGHTS")
    or _MODELS / _VARIANT / ("unet.pth" if MUSETALK_VERSION == "v15" else "pytorch_model.bin")
)
WHISPER_DIR = Path(os.environ.get("MUSETALK_WHISPER_DIR") or _MODELS / "whisper")
VAE_TYPE = os.environ.get("MUSETALK_VAE_TYPE", "sd-vae")


def _run(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None,
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command[:6])}...\n"
            f"{completed.stderr[-2000:]}"
        )


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(completed.stdout.strip())


def load_model():
    """Warm MuseTalk. Importing and instantiating here means the 60-90 second
    load happens once per container lifetime, not once per episode."""
    if not MUSETALK_HOME.exists():
        raise RuntimeError(f"MuseTalk not found at {MUSETALK_HOME}")

    # Check the render-time paths here, not on the first job. Rendering happens
    # in a subprocess with its own model load, so a missing file there surfaces
    # forty minutes into an episode as a subprocess exit code -- while /health
    # has been reporting green the whole time. This is the same failure shape
    # that left two nodes "healthy" for days holding zero bytes of weights.
    missing = [
        str(p) for p in (UNET_CONFIG, UNET_WEIGHTS, WHISPER_DIR) if not p.exists()
    ]
    if missing:
        raise RuntimeError(
            f"MuseTalk {MUSETALK_VERSION} is missing {', '.join(missing)}. "
            "Run scripts/fetch_musetalk_weights.py inside this container, then "
            "restart it. Override individual paths with MUSETALK_UNET_CONFIG, "
            "MUSETALK_UNET_WEIGHTS or MUSETALK_WHISPER_DIR if a MuseTalk update "
            "moves them."
        )

    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MUSETALK_HOME))

    import torch
    from musetalk.utils.utils import load_all_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_processor, vae, unet, pe = load_all_model()
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    return {
        "audio_processor": audio_processor,
        "vae": vae,
        "unet": unet,
        "pe": pe,
        "device": device,
    }


def ensure_driving_video(base_loop: Path, fps: int, needed_seconds: float) -> Path:
    """Return a node-local driving video at least ``needed_seconds`` long.

    Built once per (base loop, fps) and reused for every episode thereafter. The
    ping-pong construction — forward then reversed — removes the visible jump at
    the loop point, because the last forward frame is the first reverse frame.
    """
    key = hashlib.sha256(f"{base_loop}:{fps}".encode()).hexdigest()[:12]
    pingpong = LOCAL_CACHE / f"pingpong_{key}.mp4"
    extended = LOCAL_CACHE / f"driving_{key}.mp4"

    if not pingpong.exists():
        local_base = LOCAL_CACHE / f"base_{key}{base_loop.suffix}"
        if not local_base.exists():
            # One copy off the share, ever. Everything after this is local.
            shutil.copy2(base_loop, local_base)
        _run([
            "ffmpeg", "-y", "-i", str(local_base),
            "-filter_complex",
            "[0:v]split[fwd][tmp];[tmp]reverse[rev];[fwd][rev]concat=n=2:v=1:a=0,"
            f"fps={fps},setpts=N/{fps}/TB[out]",
            "-map", "[out]", "-an",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            str(pingpong),
        ])

    have = _probe_duration(extended) if extended.exists() else 0.0
    if have < needed_seconds:
        # Grow with headroom so a slightly longer episode tomorrow does not
        # trigger a rebuild.
        target = max(needed_seconds * 1.5, 300.0)
        _run([
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(pingpong),
            "-t", f"{target:.3f}", "-r", str(fps), "-an",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            str(extended),
        ])
    return extended


def handle(model, job: dict) -> dict:
    audio = Path(job["audio"])
    out_path = Path(job["out_path"])
    base_loop = Path(job["base_loop"])
    fps = int(job.get("fps", 25))
    bbox_shift = int(job.get("bbox_shift", 0))

    if not audio.exists():
        raise FileNotFoundError(f"audio window not found at {audio} inside the container")
    if not base_loop.exists():
        raise FileNotFoundError(f"base loop not found at {base_loop} inside the container")

    duration = _probe_duration(audio)
    driving = ensure_driving_video(base_loop, fps, duration + 5)

    with tempfile.TemporaryDirectory(dir=str(LOCAL_CACHE), prefix="job_") as tmp:
        workdir = Path(tmp)
        segment = workdir / "driving.mp4"

        # Offset varies per job so consecutive windows do not all start from the
        # same frame of the loop, which would make the head motion visibly repeat
        # on a fixed cycle.
        offset = float(job.get("loop_offset", 0.0)) % max(
            _probe_duration(driving) - duration - 1, 1.0
        )
        _run([
            "ffmpeg", "-y", "-i", str(driving),
            "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}",
            "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            str(segment),
        ])

        config = workdir / "task.yaml"
        config.write_text(
            "task_0:\n"
            f'  video_path: "{segment}"\n'
            f'  audio_path: "{audio}"\n'
            f"  bbox_shift: {bbox_shift}\n",
            encoding="utf-8",
        )
        results = workdir / "results"
        _run(
            [sys.executable, "-m", "scripts.inference",
             "--inference_config", str(config),
             "--result_dir", str(results),
             "--fps", str(fps),
             "--version", MUSETALK_VERSION,
             "--unet_config", str(UNET_CONFIG),
             "--unet_model_path", str(UNET_WEIGHTS),
             "--whisper_dir", str(WHISPER_DIR),
             "--vae_type", VAE_TYPE,
             # Also in task.yaml. Which of the two this MuseTalk reads has moved
             # between versions, and passing both costs nothing.
             "--bbox_shift", str(bbox_shift),
             "--use_float16"],
            cwd=MUSETALK_HOME,
        )

        produced = sorted(results.rglob("*.mp4"))
        if not produced:
            raise RuntimeError("MuseTalk produced no output for this window")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        best = max(produced, key=lambda p: p.stat().st_size)
        # Copy then replace: the destination is on the network share, and a
        # partial file there would be picked up as a finished window.
        staging = out_path.with_suffix(".partial.mp4")
        shutil.copy2(best, staging)
        staging.replace(out_path)

    return {
        "ok": True,
        "id": job.get("id", ""),
        "out_path": str(out_path),
        "duration": round(duration, 3),
        "frames": int(duration * fps),
    }


app = build_app(
    name="avatar-worker",
    role="avatar",
    loader=load_model,
    handler=handle,
    required_vram_mb=REQUIRED_VRAM_MB,
)

if __name__ == "__main__":
    serve(app, DEFAULT_PORT)
