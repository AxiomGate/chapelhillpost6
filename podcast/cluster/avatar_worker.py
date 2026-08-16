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
    if completed.returncode == 0:
        return

    # A killed process leaves no traceback -- it just stops, usually most of the
    # way through a render, and the only evidence is the exit code. Saying what
    # -9 means here is the difference between a two-minute fix and an afternoon.
    if completed.returncode == -9:
        raise RuntimeError(
            "MuseTalk was killed (SIGKILL), which is almost always this "
            "container's memory limit rather than anything wrong with the job. "
            "MuseTalk holds decoded frames in RAM -- about 6.2 MB each at 1080p, "
            "so a 120-second window at 25 fps needs roughly 18.7 GB. Lower "
            "avatar.chunk_seconds in show.yaml, or raise mem_limit in this "
            "node's compose file.\n"
            f"{completed.stderr[-2000:]}"
        )
    raise RuntimeError(
        f"command failed ({completed.returncode}): {' '.join(command[:6])}...\n"
        f"{completed.stderr[-2000:]}"
    )


def _probe_frames(path: Path) -> int:
    """Frame count from the container, or -1 when it does not carry one.

    Returning -1 rather than raising: a missing nb_frames tag is a container
    quirk, not a bad render, and refusing the job over it would be worse than
    the problem it guards against.
    """
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return -1


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(completed.stdout.strip())


def load_model():
    """Verify this node can render. Deliberately loads nothing into VRAM.

    Rendering runs in a subprocess -- scripts.inference -- which loads its own
    copy of every model. The resident copy this used to hold was never read:
    handle() takes a ``model`` argument and ignores it. It cost about 7 GB of
    VRAM that the render subprocess then had to fit around, plus 60-90 seconds
    of startup, for nothing.

    It was also the wrong 7 GB. It called load_all_model() with no arguments, so
    MuseTalk's relative defaults resolved against this worker's working
    directory instead of MuseTalk's, and `models/sd-vae` came back as a
    HuggingFace repo id that does not exist. handle() already gets this right by
    running the subprocess with cwd=MUSETALK_HOME.

    What /health should mean here is "a render started now would succeed".
    Checking that every file the subprocess needs is present says exactly that,
    and says it in milliseconds. It is also the check that was missing when two
    nodes reported healthy for days holding zero bytes of weights.
    """
    if not MUSETALK_HOME.exists():
        raise RuntimeError(f"MuseTalk not found at {MUSETALK_HOME}")

    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

    required = {
        "unet config": UNET_CONFIG,
        "unet weights": UNET_WEIGHTS,
        "whisper": WHISPER_DIR,
        "vae": _MODELS / VAE_TYPE,
        "face parsing": _MODELS / "face-parse-bisent" / "79999_iter.pth",
        "dwpose": _MODELS / "dwpose" / "dw-ll_ucoco_384.pth",
    }
    missing = [f"{label} ({path})" for label, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"MuseTalk {MUSETALK_VERSION} cannot render -- missing "
            + "; ".join(missing)
            + ". Run scripts/fetch_musetalk_weights.py inside this container and "
            "restart it. Individual paths can be overridden with "
            "MUSETALK_UNET_CONFIG, MUSETALK_UNET_WEIGHTS or MUSETALK_WHISPER_DIR "
            "if a MuseTalk update moves them."
        )

    return {
        "version": MUSETALK_VERSION,
        "unet_config": str(UNET_CONFIG),
        "unet_weights": str(UNET_WEIGHTS),
        "whisper_dir": str(WHISPER_DIR),
        "vae_type": VAE_TYPE,
    }


def ensure_driving_video(base_loop: Path, fps: int, needed_seconds: float) -> Path:
    """Return a node-local driving video at least ``needed_seconds`` long.

    Built once per (base loop contents, fps) and reused for every episode
    thereafter. The ping-pong construction — forward then reversed — removes the
    visible jump at the loop point, because the last forward frame is the first
    reverse frame.
    """
    # Key on the file's content, not just its path. Everything here is derived
    # from the base loop and reused for every later episode, so a key that only
    # covers the path means replacing assets/avatar/base_loop.mp4 -- the normal
    # way anyone changes their footage -- silently keeps rendering against the
    # old clip, on every node, until someone deletes the scratch by hand.
    # Size and mtime are enough and cost a stat; hashing gigabytes per job is not.
    stat = base_loop.stat()
    key = hashlib.sha256(
        f"{base_loop}:{fps}:{stat.st_size}:{int(stat.st_mtime)}".encode()
    ).hexdigest()[:12]
    pingpong = LOCAL_CACHE / f"pingpong_{key}.mp4"
    extended = LOCAL_CACHE / f"driving_{key}.mp4"

    # Drop clips built from a previous base loop. Each pair is hundreds of MB
    # and nothing will ever ask for them again.
    for stale in LOCAL_CACHE.glob("*_*.mp4"):
        if stale.name.startswith(("base_", "pingpong_", "driving_")) and key not in stale.name:
            stale.unlink(missing_ok=True)

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
             # How much face gets repainted and how it blends back. The
             # defaults leave a wider blended region than most footage needs.
             "--parsing_mode", str(job.get("parsing_mode", "jaw")),
             "--left_cheek_width", str(int(job.get("left_cheek_width", 90))),
             "--right_cheek_width", str(int(job.get("right_cheek_width", 90))),
             "--extra_margin", str(int(job.get("extra_margin", 10))),
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

    # MuseTalk can exit 0 having written almost nothing -- one window came back
    # with 38 frames where 1500 were due, and the worker accepted it because the
    # file existed. That shipped a frozen minute into a finished episode with no
    # error anywhere. Count the frames and fail instead, so the scheduler retries
    # on another node.
    expected = int(duration * fps)
    frames = _probe_frames(out_path)
    if 0 <= frames < expected * 0.9:
        # Remove it, or the orchestrator's content-hash cache treats this window
        # as already rendered and the retry never happens.
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MuseTalk wrote only {frames} frames for a {duration:.1f}s window "
            f"needing about {expected}, but exited successfully. The output has "
            "been discarded so this window can be retried."
        )

    return {
        "ok": True,
        "id": job.get("id", ""),
        "out_path": str(out_path),
        "duration": round(duration, 3),
        "frames": frames if frames >= 0 else expected,
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
