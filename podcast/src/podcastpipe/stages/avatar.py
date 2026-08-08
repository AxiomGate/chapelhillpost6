"""Stage 4 — render the talking head.

The strategy, argued in docs/MODEL_CHOICES.md: do not synthesize frames. Take a
short real recording of the host sitting and listening, ping-pong-loop it to the
length of the episode, and let MuseTalk inpaint the mouth to match the voice
track. Head motion, lighting, wardrobe and background are genuine footage, so
identity never drifts and a 25-minute episode renders in well under an hour on
one 3090.

Rendering is chunked and cached. A re-run after editing one paragraph re-renders
the chunks that paragraph touches.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..config import Config
from ..proc import ffprobe_duration, run, venv_python

ADAPTERS = {
    "musetalk": "musetalk_render.py",
    "latentsync": "latentsync_render.py",
}


def pingpong_command(base_loop: str | Path, output: str | Path, fps: int) -> list[str]:
    """Build a seamless A→B→A clip from the base loop.

    Playing a clip forward then backward removes the visible jump-cut at the
    loop point: the last frame forward is the first frame of the reverse, so the
    seam is continuous motion rather than a teleport.
    """
    return [
        "ffmpeg", "-y",
        "-i", str(base_loop),
        "-filter_complex",
        "[0:v]split[fwd][tmp];[tmp]reverse[rev];[fwd][rev]concat=n=2:v=1:a=0,"
        f"fps={fps},setpts=N/{fps}/TB[out]",
        "-map", "[out]",
        "-an",
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        str(output),
    ]


def extend_command(
    pingpong: str | Path, output: str | Path, duration: float, fps: int
) -> list[str]:
    """Loop the ping-pong clip out to the full episode duration."""
    return [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(pingpong),
        "-t", f"{duration:.3f}",
        "-r", str(fps),
        "-an",
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        str(output),
    ]


def slice_command(
    source: str | Path, output: str | Path, start: float, duration: float, is_audio: bool
) -> list[str]:
    """Cut one chunk. ``-ss`` before ``-i`` seeks on keyframes and would drift
    against the audio, so it goes after."""
    command = ["ffmpeg", "-y", "-i", str(source), "-ss", f"{start:.3f}", "-t", f"{duration:.3f}"]
    if is_audio:
        command += ["-c:a", "pcm_s16le"]
    else:
        command += ["-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
    command.append(str(output))
    return command


def plan_chunks(duration: float, chunk_seconds: int) -> list[tuple[float, float]]:
    """Split a duration into (start, length) chunks.

    A trailing remainder shorter than a quarter of a chunk is folded into the
    previous chunk instead of being rendered alone — the models produce a
    noticeable warm-up wobble in their first frames, and a three-second chunk is
    almost entirely warm-up.
    """
    if duration <= 0:
        return []
    chunk_seconds = max(int(chunk_seconds), 1)
    count = max(1, math.ceil(duration / chunk_seconds))
    chunks: list[tuple[float, float]] = []
    for index in range(count):
        start = index * chunk_seconds
        length = min(chunk_seconds, duration - start)
        if length <= 0:
            break
        chunks.append((round(start, 3), round(length, 3)))

    if len(chunks) > 1 and chunks[-1][1] < chunk_seconds * 0.25:
        start, length = chunks.pop()
        prev_start, prev_length = chunks[-1]
        chunks[-1] = (prev_start, round(prev_length + length, 3))
    return chunks


def chunk_key(config: Config, audio_path: Path, start: float, length: float) -> str:
    """Cache key covering the audio content and every render parameter."""
    digest = hashlib.sha256()
    with audio_path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    digest.update(
        json.dumps(
            {
                "renderer": config.avatar.renderer,
                "base_loop": config.avatar.base_loop,
                "fps": config.avatar.fps,
                "bbox_shift": config.avatar.bbox_shift,
                "start": start,
                "length": length,
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()[:16]


def render(config: Config, episode_dir: Path, audio_path: str | Path) -> str:
    """Render the full talking-head track. Returns the path to the silent video."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"voice track not found: {audio_path}")

    if config.avatar.renderer == "none":
        raise ValueError("avatar.renderer is 'none'; run assemble with a static layout instead")

    base_loop = config.path(config.avatar.base_loop)
    if not base_loop.exists():
        raise FileNotFoundError(
            f"base loop not found at {base_loop}. Record 3-5 minutes of yourself "
            "on camera, listening rather than speaking. See docs/RUNBOOK.md."
        )

    work = episode_dir / "avatar"
    work.mkdir(parents=True, exist_ok=True)
    cache = config.work_dir / "_avatar_cache"
    cache.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(audio_path)
    fps = config.avatar.fps

    # The ping-pong clip depends only on the base loop, so it is built once and
    # reused for every episode until the loop itself changes.
    pingpong = cache / f"pingpong_{hashlib.sha256(str(base_loop).encode()).hexdigest()[:12]}_{fps}.mp4"
    if not pingpong.exists():
        print("  building ping-pong base loop (one time per base clip)")
        run(pingpong_command(base_loop, pingpong, fps), log_path=work / "logs_pingpong.txt")

    driving = work / "driving.mp4"
    run(extend_command(pingpong, driving, duration, fps), log_path=work / "logs_extend.txt")

    chunks = plan_chunks(duration, config.avatar.chunk_seconds)
    print(f"  {duration / 60:.1f} min in {len(chunks)} chunk(s)")

    adapter_name = ADAPTERS.get(config.avatar.renderer)
    if adapter_name is None:
        raise ValueError(f"unknown avatar.renderer {config.avatar.renderer!r}")

    rendered: list[Path] = []
    jobs = []
    for index, (start, length) in enumerate(chunks):
        audio_chunk = work / f"chunk_{index:03d}.wav"
        video_chunk = work / f"chunk_{index:03d}.mp4"
        run(slice_command(audio_path, audio_chunk, start, length, is_audio=True))

        key = chunk_key(config, audio_chunk, start, length)
        cached = cache / f"{key}.mp4"
        rendered.append(cached)
        if cached.exists():
            print(f"  chunk {index + 1}/{len(chunks)}: cached")
            continue

        run(slice_command(driving, video_chunk, start, length, is_audio=False))
        jobs.append(
            {
                "id": f"chunk_{index:03d}",
                "video": str(video_chunk),
                "audio": str(audio_chunk),
                "out_path": str(cached),
            }
        )

    if jobs:
        job_file = work / "render_jobs.json"
        job_file.write_text(
            json.dumps(
                {
                    "fps": fps,
                    "bbox_shift": config.avatar.bbox_shift,
                    "jobs": jobs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  rendering {len(jobs)} chunk(s) on GPU{config.gpu.avatar}")
        run(
            [
                venv_python(config.avatar.venv),
                str(config.root / "adapters" / adapter_name),
                str(job_file),
            ],
            env_overlay=config.gpu.env_for("avatar"),
            log_path=episode_dir / "logs" / "avatar.log",
            timeout=21600,
        )

    missing = [str(p) for p in rendered if not p.exists()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} avatar chunk(s) missing after render, first: {missing[0]}. "
            "See logs/avatar.log."
        )

    output = work / "avatar.mp4"
    concat_videos(rendered, output)
    return str(output)


def concat_videos(paths: list[Path], output: Path) -> Path:
    """Stream-copy concat via the demuxer. All chunks share an encoder and
    parameters, so no re-encode is needed here."""
    listing = output.parent / "concat_list.txt"
    listing.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in paths), encoding="utf-8"
    )
    run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(listing),
        "-c", "copy",
        str(output),
    ])
    return output
