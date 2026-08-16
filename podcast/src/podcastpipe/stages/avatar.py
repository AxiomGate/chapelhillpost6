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


def _render_via_cluster(cluster, config: Config, base_loop: Path, todo: list[dict]) -> None:
    """Render chunks on the avatar workers, several at a time.

    Concurrent, unlike voice. There is more than one avatar worker, each render
    runs for minutes, and this is the longest stage of the episode -- dispatching
    one chunk at a time would leave half the cluster idle for most of it. The
    threads are nearly free: each spends its life blocked on an HTTP call while a
    GPU on another machine does the work. Width comes from the cluster's own
    declared capacity, so adding a third avatar node needs no change here.

    The payload names the base loop rather than shipping any video. Each worker
    keeps its own local copy, builds its own ping-pong clip once, and slices
    windows from it -- which is what keeps gigabytes off the 1 GbE.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ..cluster import ClusterError

    def render_one(job: dict) -> str:
        result = cluster.submit(
            "avatar",
            {
                "id": job["id"],
                "audio": job["audio"],
                "out_path": job["out_path"],
                "base_loop": str(base_loop),
                "fps": config.avatar.fps,
                "bbox_shift": config.avatar.bbox_shift,
                # Each window starts at a different point in the loop, so head
                # motion does not restart in lockstep every chunk and give the
                # whole episode a visible two-minute cycle.
                "loop_offset": job["start"],
            },
            timeout=7200,
        )
        if not Path(job["out_path"]).exists():
            raise ClusterError(
                f"{result.get('node', 'worker')} reported success but "
                f"{job['out_path']} does not exist"
            )
        return result.get("node", "worker")

    width = max(1, cluster.capacity("avatar"))
    print(f"  rendering {len(todo)} chunk(s) across {width} worker(s)")

    failures: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures = {pool.submit(render_one, job): job for job in todo}
        for future in as_completed(futures):
            job = futures[future]
            done += 1
            try:
                node = future.result()
            except Exception as exc:
                failures.append(f"{job['id']}: {exc}")
                print(f"    {done}/{len(todo)}  {job['id']} FAILED")
            else:
                print(f"    {done}/{len(todo)}  {job['id']} on {node}")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(todo)} avatar chunk(s) failed on the cluster. "
            f"First: {failures[0]}"
        )


def render(
    config: Config, episode_dir: Path, audio_path: str | Path, cluster=None
) -> str:
    """Render the full talking-head track. Returns the path to the silent video.

    With ``cluster`` set, chunks go to the avatar workers and each builds its own
    driving video locally. Without it, the driving video is built here and a
    local venv does the rendering -- the single-machine path.
    """
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

    adapter_name = ADAPTERS.get(config.avatar.renderer)
    if adapter_name is None:
        raise ValueError(f"unknown avatar.renderer {config.avatar.renderer!r}")

    chunks = plan_chunks(duration, config.avatar.chunk_seconds)
    print(f"  {duration / 60:.1f} min in {len(chunks)} chunk(s)")

    # Only the local path needs a driving video here. Cluster workers build
    # their own from a node-local copy of the base loop, which is the whole
    # reason gigabytes of footage never cross the network.
    driving = None
    if cluster is None:
        # The ping-pong clip depends only on the base loop, so it is built once
        # and reused for every episode until the loop itself changes.
        pingpong = (
            cache
            / f"pingpong_{hashlib.sha256(str(base_loop).encode()).hexdigest()[:12]}_{fps}.mp4"
        )
        if not pingpong.exists():
            print("  building ping-pong base loop (one time per base clip)")
            run(pingpong_command(base_loop, pingpong, fps), log_path=work / "logs_pingpong.txt")

        driving = work / "driving.mp4"
        run(extend_command(pingpong, driving, duration, fps), log_path=work / "logs_extend.txt")

    rendered: list[Path] = []
    jobs = []
    for index, (start, length) in enumerate(chunks):
        audio_chunk = work / f"chunk_{index:03d}.wav"
        run(slice_command(audio_path, audio_chunk, start, length, is_audio=True))

        key = chunk_key(config, audio_chunk, start, length)
        cached = cache / f"{key}.mp4"
        rendered.append(cached)
        if cached.exists():
            print(f"  chunk {index + 1}/{len(chunks)}: cached")
            continue

        job = {
            "id": f"chunk_{index:03d}",
            "audio": str(audio_chunk),
            "out_path": str(cached),
            "start": start,
        }
        if cluster is None:
            video_chunk = work / f"chunk_{index:03d}.mp4"
            run(slice_command(driving, video_chunk, start, length, is_audio=False))
            job["video"] = str(video_chunk)
        jobs.append(job)

    if jobs and cluster is not None:
        _render_via_cluster(cluster, config, base_loop, jobs)
    elif jobs:
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
