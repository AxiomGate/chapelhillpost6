#!/usr/bin/env python3.10
"""Render one real chunk through InfiniteTalk and report a measured fps.

Not part of the podcastpipe package on purpose -- this pilot owns nothing
outside its own directory, so it does not import from cluster/ or src/. It
exists to answer one question with a number instead of a guess: what does
InfiniteTalk actually render at on this hardware, for this show.

Usage (inside the pilot container -- see README.md for the full docker run):

    python3.10 run_pilot.py \
        --base-loop /weights_or_share/base_loop.mp4 \
        --audio /share/chunk_60s.wav \
        --out /out/infinitetalk_pilot.mp4 \
        --ckpt-dir /weights/Wan2.1-I2V-14B-480P \
        --wav2vec-dir /weights/chinese-wav2vec2-base \
        --infinitetalk-dir /weights/InfiniteTalk/single/infinitetalk.safetensors

Prints fps_effective in the same terms avatar_worker.py already reports for
MuseTalk (frames / render_seconds, load time excluded) so the two numbers are
directly comparable without unit conversion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def ffprobe_frames_and_duration(path: Path) -> tuple[int, float]:
    """Frame count and duration of a video file, via ffprobe.

    Counted after the render, from the output file itself, rather than assumed
    from the audio length -- this project has already shipped one avatar chunk
    that exited 0 with a fraction of the frames it should have had, and the
    only thing that caught it was checking the actual output.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    frames = int(stream.get("nb_read_frames", 0) or 0)
    duration = float(stream.get("duration", 0) or 0)
    return frames, duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-loop", required=True, type=Path,
                         help="Driving video -- the same base_loop.mp4 already recorded for MuseTalk.")
    parser.add_argument("--audio", required=True, type=Path,
                         help="One real audio chunk, ideally 60s to match MuseTalk's chunk_seconds default.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompt", default="A man sits at a desk, talking directly to the camera.",
                         help="Text conditioning. Keep it a plain, accurate description of the actual "
                              "footage -- this is not a creative prompt, it is telling a video diffusion "
                              "model what it is looking at.")
    parser.add_argument("--ckpt-dir", required=True, type=Path, help="Wan2.1-I2V-14B-480P weights.")
    parser.add_argument("--wav2vec-dir", required=True, type=Path, help="chinese-wav2vec2-base weights.")
    parser.add_argument("--infinitetalk-dir", required=True, type=Path,
                         help="InfiniteTalk conditioning weights (.safetensors).")
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument(
        "--low-vram", action="store_true", default=True,
        help="--num_persistent_param_in_dit 0. On by default: the 14B base model is not confirmed to "
             "fit a 24GB 3090 without this, and the README's own guidance is to reach for it on an "
             "OOM. Pass --no-low-vram to test full-VRAM mode once this succeeds once.",
    )
    parser.add_argument("--no-low-vram", dest="low_vram", action="store_false")
    args = parser.parse_args()

    for label, path in (("base loop", args.base_loop), ("audio", args.audio)):
        if not path.exists():
            print(f"error: {label} not found at {path}", file=sys.stderr)
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    job = {
        "prompt": args.prompt,
        "cond_video": str(args.base_loop),
        "cond_audio": {"person1": str(args.audio)},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(job, handle)
        job_path = Path(handle.name)
    print(f"job config: {job_path}\n  {json.dumps(job, indent=2)}")

    save_stem = args.out.with_suffix("")
    command = [
        "python3.10", "generate_infinitetalk.py",
        "--ckpt_dir", str(args.ckpt_dir),
        "--wav2vec_dir", str(args.wav2vec_dir),
        "--infinitetalk_dir", str(args.infinitetalk_dir),
        "--input_json", str(job_path),
        "--size", "infinitetalk-480",
        "--sample_steps", str(args.sample_steps),
        "--mode", "streaming",
        "--motion_frame", "9",
        "--save_file", str(save_stem),
    ]
    if args.low_vram:
        command += ["--num_persistent_param_in_dit", "0"]

    print(f"\nrunning: {' '.join(command)}\n")
    started = time.monotonic()
    completed = subprocess.run(command)
    render_seconds = time.monotonic() - started

    if completed.returncode != 0:
        print(
            f"\nInfiniteTalk exited {completed.returncode} after {render_seconds:.1f}s. "
            "A -9 is the OOM signature this project has hit before with MuseTalk -- if you "
            "see it here, --low-vram is already on by default, so the next thing to try is "
            "the fp8 quantized weights (see README.md).",
            file=sys.stderr,
        )
        return completed.returncode

    # generate_infinitetalk.py appends its own extension; find what it actually wrote
    # rather than assume .mp4, since a wrong guess here would misreport a real success
    # as a missing-output failure.
    produced = save_stem.parent.glob(save_stem.name + "*")
    candidates = sorted((p for p in produced if p.is_file()), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        print(f"\nInfiniteTalk exited 0 but nothing matching {save_stem}* was found.", file=sys.stderr)
        return 1
    output = candidates[0]
    if output != args.out:
        output.replace(args.out)
        output = args.out

    frames, duration = ffprobe_frames_and_duration(output)
    fps_effective = round(frames / render_seconds, 3) if render_seconds else 0

    print(f"""
=== pilot result ===
output:          {output}
render_seconds:  {render_seconds:.1f}
frames:          {frames}
output duration: {duration:.1f}s
fps_effective:   {fps_effective}   (MuseTalk baseline on the same hardware: 2.93)
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
