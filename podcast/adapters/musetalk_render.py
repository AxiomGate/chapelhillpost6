#!/usr/bin/env python
"""MuseTalk adapter — runs inside the musetalk virtualenv.

Reads a job file from ``podcastpipe.stages.avatar`` and drives MuseTalk's own
inference entrypoint once per chunk. MuseTalk ships as a repo with a CLI rather
than an importable package, so this shells out to it and then moves the result
to the cache path the orchestrator expects.

    python adapters/musetalk_render.py work/2026-08-08/avatar/render_jobs.json

Environment:
    MUSETALK_HOME   path to the cloned MuseTalk repo (default ~/src/MuseTalk)

Install:
    git clone https://github.com/TMElyralab/MuseTalk ~/src/MuseTalk
    python -m venv ~/envs/musetalk
    ~/envs/musetalk/bin/pip install -r ~/src/MuseTalk/requirements.txt
    cd ~/src/MuseTalk && bash download_weights.sh
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MUSETALK_HOME = Path(os.environ.get("MUSETALK_HOME", "~/src/MuseTalk")).expanduser()


def render_one(job: dict, fps: int, bbox_shift: int, workdir: Path) -> bool:
    """Render a single chunk. Returns True on success."""
    config_path = workdir / f"{job['id']}.yaml"
    config_path.write_text(
        "task_0:\n"
        f"  video_path: \"{job['video']}\"\n"
        f"  audio_path: \"{job['audio']}\"\n"
        f"  bbox_shift: {bbox_shift}\n",
        encoding="utf-8",
    )

    result_dir = workdir / f"results_{job['id']}"
    command = [
        sys.executable,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config_path),
        "--result_dir",
        str(result_dir),
        "--fps",
        str(fps),
        "--use_float16",
    ]

    print(f"  $ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(MUSETALK_HOME), text=True)
    if completed.returncode != 0:
        print(f"  MuseTalk exited {completed.returncode} for {job['id']}", file=sys.stderr)
        return False

    produced = sorted(result_dir.rglob("*.mp4"))
    if not produced:
        print(f"  MuseTalk produced no mp4 for {job['id']}", file=sys.stderr)
        return False

    out_path = Path(job["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Largest output is the full render; MuseTalk also writes small previews.
    best = max(produced, key=lambda p: p.stat().st_size)
    shutil.move(str(best), str(out_path))
    shutil.rmtree(result_dir, ignore_errors=True)
    return True


def main(job_path: str) -> int:
    if not MUSETALK_HOME.exists():
        print(
            f"MuseTalk repo not found at {MUSETALK_HOME}. Clone it, or set "
            "MUSETALK_HOME to where it lives.",
            file=sys.stderr,
        )
        return 2

    payload = json.loads(Path(job_path).read_text(encoding="utf-8"))
    jobs = payload["jobs"]
    fps = int(payload.get("fps", 25))
    bbox_shift = int(payload.get("bbox_shift", 0))

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="musetalk_") as tmp:
        workdir = Path(tmp)
        for index, job in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {job['id']}", flush=True)
            if not render_one(job, fps, bbox_shift, workdir):
                failures.append(job["id"])

    if failures:
        print(f"{len(failures)} chunk(s) failed: {failures}", file=sys.stderr)
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
