#!/usr/bin/env python
"""LatentSync adapter — runs inside the latentsync virtualenv.

Same job-file contract as the MuseTalk adapter. LatentSync renders a sharper
mouth interior (teeth, tongue) and holds up better at 720p and above, at roughly
10-20x the render time. Use it for short hero segments — a cold open, a promo
cut — not for a full daily episode.

Environment:
    LATENTSYNC_HOME  path to the cloned repo (default ~/src/LatentSync)

Install:
    git clone https://github.com/bytedance/LatentSync ~/src/LatentSync
    python -m venv ~/envs/latentsync
    ~/envs/latentsync/bin/pip install -r ~/src/LatentSync/requirements.txt
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

LATENTSYNC_HOME = Path(os.environ.get("LATENTSYNC_HOME", "~/src/LatentSync")).expanduser()


def main(job_path: str) -> int:
    if not LATENTSYNC_HOME.exists():
        print(f"LatentSync repo not found at {LATENTSYNC_HOME}", file=sys.stderr)
        return 2

    payload = json.loads(Path(job_path).read_text(encoding="utf-8"))
    jobs = payload["jobs"]
    steps = int(payload.get("inference_steps", 20))
    guidance = float(payload.get("guidance_scale", 1.5))

    failures: list[str] = []
    for index, job in enumerate(jobs, 1):
        out_path = Path(job["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(jobs)}] {job['id']}", flush=True)

        command = [
            sys.executable,
            "-m",
            "scripts.inference",
            "--unet_config_path",
            "configs/unet/stage2.yaml",
            "--inference_ckpt_path",
            "checkpoints/latentsync_unet.pt",
            "--video_path",
            job["video"],
            "--audio_path",
            job["audio"],
            "--video_out_path",
            str(out_path),
            "--inference_steps",
            str(steps),
            "--guidance_scale",
            str(guidance),
        ]
        completed = subprocess.run(command, cwd=str(LATENTSYNC_HOME), text=True)
        if completed.returncode != 0 or not out_path.exists():
            print(f"  failed: {job['id']}", file=sys.stderr)
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
