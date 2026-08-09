#!/usr/bin/env python
"""Media worker — captions and video encoding. Runs on node-a's RTX A1000.

Two jobs share this container because they share a card and never overlap in the
pipeline: captions run once the voice track is complete, encoding runs once the
avatar is. Whisper stays resident between episodes; ffmpeg is invoked per job.

The A1000 is a 50 W, 8 GB card — not a compute card. It earns its place by
keeping the desktop and all video encoding off the 3090s, so those stay clean
compute. Whisper large-v3 in int8 fits comfortably and transcribes a full episode
in a couple of minutes.

    POST /run  {"task": "captions", "audio": "...", "out_dir": "...", "prompt": "..."}
    POST /run  {"task": "encode", "args": ["-i", ...], "out_path": "..."}
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/opt/podcastpipe/src")

from worker_common import build_app, serve  # noqa: E402

DEFAULT_PORT = 8082
REQUIRED_VRAM_MB = int(os.environ.get("REQUIRED_VRAM_MB", "2500"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8_float16")


def load_model():
    from faster_whisper import WhisperModel

    return {
        "whisper": WhisperModel(
            WHISPER_MODEL, device="cuda", compute_type=WHISPER_COMPUTE
        )
    }


def _captions(model, job: dict) -> dict:
    from podcastpipe.stages.captions import Word, group_words, to_ass, to_srt

    audio = Path(job["audio"])
    out_dir = Path(job["out_dir"])
    if not audio.exists():
        raise FileNotFoundError(f"audio not found at {audio} inside the container")
    out_dir.mkdir(parents=True, exist_ok=True)

    segments, _ = model["whisper"].transcribe(
        str(audio),
        language=job.get("language", "en"),
        word_timestamps=True,
        vad_filter=True,
        # Biasing on the script sharply improves spelling of local proper nouns.
        initial_prompt=(job.get("prompt") or "")[:900] or None,
    )

    words = [
        Word(text=w.word.strip(), start=float(w.start), end=float(w.end))
        for segment in segments
        for w in (segment.words or [])
        if w.word.strip()
    ]
    cues = group_words(words)

    srt_path = out_dir / "captions.srt"
    ass_path = out_dir / "captions.ass"
    srt_path.write_text(to_srt(cues), encoding="utf-8")
    ass_path.write_text(
        to_ass(cues, int(job.get("width", 1920)), int(job.get("height", 1080))),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "srt": str(srt_path),
        "ass": str(ass_path),
        "words": len(words),
        "cues": len(cues),
    }


def _encode(model, job: dict) -> dict:
    """Run a prepared ffmpeg command. The orchestrator builds the argv — that
    logic is unit-tested there and has no business being duplicated here."""
    args = job.get("args")
    if not isinstance(args, list) or not args:
        raise ValueError("encode job needs an 'args' list")

    command = ["ffmpeg", "-y", *[str(a) for a in args]]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({completed.returncode}):\n{completed.stderr[-3000:]}"
        )

    out_path = job.get("out_path")
    if out_path and not Path(out_path).exists():
        raise RuntimeError(f"ffmpeg reported success but {out_path} does not exist")

    return {"ok": True, "out_path": out_path or ""}


TASKS = {"captions": _captions, "encode": _encode}


def handle(model, job: dict) -> dict:
    task = job.get("task", "")
    runner = TASKS.get(task)
    if runner is None:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
    return runner(model, job)


app = build_app(
    name="media-worker",
    role="media",
    loader=load_model,
    handler=handle,
    required_vram_mb=REQUIRED_VRAM_MB,
)

if __name__ == "__main__":
    serve(app, DEFAULT_PORT)
