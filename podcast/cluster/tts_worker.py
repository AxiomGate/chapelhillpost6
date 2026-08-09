#!/usr/bin/env python
"""Voice worker — Chatterbox, warm. Runs on node-b (10.10.5.16), RTX 3090.

Loads the model once at container start and holds it. A job synthesizes one text
chunk and writes a WAV to the shared export; the response carries the duration so
the scheduler can decide when an avatar window is full.

    POST /run  {"id": "...", "text": "...", "out_path": "/pipeline/...",
                "reference_audio": "/pipeline/assets/voice/reference.wav"}
    -> {"ok": true, "duration": 17.4, "out_path": "..."}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from worker_common import build_app, serve  # noqa: E402

DEFAULT_PORT = 8080
REQUIRED_VRAM_MB = int(os.environ.get("REQUIRED_VRAM_MB", "6000"))


def load_model():
    import torch
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(os.environ.get("TTS_SEED", "1234")))
    return {"tts": ChatterboxTTS.from_pretrained(device=device), "device": device}


def handle(model, job: dict) -> dict:
    import torchaudio

    text = (job.get("text") or "").strip()
    out_path = Path(job["out_path"])
    if not text:
        raise ValueError("job has no text")

    reference = job.get("reference_audio")
    if not reference or not Path(reference).exists():
        raise FileNotFoundError(
            f"voice reference not found at {reference!r} inside the container. "
            "Check that the shared export is mounted at /pipeline."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_sr = int(job.get("sample_rate", 24000))

    wav = model["tts"].generate(
        text,
        audio_prompt_path=reference,
        exaggeration=float(job.get("exaggeration", 0.5)),
        cfg_weight=float(job.get("cfg_weight", 0.5)),
    )

    sample_rate = model["tts"].sr
    if sample_rate != target_sr:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sr)
        sample_rate = target_sr

    # Write beside the target and rename, so an interrupted job never leaves a
    # truncated file that the orchestrator's cache would later trust.
    tmp = out_path.with_suffix(".partial.wav")
    torchaudio.save(str(tmp), wav.cpu(), sample_rate)
    tmp.replace(out_path)

    duration = wav.shape[-1] / sample_rate
    return {
        "ok": True,
        "id": job.get("id", ""),
        "out_path": str(out_path),
        "duration": round(float(duration), 3),
    }


app = build_app(
    name="tts-worker",
    role="tts",
    loader=load_model,
    handler=handle,
    required_vram_mb=REQUIRED_VRAM_MB,
)

if __name__ == "__main__":
    serve(app, DEFAULT_PORT)
