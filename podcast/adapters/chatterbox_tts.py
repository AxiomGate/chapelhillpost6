#!/usr/bin/env python
"""Chatterbox TTS adapter — runs inside the chatterbox virtualenv.

Reads a job file written by ``podcastpipe.stages.tts`` and synthesizes every
chunk in it. The model is loaded once for the whole batch, which is the entire
reason this is a batch adapter rather than a per-chunk call.

    python adapters/chatterbox_tts.py work/2026-08-08/tts_jobs.json

Install:
    python -m venv ~/envs/chatterbox
    ~/envs/chatterbox/bin/pip install chatterbox-tts torch torchaudio
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    chunks = job["chunks"]
    if not chunks:
        print("nothing to do")
        return 0

    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device visible; this will be very slow", file=sys.stderr)

    torch.manual_seed(int(job.get("seed", 1234)))

    print(f"loading Chatterbox on {device} ...", flush=True)
    model = ChatterboxTTS.from_pretrained(device=device)

    reference = job["reference_audio"]
    exaggeration = float(job.get("exaggeration", 0.5))
    cfg_weight = float(job.get("cfg_weight", 0.5))
    target_sr = int(job.get("sample_rate", 24000))

    failures: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        out_path = Path(chunk["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        text = chunk["text"].strip()
        if not text:
            continue

        print(f"[{index}/{len(chunks)}] {chunk['id']}: {text[:60]!r}", flush=True)
        try:
            wav = model.generate(
                text,
                audio_prompt_path=reference,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            sample_rate = model.sr
            if sample_rate != target_sr:
                wav = torchaudio.functional.resample(wav, sample_rate, target_sr)
                sample_rate = target_sr
            # Write to a temp name and rename, so an interrupted run never
            # leaves a truncated file that the cache would later trust.
            tmp = out_path.with_suffix(".partial.wav")
            torchaudio.save(str(tmp), wav.cpu(), sample_rate)
            tmp.replace(out_path)
        except Exception as exc:  # keep going; the orchestrator reports gaps
            print(f"  FAILED {chunk['id']}: {exc}", file=sys.stderr, flush=True)
            failures.append(chunk["id"])

    if failures:
        print(f"{len(failures)} chunk(s) failed: {failures[:5]}", file=sys.stderr)
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
