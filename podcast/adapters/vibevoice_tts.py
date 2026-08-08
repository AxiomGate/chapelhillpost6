#!/usr/bin/env python
"""VibeVoice TTS adapter — runs inside the vibevoice virtualenv.

Same job-file contract as the Chatterbox adapter, so the two are interchangeable
via ``tts.engine`` in show.yaml. Use this one when you want a co-host: VibeVoice
generates multi-speaker dialogue with real turn-taking in a single pass, which
chunked single-speaker synthesis cannot imitate.

A chunk may carry ``speaker`` (0-indexed) to select among the reference voices
listed in ``speakers``; it defaults to speaker 0 and the single
``reference_audio``.

Install (community fork; Microsoft pulled the code from their own repo):
    python -m venv ~/envs/vibevoice
    ~/envs/vibevoice/bin/pip install git+https://github.com/vibevoice-community/VibeVoice.git
    # weights: microsoft/VibeVoice-1.5B (fits easily) or VibeVoice-7B (wants ~24GB)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_MODEL = "microsoft/VibeVoice-1.5B"


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    chunks = job["chunks"]
    if not chunks:
        print("nothing to do")
        return 0

    import soundfile as sf
    import torch
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = job.get("model", DEFAULT_MODEL)
    print(f"loading {model_id} on {device} ...", flush=True)

    processor = VibeVoiceProcessor.from_pretrained(model_id)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=int(job.get("diffusion_steps", 10)))

    voices = job.get("speakers") or [job["reference_audio"]]
    target_sr = int(job.get("sample_rate", 24000))

    failures: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        out_path = Path(chunk["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        text = chunk["text"].strip()
        if not text:
            continue

        speaker = min(int(chunk.get("speaker", 0)), len(voices) - 1)
        print(f"[{index}/{len(chunks)}] {chunk['id']}: {text[:60]!r}", flush=True)
        try:
            inputs = processor(
                text=[f"Speaker {speaker}: {text}"],
                voice_samples=[[voices[speaker]]],
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    tokenizer=processor.tokenizer,
                    cfg_scale=float(job.get("cfg_weight", 1.3)),
                )
            audio = output.speech_outputs[0].float().cpu().numpy().squeeze()
            tmp = out_path.with_suffix(".partial.wav")
            sf.write(str(tmp), audio, target_sr)
            tmp.replace(out_path)
        except Exception as exc:
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
