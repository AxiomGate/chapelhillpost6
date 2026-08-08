# Model & software choices

Research notes behind every tool this pipeline calls. Written August 2026.
The short version is in the table; the reasoning and the rejected alternatives
follow.

## The stack

| Stage | Choice | License | Where it runs |
|---|---|---|---|
| Research + scripting | Claude API (`claude-sonnet-5`), local vLLM + Qwen3-32B-AWQ as fallback | commercial API / Apache-2.0 | network or GPU1 |
| Voice | Chatterbox (Resemble AI) zero-shot clone | MIT | GPU1 |
| Voice (2-host option) | VibeVoice-1.5B / 7B | MIT (community fork) | GPU1 |
| Talking head | MuseTalk 1.5 mouth inpainting over a recorded base loop | see below | GPU0 |
| Talking head (hero shots) | LatentSync 1.6 | Apache-2.0 | GPU0 |
| Captions / alignment | faster-whisper (large-v3) | MIT | A1000 |
| Assembly + encode | ffmpeg 7.x, NVENC | LGPL/GPL | A1000 |
| Orchestration | this repo, Python 3.11 | — | CPU |

## Talking head — the decision that matters most

This is where naive plans fail. The exciting 2025–2026 models (InfiniteTalk,
Wan 2.2-S2V, HunyuanVideo-Avatar, Hallo3) are *video diffusion* models: they
synthesize every frame from scratch. They look superb and they are far too slow
for a daily show on a 3090.

Rough throughput on a single RTX 3090 at 480p:

| Model | Approach | Speed | 25-min episode |
|---|---|---|---|
| MuseTalk 1.5 | latent inpainting of the mouth region only | ~30+ fps | **~30–45 min** |
| LatentSync 1.6 | latent diffusion, full face | ~1–3 fps | ~6–10 h |
| InfiniteTalk / Wan-S2V | full video diffusion, 81-frame rolling window | ~0.05–0.1 fps | 10–20 h+ |

A daily podcast has a hard deadline every single day. That rules out anything
that cannot finish an episode in under an hour on one card, which leaves
MuseTalk.

**So the design is: you record a base loop once, and MuseTalk drives the mouth.**

You sit down once and record 3–5 minutes of yourself at the desk, on camera,
just listening and nodding — no speech, natural micro-movements, occasional
blinks and small posture shifts. The pipeline ping-pong-loops that clip to
whatever length the episode needs and MuseTalk inpaints your mouth to match the
synthesized audio. The head motion, lighting, background and body are real
footage of you, so the result sits well outside the uncanny valley that
single-photo animators land in. It also completely avoids the identity drift
that plagues long diffusion generations.

If you would rather not be on camera at all, the same loop can be generated once
from a single portrait with LivePortrait or SadTalker and reused forever — the
cost is paid once, not per episode.

LatentSync 1.6 stays wired in for short hero segments (a 30-second cold open, a
promo cut) where the sharper mouth interior and teeth are worth ten minutes of
render. `avatar.renderer: latentsync` per-segment switches it on.

Licensing note: MuseTalk's own code is Apache-2.0-ish but it pulls
`sd-vae-ft-mse` and Whisper weights, and TMElyralab's model card asks that
outputs not be used to impersonate people without consent. You are cloning
yourself with your own footage, which is exactly the intended use, but keep a
signed consent note for anyone else who appears. LatentSync is Apache-2.0 and
unencumbered.

Rejected: **Wav2Lip** — still the most-recommended tool on the open web and it
should not be. It renders a 96×96 mouth patch that looks blurry and dated
against 2026 output. **SadTalker** — good for the one-time base loop, not for
per-episode use. **HeyGen / Synthesia** — excellent quality, but $30–$500/mo
forever and your voice and likeness live on their servers; you are buying
hardware specifically to avoid that.

## Voice — Chatterbox

Chatterbox is Resemble AI's open model, MIT-licensed, ~8 GB VRAM, zero-shot
cloning from a short reference, with an emotion-exaggeration dial that is
genuinely useful for keeping a 25-minute read from going flat. Independent
listening tests through 2026 have it beating ElevenLabs in side-by-side
preference more often than not. MIT means no license question about monetizing
the show.

Give it a good reference sample and it gets much better: 60–120 seconds of you
reading calmly in the voice you want, recorded on the mic you actually own, in
the room you actually record in, no music, no clipping, no room echo. The
pipeline expects that file at `assets/voice/reference.wav` (24 kHz+ mono WAV).

Long-form is handled by chunking, not by asking the model for 25 minutes at
once. `stages/tts.py` splits on sentence boundaries into ~300-character units,
synthesizes each with the same reference and a fixed seed, then concatenates
with short crossfades. This keeps timbre stable and means a single bad chunk is
re-rollable without regenerating the episode.

**VibeVoice** is the alternative and it is the right choice if you ever add a
co-host. It generates up to 90 minutes with four distinct speakers in one pass
with real turn-taking, which chunked single-speaker TTS cannot imitate. The
1.5B fits comfortably; the 7B wants the full 24 GB. Microsoft pulled the code
from their own repo in late 2025 citing misuse; the weights are still on
Hugging Face and a community fork maintains the code, all MIT. It is wired in
as `tts.engine: vibevoice`.

Rejected: **XTTS-v2** — still good, but Coqui's CPML license forbids commercial
use, which is a problem the moment the show takes a sponsor. **Kokoro** — lovely
and tiny, but no voice cloning. **Higgs Audio v3** — technically impressive,
research/non-commercial license. **ElevenLabs** — the quality bar, but recurring
cost and your voice print sits with a third party.

## Research and scripting — Claude API, with a local escape hatch

This is the one place where I recommend spending money, and it is a few cents an
episode.

A daily local-news podcast lives or dies on not saying false things about real
people in your town. Summarizing a dozen sources into a factually tight script
with correct attribution is precisely where a 32B local model degrades in ways
that are hard to spot — it stays fluent while quietly inventing a detail. On a
show carrying the American Legion Post 6 name, that is the expensive failure.

So: `claude-sonnet-5` for the research brief and the script draft, at roughly
2–5¢ per episode. Both 3090s stay free for TTS and video, which is what they are
actually good for.

The local path is real, not decorative — set `llm.provider: local` and the same
prompts run against a vLLM server on GPU1. Qwen3-32B-AWQ fits in 24 GB with room
for a long context. Use it for cost control, for offline operation, or if you
simply prefer nothing leaving the house. The two paths share one interface in
`llm.py`, so switching is a config line.

Every claim in a generated brief carries a source URL, and the review UI shows
them inline next to the sentence they support. That check is the actual safety
mechanism regardless of which model wrote the draft.

## Captions

faster-whisper large-v3 on the A1000, run against the *synthesized* audio rather
than the script. That matters: TTS occasionally elides or reflows a word, and
aligning to what was actually spoken keeps captions frame-accurate. Word-level
timestamps drive both the burned-in karaoke captions and the SRT sidecar that
YouTube ingests.

## Encoding

Both 3090s and the A1000 are Ampere, which means 7th-gen NVENC: H.264 and HEVC
hardware encode, **no AV1** — AV1 encode starts at Ada (RTX 40-series). Not a
problem: YouTube wants H.264 High profile for 1080p ingest anyway, and it
transcodes to AV1 itself on the serving side.

Encoding runs on the A1000 (`hevc_nvenc` / `h264_nvenc`) so a 25-minute export
never competes with the avatar render for a 3090's time.

## Sources

- [MuseTalk (TMElyralab)](https://github.com/TMElyralab/MuseTalk)
- [Open-source lip sync tools compared, 2026](https://lipsync.com/blog/open-source-lip-sync)
- [Best free open-source lip-sync models, ranked](https://www.pixazo.ai/blog/best-open-source-ai-lip-sync-models)
- [InfiniteTalk (MeiGen-AI)](https://github.com/MeiGen-AI/InfiniteTalk)
- [Best open-source TTS models 2026 (BentoML)](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- [Chatterbox benchmark vs ElevenLabs](https://findskill.ai/blog/best-open-source-tts-2026/)
- [Open-source voice cloning tools (Resemble AI)](https://www.resemble.ai/resources/best-open-source-ai-voice-cloning-tools)
- [VibeVoice community fork](https://github.com/vibevoice-community/VibeVoice)
- [NVIDIA RTX A1000 product page](https://www.nvidia.com/en-us/products/workstations/rtx-a1000/)
