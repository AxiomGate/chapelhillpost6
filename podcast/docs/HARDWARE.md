# Hardware plan

Target machine: 2× RTX 3090 (24 GB each) + 1× RTX A1000 (8 GB), Ubuntu 24.04 LTS.

## What each card does

| Card | VRAM | Job | Why |
|---|---|---|---|
| GPU0 — 3090 | 24 GB | MuseTalk / LatentSync avatar render | Longest single stage; gets a card to itself |
| GPU1 — 3090 | 24 GB | Chatterbox TTS, and vLLM if running local LLM | TTS is bursty; shares well with an idle LLM |
| GPU2 — A1000 | 8 GB | Displays, faster-whisper, NVENC encode/mux | Keeps the desktop and ffmpeg off the 3090s |

Assignment is not hand-waving — `podcastpipe/gpu.py` reads it from
`config/show.yaml` and sets `CUDA_VISIBLE_DEVICES` on each subprocess, so a
stage physically cannot wander onto the wrong card.

### Be honest about the A1000

It is a 50 W, single-slot, 2304-core card. It is not a compute card and nothing
heavy should be scheduled on it. What it is genuinely good for here:

1. **Driving your monitors.** A desktop session on a 3090 costs 1–2 GB of VRAM
   and stutters whenever a render is running. Move displays to the A1000 and
   both 3090s become clean headless compute.
2. **NVENC.** Its 7th-gen encoder is the same silicon as the 3090s'. Export a
   25-minute 1080p episode there and it costs the 3090s nothing.
3. **faster-whisper.** large-v3 in int8 fits in 8 GB and caption alignment for a
   full episode takes a couple of minutes.

Do not try to run TTS or the avatar model on it. 8 GB is too tight and it will
be 4–6× slower than a 3090 even when it fits.

## Before you install the second 3090

**Power.** Two 3090s are 350 W each at stock, with transient spikes well above
that. Add CPU, drives and the A1000 and you want a **1200 W** PSU minimum,
1300–1500 W if the CPU is a high-core-count part. A quality single-rail unit
with three separate 8-pin PCIe cables — do not run both cards off daisy-chained
pigtails from one cable.

If your PSU is smaller than that, power-limit the cards instead of buying one:

```bash
sudo nvidia-smi -i 0 -pl 280
sudo nvidia-smi -i 1 -pl 280
```

280 W costs roughly 5–8% of performance and drops peak draw by 140 W across the
pair. For this workload that is a very good trade, and it is worth doing even
with a big PSU — see thermals.

**Thermals and spacing.** Blower-style 3090s are rare; most are 2.5–3 slot
open-air coolers that dump heat sideways into whatever is above them. Two of
them adjacent will thermal-throttle, and 3090s have a specific weakness: the
GDDR6X memory on the back of the board hits 100–110 °C and throttles long before
the core does. Watch memory junction temperature, not core:

```bash
nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory,power.draw --format=csv -l 5
```

Keep at least one empty slot between the cards, run strong front intake, and if
the top card still runs hot, a PCIe riser mounting it vertically or in a
different slot is the usual fix.

**PCIe lanes.** x8/x8 is fine. Nothing here streams enough data between host and
device for x16 to matter; a 25-minute avatar render moves latents, not raw
frames.

**NVLink: skip it.** You will read that 3090s support NVLink and can "pool"
48 GB. For this pipeline that is not useful — no model in the stack needs more
than 24 GB, and the design deliberately runs *different* models on the two cards
concurrently rather than one model across both. NVLink would only help if you
later want to serve a 70B LLM locally with tensor parallelism, and even then
PCIe works, just slower.

## Storage

Episodes are large. Per episode, roughly:

- base loop frames + intermediate PNGs: 8–20 GB (deleted after assembly)
- avatar render (lossless intermediate): 3–6 GB
- final 1080p H.264: 300–800 MB
- audio masters: ~100 MB

Budget **2 TB NVMe** as scratch and keep it separate from the OS drive. The
pipeline writes everything under `work/<episode-id>/` and `podcastpipe gc`
prunes intermediates older than N days (default 14) while keeping masters.

Masters belong somewhere durable — the config has a `publish.archive_dir` that
should point at a NAS, an external drive, or rclone-mounted cloud storage. The
scratch NVMe is not a backup.

## System RAM

64 GB is comfortable. 32 GB works but ffmpeg filtergraph assembly on a
25-minute 1080p timeline plus a model loading concurrently will swap. If you are
buying anyway, 64 GB.

## Rough daily wall-clock, 25-minute episode

| Stage | Card | Time |
|---|---|---|
| Research (fetch + brief) | — | 2–4 min |
| Script draft | — | 1–2 min |
| **Your review** | — | 10–20 min |
| TTS | GPU1 | 4–8 min |
| Avatar render | GPU0 | 30–45 min |
| Captions | A1000 | 2–3 min |
| Assembly + encode | A1000 | 5–10 min |
| Upload | — | 3–10 min |

About 55–80 minutes of machine time, of which you are present for the review
step only. Research and script drafting are scheduled overnight so the draft is
waiting when you sit down.

TTS and the avatar render pipeline against each other by chunk — the avatar
starts on chunk 1 while TTS is still working on chunk 5 — which is where the
two-card split earns its keep.
