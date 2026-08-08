# The Post 6 Daily — video pipeline

Turns a daily local-news podcast into video: gather sources, draft a script, get
your approval, synthesize your cloned voice, render a talking head, cut the
video, publish.

Runs entirely on your own hardware except the scripting model, which is a config
line away from local too.

```
sources ──► brief ──► script ──► [YOU APPROVE] ──► voice ──► avatar ──► video ──► publish
  RSS        LLM       LLM          web UI       Chatterbox  MuseTalk   ffmpeg    YouTube
                                                    GPU1       GPU0     A1000      + RSS
```

## Quick start

```bash
cd podcast
bash scripts/setup_ubuntu.sh      # system packages, ffmpeg, orchestrator venv
bash scripts/install_models.sh    # Chatterbox + MuseTalk (40-60 GB, takes a while)

cp config/env.example .env && $EDITOR .env
source .venv/bin/activate
podcastpipe doctor                # tells you exactly what is still missing
```

Three assets you have to make yourself:

| File | What | How long |
|---|---|---|
| `assets/voice/reference.wav` | You reading calmly, clean audio, no music | 60–120 s, once |
| `assets/avatar/base_loop.mp4` | You on camera *listening*, not speaking | 3–5 min, once |
| `assets/brand/background.png` | 1920×1080 backdrop | any image to start |

Then edit `config/show.yaml` (your name, style guide, segments) and
`config/sources.yaml` (feeds — verify each one actually resolves).

## Daily use

```bash
podcastpipe research     # fetch feeds, dedupe, build a sourced brief
podcastpipe script       # draft the episode
podcastpipe review       # http://127.0.0.1:8420 — edit, check sources, approve
podcastpipe finish       # voice + avatar + captions + assembly + publish
```

Or let cron do the first two overnight (`scripts/daily.sh`) so a draft is waiting
when you sit down. About 10–20 minutes of your time; 55–80 minutes of machine
time you are not present for.

## Documentation

| Doc | What's in it |
|---|---|
| [MODEL_CHOICES.md](docs/MODEL_CHOICES.md) | Every tool, why it beat the alternatives, licenses |
| [HARDWARE.md](docs/HARDWARE.md) | GPU roles, PSU, thermals, storage, timings |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the stages fit together, caching, failure modes |
| [RUNBOOK.md](docs/RUNBOOK.md) | Recording the assets, tuning the voice, troubleshooting |

## Commands

| Command | Does |
|---|---|
| `research` | Fetch feeds, dedupe against 3 weeks of history, build the brief |
| `script` | Draft the episode from the brief |
| `review` | Serve the approval UI |
| `approve` | Approve from the CLI instead |
| `tts` | Synthesize the voice track |
| `avatar` | Render the talking head |
| `captions` | Transcribe the audio, write SRT + ASS |
| `assemble` | Composite the final video |
| `publish` | Upload to YouTube, rebuild the RSS feed, archive |
| `run` | research + script, stops at the approval gate |
| `finish` | Everything after approval |
| `status` | Recent episodes and their stage |
| `doctor` | Check binaries, GPUs, environments, assets, keys |
| `gc` | Prune intermediates older than N days |

Every stage reads from disk and writes back, so any stage can be re-run alone
after an edit.

## Tests

```bash
make test    # 191 tests, no GPU or ffmpeg needed
```

They cover the logic that can be verified without hardware: text normalization,
chunking, dedupe and scoring, caption timing, ffmpeg filtergraph construction,
chunk planning, cache invalidation, RSS generation, the review UI driven through
a real HTTP client, and an end-to-end research→script run with the LLM stubbed —
including the check that a model-invented citation gets stripped rather than
passed through as a source.

They earn their keep: writing them surfaced four real bugs, two of which
(a SQLite connection unusable from the web server's threadpool, and a FastAPI
annotation-resolution failure that made every save return 422) would have broken
the review UI on the first day of use.

## A word on the editorial risk

The pipeline generates a draft. It does not know anything. On a show carrying
the American Legion Post 6 name, in a town where the people in the stories are
your neighbors, the approval step is not a formality — it is the product.

The design supports that: every claim carries its sources, the review UI puts
them next to the sentence they justify, unsourced blocks are flagged, and a URL
the model invented is stripped before it can look verified. Read the drafts.
