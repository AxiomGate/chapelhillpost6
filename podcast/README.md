# podcastpipe — multi-tenant podcast video pipeline

Turns a daily audio podcast into video, for any number of client shows off one codebase: gather sources, draft a script, get
your approval, synthesize your cloned voice, render a talking head, cut the
video, publish.

Runs entirely on your own hardware except the scripting model, which is a config
line away from local too. Nothing in the codebase is branded: `config/show.yaml`
holds engineering defaults only, and each show's name, voice, likeness, colours,
domain, feeds and editorial style live in its own profile under `clients/`.

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

## Adding a client

```bash
cp -r clients/example clients/acme
$EDITOR clients/acme/show.yaml      # name, host, style guide, segments, colours
$EDITOR clients/acme/sources.yaml   # this show's feeds and keywords
podcastpipe clients                 # confirm it is picked up
```

A profile states only what differs from the base; everything else is inherited.
Then add that client's three assets, which are the only things you cannot
generate:

| File (under `clients/<name>/`) | What | How long |
|---|---|---|
| `assets/voice/reference.wav` | The host reading calmly, clean audio, no music | 60–120 s, once |
| `assets/avatar/base_loop.mp4` | The host on camera *listening*, not speaking | 3–5 min, once |
| `assets/brand/background.png` | 1920×1080 backdrop | any image to start |

Client assets shadow the shared ones, so a client without its own background
falls back to a house default — but a client's voice and likeness never reach
another show.

## Daily use

```bash
podcastpipe --client acme research   # fetch feeds, dedupe, build a sourced brief
podcastpipe --client acme script     # draft the episode
podcastpipe --client acme review     # http://127.0.0.1:8420 — edit, approve
podcastpipe --client acme finish     # voice + avatar + captions + assembly + publish
```

Or set `PODCASTPIPE_CLIENT=acme` once and drop the flag. Each client gets its own
`work/<client>/` and `output/<client>/` tree, so two shows produced the same day
cannot collide.

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
| `clients` | List configured client profiles |
| `cluster` | Show worker node health across the GPU nodes |

Every stage reads from disk and writes back, so any stage can be re-run alone
after an edit.

## Tests

```bash
make test    # 281 tests, no GPU or ffmpeg needed
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

The pipeline generates a draft. It does not know anything. When a show runs under
a client's name and the people in the stories are real, the approval step is not
a formality — it is the product.

The design supports that: every claim carries its sources, the review UI puts
them next to the sentence they justify, unsourced blocks are flagged, and a URL
the model invented is stripped before it can look verified. Read the drafts.

## Tenant isolation

Running many clients off one codebase makes cross-tenant leakage the failure that
matters most. Four boundaries enforce it, each with tests:

- **Config** — the base file carries no client name, domain or feeds, and a test
  fails the build if one appears.
- **Assets** — resolved from the client directory first; a client with no voice
  reference gets a missing-file error, never another client's voice.
- **Work and output** — namespaced per client, so same-day episodes cannot collide.
- **Credentials** — YouTube secrets are per client, since a shared token would
  upload every show to one channel.
