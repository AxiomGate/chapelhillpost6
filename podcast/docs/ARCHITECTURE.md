# Architecture

## Shape

Seven stages. Each reads its input from disk, writes its output to disk, and
records what it produced in SQLite. Nothing is held in memory between stages, so
any stage can be re-run in isolation after you edit its input by hand.

```
config/sources.yaml
        │
        ▼
  ┌───────────┐   feeds, dedupe, score
  │ research  │──► work/<date>/brief.json
  └───────────┘
        │
        ▼
  ┌───────────┐   LLM writes segments and blocks
  │  script   │──► work/<date>/script.json
  └───────────┘
        │
        ▼
  ┌───────────┐   ◄── YOU. The only required human step.
  │  review   │──► script.json { approved: true }
  └───────────┘
        │
        ▼
  ┌───────────┐   Chatterbox on GPU1, chunk-cached
  │    tts    │──► audio/{voice_raw,master_podcast,master_youtube}.wav, episode.mp3
  └───────────┘       and block durations written back into script.json
        │
        ├──────────────────────┐
        ▼                      ▼
  ┌───────────┐          ┌───────────┐   faster-whisper on the A1000
  │  avatar   │          │ captions  │──► captions/{captions.srt,captions.ass}
  │ MuseTalk  │          └───────────┘
  │   GPU0    │
  └───────────┘
        │  avatar/avatar.mp4   │
        └──────────┬───────────┘
                   ▼
             ┌───────────┐   one ffmpeg pass, NVENC on the A1000
             │ assemble  │──► video/episode.mp4
             └───────────┘
                   │
                   ▼
             ┌───────────┐
             │  publish  │──► YouTube, output/feed.xml, archive
             └───────────┘
```

`avatar` and `captions` are independent once the voice track exists and can run
concurrently — they are on different cards.

## Why blocks are the unit

A *block* is one contiguous run of speech, typically a paragraph. It is the unit
of:

- **TTS** — synthesized separately, cached by content hash
- **Timeline** — every block has a `start` and `duration` once TTS runs
- **Visual direction** — each block carries a visual cue the assembler renders
- **Sourcing** — each block lists the URLs supporting it
- **Editing** — the review UI edits one block at a time

Block ids are `<segment>-<n>` and stay stable across re-runs, which is what
makes incremental re-rendering work. Editing one paragraph in the review UI
changes that block's text, which changes its content hash, which invalidates
exactly that block's audio and the avatar chunks overlapping it. Everything else
is reused from cache.

## Caching

Two content-addressed caches under `work/`:

**`_tts_cache/<sha>.wav`** — key covers the chunk text plus every parameter that
changes the audio: engine, reference file, seed, exaggeration, cfg weight.
Changing the voice reference correctly invalidates the whole cache; fixing a
typo in paragraph nine invalidates one chunk. Two blocks with identical text
(a recurring sign-off) synthesize once.

**`_avatar_cache/<sha>.mp4`** — key covers the audio chunk's bytes plus renderer,
base loop, fps and bbox shift. It also holds the ping-pong base clip, which is
built once per base loop rather than once per episode.

`podcastpipe gc --days 14` prunes episode intermediates while keeping masters,
scripts and briefs.

## Process isolation

MuseTalk, Chatterbox, VibeVoice and LatentSync pin mutually incompatible
versions of torch, diffusers and transformers. There is no single environment
that satisfies all of them, and pretending otherwise wastes a weekend.

So each lives in its own virtualenv and the orchestrator drives it as a
subprocess (`podcastpipe/proc.py`), passing a JSON job file and setting
`CUDA_VISIBLE_DEVICES` to pin the card. The child sees exactly one GPU as
`cuda:0`, so tool defaults land correctly even for tools with no device flag.

Job files are batched — one file describing every chunk — so the model loads
once per episode instead of once per chunk. That is the difference between a
6-minute TTS stage and a 40-minute one.

Adding an engine means writing one adapter that reads the job-file format and
adding a line to the `ADAPTERS` map. Nothing else changes.

## Where the state lives

`work/pipeline.db` (SQLite) holds *state*, not content:

- `episodes` — id, date, number, title, status, artifact paths
- `stories` — every story attached to every episode, which is what makes
  "did we already cover this?" answerable
- `events` — an append-only log of what each stage did and when

Content lives in files. If the database is lost, the episodes are still on disk;
if the files are lost, the database tells you what was published.

## Failure modes and what happens

| Failure | Behavior |
|---|---|
| One RSS feed is down | Logged with its name, skipped. Never fatal |
| All feeds return nothing | `research` exits 1 rather than drafting from nothing |
| Model invents a citation | URL is stripped; the claim surfaces as unsupported |
| One TTS chunk fails | Adapter continues, reports which; orchestrator names the first missing file and stops before assembly |
| Interrupted TTS | Chunks are written to `.partial.wav` and renamed on completion, so a truncated file never poisons the cache |
| Avatar chunk fails | Same pattern: named, and assembly does not proceed on a gap |
| ffmpeg fails | Full command and stderr in `work/<date>/logs/<stage>.log` |
| Bad visual cue JSON in the UI | Previous cue is kept rather than lost to a typo |

## Extending it

**Second host.** Set `tts.engine: vibevoice`, add `speakers` to the job payload,
and give blocks a `speaker` field. The adapter already handles it; the script
prompt needs speaker labels added.

**Different video look.** `assemble.build_video_filtergraph` is a pure function
with tests. Change the composition there and the tests tell you if the graph is
still well-formed before you spend forty minutes rendering.

**More visual cue types.** Add a branch in `plan_overlays`. Cues are free-form
JSON, so the scripting prompt and the assembler are the only two places that
need to agree.

**Local-only operation.** `llm.provider: local` plus a vLLM server on GPU1. Read
the tradeoff in MODEL_CHOICES.md first — it is a real one for a news show.
