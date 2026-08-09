# Runbook

## Recording the two assets that matter

Everything downstream inherits the quality of these two files. An hour spent
here is worth more than any amount of parameter tuning.

### `assets/voice/reference.wav`

60–120 seconds of you reading calmly.

- Same mic, same room, same distance you would use for the show. The clone
  inherits the room, so record where you actually record.
- No music, no background noise, no clipping. Watch the meter; peaks around
  −6 dB.
- Read at the pace and energy you want on air. If you read the reference in a
  flat monotone, every episode will be flat.
- Include a couple of questions and a couple of emphatic sentences so the model
  has intonation range to work from.
- Mono, 24 kHz or higher, WAV. Trim silence from both ends.

Re-record it if you change mics or rooms. Then delete `work/_tts_cache/` — the
cache key includes the reference path but not its contents, so an in-place
replacement will not invalidate it.

### `assets/avatar/base_loop.mp4`

3–5 minutes of you on camera, **listening rather than speaking**.

- Sit as you would while presenting. Look at the lens, not the screen.
- Blink naturally. Small head movements, occasional posture shifts, the odd nod.
- Keep your mouth closed and relaxed. MuseTalk replaces the mouth region, and
  it has an easier time starting from a neutral closed mouth.
- Steady lighting, no flicker, no one walking behind you.
- Frame head-and-shoulders with some headroom. Avoid resting a hand near your
  face — anything crossing the mouth region confuses the inpainting.
- 1080p, 25 or 30 fps, locked-off camera on a tripod.

Longer is better: the pipeline ping-pong-loops this clip, and a 5-minute source
loops far less visibly across a 25-minute episode than a 90-second one.

If you would rather not be on camera, generate the loop once from a single
portrait with LivePortrait or SadTalker and reuse it forever.

## First run

```bash
source .venv/bin/activate
podcastpipe doctor           # fix everything it flags first

podcastpipe research --lookback 72
```

Read the per-feed counts. Any feed printing `0 recent items` or an error needs
its URL fixed in `config/sources.yaml` or removing. Ten working feeds beat
thirty half-working ones.

```bash
podcastpipe script
podcastpipe review           # http://127.0.0.1:8420
```

Read the whole draft. Check that every factual sentence has a source chip next
to it, and open a couple. Fix what is wrong, approve.

```bash
podcastpipe tts              # listen to work/<date>/audio/master_podcast.wav
```

Do not render video until the audio sounds right. Tune the voice first (below),
then:

```bash
podcastpipe avatar
podcastpipe captions
podcastpipe assemble
```

Watch the result before enabling publishing. Leave `youtube_privacy: private`
for the first week.

## Tuning the voice

In `config/show.yaml` under `tts`:

| Symptom | Change |
|---|---|
| Flat, monotone | `exaggeration` 0.45 → 0.6 |
| Over-acted, singsong | `exaggeration` → 0.35 |
| Does not sound like you | `cfg_weight` → 0.3, and re-record the reference |
| Rushed | `cfg_weight` → 0.6 |
| Odd pauses mid-sentence | lower `max_chars_per_chunk` to 220 |
| Audible seams between chunks | raise `crossfade_ms` to 60–80 |
| Not enough space between stories | raise `block_gap_seconds` to 0.6 |

**Mispronounced local names are a text problem, not a model problem.** Add them
to the `pronunciations` map in `show.yaml`:

```yaml
pronunciations:
  "Efland": "EFF-land"
  "Haw River": "Hah River"
```

Spell it the way it sounds. This is the single highest-value tuning in the whole
config, and it is worth adding to every time you hear something wrong.

## Tuning the avatar

| Symptom | Change |
|---|---|
| Lips look too closed / barely move | `bbox_shift` −5 to −10 |
| Mouth region too large, jaw artifacts | `bbox_shift` +5 to +10 |
| Visible seam at the loop point | re-record a longer base loop |
| Identity looks off | wrong tool for the job — check `renderer` is `musetalk` |
| Render is far too slow | confirm it is on the 3090: `nvidia-smi` during a render |

After changing `bbox_shift`, clear `work/_avatar_cache/` or the old renders will
be reused.

## Daily rhythm

Cron does research and drafting at 4:30am:

```
30 4 * * * /opt/podcastpipe/scripts/daily.sh >> ~/podcast-cron.log 2>&1
```

You sit down, run `podcastpipe review`, spend 10–20 minutes editing, approve, and
run `podcastpipe finish`. It takes 55–80 minutes unattended.

If you want it to publish while you eat breakfast, run `finish` in tmux:

```bash
tmux new -d -s ep 'source .venv/bin/activate && podcastpipe finish'
tmux attach -t ep
```

## Troubleshooting

**`research` finds nothing.** Feeds have moved. Run with `--lookback 168` to
check whether it is a recency problem or a URL problem, and fix
`config/sources.yaml`.

**Everything is a duplicate.** `--dedupe-days` is too generous for how fast your
sources move. Drop it to 7.

**TTS chunk failures.** Read `work/<date>/logs/tts.log`. Out-of-memory means
something else is on GPU1 — check `nvidia-smi`. The most common cause is a
desktop session that was never moved to the A1000.

**MuseTalk cannot find weights.** `bash download_weights.sh` in the MuseTalk
repo, and confirm `MUSETALK_HOME` points at it.

**ffmpeg: "Unknown encoder h264_nvenc".** Your ffmpeg build lacks NVENC. Install
a full build or set `video.encoder: libx264` — slower, works everywhere.

**Video and audio drift apart.** Almost always a base loop whose fps does not
match `avatar.fps`. Set `avatar.fps` to the base loop's actual rate:
`ffprobe -show_streams assets/avatar/base_loop.mp4 | grep r_frame_rate`.

**Captions are out of sync.** They are aligned to the synthesized audio, so a
mismatch means `assemble` used a different audio file than `captions` did.
Re-run both.

**Out of disk.** `podcastpipe gc --days 7`. If that is not enough,
`rm -rf work/_avatar_cache` — it rebuilds, it just costs render time.

## Before you go public

- [ ] Listen to a full episode end to end, not a sample
- [ ] Confirm every name and number in the script against its source
- [ ] Check the captions on the finished video, not just the SRT
- [ ] Set `youtube_privacy: public` only after a week of private test uploads
- [ ] Point `publish.archive_dir` at real backup storage
- [ ] Validate the RSS feed at podba.se/validate before submitting it anywhere
- [ ] Decide how you will run corrections, and say so on air in episode one
