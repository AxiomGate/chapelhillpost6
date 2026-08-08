"""Stage 3 — synthesize the voice track.

Each speech block is normalized, split into sentence-safe chunks, and handed to
the TTS engine as one batch job so the model loads once per episode rather than
once per chunk. Chunk audio is cached by content hash: editing one paragraph in
the review UI re-synthesizes that paragraph and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..models import Script, assign_timeline
from ..proc import concat_audio, ffprobe_duration, loudnorm, run, venv_python
from ..textnorm import chunk_text, normalize_for_tts

ADAPTERS = {
    "chatterbox": "chatterbox_tts.py",
    "vibevoice": "vibevoice_tts.py",
}


@dataclass
class ChunkJob:
    id: str
    block_id: str
    text: str
    out_path: str


def chunk_hash(text: str, config: Config) -> str:
    """Cache key for a synthesized chunk.

    Includes the parameters that change the audio, so bumping exaggeration or
    swapping the reference sample correctly invalidates the cache while an
    unrelated edit elsewhere in the script does not.
    """
    payload = "|".join(
        [
            text,
            config.tts.engine,
            config.tts.reference_audio,
            str(config.tts.seed),
            str(config.tts.exaggeration),
            str(config.tts.cfg_weight),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def plan_chunks(script: Script, config: Config, cache_dir: Path) -> list[ChunkJob]:
    """Expand the script into the flat list of chunks to synthesize."""
    pronunciations = config.raw.get("pronunciations", {}) or {}
    jobs: list[ChunkJob] = []
    for block in script.speech_blocks():
        spoken = normalize_for_tts(block.text, pronunciations)
        for index, chunk in enumerate(chunk_text(spoken, config.tts.max_chars_per_chunk)):
            digest = chunk_hash(chunk, config)
            jobs.append(
                ChunkJob(
                    id=f"{block.id}-{index:03d}-{digest}",
                    block_id=block.id,
                    text=chunk,
                    out_path=str(cache_dir / f"{digest}.wav"),
                )
            )
    return jobs


def pending_jobs(jobs: list[ChunkJob]) -> list[ChunkJob]:
    """Jobs whose audio is not already cached.

    Deduplicated by output path: two blocks containing the identical sentence
    (a recurring sign-off, a repeated station ID) synthesize once.
    """
    pending: list[ChunkJob] = []
    claimed: set[str] = set()
    for job in jobs:
        if job.out_path in claimed or Path(job.out_path).exists():
            continue
        claimed.add(job.out_path)
        pending.append(job)
    return pending


def synthesize(config: Config, script: Script, episode_dir: Path) -> dict[str, str]:
    """Run TTS for the whole script. Returns the artifact paths it produced."""
    cache_dir = config.work_dir / "_tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = episode_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    reference = config.path(config.tts.reference_audio)
    if not reference.exists():
        raise FileNotFoundError(
            f"voice reference not found at {reference}. Record 60-120 seconds of "
            "clean speech and save it there. See docs/RUNBOOK.md."
        )

    jobs = plan_chunks(script, config, cache_dir)
    if not jobs:
        raise ValueError("script contains no speech blocks")

    todo = pending_jobs(jobs)
    print(f"  {len(jobs)} chunks, {len(todo)} to synthesize, {len(jobs) - len(todo)} cached")

    if todo:
        job_file = episode_dir / "tts_jobs.json"
        job_file.write_text(
            json.dumps(
                {
                    "reference_audio": str(reference),
                    "sample_rate": config.tts.sample_rate,
                    "seed": config.tts.seed,
                    "exaggeration": config.tts.exaggeration,
                    "cfg_weight": config.tts.cfg_weight,
                    "chunks": [
                        {"id": j.id, "text": j.text, "out_path": j.out_path} for j in todo
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        adapter = ADAPTERS.get(config.tts.engine)
        if adapter is None:
            raise ValueError(f"unknown tts.engine {config.tts.engine!r}")

        run(
            [
                venv_python(config.tts.venv),
                str(config.root / "adapters" / adapter),
                str(job_file),
            ],
            env_overlay=config.gpu.env_for("tts"),
            log_path=episode_dir / "logs" / "tts.log",
            timeout=7200,
        )

        missing = [j.out_path for j in todo if not Path(j.out_path).exists()]
        if missing:
            raise RuntimeError(
                f"TTS adapter finished but {len(missing)} chunk(s) are missing, "
                f"first: {missing[0]}. See logs/tts.log."
            )

    # Join chunks per block so each block has one audio file and a real duration.
    by_block: dict[str, list[str]] = {}
    for job in jobs:
        by_block.setdefault(job.block_id, []).append(job.out_path)

    for block in script.speech_blocks():
        paths = by_block.get(block.id, [])
        if not paths:
            continue
        block_wav = audio_dir / f"{block.id}.wav"
        concat_audio(paths, block_wav, config.tts.crossfade_ms, config.tts.sample_rate)
        block.audio_path = str(block_wav)
        block.duration = ffprobe_duration(block_wav)

    gap = float(config.raw.get("block_gap_seconds", 0.35))
    total = assign_timeline(script.speech_blocks(), gap=gap)
    print(f"  voice track: {total / 60:.1f} min")

    # One continuous track, then two masters at different loudness targets.
    raw_track = audio_dir / "voice_raw.wav"
    concat_with_gaps(
        [b.audio_path for b in script.speech_blocks()], raw_track, gap, config.tts.sample_rate
    )

    podcast_master = audio_dir / "master_podcast.wav"
    youtube_master = audio_dir / "master_youtube.wav"
    loudnorm(raw_track, podcast_master, config.audio.podcast_lufs, config.audio.true_peak,
             config.audio.loudness_range)
    loudnorm(raw_track, youtube_master, config.audio.youtube_lufs, config.audio.true_peak,
             config.audio.loudness_range)

    mp3 = audio_dir / "episode.mp3"
    run(
        ["ffmpeg", "-y", "-i", str(podcast_master), "-c:a", "libmp3lame", "-b:a", "128k",
         "-ac", "1", str(mp3)]
    )

    return {
        "voice_raw": str(raw_track),
        "master_podcast": str(podcast_master),
        "master_youtube": str(youtube_master),
        "mp3": str(mp3),
        "duration": f"{total:.3f}",
    }


def concat_with_gaps(
    paths: list[str], output: Path, gap: float, sample_rate: int
) -> Path:
    """Concatenate block audio with a fixed silence between blocks.

    The pause is inserted here rather than asked of the TTS model, which gives a
    consistent rhythm between stories and keeps block durations honest for the
    video timeline.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if gap <= 0:
        return concat_audio(paths, output, 0, sample_rate)

    command: list[str] = ["ffmpeg", "-y"]
    for path in paths:
        command += ["-i", path]

    parts = []
    labels = []
    for index in range(len(paths)):
        label = f"a{index}"
        pad = f"apad=pad_dur={gap}" if index < len(paths) - 1 else "anull"
        parts.append(f"[{index}:a]aresample={sample_rate},{pad}[{label}]")
        labels.append(f"[{label}]")
    parts.append(f"{''.join(labels)}concat=n={len(paths)}:v=0:a=1[out]")

    command += [
        "-filter_complex", ";".join(parts),
        "-map", "[out]",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output),
    ]
    run(command)
    return output
