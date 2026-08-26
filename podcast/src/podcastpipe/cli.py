"""Command line interface.

Every stage is a separate command that reads its input from disk and writes its
output back, so any stage can be re-run in isolation after an edit. ``run``
chains them with the approval gate in the middle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from .config import Config, load_config
from .db import Database
from .models import Brief, Claim, Episode, Script, Story
from .stages import assemble as assemble_stage
from .stages import captions as captions_stage
from .stages import publish as publish_stage
from .stages import research as research_stage
from .stages import script as script_stage
from .stages import tts as tts_stage

STAGE_ORDER = ["new", "researched", "scripted", "approved", "voiced", "rendered", "published"]


def _context(args: argparse.Namespace) -> tuple[Config, Database, str]:
    config = load_config(args.config, getattr(args, "client", None))
    database = Database(config.work_dir / "pipeline.db")
    episode_id = args.date or Episode.make_id()
    return config, database, episode_id


def _episode_dir(config: Config, episode_id: str) -> Path:
    path = config.episode_dir(episode_id)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    return path


def _load_brief(path: Path) -> Brief:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Brief(
        episode_id=data["episode_id"],
        generated_at=data.get("generated_at", ""),
        headline=data.get("headline", ""),
        stories=[Story(**s) for s in data.get("stories", [])],
        claims=[Claim(**c) for c in data.get("claims", [])],
        notes=data.get("notes", ""),
    )


def _save_brief(brief: Brief, path: Path) -> None:
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(brief), indent=2, ensure_ascii=False), encoding="utf-8")


# ---- commands -----------------------------------------------------------


ARCHIVE_ARTIFACTS = (
    "video",
    "mp3",
    "master_podcast",
    "master_youtube",
    "captions_srt",
    "captions_ass",
    "avatar",
)
ARCHIVE_FILES = ("script.json", "brief.json")


def _archive_run(
    config: Config,
    database: Database,
    episode_id: str,
    episode_dir: Path,
    label: str = "",
) -> Path:
    """Snapshot one finished run into its own timestamped directory.

    Runs are kept individually rather than overwriting a per-episode folder,
    because the reason to keep a good one is to compare it against the next.
    A directory of identical-looking episode.mp4 files is nearly useless for
    that, so each snapshot carries a manifest of the settings that produced it
    — voice, blend width, batch size, style guide, the lot. When one run sounds
    or looks better than another, the manifest is what says why.
    """
    episode = database.get_episode(episode_id)
    now = datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    # Last four digits of the run time, HHMM. Stamped onto every archived file,
    # not just the folder: two generations both called episode.mp4 collide the
    # moment anyone drags them into the same directory to compare them, or
    # uploads them, which is the whole reason for keeping generations apart.
    short = now.strftime("%H%M")
    name = f"{episode_id}_{stamp}" + (f"_{label}" if label else "")

    root = (
        Path(config.publish.archive_dir).expanduser()
        if config.publish.archive_dir
        else config.output_dir / "archive"
    )
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)

    def stamped(filename: str) -> str:
        """episode.mp4 -> episode_1822.mp4, keeping the extension usable."""
        source = Path(filename)
        return f"{source.stem}_{short}{source.suffix}"

    copied: dict[str, str] = {}
    for key in ARCHIVE_ARTIFACTS:
        source = (episode.artifacts.get(key) if episode else None) or ""
        if source and Path(source).exists():
            target = destination / stamped(Path(source).name)
            shutil.copy2(source, target)
            copied[key] = target.name
    for filename in ARCHIVE_FILES:
        if (episode_dir / filename).exists():
            target = destination / stamped(filename)
            shutil.copy2(episode_dir / filename, target)
            copied[filename] = target.name

    raw = config.raw
    manifest = {
        "episode_id": episode_id,
        "archived_at": now.isoformat(timespec="seconds"),
        # The suffix every file in this folder carries, so the manifest can be
        # matched back to a loose file someone copied out of here.
        "run_stamp": short,
        "label": label,
        "client": config.client or "",
        "title": (episode.title if episode else "") or "",
        "status": (episode.status if episode else "") or "",
        "files": copied,
        "sizes_bytes": {
            path.name: path.stat().st_size
            for path in sorted(destination.iterdir())
            if path.is_file()
        },
        # The settings that made this run what it is. Snapshotted rather than
        # referenced: config/show.yaml will have moved on by the time anyone
        # asks why an old run sounded better.
        "settings": {
            "show": {
                k: raw.get("show", {}).get(k)
                for k in ("name", "host", "tagline", "target_minutes")
            },
            "style_guide": raw.get("show", {}).get("style_guide", ""),
            "segments": [
                {"id": s.get("id"), "target_seconds": s.get("target_seconds")}
                for s in raw.get("show", {}).get("segments", []) or []
            ],
            "llm": {"model": config.llm.model, "max_tokens": config.llm.max_tokens},
            "tts": {
                "engine": config.tts.engine,
                "reference_audio": config.tts.reference_audio,
                "exaggeration": config.tts.exaggeration,
                "cfg_weight": config.tts.cfg_weight,
                "seed": config.tts.seed,
            },
            "avatar": {
                "renderer": config.avatar.renderer,
                "base_loop": config.avatar.base_loop,
                "fps": config.avatar.fps,
                "chunk_seconds": config.avatar.chunk_seconds,
                "bbox_shift": config.avatar.bbox_shift,
                "parsing_mode": config.avatar.parsing_mode,
                "left_cheek_width": config.avatar.left_cheek_width,
                "right_cheek_width": config.avatar.right_cheek_width,
                "extra_margin": config.avatar.extra_margin,
                "batch_size": config.avatar.batch_size,
            },
            "video": {
                "width": config.video.width,
                "height": config.video.height,
                "encoder": config.video.encoder,
                "cq": config.video.cq,
                "burn_captions": config.video.burn_captions,
            },
            "audio": {
                "podcast_lufs": config.audio.podcast_lufs,
                "youtube_lufs": config.audio.youtube_lufs,
            },
        },
    }
    # Stamped like everything else. A manifest that travels with a file someone
    # copied out is only useful if it does not collide with the last one.
    (destination / stamped("manifest.json")).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def cmd_archive(args: argparse.Namespace) -> int:
    """Keep this run. Copies the deliverables somewhere they will not be
    overwritten by the next attempt at the same episode."""
    config, database, episode_id = _context(args)
    episode_dir = _episode_dir(config, episode_id)

    if database.get_episode(episode_id) is None:
        print(f"Unknown episode {episode_id}", file=sys.stderr)
        return 1

    destination = _archive_run(config, database, episode_id, episode_dir, args.label)
    files = sorted(p.name for p in destination.iterdir() if p.is_file())
    # The manifest is always written, so it is not evidence that anything was
    # produced. Anything else in the folder is.
    if not [f for f in files if not f.startswith("manifest")]:
        print(
            f"Nothing to archive for {episode_id} — no rendered artifacts found. "
            "Run the pipeline through assemble or publish first.",
            file=sys.stderr,
        )
        return 1

    print(f"Archived {episode_id} to {destination}")
    for name in files:
        print(f"  {name}")
    database.log(episode_id, "archive", "ok", str(destination))
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)
    episode_dir = _episode_dir(config, episode_id)

    episode = database.get_episode(episode_id) or Episode(id=episode_id, date=episode_id)
    database.upsert_episode(episode)

    print(f"Researching {episode_id} for {config.label()}")
    sources = research_stage.load_sources(config.sources_path())
    raw = research_stage.fetch_feeds(sources, args.lookback)
    print(f"  {len(raw)} raw items")

    seen = database.seen_urls(args.dedupe_days)
    stories = research_stage.dedupe_stories(raw, seen)
    window = "all history" if args.dedupe_days <= 0 else f"{args.dedupe_days}d"
    print(
        f"  {len(stories)} after dedupe "
        f"({len(seen)} previously covered, window {window})"
    )
    if not stories:
        print("No stories found. Check config/sources.yaml and your network.", file=sys.stderr)
        database.log(episode_id, "research", "failed", "no stories")
        return 1

    brief = research_stage.build_brief(config, episode_id, stories, args.max_stories)
    _save_brief(brief, episode_dir / "brief.json")
    database.add_stories(episode_id, brief.stories)

    unsupported = research_stage.unsupported_claims(brief)
    if unsupported:
        print(f"  ! {len(unsupported)} claim(s) lost their sources — review these carefully")

    database.set_status(episode_id, "researched")
    database.set_artifact(episode_id, "brief", str(episode_dir / "brief.json"))
    database.log(episode_id, "research", "ok", f"{len(brief.stories)} stories")
    print(f"  brief: {episode_dir / 'brief.json'}")
    print(f"  headline: {brief.headline}")
    return 0


def cmd_script(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)
    episode_dir = _episode_dir(config, episode_id)

    brief_path = episode_dir / "brief.json"
    if not brief_path.exists():
        print(f"No brief at {brief_path}. Run 'research' first.", file=sys.stderr)
        return 1

    print(f"Writing script for {episode_id}")
    brief = _load_brief(brief_path)
    script = script_stage.generate_script(config, brief)
    script.save(episode_dir / "script.json")

    warnings = script_stage.validate_script(script, config)
    for warning in warnings:
        print(f"  ! {warning}")

    episode = database.get_episode(episode_id) or Episode(id=episode_id, date=episode_id)
    episode.title = script.title
    episode.status = "scripted"
    database.upsert_episode(episode)
    database.set_artifact(episode_id, "script", str(episode_dir / "script.json"))
    database.log(episode_id, "script", "ok", f"{script.word_count()} words")

    print(f"  title: {script.title}")
    print(f"  {script.word_count()} words, about {script.estimated_minutes():.1f} min")
    print(f"  script: {episode_dir / 'script.json'}")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)
    path = _episode_dir(config, episode_id) / "script.json"
    if not path.exists():
        print(f"No script at {path}", file=sys.stderr)
        return 1
    script = Script.load(path)
    script.approved = True
    script.save(path)
    database.set_status(episode_id, "approved")
    database.log(episode_id, "approve", "ok", "approved from CLI")
    print(f"Approved {episode_id}: {script.title}")
    return 0


def cmd_tts(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)
    episode_dir = _episode_dir(config, episode_id)
    path = episode_dir / "script.json"
    if not path.exists():
        print(f"No script at {path}", file=sys.stderr)
        return 1

    script = Script.load(path)
    if not script.approved and not args.force:
        print("Script is not approved. Run 'approve', or pass --force.", file=sys.stderr)
        return 1

    print(f"Synthesizing voice for {episode_id}")
    started = time.time()
    artifacts = tts_stage.synthesize(
        config, script, episode_dir, cluster=_cluster_client(config)
    )
    script.save(path)  # durations and timeline are now filled in

    for key, value in artifacts.items():
        database.set_artifact(episode_id, key, value)
    database.set_status(episode_id, "voiced")
    database.log(episode_id, "tts", "ok", f"{time.time() - started:.0f}s")
    print(f"  done in {time.time() - started:.0f}s")
    return 0


def cmd_avatar(args: argparse.Namespace) -> int:
    from .stages import avatar as avatar_stage

    config, database, episode_id = _context(args)

    # renderer: none is the audio-first configuration, not an error. Skip here
    # rather than raising, so `finish` runs straight through to captions and
    # publish on an episode that was never meant to have video.
    if config.avatar.renderer == "none":
        print("Avatar renderer is 'none' — audio-only episode, skipping.")
        return 0

    episode_dir = _episode_dir(config, episode_id)
    episode = database.get_episode(episode_id)
    audio = (episode.artifacts.get("voice_raw") if episode else None) or str(
        episode_dir / "audio" / "voice_raw.wav"
    )

    print(f"Rendering avatar for {episode_id}")
    started = time.time()
    path = avatar_stage.render(config, episode_dir, audio, cluster=_cluster_client(config))
    elapsed = time.time() - started
    database.set_artifact(episode_id, "avatar", path)
    database.log(episode_id, "avatar", "ok", f"{elapsed:.0f}s")
    print(f"  {path} in {elapsed / 60:.1f} min")
    return 0


def cmd_captions(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)
    episode_dir = _episode_dir(config, episode_id)
    episode = database.get_episode(episode_id)
    audio = (episode.artifacts.get("master_youtube") if episode else None) or str(
        episode_dir / "audio" / "master_youtube.wav"
    )
    script_path = episode_dir / "script.json"
    script_text = Script.load(script_path).full_text() if script_path.exists() else ""

    print(f"Generating captions for {episode_id}")
    cluster = _cluster_client(config)
    if cluster is not None:
        # faster-whisper lives in the media worker's image, not the
        # orchestrator's -- the orchestrator runs no models by design.
        result = cluster.submit(
            "media",
            {
                "task": "captions",
                "audio": audio,
                "out_dir": str(episode_dir / "captions"),
                "prompt": script_text,
                "width": config.video.width,
                "height": config.video.height,
            },
            timeout=1800,
        )
        artifacts = {"srt": result["srt"], "ass": result["ass"]}
        print(
            f"  {result.get('words', 0)} words, {result.get('cues', 0)} cues "
            f"on {result.get('node', 'worker')}"
        )
    else:
        artifacts = captions_stage.generate(
            audio,
            episode_dir / "captions",
            script_text=script_text,
            width=config.video.width,
            height=config.video.height,
            font=config.video.caption_font,
            highlight=config.video.caption_highlight,
        )
    for key, value in artifacts.items():
        database.set_artifact(episode_id, f"captions_{key}", value)
    database.log(episode_id, "captions", "ok", "")
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)

    # Nothing to composite without an avatar render. The audio masters are
    # already the deliverable for an audio-first episode.
    if config.avatar.renderer == "none":
        print("Avatar renderer is 'none' — no video to assemble, skipping.")
        return 0

    episode_dir = _episode_dir(config, episode_id)
    episode = database.get_episode(episode_id)
    if episode is None:
        print(f"Unknown episode {episode_id}", file=sys.stderr)
        return 1

    script = Script.load(episode_dir / "script.json")
    avatar = episode.artifacts.get("avatar") or str(episode_dir / "avatar" / "avatar.mp4")
    audio = episode.artifacts.get("master_youtube") or str(
        episode_dir / "audio" / "master_youtube.wav"
    )
    ass_path = episode.artifacts.get("captions_ass")

    print(f"Assembling {episode_id}")
    started = time.time()
    output = assemble_stage.assemble(
        config, script, episode_dir, avatar, audio, ass_path,
        cluster=_cluster_client(config),
    )
    database.set_artifact(episode_id, "video", output)
    database.set_status(episode_id, "rendered")
    database.log(episode_id, "assemble", "ok", f"{time.time() - started:.0f}s")
    print(f"  {output}")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)
    episode_dir = _episode_dir(config, episode_id)
    episode = database.get_episode(episode_id)
    if episode is None:
        print(f"Unknown episode {episode_id}", file=sys.stderr)
        return 1

    script = Script.load(episode_dir / "script.json")
    brief_path = episode_dir / "brief.json"
    sources = _load_brief(brief_path).source_urls() if brief_path.exists() else []
    description = publish_stage.build_youtube_description(
        script.description, sources, config.publish.site_base_url
    )

    if config.publish.youtube_enabled and not args.skip_youtube:
        video = episode.artifacts.get("video")
        if not video or not Path(video).exists():
            print("No rendered video to upload.", file=sys.stderr)
            return 1
        print("Uploading to YouTube")
        video_id = publish_stage.upload_to_youtube(
            config,
            video,
            script.title,
            description,
            script.tags,
            episode.artifacts.get("captions_srt"),
        )
        database.set_artifact(episode_id, "youtube_id", video_id)

    if config.publish.rss_enabled:
        entries = []
        for record in database.list_episodes(limit=500):
            mp3 = record.artifacts.get("mp3")
            if not mp3 or not Path(mp3).exists():
                continue
            entries.append(
                {
                    "id": record.id,
                    "date": record.date,
                    "number": record.number,
                    "title": record.title,
                    "description": record.artifacts.get("description", ""),
                    "duration": float(record.artifacts.get("duration", 0) or 0),
                    "audio_url": f"{config.publish.media_base_url.rstrip('/')}/{record.id}.mp3",
                    "audio_bytes": Path(mp3).stat().st_size,
                }
            )
        path = publish_stage.write_feed(config, entries)
        print(f"  feed: {path} ({len(entries)} episodes)")

    # Always snapshot on publish, timestamped. Publishing the same episode twice
    # used to overwrite the first archive, which quietly destroyed the only copy
    # of whatever the earlier attempt produced.
    print(f"  archived to {_archive_run(config, database, episode_id, episode_dir, 'published')}")

    database.set_artifact(episode_id, "description", script.description)
    database.set_status(episode_id, "published")
    database.log(episode_id, "publish", "ok", "")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Research and script, then stop at the approval gate unless it is passed."""
    steps = [cmd_research, cmd_script]
    for step in steps:
        code = step(args)
        if code != 0:
            return code

    config, database, episode_id = _context(args)
    script_path = _episode_dir(config, episode_id) / "script.json"
    approved = Script.load(script_path).approved if script_path.exists() else False

    if not approved and not args.yes:
        print()
        print("Script is ready for review.")
        print(f"  podcastpipe review          # open the review UI")
        print(f"  podcastpipe approve --date {episode_id}")
        print(f"  podcastpipe finish --date {episode_id}")
        return 0

    if args.yes:
        cmd_approve(args)
    return cmd_finish(args)


def cmd_finish(args: argparse.Namespace) -> int:
    """Everything after approval: voice, avatar, captions, assembly, publish."""
    from .stages import avatar as avatar_stage  # noqa: F401  (import cost is real)

    for step in (cmd_tts, cmd_avatar, cmd_captions, cmd_assemble, cmd_publish):
        code = step(args)
        if code != 0:
            return code
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config, database, _ = _context(args)
    episodes = database.list_episodes(limit=args.limit)
    if not episodes:
        print("No episodes yet. Run 'podcastpipe research'.")
        return 0
    print(f"{'DATE':<12} {'#':>4}  {'STATUS':<11} TITLE")
    for episode in episodes:
        print(f"{episode.date:<12} {episode.number:>4}  {episode.status:<11} {episode.title[:60]}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check that everything the pipeline needs is actually present."""
    config = load_config(args.config, getattr(args, "client", None))
    problems = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal problems
        print(f"  [{'ok' if ok else 'XX'}] {label}{f' — {detail}' if detail else ''}")
        if not ok:
            problems += 1

    print("Binaries")
    for binary in ("ffmpeg", "ffprobe"):
        check(binary, shutil.which(binary) is not None, "not on PATH")

    print("GPUs")
    nvidia_smi = shutil.which("nvidia-smi")
    check("nvidia-smi", nvidia_smi is not None, "not on PATH")
    if nvidia_smi:
        from .proc import run as run_command

        result = run_command(
            [nvidia_smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            check=False,
        )
        for line in result.stdout.strip().splitlines():
            print(f"       {line}")

    print("Environments")
    for label, venv in (("tts", config.tts.venv), ("avatar", config.avatar.venv)):
        check(f"{label} venv ({venv})", (Path(venv).expanduser() / "bin" / "python").exists())

    print("Assets")
    for label, value in (
        ("voice reference", config.tts.reference_audio),
        ("avatar base loop", config.avatar.base_loop),
        ("background", config.video.background),
    ):
        check(f"{label} ({value})", config.path(value).exists())

    print("LLM")
    if config.llm.provider == "anthropic":
        import os

        check("ANTHROPIC_API_KEY", bool(os.environ.get("ANTHROPIC_API_KEY")), "unset")
    else:
        check(f"local endpoint ({config.llm.local_base_url})", True, "not probed")

    print(f"\n{problems} problem(s)" if problems else "\nAll checks passed.")
    return 1 if problems else 0


def cmd_gc(args: argparse.Namespace) -> int:
    """Delete intermediates older than N days, keeping masters and metadata."""
    config = load_config(args.config, getattr(args, "client", None))
    cutoff = datetime.now() - timedelta(days=args.days)
    keep = {"script.json", "brief.json", "episode.mp4", "episode.mp3"}
    freed = 0

    for episode_dir in sorted(config.work_dir.glob("*")):
        if not episode_dir.is_dir() or episode_dir.name.startswith("_"):
            continue
        try:
            when = datetime.strptime(episode_dir.name, "%Y-%m-%d")
        except ValueError:
            continue
        if when >= cutoff:
            continue
        for path in episode_dir.rglob("*"):
            if path.is_file() and path.name not in keep:
                freed += path.stat().st_size
                if not args.dry_run:
                    path.unlink()

    print(f"{'Would free' if args.dry_run else 'Freed'} {freed / 1e9:.2f} GB")
    return 0


def _cluster_client(config):
    """Build a ClusterClient from config/cluster.yaml, or None if disabled."""
    import yaml

    from .cluster import ClusterClient, load_nodes

    path = config.root / "config" / "cluster.yaml"
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not raw.get("enabled"):
        return None

    storage = raw.get("storage", {})
    scheduling = raw.get("scheduling", {})
    return ClusterClient(
        load_nodes(raw.get("nodes", {})),
        shared_root_local=storage.get("local_root", "/pipeline"),
        shared_root_container=storage.get("container_root", "/pipeline"),
        max_failures=int(scheduling.get("max_failures", 3)),
    )


def cmd_clients(args: argparse.Namespace) -> int:
    """List client profiles and their show names."""
    from .config import list_clients

    config = load_config(args.config)
    names = list_clients(config.root)
    if not names:
        print("No client profiles. Copy clients/example/ to clients/<name>/.")
        return 0

    print(f"{'CLIENT':<18} {'SHOW':<34} SUBJECT FEEDS")
    for name in names:
        try:
            client_config = load_config(args.config, name)
            feeds = 0
            path = client_config.sources_path()
            if path.exists():
                import yaml as _yaml

                feeds = len((_yaml.safe_load(path.read_text()) or {}).get("feeds", []) or [])
            print(f"{name:<18} {client_config.show.name[:34]:<34} {feeds}")
        except Exception as exc:
            print(f"{name:<18} {'(config error)':<34} {exc}")
    return 0


def cmd_feeds(args: argparse.Namespace) -> int:
    """Check every configured feed and report which ones actually work.

    A dead feed does not fail a run -- research skips it and carries on, which is
    correct behaviour and also why a show can quietly lose half its sources and
    still produce a brief every morning. This makes that state something you can
    look at directly instead of inferring from a thin episode.

    Exits non-zero if any feed is broken, so it can gate a deploy.
    """
    config = load_config(args.config, getattr(args, "client", None))
    sources = research_stage.load_sources(config.sources_path())
    feeds = sources.get("feeds", []) or []
    if not feeds:
        print(f"No feeds configured in {config.sources_path()}", file=sys.stderr)
        return 1

    print(f"Checking {len(feeds)} feeds for {config.label()}")
    broken: list[tuple[str, str]] = []
    for feed in feeds:
        url = feed.get("url")
        name = feed.get("name", url)
        if not url:
            continue
        parsed, error = research_stage.fetch_feed(url, timeout=args.timeout)
        if error is not None:
            print(f"  ! {name}\n      {url}\n      {error}")
            broken.append((name, url))
            continue
        recent = sum(
            1
            for entry in parsed.entries
            if research_stage.is_recent(
                research_stage.parse_entry_date(
                    entry.get("published")
                    or entry.get("updated")
                    or entry.get("published_parsed")
                ),
                args.lookback,
            )
        )
        print(f"  · {name}: {len(parsed.entries)} items, {recent} within {args.lookback}h")

    if broken:
        print(
            f"\n{len(broken)} of {len(feeds)} feeds are broken. Fix the URL in "
            f"{config.sources_path()} or delete the entry — a source that is "
            "listed but never returns anything is worse than no source at all.",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(feeds)} feeds OK.")
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    """Show every worker's health, GPU, VRAM and throughput."""
    config = load_config(args.config, getattr(args, "client", None))
    client = _cluster_client(config)
    if client is None:
        print("Cluster mode is off. Set 'enabled: true' in config/cluster.yaml.")
        return 0

    report = client.check_health(timeout=args.timeout)
    if not report:
        print("No nodes configured.", file=sys.stderr)
        return 1

    print(f"{'NODE':<12} {'ROLE':<8} {'STATE':<8} {'GPU':<24} {'VRAM FREE':>10}")
    problems = 0
    for name, entry in report.items():
        ok = entry.get("ok")
        problems += 0 if ok else 1
        vram = entry.get("vram_free_mb")
        print(
            f"{name:<12} {entry.get('role', ''):<8} {'up' if ok else 'DOWN':<8} "
            f"{(entry.get('gpu') or '')[:24]:<24} "
            f"{(f'{vram} MB' if vram else '-'):>10}"
        )
        if not ok and entry.get("detail"):
            print(f"             └─ {entry['detail'][:100]}")

    for role in ("tts", "avatar", "media"):
        if client.nodes_for(role):
            print(f"\n{role} capacity: {client.capacity(role)} concurrent slot(s)")

    print(f"\n{problems} node(s) down" if problems else "\nAll nodes healthy.")
    return 1 if problems else 0


def cmd_review(args: argparse.Namespace) -> int:
    from .review.app import serve

    config = load_config(args.config, getattr(args, "client", None))
    serve(config, host=args.host, port=args.port)
    return 0


# ---- parser -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcastpipe", description="Daily podcast video pipeline"
    )
    parser.add_argument("--config", help="path to the base show.yaml", default=None)
    parser.add_argument(
        "--client",
        default=None,
        help="client profile under clients/ (or set PODCASTPIPE_CLIENT)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--date", help="episode id, YYYY-MM-DD (default: today)")
        sp.set_defaults(handler=handler)
        return sp

    research = add("research", cmd_research, "gather sources and build the brief")
    research.add_argument("--lookback", type=int, default=36, help="feed window in hours")
    research.add_argument(
        "--dedupe-days",
        type=int,
        default=21,
        help="days of history to exclude; 0 means never repeat a story",
    )
    research.add_argument("--max-stories", type=int, default=25)

    add("script", cmd_script, "write the script from the brief")
    add("approve", cmd_approve, "mark the script approved")

    tts = add("tts", cmd_tts, "synthesize the voice track")
    tts.add_argument("--force", action="store_true", help="run without approval")

    add("avatar", cmd_avatar, "render the talking head")
    add("captions", cmd_captions, "transcribe and build caption files")
    add("assemble", cmd_assemble, "composite the final video")

    publish = add("publish", cmd_publish, "upload and update the feed")
    publish.add_argument("--skip-youtube", action="store_true")

    archive = add("archive", cmd_archive, "keep this run in its own timestamped folder")
    archive.add_argument(
        "--label",
        default="",
        help="short tag appended to the folder name, e.g. --label cheek50",
    )

    run_cmd = add("run", cmd_run, "research + script, then stop for review")
    run_cmd.add_argument("--yes", action="store_true", help="skip the approval gate")
    run_cmd.add_argument("--lookback", type=int, default=36)
    run_cmd.add_argument("--dedupe-days", type=int, default=21,
                         help="days of history to exclude; 0 means never repeat")
    run_cmd.add_argument("--max-stories", type=int, default=25)
    run_cmd.add_argument("--force", action="store_true")
    run_cmd.add_argument("--skip-youtube", action="store_true")

    finish = add("finish", cmd_finish, "run everything after approval")
    finish.add_argument("--force", action="store_true")
    finish.add_argument("--skip-youtube", action="store_true")

    status = sub.add_parser("status", help="list recent episodes")
    status.add_argument("--limit", type=int, default=20)
    status.add_argument("--date", default=None, help=argparse.SUPPRESS)
    status.set_defaults(handler=cmd_status)

    doctor = sub.add_parser("doctor", help="check the local setup")
    doctor.set_defaults(handler=cmd_doctor)

    gc_cmd = sub.add_parser("gc", help="prune old intermediates")
    gc_cmd.add_argument("--days", type=int, default=14)
    gc_cmd.add_argument("--dry-run", action="store_true")
    gc_cmd.set_defaults(handler=cmd_gc)

    clients_cmd = sub.add_parser("clients", help="list configured client profiles")
    clients_cmd.set_defaults(handler=cmd_clients)

    feeds_cmd = sub.add_parser("feeds", help="check every configured news feed")
    feeds_cmd.add_argument("--lookback", type=int, default=36)
    feeds_cmd.add_argument("--timeout", type=int, default=20)
    feeds_cmd.set_defaults(handler=cmd_feeds)

    cluster = sub.add_parser("cluster", help="show worker node health")
    cluster.add_argument("--timeout", type=int, default=5)
    cluster.set_defaults(handler=cmd_cluster)

    review = sub.add_parser("review", help="serve the approval UI")
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8420)
    review.set_defaults(handler=cmd_review)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
