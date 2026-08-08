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
    config = load_config(args.config)
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


def cmd_research(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)
    episode_dir = _episode_dir(config, episode_id)

    episode = database.get_episode(episode_id) or Episode(id=episode_id, date=episode_id)
    database.upsert_episode(episode)

    print(f"Researching {episode_id}")
    sources = research_stage.load_sources(config.root / "config" / "sources.yaml")
    raw = research_stage.fetch_feeds(sources, args.lookback)
    print(f"  {len(raw)} raw items")

    stories = research_stage.dedupe_stories(raw, database.seen_urls(args.dedupe_days))
    print(f"  {len(stories)} after dedupe")
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
    artifacts = tts_stage.synthesize(config, script, episode_dir)
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
    episode_dir = _episode_dir(config, episode_id)
    episode = database.get_episode(episode_id)
    audio = (episode.artifacts.get("voice_raw") if episode else None) or str(
        episode_dir / "audio" / "voice_raw.wav"
    )

    print(f"Rendering avatar for {episode_id}")
    started = time.time()
    path = avatar_stage.render(config, episode_dir, audio)
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
    artifacts = captions_stage.generate(
        audio,
        episode_dir / "captions",
        script_text=script_text,
        width=config.video.width,
        height=config.video.height,
    )
    for key, value in artifacts.items():
        database.set_artifact(episode_id, f"captions_{key}", value)
    database.log(episode_id, "captions", "ok", "")
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    config, database, episode_id = _context(args)
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
    output = assemble_stage.assemble(config, script, episode_dir, avatar, audio, ass_path)
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

    if config.publish.archive_dir:
        archive = Path(config.publish.archive_dir).expanduser() / episode_id
        archive.mkdir(parents=True, exist_ok=True)
        for key in ("video", "mp3", "master_podcast", "captions_srt"):
            source = episode.artifacts.get(key)
            if source and Path(source).exists():
                shutil.copy2(source, archive / Path(source).name)
        for name in ("script.json", "brief.json"):
            if (episode_dir / name).exists():
                shutil.copy2(episode_dir / name, archive / name)
        print(f"  archived to {archive}")

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
    config = load_config(args.config)
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
    config = load_config(args.config)
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


def cmd_review(args: argparse.Namespace) -> int:
    from .review.app import serve

    config = load_config(args.config)
    serve(config, host=args.host, port=args.port)
    return 0


# ---- parser -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcastpipe", description="Daily podcast video pipeline"
    )
    parser.add_argument("--config", help="path to show.yaml", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--date", help="episode id, YYYY-MM-DD (default: today)")
        sp.set_defaults(handler=handler)
        return sp

    research = add("research", cmd_research, "gather sources and build the brief")
    research.add_argument("--lookback", type=int, default=36, help="feed window in hours")
    research.add_argument("--dedupe-days", type=int, default=21)
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

    run_cmd = add("run", cmd_run, "research + script, then stop for review")
    run_cmd.add_argument("--yes", action="store_true", help="skip the approval gate")
    run_cmd.add_argument("--lookback", type=int, default=36)
    run_cmd.add_argument("--dedupe-days", type=int, default=21)
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
