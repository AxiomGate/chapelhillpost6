"""Stage 6 — composite the finished video.

One ffmpeg pass builds the whole frame: branded background, the avatar render
inset, timed lower-thirds and title cards, B-roll cutaways, and burned-in
captions. Encoding is NVENC on the A1000 so it never queues behind a 3090.

The filtergraph is built by pure functions so the composition can be tested
without ffmpeg present, which matters because a filtergraph typo surfaces as an
opaque error forty minutes into a render otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..models import Block, Script
from ..proc import run

STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "for", "and", "to", "with"}


@dataclass
class Overlay:
    """A timed image laid over the base composition."""

    path: str
    start: float
    end: float
    x: str = "0"
    y: str = "0"
    fade: float = 0.3


def _escape_filter_path(path: str | Path) -> str:
    """Escape a path for use inside a filtergraph argument.

    ffmpeg parses ``:`` as an option separator and ``\\`` as an escape inside
    filter arguments, so a Windows-style or colon-bearing path silently breaks
    the graph.
    """
    text = str(path)
    return text.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def build_video_filtergraph(
    width: int,
    height: int,
    avatar_input: int,
    background_input: int,
    overlays: list[Overlay],
    overlay_input_start: int,
    avatar_box: tuple[int, int, int, int],
    ass_path: str | Path | None = None,
) -> str:
    """Compose background + avatar + overlays + captions into one graph.

    ``avatar_box`` is (x, y, w, h) in output pixels. The avatar is scaled to
    cover that box and center-cropped, so a source clip of any aspect ratio
    fills the frame without letterboxing or distortion.
    """
    box_x, box_y, box_w, box_h = avatar_box
    parts = [
        f"[{background_input}:v]scale={width}:{height},setsar=1[bg]",
        f"[{avatar_input}:v]scale={box_w}:{box_h}:force_original_aspect_ratio=increase,"
        f"crop={box_w}:{box_h},setsar=1[av]",
        f"[bg][av]overlay={box_x}:{box_y}:shortest=1[v0]",
    ]

    current = "v0"
    for index, overlay in enumerate(overlays):
        input_index = overlay_input_start + index
        label_in = f"ov{index}"
        label_out = f"v{index + 1}"
        duration = max(overlay.end - overlay.start, 0.1)
        fade = min(overlay.fade, duration / 2)
        parts.append(
            f"[{input_index}:v]format=rgba,"
            f"fade=t=in:st=0:d={fade:.2f}:alpha=1,"
            f"fade=t=out:st={max(duration - fade, 0):.2f}:d={fade:.2f}:alpha=1,"
            f"setpts=PTS-STARTPTS+{overlay.start:.3f}/TB[{label_in}]"
        )
        parts.append(
            f"[{current}][{label_in}]overlay={overlay.x}:{overlay.y}:"
            f"enable='between(t,{overlay.start:.3f},{overlay.end:.3f})'[{label_out}]"
        )
        current = label_out

    if ass_path:
        parts.append(f"[{current}]ass='{_escape_filter_path(ass_path)}'[vout]")
    else:
        parts.append(f"[{current}]null[vout]")

    return ";".join(parts)


def build_encode_command(
    background: str | Path,
    avatar: str | Path,
    audio: str | Path,
    overlays: list[Overlay],
    output: str | Path,
    config: Config,
    ass_path: str | Path | None = None,
) -> list[str]:
    """Full ffmpeg argv for the composite render."""
    video = config.video
    avatar_box = (
        int(video.width * 0.30),
        int(video.height * 0.08),
        int(video.width * 0.40),
        int(video.height * 0.72),
    )

    command = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(background),   # 0
        "-i", str(avatar),                     # 1
        "-i", str(audio),                      # 2
    ]
    for overlay in overlays:
        command += ["-loop", "1", "-i", overlay.path]

    filtergraph = build_video_filtergraph(
        width=video.width,
        height=video.height,
        avatar_input=1,
        background_input=0,
        overlays=overlays,
        overlay_input_start=3,
        avatar_box=avatar_box,
        ass_path=ass_path if video.burn_captions else None,
    )

    command += [
        "-filter_complex", filtergraph,
        "-map", "[vout]",
        "-map", "2:a",
        "-shortest",
        "-r", str(video.fps),
        "-c:v", video.encoder,
        "-preset", video.preset,
        "-rc", "vbr",
        "-cq", str(video.cq),
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        str(output),
    ]
    return command


def keyword_score(query: str, candidate: str) -> int:
    """Overlap between a B-roll request and an asset filename.

    Deliberately dumb: local B-roll libraries are small and hand-named, and a
    word-overlap count beats an embedding index that has to be rebuilt whenever
    a file is added.
    """
    def tokens(text: str) -> set[str]:
        return {
            w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in STOPWORDS
        }

    return len(tokens(query) & tokens(candidate))


def find_broll(query: str, library: Path) -> Path | None:
    """Best-matching asset in the B-roll library, or None if nothing matches."""
    if not query or not library.exists():
        return None
    candidates = [
        p for p in library.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".mov"}
    ]
    scored = [(keyword_score(query, p.stem), p) for p in candidates]
    scored = [(score, path) for score, path in scored if score > 0]
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[0])[1]


def render_lower_third(
    text: str,
    subtitle: str,
    output: Path,
    width: int,
    height: int,
    accent: str = "#2F6F7E",
) -> Path:
    """Draw a lower-third strap to a transparent PNG.

    Pillow rather than ffmpeg's drawtext: real font metrics, so the plate is
    sized to the text instead of the text being sized to a guessed plate.
    """
    from PIL import Image, ImageDraw, ImageFont

    plate_w = int(width * 0.52)
    plate_h = int(height * 0.13)
    image = Image.new("RGBA", (plate_w, plate_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    title_size = max(20, plate_h // 3)
    sub_size = max(16, plate_h // 5)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", title_size)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", sub_size)
    except OSError:  # fall back rather than fail a render over a missing font
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    draw.rectangle([0, 0, plate_w, plate_h], fill=(16, 18, 24, 225))
    draw.rectangle([0, 0, max(6, plate_w // 120), plate_h], fill=accent)

    pad = plate_h // 6
    draw.text((pad * 2, pad), text[:64], font=title_font, fill=(255, 255, 255, 255))
    if subtitle:
        draw.text(
            (pad * 2, pad + title_size + pad // 2),
            subtitle[:80],
            font=sub_font,
            fill=(190, 196, 210, 255),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def plan_overlays(script: Script, config: Config, output_dir: Path) -> list[Overlay]:
    """Turn each block's visual cue into a timed overlay.

    Straps are held for at most 6 seconds — long enough to read twice, short
    enough that they do not become furniture — and never past the end of the
    block that owns them.
    """
    overlays: list[Overlay] = []
    video = config.video
    broll_library = config.path("assets/broll")

    for block in script.speech_blocks():
        visual = block.visual or {}
        kind = visual.get("type")

        if kind in {"lower_third", "title_card"} and visual.get("text"):
            path = output_dir / f"lt_{block.id}.png"
            render_lower_third(
                visual["text"],
                visual.get("subtitle", ""),
                path,
                video.width,
                video.height,
                accent=video.accent,
            )
            hold = min(6.0, max(2.5, block.duration))
            overlays.append(
                Overlay(
                    path=str(path),
                    start=round(block.start + 0.4, 3),
                    end=round(min(block.start + 0.4 + hold, block.end), 3),
                    x=str(int(video.width * 0.06)),
                    y=str(int(video.height * 0.74)),
                )
            )

        elif kind == "broll":
            asset = find_broll(visual.get("query", ""), broll_library)
            if asset and asset.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                overlays.append(
                    Overlay(
                        path=str(asset),
                        start=round(block.start + 0.5, 3),
                        end=round(min(block.start + 0.5 + 5.0, block.end), 3),
                        x=str(int(video.width * 0.06)),
                        y=str(int(video.height * 0.10)),
                    )
                )

    return [o for o in overlays if o.end > o.start + 0.2]


def mix_music(
    voice: Path, music: Path, output: Path, gain_db: float, duck_db: float = -12.0
) -> Path:
    """Lay a music bed under the voice with sidechain ducking.

    Sidechaining rather than a fixed low level: the bed sits up in the pauses
    and gets out of the way under speech, which is what makes a bed feel
    intentional instead of like a mistake.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y",
        "-i", str(voice),
        "-stream_loop", "-1", "-i", str(music),
        "-filter_complex",
        f"[1:a]volume={gain_db}dB[bed];"
        f"[bed][0:a]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400:"
        f"makeup=1[ducked];"
        f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]",
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        str(output),
    ])
    return output


def concat_with_bumpers(
    main: Path, output: Path, config: Config, intro: Path | None, outro: Path | None
) -> Path:
    """Join intro / main / outro, re-encoding so mismatched sources still join.

    Stream-copy concat would be faster but demands identical codec parameters
    across every part, and a bumper exported from a video editor almost never
    matches an NVENC render exactly.
    """
    parts = [p for p in (intro, main, outro) if p and Path(p).exists()]
    if len(parts) == 1:
        return main

    command: list[str] = ["ffmpeg", "-y"]
    for part in parts:
        command += ["-i", str(part)]

    video = config.video
    graph = []
    labels = []
    for index in range(len(parts)):
        graph.append(
            f"[{index}:v]scale={video.width}:{video.height}:force_original_aspect_ratio=decrease,"
            f"pad={video.width}:{video.height}:-1:-1:color=black,setsar=1,fps={video.fps}[v{index}]"
        )
        graph.append(f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{index}]")
        labels.append(f"[v{index}][a{index}]")
    graph.append(f"{''.join(labels)}concat=n={len(parts)}:v=1:a=1[vout][aout]")

    command += [
        "-filter_complex", ";".join(graph),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", video.encoder, "-preset", video.preset, "-rc", "vbr",
        "-cq", str(video.cq), "-b:v", "0", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]
    run(command)
    return output


def assemble(
    config: Config,
    script: Script,
    episode_dir: Path,
    avatar_path: str,
    audio_path: str,
    ass_path: str | None,
) -> str:
    """Build the finished episode video. Returns its path."""
    video_dir = episode_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    audio = Path(audio_path)
    music = config.path(config.audio.music_bed) if config.audio.music_bed else None
    if music and music.exists():
        print("  mixing music bed")
        audio = mix_music(
            audio, music, video_dir / "mixed.wav", config.audio.music_gain_db
        )

    print("  planning overlays")
    overlays = plan_overlays(script, config, video_dir / "overlays")
    print(f"  {len(overlays)} overlay(s)")

    background = config.path(config.video.background)
    if not background.exists():
        raise FileNotFoundError(
            f"background not found at {background}. Any 1920x1080 PNG works to start."
        )

    body = video_dir / "body.mp4"
    print(f"  encoding with {config.video.encoder} on GPU{config.gpu.encode}")
    run(
        build_encode_command(background, avatar_path, audio, overlays, body, config, ass_path),
        env_overlay=config.gpu.env_for("encode"),
        log_path=episode_dir / "logs" / "assemble.log",
        timeout=14400,
    )

    intro = config.path(config.video.intro) if config.video.intro else None
    outro = config.path(config.video.outro) if config.video.outro else None
    final = video_dir / "episode.mp4"
    concat_with_bumpers(
        body,
        final,
        config,
        intro if intro and intro.exists() else None,
        outro if outro and outro.exists() else None,
    )
    if final != body and not final.exists():
        raise RuntimeError("assembly produced no output")
    return str(final if final.exists() else body)
