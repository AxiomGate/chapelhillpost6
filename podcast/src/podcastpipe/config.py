"""Configuration loading for the podcast pipeline.

Everything the pipeline does is driven by ``config/show.yaml``. Environment
variables referenced as ``${VAR}`` inside string values are expanded at load
time so secrets never live in the YAML.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ConfigError(ValueError):
    """Raised when show.yaml is missing something the pipeline needs."""


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references in strings.

    An unset variable expands to an empty string rather than raising, so a
    config can be loaded for inspection on a machine that has no secrets. The
    stages that actually need a credential check for emptiness themselves and
    give a better error than a KeyError from here would.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


@dataclass
class GpuConfig:
    avatar: int = 0
    tts: int = 1
    llm: int = 1
    whisper: int = 2
    encode: int = 2

    def env_for(self, stage: str) -> dict[str, str]:
        """CUDA_VISIBLE_DEVICES for a stage, as a subprocess env overlay.

        Remapping to a single visible device means the child process always
        sees its card as ``cuda:0``, so tool defaults land on the right GPU
        even when the tool has no device flag of its own.
        """
        index = getattr(self, stage, None)
        if index is None:
            raise ConfigError(f"no GPU assignment for stage {stage!r}")
        return {"CUDA_VISIBLE_DEVICES": str(index)}


@dataclass
class LlmConfig:
    provider: str = "anthropic"  # anthropic | local
    model: str = "claude-sonnet-5"
    local_base_url: str = "http://localhost:8000/v1"
    local_model: str = "Qwen/Qwen3-32B-AWQ"
    max_tokens: int = 8000
    temperature: float = 0.3


@dataclass
class TtsConfig:
    engine: str = "chatterbox"  # chatterbox | vibevoice
    venv: str = "~/envs/chatterbox"
    reference_audio: str = "assets/voice/reference.wav"
    max_chars_per_chunk: int = 300
    crossfade_ms: int = 40
    seed: int = 1234
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    sample_rate: int = 24000


@dataclass
class AvatarConfig:
    renderer: str = "musetalk"  # musetalk | latentsync | none
    venv: str = "~/envs/musetalk"
    base_loop: str = "assets/avatar/base_loop.mp4"
    fps: int = 25
    bbox_shift: int = 0
    chunk_seconds: int = 120


@dataclass
class VideoConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 25
    encoder: str = "h264_nvenc"
    preset: str = "p5"
    cq: int = 21
    background: str = "assets/brand/background.png"
    intro: str = "assets/brand/intro.mp4"
    outro: str = "assets/brand/outro.mp4"
    burn_captions: bool = True


@dataclass
class AudioConfig:
    # Podcast platforms want -16 LUFS mono / -19 stereo (AES/Apple guidance);
    # YouTube normalizes to about -14. Separate masters, one render.
    podcast_lufs: float = -16.0
    youtube_lufs: float = -14.0
    true_peak: float = -1.5
    loudness_range: float = 7.0
    music_bed: str = ""
    music_gain_db: float = -26.0


@dataclass
class PublishConfig:
    youtube_enabled: bool = False
    youtube_client_secrets: str = "config/youtube_client_secret.json"
    youtube_token: str = "config/youtube_token.json"
    youtube_privacy: str = "private"
    youtube_category_id: str = "25"  # News & Politics
    rss_enabled: bool = True
    rss_output: str = "output/feed.xml"
    site_base_url: str = "https://alpost6.org"
    media_base_url: str = "https://media.alpost6.org/episodes"
    archive_dir: str = ""


@dataclass
class ShowConfig:
    name: str = "The Post 6 Daily"
    tagline: str = ""
    host: str = "Host"
    author_email: str = ""
    language: str = "en-us"
    timezone: str = "America/New_York"
    target_minutes: int = 20
    segments: list[dict[str, Any]] = field(default_factory=list)
    style_guide: str = ""


@dataclass
class Config:
    root: Path
    show: ShowConfig
    gpu: GpuConfig
    llm: LlmConfig
    tts: TtsConfig
    avatar: AvatarConfig
    video: VideoConfig
    audio: AudioConfig
    publish: PublishConfig
    work_dir: Path
    output_dir: Path
    raw: dict[str, Any] = field(default_factory=dict)

    def path(self, value: str) -> Path:
        """Resolve a config path against the project root unless absolute."""
        expanded = Path(value).expanduser()
        return expanded if expanded.is_absolute() else (self.root / expanded)

    def episode_dir(self, episode_id: str) -> Path:
        return self.work_dir / episode_id


def _section(data: dict[str, Any], key: str, cls: type) -> Any:
    raw = data.get(key) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"section {key!r} must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in {key!r}: {', '.join(sorted(unknown))}")
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    """Load show.yaml into a Config. Defaults to ``config/show.yaml``."""
    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = root / "config" / "show.yaml"
    path = Path(path).resolve()
    root = path.parents[1]

    with path.open("r", encoding="utf-8") as handle:
        data = expand_env(yaml.safe_load(handle) or {})

    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")

    return Config(
        root=root,
        show=_section(data, "show", ShowConfig),
        gpu=_section(data, "gpu", GpuConfig),
        llm=_section(data, "llm", LlmConfig),
        tts=_section(data, "tts", TtsConfig),
        avatar=_section(data, "avatar", AvatarConfig),
        video=_section(data, "video", VideoConfig),
        audio=_section(data, "audio", AudioConfig),
        publish=_section(data, "publish", PublishConfig),
        work_dir=root / (data.get("work_dir") or "work"),
        output_dir=root / (data.get("output_dir") or "output"),
        raw=data,
    )
