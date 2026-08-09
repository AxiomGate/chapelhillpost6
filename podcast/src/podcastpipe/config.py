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
    # Per-client branding. Neutral defaults; every client overrides these.
    accent: str = "#2F6F7E"
    caption_font: str = "DejaVu Sans"
    caption_highlight: str = "3BE8FF"


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
    site_base_url: str = "https://example.com"
    media_base_url: str = "https://media.example.com/episodes"
    archive_dir: str = ""


@dataclass
class ShowConfig:
    name: str = "Example Daily"
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
    client: str = ""
    client_dir: Path | None = None

    def path(self, value: str) -> Path:
        """Resolve a config path, preferring the active client's directory.

        Client assets shadow the base ones: ``assets/voice/reference.wav``
        resolves to ``clients/<name>/assets/voice/reference.wav`` when that file
        exists, and falls back to the repo root otherwise. That fallback is what
        lets shared furniture (a default background, a stock bumper) live in one
        place while a client's voice and likeness never leak across tenants.
        """
        expanded = Path(value).expanduser()
        if expanded.is_absolute():
            return expanded
        if self.client_dir is not None:
            candidate = self.client_dir / expanded
            if candidate.exists():
                return candidate
        return self.root / expanded

    def client_path(self, value: str) -> Path:
        """Resolve strictly inside the client directory, whether or not it
        exists. Use when writing new client-owned files."""
        base = self.client_dir if self.client_dir is not None else self.root
        expanded = Path(value).expanduser()
        return expanded if expanded.is_absolute() else (base / expanded)

    def sources_path(self) -> Path:
        """Feed list for the active client, falling back to the base template.

        Each client researches its own subject, so this almost always resolves
        into the client directory; the base file exists only as a starting
        template.
        """
        if self.client_dir is not None:
            candidate = self.client_dir / "sources.yaml"
            if candidate.exists():
                return candidate
        return self.root / "config" / "sources.yaml"

    def label(self) -> str:
        """Human label for logs and the review UI."""
        return f"{self.show.name}" + (f" [{self.client}]" if self.client else "")

    def episode_dir(self, episode_id: str) -> Path:
        return self.work_dir / episode_id


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base``, recursing into nested mappings.

    Lists replace rather than concatenate. A client redefining ``segments``
    means "this is my lineup", not "append mine to the default one" — appending
    would silently give them both.
    """
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def list_clients(root: Path) -> list[str]:
    """Client profile names, i.e. directories under ``clients/`` holding a
    show.yaml."""
    clients_dir = root / "clients"
    if not clients_dir.is_dir():
        return []
    return sorted(
        p.name for p in clients_dir.iterdir()
        if p.is_dir() and (p / "show.yaml").exists()
    )


def _section(data: dict[str, Any], key: str, cls: type) -> Any:
    raw = data.get(key) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"section {key!r} must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in {key!r}: {', '.join(sorted(unknown))}")
    return cls(**raw)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: str | Path | None = None, client: str | None = None) -> Config:
    """Load the base config, then deep-merge the active client's profile.

    The client comes from the ``client`` argument, else ``PODCASTPIPE_CLIENT``.
    With neither, the base config is used alone — useful for smoke tests, but it
    carries no branding by design, so real runs should always name a client.

    Environment expansion happens *after* the merge so a client can set
    placeholders the base file references.
    """
    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = root / "config" / "show.yaml"
    path = Path(path).resolve()
    root = path.parents[1]

    data = _read_yaml(path)

    client = client or os.environ.get("PODCASTPIPE_CLIENT", "") or ""
    client_dir: Path | None = None
    if client:
        client_dir = root / "clients" / client
        client_config = client_dir / "show.yaml"
        if not client_config.exists():
            available = list_clients(root)
            raise ConfigError(
                f"no client profile at {client_config}. "
                + (f"Available: {', '.join(available)}" if available
                   else "Create one by copying clients/example/.")
            )
        data = deep_merge(data, _read_yaml(client_config))

    data = expand_env(data)

    # Each client gets its own work and output trees, so two shows running the
    # same day cannot collide on an episode id.
    work_dir = root / (data.get("work_dir") or "work")
    output_dir = root / (data.get("output_dir") or "output")
    if client:
        work_dir = work_dir / client
        output_dir = output_dir / client

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
        work_dir=work_dir,
        output_dir=output_dir,
        raw=data,
        client=client,
        client_dir=client_dir,
    )
