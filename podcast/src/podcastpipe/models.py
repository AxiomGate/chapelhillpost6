"""Core data types that move between pipeline stages.

These are plain dataclasses with JSON round-tripping so every stage boundary is
inspectable on disk. If a stage misbehaves you can read its output, fix it in a
text editor, and resume from there.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

# Measured pace of the synthesized voice, in words per minute of finished audio.
#
# Not a textbook figure. 1961 words of script produced an 11.1 minute voice
# track on 2026-08-26, which is 177 wpm -- and that number includes the 0.35s
# gap between blocks, so it is the rate that actually converts a word count into
# runtime. The textbook 150 was here first and is why a script written to fill
# 15 minutes came out at 11: every budget was about 15% short.
#
# Re-measure if the voice changes. `podcastpipe tts` prints the track length and
# `status` has the word count; divide one by the other. A reference recording at
# a different speaking pace moves this number.
#
# Used in two places that must agree: the word budgets handed to the model when
# writing a script, and the runtime estimate reported after one is written. If
# they drift apart, a correct script gets flagged as the wrong length.
WORDS_PER_MINUTE = 175

EpisodeStatus = str  # new | researched | scripted | approved | rendered | published


def slugify(text: str, max_length: int = 60) -> str:
    """URL/filename-safe slug. Collapses punctuation, never returns empty."""
    lowered = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "-", lowered).strip("-")
    slug = slug[:max_length].rstrip("-")
    return slug or "untitled"


@dataclass
class Story:
    """One candidate news item gathered during research."""

    title: str
    url: str
    source: str
    summary: str = ""
    published_at: str = ""
    score: float = 0.0
    cluster: int = -1

    @property
    def domain(self) -> str:
        match = re.search(r"https?://(?:www\.)?([^/]+)", self.url)
        return match.group(1).lower() if match else ""


@dataclass
class Claim:
    """A factual assertion in the brief, tied to the sources that support it."""

    text: str
    source_urls: list[str] = field(default_factory=list)


@dataclass
class Brief:
    """Research output: the clustered, verified material a script is built on."""

    episode_id: str
    generated_at: str
    headline: str = ""
    stories: list[Story] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    notes: str = ""

    def source_urls(self) -> list[str]:
        seen: dict[str, None] = {}
        for story in self.stories:
            seen.setdefault(story.url, None)
        return list(seen)


@dataclass
class Block:
    """A contiguous run of speech, the unit of TTS and of visual direction."""

    id: str
    text: str
    kind: str = "speech"  # speech | music | silence
    visual: dict[str, Any] = field(default_factory=dict)
    source_urls: list[str] = field(default_factory=list)
    # Filled in by later stages.
    audio_path: str = ""
    duration: float = 0.0
    start: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Segment:
    id: str
    name: str
    blocks: list[Block] = field(default_factory=list)

    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks if b.kind == "speech")


@dataclass
class Script:
    episode_id: str
    title: str
    segments: list[Segment] = field(default_factory=list)
    description: str = ""
    tags: list[str] = field(default_factory=list)
    approved: bool = False

    def blocks(self) -> list[Block]:
        return [b for segment in self.segments for b in segment.blocks]

    def speech_blocks(self) -> list[Block]:
        return [b for b in self.blocks() if b.kind == "speech" and b.text.strip()]

    def word_count(self) -> int:
        return sum(s.word_count() for s in self.segments)

    def estimated_minutes(self, words_per_minute: int = WORDS_PER_MINUTE) -> float:
        """Rough runtime estimate. The real number comes from the TTS output."""
        return self.word_count() / max(words_per_minute, 1)

    def full_text(self) -> str:
        return "\n\n".join(b.text.strip() for b in self.speech_blocks())

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Script":
        segments = [
            Segment(
                id=s["id"],
                name=s.get("name", s["id"]),
                blocks=[Block(**b) for b in s.get("blocks", [])],
            )
            for s in data.get("segments", [])
        ]
        return cls(
            episode_id=data["episode_id"],
            title=data.get("title", ""),
            segments=segments,
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            approved=bool(data.get("approved", False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Script":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Episode:
    id: str
    date: str
    title: str = ""
    status: EpisodeStatus = "new"
    number: int = 0
    created_at: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def make_id(when: date | datetime | None = None) -> str:
        when = when or datetime.now()
        return when.strftime("%Y-%m-%d")

    def slug(self) -> str:
        return f"{self.date}-{slugify(self.title)}" if self.title else self.date


def assign_timeline(blocks: Iterable[Block], gap: float = 0.0) -> float:
    """Lay blocks end to end, setting ``start`` on each. Returns total duration.

    ``gap`` inserts a fixed pause between blocks, which is how the show gets
    breathing room between stories without baking silence into the TTS.
    """
    cursor = 0.0
    total = 0.0
    for index, block in enumerate(blocks):
        if index > 0:
            cursor += gap
        block.start = round(cursor, 3)
        cursor += block.duration
        total = cursor
    return round(total, 3)
