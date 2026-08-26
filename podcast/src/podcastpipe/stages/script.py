"""Stage 2 — turn the brief into a broadcast script.

Output is a ``Script``: segments of blocks, where a block is one contiguous run
of speech and the unit that TTS, the avatar renderer and the visual director all
key off. Blocks carry their supporting source URLs so the review UI can show,
line by line, what a sentence is standing on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import Config
from ..llm import LlmClient
from ..models import WORDS_PER_MINUTE, Block, Brief, Script, Segment

SYSTEM_PROMPT = """You write scripts for a daily local news podcast that is also
published as video. You are writing words a human will speak aloud, not prose to
be read.

Craft rules:
- Short sentences. One idea each. A sentence that needs a comma to survive
  probably wants to be two sentences.
- Spoken register: contractions, plain words, active voice. No headline-ese, no
  "in a stunning development", no throat-clearing.
- Attribute in the sentence, the way a broadcaster does: "The town council voted
  Tuesday, according to the meeting agenda."
- Numbers as a person would say them. "About twelve hundred" beats "1,187"
  unless the exact figure is the point.
- Never state a fact that is not in the brief. If the brief flags uncertainty,
  keep the uncertainty in the script.
- No stage directions, no speaker labels, no markdown, no emoji. Just the words
  to be spoken.

Length is a hard requirement, not a suggestion:
- Every segment carries a word budget. Hit it. A segment that lands under its
  budget is the single most common failure here, and it is not a stylistic
  choice -- it produces an episode that ends early with airtime left over.
- Write to the budget by going deeper, never by padding. Depth means: what
  actually changed, who it applies to, what the eligibility or deadline is, what
  a listener should do about it, what is still unresolved, and what the counter
  argument or the caveat is. All of that is in the source material and most of it
  does not survive a first pass.
- If the material for a segment feels thin, that usually means the obvious facts
  were stated and the consequences were not. Say what it means for someone it
  applies to.
- Do not announce the budget, the runtime, or the structure on air.

For each block, supply a visual cue the video pipeline can render:
- {"type": "lower_third", "text": "..."} for a name/topic strap
- {"type": "title_card", "text": "..."} for segment openers
- {"type": "broll", "query": "..."} to suggest imagery
- {} for a plain shot of the host

Return JSON only:
{
  "title": "episode title, under 70 characters, specific not generic",
  "description": "3-5 sentence show-notes paragraph",
  "tags": ["...", "..."],
  "segments": [
    {
      "id": "matches the requested segment id",
      "name": "display name",
      "blocks": [
        {"text": "spoken words", "visual": {...}, "source_urls": ["..."]}
      ]
    }
  ]
}"""


def build_prompt(config: Config, brief: Brief) -> str:
    segments = config.show.segments or [
        {"id": "cold_open", "name": "Cold Open", "target_seconds": 30},
        {"id": "main", "name": "Today's Stories", "target_seconds": 600},
        {"id": "outro", "name": "Outro", "target_seconds": 45},
    ]

    # Seconds are what the show is designed in; words are what the model can
    # actually aim at. Asking for "~240s" and hoping produced a script that ran
    # 8.5 minutes against a 15 minute target -- the model has no reliable way to
    # convert one to the other, so the conversion happens here.
    def words_for(seconds: float) -> int:
        return int(round(seconds / 60 * WORDS_PER_MINUTE / 10) * 10)

    segment_spec = "\n".join(
        f"- id: {s['id']} | {s.get('name', s['id'])} | "
        f"~{s.get('target_seconds', 120)}s = write about "
        f"{words_for(float(s.get('target_seconds', 120)))} words"
        + (f" | {s['instructions']}" if s.get("instructions") else "")
        for s in segments
    )

    total_seconds = sum(float(s.get("target_seconds", 120)) for s in segments)
    total_words = words_for(total_seconds)

    story_block = "\n\n".join(
        f"[{i}] {s.title}\n    url: {s.url}\n    source: {s.source}\n    summary: {s.summary}"
        for i, s in enumerate(brief.stories, 1)
    )

    claim_block = "\n".join(
        f"- {c.text}  [{', '.join(c.source_urls) or 'UNSOURCED — do not state as fact'}]"
        for c in brief.claims
    )

    return f"""Show: {config.show.name}
Host: {config.show.host}
Tagline: {config.show.tagline}
Episode date: {brief.episode_id}
Target runtime: {config.show.target_minutes} minutes.
Total script length: about {total_words} words across all segments. This is the
number to hit. Coming in short is the failure mode to avoid.

Style guide:
{config.show.style_guide or '(none specified)'}

Segments to write, in order (each with its own word budget):
{segment_spec}

Day's headline: {brief.headline}

Editorial notes from the research desk:
{brief.notes}

Verified claims:
{claim_block or '(none)'}

Source material:
{story_block}
"""


def generate_script(config: Config, brief: Brief) -> Script:
    client = LlmClient(config)
    data = client.complete_json(SYSTEM_PROMPT, build_prompt(config, brief))
    return script_from_payload(brief.episode_id, data)


def script_from_payload(episode_id: str, data: dict) -> Script:
    """Convert the model's JSON into a Script, assigning stable block ids.

    Block ids are ``<segment>-<n>`` and are the cache key for TTS and avatar
    renders, so they must stay stable across re-runs. Editing a block's text in
    the review UI keeps the id and invalidates only that block's audio.
    """
    segments: list[Segment] = []
    for raw_segment in data.get("segments", []):
        segment_id = raw_segment.get("id") or f"seg{len(segments) + 1}"
        blocks: list[Block] = []
        for index, raw_block in enumerate(raw_segment.get("blocks", []), 1):
            text = clean_spoken_text(raw_block.get("text", ""))
            if not text:
                continue
            blocks.append(
                Block(
                    id=f"{segment_id}-{index}",
                    text=text,
                    kind=raw_block.get("kind", "speech"),
                    visual=raw_block.get("visual") or {},
                    source_urls=list(raw_block.get("source_urls") or []),
                )
            )
        segments.append(
            Segment(id=segment_id, name=raw_segment.get("name", segment_id), blocks=blocks)
        )

    return Script(
        episode_id=episode_id,
        title=data.get("title", "").strip(),
        description=data.get("description", "").strip(),
        tags=[t for t in data.get("tags", []) if isinstance(t, str)],
        segments=segments,
    )


_MARKDOWN = re.compile(r"[*_`#]+")
_SPEAKER_LABEL = re.compile(r"^\s*(?:HOST|NARRATOR|ANCHOR)\s*:\s*", re.IGNORECASE)
_STAGE_DIRECTION = re.compile(r"[\[(](?:music|sfx|pause|beat|sound)[^\])]*[\])]", re.IGNORECASE)


def clean_spoken_text(text: str) -> str:
    """Strip artifacts that leak into generated scripts and would be read aloud.

    Models are trained on written text, so markdown emphasis, speaker labels and
    bracketed stage directions show up even when the prompt forbids them. They
    are silent on the page and embarrassing in a voice track.
    """
    text = _STAGE_DIRECTION.sub(" ", text or "")
    text = _SPEAKER_LABEL.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_script(script: Script, config: Config) -> list[str]:
    """Editorial checks worth surfacing before a human approves. Warnings, not
    errors — the host makes the call."""
    warnings: list[str] = []

    if not script.title:
        warnings.append("Episode has no title.")

    minutes = script.estimated_minutes()
    target = config.show.target_minutes
    if minutes < target * 0.6:
        warnings.append(
            f"Script runs about {minutes:.1f} min against a {target} min target — short."
        )
    elif minutes > target * 1.4:
        warnings.append(
            f"Script runs about {minutes:.1f} min against a {target} min target — long."
        )

    for block in script.speech_blocks():
        if len(block.text) > 1200:
            warnings.append(f"Block {block.id} is very long ({len(block.text)} chars).")
        if re.search(r"\b(?:reports say|sources say|it is said|some say)\b", block.text, re.I):
            warnings.append(f"Block {block.id} uses vague attribution.")
        if re.search(r"\bhttps?://", block.text):
            warnings.append(f"Block {block.id} contains a raw URL, which will be read aloud.")

    unsourced = [b.id for b in script.speech_blocks() if not b.source_urls]
    if len(unsourced) > len(script.speech_blocks()) * 0.5:
        warnings.append(
            f"{len(unsourced)} of {len(script.speech_blocks())} blocks cite no source."
        )

    return warnings


def save(script: Script, path: str | Path) -> Path:
    return script.save(path)


def load(path: str | Path) -> Script:
    return Script.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
