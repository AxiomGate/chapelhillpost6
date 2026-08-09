"""Stage 5 — captions, aligned to the audio that actually exists.

Transcription runs against the *synthesized* voice track rather than the script.
TTS occasionally elides a word or reflows a clause, and captions that follow the
script instead of the audio drift in exactly the places a viewer notices. The
script is still used, as an initial-prompt hint, which sharply improves how the
model spells local names.

Produces an SRT sidecar for YouTube and an ASS file for burned-in captions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """SRT-style ``HH:MM:SS,mmm``. Negative input clamps to zero rather than
    producing a timestamp ffmpeg will reject."""
    seconds = max(0.0, seconds)
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def format_ass_timestamp(seconds: float) -> str:
    """ASS uses ``H:MM:SS.cc`` with centiseconds and a single-digit hour."""
    seconds = max(0.0, seconds)
    centiseconds = int(round(seconds * 100))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def group_words(
    words: list[Word],
    max_chars: int = 42,
    max_duration: float = 5.0,
    max_gap: float = 0.7,
) -> list[Cue]:
    """Group word timings into readable caption cues.

    A cue breaks on any of: line length, elapsed time, a pause long enough to be
    a natural beat, or sentence-ending punctuation. Two lines of ~42 characters
    is the long-standing broadcast readability limit and it is what YouTube's
    own player is laid out for.
    """
    cues: list[Cue] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        cues.append(
            Cue(
                start=current[0].start,
                end=current[-1].end,
                text=" ".join(w.text for w in current).strip(),
                words=list(current),
            )
        )
        current.clear()

    for word in words:
        if current:
            length = sum(len(w.text) + 1 for w in current) + len(word.text)
            elapsed = word.end - current[0].start
            gap = word.start - current[-1].end
            if length > max_chars or elapsed > max_duration or gap > max_gap:
                flush()
        current.append(word)
        if word.text.rstrip().endswith((".", "!", "?")):
            flush()

    flush()
    return cues


def to_srt(cues: list[Cue]) -> str:
    lines: list[str] = []
    for index, cue in enumerate(cues, 1):
        lines.append(str(index))
        lines.append(
            f"{format_timestamp(cue.start)} --> {format_timestamp(max(cue.end, cue.start + 0.1))}"
        )
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines)


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},&H00FFFFFF,&H00{highlight},&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,80,80,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def to_ass(
    cues: list[Cue],
    width: int = 1920,
    height: int = 1080,
    font: str = "Inter SemiBold",
    size: int = 54,
    highlight: str = "3BE8FF",
    karaoke: bool = True,
) -> str:
    """Burned-in caption file.

    With ``karaoke`` on, each word is wrapped in an ASS ``\\k`` tag so the
    active word highlights as it is spoken. That single effect is the largest
    measurable driver of retention on talk-format video, and it costs nothing
    at render time because libass does the work.
    """
    body = ASS_HEADER.format(
        width=width,
        height=height,
        font=font,
        size=size,
        highlight=highlight,
        outline=max(2, size // 18),
        margin=max(60, height // 12),
    )

    lines: list[str] = []
    for cue in cues:
        end = max(cue.end, cue.start + 0.1)
        if karaoke and cue.words:
            parts = []
            for word in cue.words:
                centiseconds = max(1, int(round((word.end - word.start) * 100)))
                parts.append(f"{{\\k{centiseconds}}}{word.text}")
            text = "".join(parts)
        else:
            text = cue.text
        lines.append(
            f"Dialogue: 0,{format_ass_timestamp(cue.start)},{format_ass_timestamp(end)},"
            f"Caption,,0,0,0,,{text}"
        )
    return body + "\n".join(lines) + "\n"


def transcribe(
    audio_path: str | Path,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8_float16",
    initial_prompt: str = "",
    language: str = "en",
) -> list[Word]:
    """Word-level timings via faster-whisper.

    ``initial_prompt`` should be a slice of the script: it biases spelling
    toward the proper nouns this show actually uses, which is what makes
    "Springfield" not come back as "springfield" or worse.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        initial_prompt=initial_prompt[:900] or None,
    )

    words: list[Word] = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                words.append(Word(text=text, start=float(word.start), end=float(word.end)))
    return words


def generate(
    audio_path: str | Path,
    output_dir: str | Path,
    script_text: str = "",
    width: int = 1920,
    height: int = 1080,
    device: str = "cuda",
    font: str = "DejaVu Sans",
    highlight: str = "3BE8FF",
) -> dict[str, str]:
    """Transcribe and write both caption formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    words = transcribe(audio_path, initial_prompt=script_text, device=device)
    cues = group_words(words)

    srt_path = output_dir / "captions.srt"
    ass_path = output_dir / "captions.ass"
    srt_path.write_text(to_srt(cues), encoding="utf-8")
    ass_path.write_text(
        to_ass(cues, width, height, font=font, highlight=highlight), encoding="utf-8"
    )

    print(f"  {len(words)} words in {len(cues)} cues")
    return {"srt": str(srt_path), "ass": str(ass_path)}
