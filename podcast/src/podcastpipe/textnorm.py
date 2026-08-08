"""Text normalization for TTS, plus sentence-aware chunking.

Neural TTS reads what you give it fairly literally. "$1.2M" comes out as
"dollar one point two em", "NCDOT" as a mangled word, and "2026" as "two
thousand and twenty six" when a broadcaster says "twenty twenty-six". Fixing
this in text is far cheaper than fixing it in audio.

The lexicon is extendable from ``config/show.yaml`` under ``pronunciations``,
which is where local names go — the ones no general rule will ever get right.
"""

from __future__ import annotations

import re
from typing import Iterable

ABBREVIATIONS: dict[str, str] = {
    r"\bMr\.": "Mister",
    r"\bMrs\.": "Missus",
    r"\bMs\.": "Miz",
    r"\bDr\.": "Doctor",
    r"\bProf\.": "Professor",
    r"\bSgt\.": "Sergeant",
    r"\bCol\.": "Colonel",
    r"\bCapt\.": "Captain",
    r"\bLt\.": "Lieutenant",
    r"\bGen\.": "General",
    r"\bGov\.": "Governor",
    r"\bSen\.": "Senator",
    r"\bRep\.": "Representative",
    r"\bSt\. ": "Saint ",
    r"\bAve\.": "Avenue",
    r"\bBlvd\.": "Boulevard",
    r"\bRd\.": "Road",
    r"\bMt\.": "Mount",
    r"\bvs\.": "versus",
    r"\betc\.": "et cetera",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\bapprox\.": "approximately",
    r"\bU\.S\.A\.": "U S A",
    r"\bU\.S\.": "U S",
    r"\bN\.C\.": "North Carolina",
    r"\bD\.C\.": "D C",
}

UNITS = {
    "K": "thousand",
    "M": "million",
    "B": "billion",
    "T": "trillion",
}

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]
_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def number_to_words(value: int) -> str:
    """Spell out an integer below one million. Above that, digits are kept —
    a nine-digit number read aloud is a mistake in the script, not here."""
    if value < 0:
        return "negative " + number_to_words(-value)
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        out = f"{_ONES[hundreds]} hundred"
        return f"{out} {number_to_words(rest)}" if rest else out
    if value < 1_000_000:
        thousands, rest = divmod(value, 1000)
        out = f"{number_to_words(thousands)} thousand"
        return f"{out} {number_to_words(rest)}" if rest else out
    return str(value)


def ordinal_to_words(value: int) -> str:
    words = number_to_words(value)
    head, _, tail = words.rpartition("-")
    last = tail or words
    if last in _ORDINALS:
        converted = _ORDINALS[last]
    elif last.endswith("y"):
        converted = last[:-1] + "ieth"
    else:
        converted = last + "th"
    return f"{head}-{converted}" if head else converted


def year_to_words(year: int) -> str:
    """Broadcast convention: 1998 is "nineteen ninety-eight", 2005 is "two
    thousand five", 2026 is "twenty twenty-six"."""
    if not 1000 <= year <= 2999:
        return number_to_words(year)
    high, low = divmod(year, 100)
    # 2000-2009 first: "two thousand", not "twenty hundred".
    if 2000 <= year <= 2009:
        return "two thousand" if low == 0 else f"two thousand {number_to_words(low)}"
    if low == 0:
        return f"{number_to_words(high)} hundred"
    if low < 10:
        return f"{number_to_words(high)} oh {number_to_words(low)}"
    return f"{number_to_words(high)} {number_to_words(low)}"


def _money(match: re.Match) -> str:
    amount = match.group("amount").replace(",", "")
    unit = (match.group("unit") or "").upper()
    spoken_unit = f" {UNITS[unit]}" if unit in UNITS else ""

    if "." in amount:
        whole, _, frac = amount.partition(".")
        whole_words = number_to_words(int(whole)) if whole else "zero"
        if spoken_unit:  # $1.2 million -> one point two million dollars
            digits = " ".join(_ONES[int(d)] for d in frac)
            return f"{whole_words} point {digits}{spoken_unit} dollars"
        cents = int(frac.ljust(2, "0")[:2])
        if cents:
            return f"{whole_words} dollars and {number_to_words(cents)} cents"
        return f"{whole_words} dollars"

    value = int(amount)
    if spoken_unit:
        return f"{number_to_words(value)}{spoken_unit} dollars"
    return f"{number_to_words(value)} dollar" + ("" if value == 1 else "s")


_MONEY_RE = re.compile(r"\$(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>[KMBT])?\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*%")
_ORDINAL_RE = re.compile(r"\b(\d{1,4})(?:st|nd|rd|th)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap])\.?m\.?", re.IGNORECASE)
_INT_RE = re.compile(r"\b\d[\d,]*\b")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")


def _percent(match: re.Match) -> str:
    amount = match.group("amount").replace(",", "")
    if "." in amount:
        whole, _, frac = amount.partition(".")
        digits = " ".join(_ONES[int(d)] for d in frac)
        return f"{number_to_words(int(whole))} point {digits} percent"
    return f"{number_to_words(int(amount))} percent"


def _time(match: re.Match) -> str:
    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3).lower()
    suffix = "A M" if meridiem == "a" else "P M"
    if minute == 0:
        return f"{number_to_words(hour)} {suffix}"
    if minute < 10:
        return f"{number_to_words(hour)} oh {number_to_words(minute)} {suffix}"
    return f"{number_to_words(hour)} {number_to_words(minute)} {suffix}"


def normalize_for_tts(
    text: str,
    pronunciations: dict[str, str] | None = None,
    spell_acronyms: bool = True,
    keep_acronyms: Iterable[str] = ("NASA", "NATO", "AIDS", "OSHA", "FEMA", "SWAT", "VA"),
) -> str:
    """Rewrite text into something a TTS model reads correctly out loud.

    Order matters: custom pronunciations first (so a local name is protected
    from later rules), then money and percentages before bare integers, then
    years before generic numbers.
    """
    if not text:
        return ""

    for phrase, replacement in (pronunciations or {}).items():
        text = re.sub(rf"\b{re.escape(phrase)}\b", replacement, text)

    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    text = _MONEY_RE.sub(_money, text)
    text = _PERCENT_RE.sub(_percent, text)
    text = _TIME_RE.sub(_time, text)
    text = _ORDINAL_RE.sub(lambda m: ordinal_to_words(int(m.group(1))), text)
    text = _YEAR_RE.sub(lambda m: year_to_words(int(m.group(1))), text)
    text = _INT_RE.sub(lambda m: number_to_words(int(m.group(0).replace(",", ""))), text)

    if spell_acronyms:
        protected = {a.upper() for a in keep_acronyms}
        text = _ACRONYM_RE.sub(
            lambda m: m.group(1) if m.group(1) in protected else " ".join(m.group(1)),
            text,
        )

    text = text.replace("&", " and ").replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“])")


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, protecting common abbreviations first so
    "Mister Smith went" does not break after "Mr."."""
    if not text.strip():
        return []
    guarded = text
    for abbr in ("Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Jr.", "Sr.", "vs.", "U.S."):
        guarded = guarded.replace(abbr, abbr.replace(".", "\x00"))
    parts = _SENTENCE_RE.split(guarded)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Group sentences into TTS-sized chunks without ever splitting mid-sentence.

    Chunking keeps timbre stable across a long read and makes a bad take
    re-rollable in isolation. A single sentence longer than ``max_chars`` is
    emitted whole rather than cut — clipping a sentence in half produces an
    audible swallow, which is worse than a slightly long chunk.
    """
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks
