"""Stage 1 — gather the day's material and turn it into a sourced brief.

Feeds in ``config/sources.yaml`` are fetched, filtered to the last N hours,
deduplicated against both each other and the recent archive, scored for local
relevance, then handed to the LLM to be clustered and written up. Every claim in
the resulting brief carries the URLs that support it, and the review UI shows
those next to the sentence they justify.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..config import Config
from ..llm import LlmClient
from ..models import Brief, Claim, Story

_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "the", "to", "with", "after", "over", "new", "says",
}

SYSTEM_PROMPT = """You are the research desk for a daily local news podcast.

Your job is to turn a list of raw headlines and summaries into a tight,
factually careful brief that a host will read on air. You are writing for
broadcast, so accuracy matters more than color.

Hard rules:
- Never assert anything that is not supported by the supplied source material.
  If sources conflict, say they conflict and attribute each side.
- Every claim must list the URLs (from the supplied set only) that support it.
  Never invent, guess, or shorten a URL.
- Do not add statistics, dates, dollar figures, titles, or quotes that are not
  in the source material.
- Prefer specific attribution ("according to the Springfield Town Council
  agenda") over vague attribution ("reports say").
- If a story is thin or unverifiable, mark it low priority rather than padding.

Return JSON only, matching this shape:
{
  "headline": "one line summarizing the day",
  "notes": "editorial guidance for the host: what to lead with and why, what to be careful about",
  "clusters": [
    {
      "title": "short story title",
      "priority": 1,
      "summary": "2-4 sentences of what is known",
      "source_urls": ["..."],
      "claims": [
        {"text": "a single factual assertion", "source_urls": ["..."]}
      ]
    }
  ]
}
Order clusters by priority, 1 being the lead story."""


def normalize_url(url: str) -> str:
    """Strip tracking parameters and trailing noise so the same article from
    two feeds deduplicates to one entry."""
    url = (url or "").strip()
    if not url:
        return ""
    url = re.sub(r"[?&](utm_[^=]+|fbclid|gclid|mc_cid|mc_eid|ref|source)=[^&#]*", "", url)
    url = re.sub(r"[?&]+$", "", url)
    url = url.split("#")[0]
    return url.rstrip("/")


def title_key(title: str) -> str:
    """Comparable form of a headline: lowercased, punctuation and stopwords out."""
    words = re.sub(r"[^\w\s]", " ", (title or "").lower()).split()
    return " ".join(w for w in words if w not in _STOPWORDS)


def titles_match(a: str, b: str, threshold: float = 0.82) -> bool:
    """Whether two headlines are the same story reworded by another outlet."""
    key_a, key_b = title_key(a), title_key(b)
    if not key_a or not key_b:
        return False
    if key_a == key_b:
        return True
    return SequenceMatcher(None, key_a, key_b).ratio() >= threshold


def dedupe_stories(stories: Iterable[Story], seen_urls: set[str] | None = None) -> list[Story]:
    """Collapse duplicate stories, keeping the highest-scoring version.

    Two passes: exact URL match (after normalization), then fuzzy headline
    match. ``seen_urls`` drops anything already covered in a recent episode.
    """
    seen_urls = {normalize_url(u) for u in (seen_urls or set())}
    kept: list[Story] = []
    kept_urls: set[str] = set()

    ordered = sorted(stories, key=lambda s: s.score, reverse=True)
    for story in ordered:
        url = normalize_url(story.url)
        if not url or url in kept_urls or url in seen_urls:
            continue
        if any(titles_match(story.title, other.title) for other in kept):
            continue
        story.url = url
        kept.append(story)
        kept_urls.add(url)
    return kept


def score_story(story: Story, keywords: dict[str, float], source_weight: float = 1.0) -> float:
    """Relevance score: keyword hits in the headline and summary, weighted by
    how much we trust the feed it came from.

    Headline hits count double — a town name in the headline means the story is
    about that town, whereas one in the body may be an aside.
    """
    title = (story.title or "").lower()
    body = (story.summary or "").lower()
    score = 0.0
    for keyword, weight in keywords.items():
        needle = keyword.lower()
        if needle in title:
            score += weight * 2.0
        elif needle in body:
            score += weight
    return round(score * source_weight, 3)


def parse_entry_date(value: Any) -> str:
    """Best-effort ISO timestamp from whatever a feed hands us."""
    if not value:
        return ""
    if isinstance(value, str):
        return value
    try:  # feedparser struct_time
        return datetime(*value[:6], tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return ""


def is_recent(published_at: str, hours: int) -> bool:
    """Whether a timestamp falls inside the lookback window. Unparseable or
    missing dates are kept — plenty of small-town feeds omit them, and dropping
    those would silently lose local coverage."""
    if not published_at:
        return True
    for parser in (
        lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")),
        lambda v: datetime.strptime(v, "%a, %d %b %Y %H:%M:%S %z"),
        lambda v: datetime.strptime(v, "%a, %d %b %Y %H:%M:%S %Z"),
    ):
        try:
            parsed = parser(published_at)
        except (ValueError, TypeError):
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed >= datetime.now(timezone.utc) - timedelta(hours=hours)
    return True


def load_sources(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def fetch_feeds(sources: dict[str, Any], lookback_hours: int = 36) -> list[Story]:
    """Pull every configured RSS/Atom feed. A broken feed is logged and skipped,
    never fatal — one dead local paper must not stop the show."""
    import feedparser  # imported here so pure logic stays testable without it

    keywords: dict[str, float] = sources.get("keywords", {}) or {}
    stories: list[Story] = []

    for feed in sources.get("feeds", []) or []:
        url = feed.get("url")
        if not url:
            continue
        name = feed.get("name", url)
        weight = float(feed.get("weight", 1.0))
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:  # network, XML, encoding — all non-fatal
            print(f"  ! {name}: {exc}")
            continue
        if getattr(parsed, "bozo", False) and not parsed.entries:
            print(f"  ! {name}: unreadable feed ({getattr(parsed, 'bozo_exception', '')})")
            continue

        count = 0
        for entry in parsed.entries:
            published = parse_entry_date(
                entry.get("published") or entry.get("updated") or entry.get("published_parsed")
            )
            if not is_recent(published, lookback_hours):
                continue
            summary = re.sub(r"<[^>]+>", " ", entry.get("summary", "") or "")
            story = Story(
                title=(entry.get("title") or "").strip(),
                url=normalize_url(entry.get("link") or ""),
                source=name,
                summary=re.sub(r"\s+", " ", summary).strip()[:1200],
                published_at=published,
            )
            if not story.title or not story.url:
                continue
            story.score = score_story(story, keywords, weight)
            stories.append(story)
            count += 1
        print(f"  · {name}: {count} recent items")

    return stories


def build_brief(
    config: Config,
    episode_id: str,
    stories: list[Story],
    max_stories: int = 25,
) -> Brief:
    """Ask the LLM to cluster and write up the shortlist."""
    shortlist = sorted(stories, key=lambda s: s.score, reverse=True)[:max_stories]
    allowed = {s.url for s in shortlist}

    lines = []
    for index, story in enumerate(shortlist, 1):
        lines.append(
            f"[{index}] {story.title}\n"
            f"    source: {story.source}\n"
            f"    url: {story.url}\n"
            f"    published: {story.published_at or 'unknown'}\n"
            f"    summary: {story.summary or '(none provided)'}"
        )

    user = (
        f"Show: {config.show.name}\n"
        f"Audience: {config.show.tagline or 'local community listeners'}\n"
        f"Episode date: {episode_id}\n"
        f"Target runtime: {config.show.target_minutes} minutes\n\n"
        f"Source material ({len(shortlist)} items):\n\n" + "\n\n".join(lines)
    )

    client = LlmClient(config)
    data = client.complete_json(SYSTEM_PROMPT, user)

    claims: list[Claim] = []
    ordered: list[Story] = []
    by_url = {s.url: s for s in shortlist}

    for cluster in data.get("clusters", []):
        for url in cluster.get("source_urls", []):
            url = normalize_url(url)
            if url in by_url and by_url[url] not in ordered:
                ordered.append(by_url[url])
        for claim in cluster.get("claims", []):
            urls = [normalize_url(u) for u in claim.get("source_urls", [])]
            # Drop any URL the model did not get from us. A fabricated citation
            # is worse than none, because it looks verified.
            urls = [u for u in urls if u in allowed]
            claims.append(Claim(text=claim.get("text", ""), source_urls=urls))

    for story in shortlist:  # anything unclustered still goes in the record
        if story not in ordered:
            ordered.append(story)

    return Brief(
        episode_id=episode_id,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        headline=data.get("headline", ""),
        stories=ordered,
        claims=claims,
        notes=data.get("notes", ""),
    )


def unsupported_claims(brief: Brief) -> list[Claim]:
    """Claims that ended up with no surviving source. These are exactly the
    lines to read twice before approving."""
    return [c for c in brief.claims if not c.source_urls]
