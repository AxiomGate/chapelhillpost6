"""Stage 7 — publish.

Builds a podcast RSS 2.0 feed with the iTunes extensions Apple and Spotify
require, and uploads the video to YouTube with captions attached. The feed is
generated from the episode database, so it is always a complete regeneration
rather than an append — that makes a bad edit recoverable by fixing the data and
rebuilding, instead of by hand-patching XML.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..config import Config

ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"


def format_rfc2822(value: str) -> str:
    """RFC 2822 date, which is what RSS requires. Accepts ISO or YYYY-MM-DD."""
    for parser in (
        lambda v: datetime.fromisoformat(v),
        lambda v: datetime.strptime(v, "%Y-%m-%d"),
    ):
        try:
            parsed = parser(value)
        except (ValueError, TypeError):
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return format_datetime(parsed)
    return format_datetime(datetime.now(timezone.utc))


def format_duration(seconds: float) -> str:
    """``HH:MM:SS`` for the itunes:duration tag."""
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_feed(config: Config, episodes: list[dict[str, Any]]) -> str:
    """Render the full podcast feed.

    ``episodes`` entries need: id, date, number, title, description, duration,
    audio_url, audio_bytes. Episodes missing an audio_url are skipped rather
    than emitted broken — a 404 enclosure gets a feed delisted.
    """
    ET.register_namespace("itunes", ITUNES_NS)
    ET.register_namespace("content", CONTENT_NS)

    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")

    show = config.show
    ET.SubElement(channel, "title").text = show.name
    ET.SubElement(channel, "link").text = config.publish.site_base_url
    ET.SubElement(channel, "description").text = show.tagline or show.name
    ET.SubElement(channel, "language").text = show.language
    ET.SubElement(channel, "lastBuildDate").text = format_datetime(datetime.now(timezone.utc))
    ET.SubElement(channel, f"{{{ITUNES_NS}}}author").text = show.host
    ET.SubElement(channel, f"{{{ITUNES_NS}}}summary").text = show.tagline or show.name
    ET.SubElement(channel, f"{{{ITUNES_NS}}}explicit").text = "false"
    ET.SubElement(channel, f"{{{ITUNES_NS}}}type").text = "episodic"

    owner = ET.SubElement(channel, f"{{{ITUNES_NS}}}owner")
    ET.SubElement(owner, f"{{{ITUNES_NS}}}name").text = show.host
    ET.SubElement(owner, f"{{{ITUNES_NS}}}email").text = show.author_email

    artwork = f"{config.publish.site_base_url.rstrip('/')}/images/podcast-cover.jpg"
    ET.SubElement(channel, f"{{{ITUNES_NS}}}image", {"href": artwork})

    for episode in episodes:
        audio_url = episode.get("audio_url")
        if not audio_url:
            continue

        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = episode.get("title") or episode["id"]
        ET.SubElement(item, "guid", {"isPermaLink": "false"}).text = episode["id"]
        ET.SubElement(item, "pubDate").text = format_rfc2822(episode.get("date", ""))
        ET.SubElement(item, "description").text = episode.get("description", "")
        ET.SubElement(item, f"{{{CONTENT_NS}}}encoded").text = (
            f"<![CDATA[{episode.get('description', '')}]]>"
        )
        ET.SubElement(
            item,
            "enclosure",
            {
                "url": audio_url,
                "length": str(int(episode.get("audio_bytes", 0))),
                "type": "audio/mpeg",
            },
        )
        ET.SubElement(item, f"{{{ITUNES_NS}}}duration").text = format_duration(
            float(episode.get("duration", 0) or 0)
        )
        ET.SubElement(item, f"{{{ITUNES_NS}}}episode").text = str(episode.get("number", 0))
        ET.SubElement(item, f"{{{ITUNES_NS}}}explicit").text = "false"

    ET.indent(rss, space="  ")
    body = ET.tostring(rss, encoding="unicode", xml_declaration=True)
    # ElementTree escapes the CDATA markers; unescape just those two tokens.
    return body.replace("&lt;![CDATA[", "<![CDATA[").replace("]]&gt;", "]]>")


def write_feed(config: Config, episodes: list[dict[str, Any]]) -> Path:
    path = config.path(config.publish.rss_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_feed(config, episodes), encoding="utf-8")
    return path


def build_youtube_description(
    description: str, sources: list[str], site_url: str, max_sources: int = 12
) -> str:
    """Show-notes text with a sources block.

    Listing sources under a local news video is not decoration — it is the thing
    that makes a correction possible when you get something wrong.
    """
    parts = [description.strip(), ""]
    if sources:
        parts.append("Sources:")
        parts.extend(f"• {url}" for url in sources[:max_sources])
        parts.append("")
    parts.append(f"More: {site_url}")
    return "\n".join(parts).strip()


def upload_to_youtube(
    config: Config,
    video_path: str | Path,
    title: str,
    description: str,
    tags: list[str],
    captions_path: str | Path | None = None,
) -> str:
    """Resumable upload via the YouTube Data API. Returns the video id."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    token_path = config.path(config.publish.youtube_token)
    secrets_path = config.path(config.publish.youtube_client_secrets)

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not secrets_path.exists():
                raise FileNotFoundError(
                    f"YouTube client secrets not found at {secrets_path}. Create an "
                    "OAuth desktop client in Google Cloud Console and download it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), scopes)
            credentials = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    youtube = build("youtube", "v3", credentials=credentials)
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:30],
            "categoryId": config.publish.youtube_category_id,
        },
        "status": {
            "privacyStatus": config.publish.youtube_privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(video_path), chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  upload {int(status.progress() * 100)}%", flush=True)

    video_id = response["id"]
    print(f"  https://youtu.be/{video_id}")

    if captions_path and Path(captions_path).exists():
        youtube.captions().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "language": "en",
                    "name": "English",
                    "isDraft": False,
                }
            },
            media_body=MediaFileUpload(str(captions_path)),
        ).execute()
        print("  captions attached")

    return video_id


def episode_page_html(title: str, description: str, video_id: str, audio_url: str) -> str:
    """A minimal episode block for the website. Escaped, because titles come
    from a language model and will eventually contain an ampersand."""
    embed = (
        f'<iframe src="https://www.youtube.com/embed/{html.escape(video_id)}" '
        'title="Episode video" frameborder="0" allowfullscreen loading="lazy"></iframe>'
        if video_id
        else ""
    )
    return (
        '<article class="episode">\n'
        f"  <h2>{html.escape(title)}</h2>\n"
        f"  <p>{html.escape(description)}</p>\n"
        f"  {embed}\n"
        f'  <audio controls preload="none" src="{html.escape(audio_url)}"></audio>\n'
        "</article>"
    )
