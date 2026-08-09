"""The approval gate — a small local web app for editing and approving scripts.

This is the one place a human is required, so it is built to make the human's
job fast: every block is editable in place, its supporting sources sit next to
it, and the warnings from ``validate_script`` are at the top. Approving writes
``approved: true`` back to script.json and the rest of the pipeline unblocks.

Binds to 127.0.0.1 by default. It has no authentication, so if you move it off
localhost, put it behind something that does.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from ..config import Config
from ..db import Database
from ..models import Episode, Script
from ..stages.script import clean_spoken_text, validate_script

# Imported at module scope on purpose. This module uses `from __future__ import
# annotations`, so FastAPI resolves handler annotations against module globals
# via get_type_hints; a function-local `Request` import resolves to nothing and
# FastAPI silently demotes the parameter to a query field, answering 422.
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    FASTAPI_AVAILABLE = True
except ImportError:  # keep the core CLI usable without the web extras
    FASTAPI_AVAILABLE = False

MISSING_DEPS = "The review UI needs fastapi and uvicorn: pip install '.[review]'"


def create_app(config: Config):
    """Build the FastAPI app. Separate from ``serve`` so tests can drive it
    without binding a port."""
    if not FASTAPI_AVAILABLE:
        raise SystemExit(MISSING_DEPS)

    app = FastAPI(title="Podcast review")
    database = Database(config.work_dir / "pipeline.db")

    def script_path(episode_id: str) -> Path:
        return config.episode_dir(episode_id) / "script.json"

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        episodes = database.list_episodes(limit=30)
        rows = "".join(
            f'<tr><td><a href="/episode/{e.id}">{e.id}</a></td>'
            f"<td>{e.number}</td><td>{html.escape(e.status)}</td>"
            f"<td>{html.escape(e.title)}</td></tr>"
            for e in episodes
        )
        return _page(
            "Episodes",
            f"<table class='list'><tr><th>Date</th><th>#</th><th>Status</th><th>Title</th></tr>{rows}</table>"
            if rows
            else "<p class='muted'>No episodes yet. Run <code>podcastpipe research</code>.</p>",
        )

    @app.get("/episode/{episode_id}", response_class=HTMLResponse)
    def episode_view(episode_id: str) -> str:
        path = script_path(episode_id)
        if not path.exists():
            return _page("Not found", f"<p>No script for {html.escape(episode_id)}.</p>")

        script = Script.load(path)
        warnings = validate_script(script, config)
        brief_path = config.episode_dir(episode_id) / "brief.json"
        notes = ""
        if brief_path.exists():
            brief = json.loads(brief_path.read_text(encoding="utf-8"))
            notes = brief.get("notes", "")

        banner = ""
        if warnings:
            items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
            banner = f"<div class='warn'><strong>Check before approving</strong><ul>{items}</ul></div>"

        blocks_html = []
        for segment in script.segments:
            blocks_html.append(f"<h2>{html.escape(segment.name)}</h2>")
            for block in segment.blocks:
                sources = "".join(
                    f'<a href="{html.escape(u)}" target="_blank" rel="noreferrer">{html.escape(_domain(u))}</a>'
                    for u in block.source_urls
                )
                sources_html = (
                    f"<div class='sources'>{sources}</div>"
                    if sources
                    else "<div class='sources nosrc'>no source cited</div>"
                )
                visual = html.escape(json.dumps(block.visual)) if block.visual else "{}"
                blocks_html.append(
                    f"<div class='block'>"
                    f"<label class='bid'>{html.escape(block.id)}</label>"
                    f"<textarea name='text__{html.escape(block.id)}' rows='4'>{html.escape(block.text)}</textarea>"
                    f"<input class='visual' name='visual__{html.escape(block.id)}' value='{visual}'>"
                    f"{sources_html}"
                    f"</div>"
                )

        body = f"""
        {banner}
        <p class='muted'>{script.word_count()} words &middot; about {script.estimated_minutes():.1f} min
        &middot; status: {'approved' if script.approved else 'awaiting approval'}</p>
        {f"<div class='notes'><strong>Desk notes</strong><p>{html.escape(notes)}</p></div>" if notes else ""}
        <form method='post' action='/episode/{html.escape(episode_id)}/save'>
          <label class='field'>Title
            <input name='title' value='{html.escape(script.title)}'>
          </label>
          <label class='field'>Description
            <textarea name='description' rows='3'>{html.escape(script.description)}</textarea>
          </label>
          {''.join(blocks_html)}
          <div class='actions'>
            <button type='submit' name='action' value='save'>Save draft</button>
            <button type='submit' name='action' value='approve' class='primary'>Save &amp; approve</button>
          </div>
        </form>
        """
        return _page(script.title or episode_id, body)

    @app.post("/episode/{episode_id}/save")
    async def save_script(episode_id: str, request: Request) -> RedirectResponse:
        # Block field names are dynamic (text__<block id>), which FastAPI cannot
        # express as typed parameters, so the raw form is read off the request.
        form = await request.form()
        path = script_path(episode_id)
        script = Script.load(path)

        script.title = str(form.get("title", script.title)).strip()
        script.description = str(form.get("description", script.description)).strip()

        for block in script.blocks():
            text_key = f"text__{block.id}"
            if text_key in form:
                new_text = clean_spoken_text(str(form[text_key]))
                if new_text != block.text:
                    # Text changed, so cached audio for this block is stale.
                    block.text = new_text
                    block.audio_path = ""
                    block.duration = 0.0
            visual_key = f"visual__{block.id}"
            if visual_key in form:
                try:
                    script_visual = json.loads(str(form[visual_key]) or "{}")
                    if isinstance(script_visual, dict):
                        block.visual = script_visual
                except json.JSONDecodeError:
                    pass  # keep the previous cue rather than losing it to a typo

        action = str(form.get("action", "save"))
        script.approved = action == "approve"
        script.save(path)

        episode = database.get_episode(episode_id) or Episode(id=episode_id, date=episode_id)
        episode.title = script.title
        episode.status = "approved" if script.approved else "scripted"
        database.upsert_episode(episode)
        database.log(episode_id, "review", action, f"{script.word_count()} words")

        return RedirectResponse(f"/episode/{episode_id}", status_code=303)

    return app


def serve(config: Config, host: str = "127.0.0.1", port: int = 8420) -> None:
    """Run the review UI. No authentication — if you move it off localhost, put
    it behind something that has some."""
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(MISSING_DEPS) from exc

    print(f"Review UI: http://{host}:{port}")
    uvicorn.run(create_app(config), host=host, port=port, log_level="warning")


def _domain(url: str) -> str:
    stripped = url.split("//")[-1]
    return stripped.split("/")[0][:40] or url[:40]


STYLE = """
:root { color-scheme: light dark; --bg:#fbfbfd; --fg:#16181d; --muted:#6b7280;
  --line:#e3e5ea; --accent:#2F6F7E; --card:#ffffff; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1115; --fg:#e8eaf0;
  --muted:#9aa1ad; --line:#242833; --card:#161922; } }
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.25rem 5rem; background:var(--bg); color:var(--fg);
  font:16px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
main { max-width:52rem; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 1.5rem; }
h2 { font-size:1rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); margin:2rem 0 .75rem; }
a { color:var(--accent); }
.muted { color:var(--muted); font-size:.9rem; }
.block { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:.85rem; margin-bottom:.85rem; }
.bid { font:600 .72rem ui-monospace,monospace; color:var(--muted); display:block;
  margin-bottom:.4rem; }
textarea, input { width:100%; font:inherit; color:inherit; background:transparent;
  border:1px solid var(--line); border-radius:7px; padding:.5rem; resize:vertical; }
textarea:focus, input:focus { outline:2px solid var(--accent); outline-offset:1px; }
.visual { margin-top:.5rem; font:.78rem ui-monospace,monospace; color:var(--muted); }
.sources { margin-top:.5rem; display:flex; flex-wrap:wrap; gap:.4rem; font-size:.78rem; }
.sources a { background:rgba(47,111,126,.12); padding:.15rem .45rem; border-radius:4px;
  text-decoration:none; }
.nosrc { color:#c2410c; }
.warn { background:rgba(234,179,8,.12); border:1px solid rgba(234,179,8,.4);
  border-radius:10px; padding:.85rem 1rem; margin-bottom:1.25rem; }
.warn ul { margin:.5rem 0 0; padding-left:1.1rem; }
.notes { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:.85rem 1rem; margin-bottom:1.5rem; }
.field { display:block; margin-bottom:1rem; font-size:.85rem; color:var(--muted); }
.field input, .field textarea { margin-top:.35rem; color:var(--fg); font-size:1rem; }
.actions { position:sticky; bottom:0; padding:1rem 0; background:linear-gradient(
  to top, var(--bg) 70%, transparent); display:flex; gap:.75rem; }
button { font:inherit; padding:.6rem 1.1rem; border-radius:8px; border:1px solid var(--line);
  background:var(--card); color:var(--fg); cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
table.list { width:100%; border-collapse:collapse; }
table.list th { text-align:left; font-size:.75rem; text-transform:uppercase;
  letter-spacing:.06em; color:var(--muted); border-bottom:1px solid var(--line); padding:.5rem; }
table.list td { padding:.5rem; border-bottom:1px solid var(--line); }
code { font:.85em ui-monospace,monospace; background:var(--card); padding:.1rem .3rem;
  border-radius:4px; }
"""


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body><main><h1><a href='/' style='text-decoration:none;color:inherit'>&#8592;</a> "
        f"{html.escape(title)}</h1>{body}</main></body></html>"
    )
