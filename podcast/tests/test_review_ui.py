"""Tests for the approval UI — the one step a human is required for.

Exercises the real app through Starlette's test client: rendering a script,
saving an edit, approving, and the cache invalidation that an edit must trigger.
"""

import json

import pytest
import yaml

from podcastpipe.config import load_config
from podcastpipe.db import Database
from podcastpipe.models import Block, Episode, Script, Segment

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from podcastpipe.review.app import create_app  # noqa: E402


@pytest.fixture
def project(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "show.yaml").write_text(
        yaml.safe_dump({"show": {"name": "Test Show", "target_minutes": 18}}),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def config(project):
    return load_config(project / "config" / "show.yaml")


@pytest.fixture
def episode(config):
    """An episode with a script and a brief on disk, registered in the db."""
    episode_id = "2026-08-08"
    episode_dir = config.episode_dir(episode_id)
    episode_dir.mkdir(parents=True)

    script = Script(
        episode_id=episode_id,
        title="Budget night",
        description="What the council did.",
        segments=[
            Segment(
                id="cold_open",
                name="Cold Open",
                blocks=[
                    Block(
                        id="cold_open-1",
                        text="The budget passed late last night.",
                        source_urls=["https://chapelboro.com/budget"],
                        visual={"type": "title_card", "text": "Budget Night"},
                        audio_path="/cached/audio.wav",
                        duration=3.2,
                    )
                ],
            ),
            Segment(
                id="main",
                name="Main",
                blocks=[Block(id="main-1", text="Here is what happened.")],
            ),
        ],
    )
    script.save(episode_dir / "script.json")
    (episode_dir / "brief.json").write_text(
        json.dumps({"notes": "Lead with the budget."}), encoding="utf-8"
    )

    with Database(config.work_dir / "pipeline.db") as db:
        db.upsert_episode(Episode(id=episode_id, date=episode_id, title="Budget night"))
    return episode_id


@pytest.fixture
def client(config):
    return TestClient(create_app(config))


class TestIndex:
    def test_empty_index(self, config):
        with TestClient(create_app(config)) as client:
            response = client.get("/")
        assert response.status_code == 200
        assert "No episodes yet" in response.text

    def test_lists_episodes(self, client, episode):
        response = client.get("/")
        assert episode in response.text
        assert "Budget night" in response.text


class TestEpisodeView:
    def test_unknown_episode(self, client):
        response = client.get("/episode/1999-01-01")
        assert response.status_code == 200
        assert "No script" in response.text

    def test_renders_blocks(self, client, episode):
        response = client.get(f"/episode/{episode}")
        assert "The budget passed late last night." in response.text
        assert "cold_open-1" in response.text
        assert "main-1" in response.text

    def test_shows_sources_next_to_text(self, client, episode):
        response = client.get(f"/episode/{episode}")
        assert "chapelboro.com" in response.text

    def test_flags_blocks_without_sources(self, client, episode):
        response = client.get(f"/episode/{episode}")
        assert "no source cited" in response.text

    def test_shows_desk_notes(self, client, episode):
        assert "Lead with the budget." in client.get(f"/episode/{episode}").text

    def test_shows_validation_warnings(self, client, episode):
        # This script is far under the 18-minute target.
        assert "Check before approving" in client.get(f"/episode/{episode}").text

    def test_escapes_html_in_titles(self, config, client, episode):
        path = config.episode_dir(episode) / "script.json"
        script = Script.load(path)
        script.title = "Budget <script>alert(1)</script>"
        script.save(path)
        response = client.get(f"/episode/{episode}")
        assert "<script>alert(1)</script>" not in response.text
        assert "&lt;script&gt;" in response.text


class TestSaving:
    def _form(self, **overrides):
        form = {
            "title": "Budget night",
            "description": "What the council did.",
            "text__cold_open-1": "The budget passed late last night.",
            "text__main-1": "Here is what happened.",
            "visual__cold_open-1": '{"type": "title_card", "text": "Budget Night"}',
            "action": "save",
        }
        form.update(overrides)
        return form

    def test_save_redirects_back(self, client, episode):
        response = client.post(
            f"/episode/{episode}/save", data=self._form(), follow_redirects=False
        )
        assert response.status_code == 303

    def test_edit_persists(self, config, client, episode):
        client.post(
            f"/episode/{episode}/save",
            data=self._form(**{"text__main-1": "Revised wording here."}),
        )
        script = Script.load(config.episode_dir(episode) / "script.json")
        assert script.blocks()[1].text == "Revised wording here."

    def test_edit_invalidates_cached_audio(self, config, client, episode):
        client.post(
            f"/episode/{episode}/save",
            data=self._form(**{"text__cold_open-1": "Completely different opening."}),
        )
        block = Script.load(config.episode_dir(episode) / "script.json").blocks()[0]
        assert block.audio_path == ""
        assert block.duration == 0.0

    def test_unchanged_block_keeps_cached_audio(self, config, client, episode):
        client.post(f"/episode/{episode}/save", data=self._form())
        block = Script.load(config.episode_dir(episode) / "script.json").blocks()[0]
        assert block.audio_path == "/cached/audio.wav"
        assert block.duration == 3.2

    def test_markdown_stripped_on_save(self, config, client, episode):
        client.post(
            f"/episode/{episode}/save",
            data=self._form(**{"text__main-1": "**Bold** claim."}),
        )
        script = Script.load(config.episode_dir(episode) / "script.json")
        assert script.blocks()[1].text == "Bold claim."

    def test_save_does_not_approve(self, config, client, episode):
        client.post(f"/episode/{episode}/save", data=self._form())
        assert Script.load(config.episode_dir(episode) / "script.json").approved is False

    def test_approve_sets_flag_and_status(self, config, client, episode):
        client.post(f"/episode/{episode}/save", data=self._form(action="approve"))
        assert Script.load(config.episode_dir(episode) / "script.json").approved is True
        with Database(config.work_dir / "pipeline.db") as db:
            assert db.get_episode(episode).status == "approved"

    def test_title_change_reaches_the_database(self, config, client, episode):
        client.post(f"/episode/{episode}/save", data=self._form(title="New title"))
        with Database(config.work_dir / "pipeline.db") as db:
            assert db.get_episode(episode).title == "New title"

    def test_malformed_visual_json_keeps_previous_cue(self, config, client, episode):
        client.post(
            f"/episode/{episode}/save",
            data=self._form(**{"visual__cold_open-1": "{not json"}),
        )
        block = Script.load(config.episode_dir(episode) / "script.json").blocks()[0]
        assert block.visual == {"type": "title_card", "text": "Budget Night"}

    def test_valid_visual_json_is_applied(self, config, client, episode):
        client.post(
            f"/episode/{episode}/save",
            data=self._form(**{"visual__cold_open-1": '{"type": "broll", "query": "town hall"}'}),
        )
        block = Script.load(config.episode_dir(episode) / "script.json").blocks()[0]
        assert block.visual == {"type": "broll", "query": "town hall"}

    def test_review_action_is_logged(self, config, client, episode):
        client.post(f"/episode/{episode}/save", data=self._form(action="approve"))
        with Database(config.work_dir / "pipeline.db") as db:
            stages = [e["stage"] for e in db.events(episode)]
        assert "review" in stages
