"""End-to-end test of research -> brief -> script with the LLM stubbed.

Covers the part of the pipeline that runs before any GPU or ffmpeg is involved,
including the citation-laundering guard: a model that returns a URL we never
gave it must not end up looking like a verified source.
"""

import json

import pytest
import yaml

from podcastpipe.config import load_config
from podcastpipe.models import Story
from podcastpipe.stages import research as research_stage
from podcastpipe.stages import script as script_stage

BRIEF_RESPONSE = {
    "headline": "Council approves the 2026 budget",
    "notes": "Lead with the budget. The tax rate is the part people will ask about.",
    "clusters": [
        {
            "title": "Budget approved",
            "priority": 1,
            "summary": "The council approved the budget on a 6-3 vote.",
            "source_urls": ["https://chapelboro.com/budget"],
            "claims": [
                {
                    "text": "The council approved the budget 6-3.",
                    "source_urls": ["https://chapelboro.com/budget"],
                },
                {
                    "text": "The tax rate rises by one cent.",
                    # A URL we never supplied. Must be stripped.
                    "source_urls": ["https://invented-source.example/made-up"],
                },
            ],
        },
        {
            "title": "VA expands benefits",
            "priority": 2,
            "summary": "New eligibility rules take effect.",
            "source_urls": ["https://news.va.gov/benefits"],
            "claims": [],
        },
    ],
}

SCRIPT_RESPONSE = {
    "title": "Budget night in Chapel Hill",
    "description": "The council approved the 2026 budget. Plus VA benefit changes.",
    "tags": ["chapel hill", "budget", "veterans"],
    "segments": [
        {
            "id": "cold_open",
            "name": "Cold Open",
            "blocks": [
                {
                    "text": "**The budget passed** late last night.",
                    "visual": {"type": "title_card", "text": "Budget Night"},
                    "source_urls": ["https://chapelboro.com/budget"],
                }
            ],
        },
        {
            "id": "headlines",
            "name": "Today's Headlines",
            "blocks": [
                {
                    "text": "HOST: Good morning. It's Saturday.",
                    "visual": {},
                    "source_urls": [],
                },
                {
                    "text": "The council voted six to three, according to the agenda.",
                    "visual": {"type": "lower_third", "text": "Town Council"},
                    "source_urls": ["https://chapelboro.com/budget"],
                },
            ],
        },
    ],
}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "show.yaml").write_text(
        yaml.safe_dump(
            {
                "show": {
                    "name": "The Post 6 Daily",
                    "host": "Host",
                    "target_minutes": 18,
                    "segments": [
                        {"id": "cold_open", "name": "Cold Open", "target_seconds": 25},
                        {"id": "headlines", "name": "Today's Headlines", "target_seconds": 180},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace the network call with canned responses, chosen by prompt."""
    calls = []

    def fake_complete(self, system, user, **kwargs):
        calls.append({"system": system, "user": user})
        if "research desk" in system:
            return f"```json\n{json.dumps(BRIEF_RESPONSE)}\n```"
        return json.dumps(SCRIPT_RESPONSE)

    monkeypatch.setattr("podcastpipe.llm.LlmClient.complete", fake_complete)
    return calls


@pytest.fixture
def stories():
    return [
        Story(
            title="Council approves 2026 budget",
            url="https://chapelboro.com/budget",
            source="Chapelboro",
            summary="A 6-3 vote.",
            score=10.0,
        ),
        Story(
            title="VA expands benefits eligibility",
            url="https://news.va.gov/benefits",
            source="VA",
            summary="New rules.",
            score=6.0,
        ),
        Story(
            title="Unrelated wire story",
            url="https://wire.example/x",
            source="Wire",
            summary="Elsewhere.",
            score=0.5,
        ),
    ]


class TestBriefBuilding:
    def test_brief_has_headline_and_notes(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        brief = research_stage.build_brief(config, "2026-08-08", stories)
        assert brief.headline == "Council approves the 2026 budget"
        assert "budget" in brief.notes.lower()

    def test_clustered_stories_come_first(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        brief = research_stage.build_brief(config, "2026-08-08", stories)
        assert brief.stories[0].url == "https://chapelboro.com/budget"

    def test_unclustered_stories_are_still_recorded(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        brief = research_stage.build_brief(config, "2026-08-08", stories)
        assert "https://wire.example/x" in brief.source_urls()

    def test_invented_citation_is_stripped(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        brief = research_stage.build_brief(config, "2026-08-08", stories)
        all_urls = [url for claim in brief.claims for url in claim.source_urls]
        assert "https://invented-source.example/made-up" not in all_urls

    def test_stripped_citation_surfaces_as_unsupported(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        brief = research_stage.build_brief(config, "2026-08-08", stories)
        unsupported = research_stage.unsupported_claims(brief)
        assert len(unsupported) == 1
        assert "tax rate" in unsupported[0].text

    def test_prompt_includes_the_source_urls(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        research_stage.build_brief(config, "2026-08-08", stories)
        assert "https://chapelboro.com/budget" in stub_llm[0]["user"]

    def test_shortlist_is_capped(self, project, stub_llm):
        config = load_config(project / "config" / "show.yaml")
        many = [
            Story(title=f"Story {i}", url=f"https://a.com/{i}", source="s", score=float(i))
            for i in range(50)
        ]
        research_stage.build_brief(config, "2026-08-08", many, max_stories=5)
        assert stub_llm[0]["user"].count("    url: ") == 5


class TestScriptGeneration:
    def _brief(self, config, stories):
        return research_stage.build_brief(config, "2026-08-08", stories)

    def test_script_structure(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        script = script_stage.generate_script(config, self._brief(config, stories))
        assert script.title == "Budget night in Chapel Hill"
        assert [s.id for s in script.segments] == ["cold_open", "headlines"]

    def test_markdown_and_labels_stripped(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        script = script_stage.generate_script(config, self._brief(config, stories))
        texts = [b.text for b in script.speech_blocks()]
        assert "The budget passed late last night." in texts
        assert "Good morning. It's Saturday." in texts
        assert not any("HOST:" in t or "**" in t for t in texts)

    def test_visual_cues_survive(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        script = script_stage.generate_script(config, self._brief(config, stories))
        kinds = [b.visual.get("type") for b in script.speech_blocks() if b.visual]
        assert "title_card" in kinds and "lower_third" in kinds

    def test_prompt_carries_the_style_guide_and_segments(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        self._brief(config, stories)
        script_stage.generate_script(config, self._brief(config, stories))
        prompt = stub_llm[-1]["user"]
        assert "id: cold_open" in prompt
        assert "id: headlines" in prompt

    def test_unsourced_claims_are_flagged_in_the_prompt(self, project, stub_llm, stories):
        config = load_config(project / "config" / "show.yaml")
        brief = self._brief(config, stories)
        prompt = script_stage.build_prompt(config, brief)
        assert "UNSOURCED" in prompt

    def test_script_persists_and_reloads(self, project, stub_llm, stories, tmp_path):
        config = load_config(project / "config" / "show.yaml")
        script = script_stage.generate_script(config, self._brief(config, stories))
        path = tmp_path / "script.json"
        script.save(path)

        from podcastpipe.models import Script

        reloaded = Script.load(path)
        assert reloaded.title == script.title
        assert reloaded.approved is False
        assert len(reloaded.speech_blocks()) == len(script.speech_blocks())
