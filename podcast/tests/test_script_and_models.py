import json

import pytest

from podcastpipe.llm import LlmError, parse_json_response
from podcastpipe.models import Block, Script, Segment, assign_timeline, slugify
from podcastpipe.stages.script import clean_spoken_text, script_from_payload, validate_script


class TestSlugify:
    def test_basic(self):
        assert slugify("Council Approves Budget") == "council-approves-budget"

    def test_strips_punctuation(self):
        assert slugify("Episode 12: What's Next?!") == "episode-12-whats-next"

    def test_never_empty(self):
        assert slugify("!!!") == "untitled"

    def test_truncates_without_trailing_dash(self):
        assert not slugify("a " * 60, max_length=20).endswith("-")


class TestCleanSpokenText:
    def test_removes_markdown(self):
        assert clean_spoken_text("**Big** news _today_") == "Big news today"

    def test_removes_speaker_label(self):
        assert clean_spoken_text("HOST: Good morning.") == "Good morning."

    def test_removes_stage_directions(self):
        assert "music" not in clean_spoken_text("[MUSIC FADES] Welcome back.").lower()

    def test_collapses_whitespace(self):
        assert clean_spoken_text("a\n\n  b") == "a b"

    def test_keeps_ordinary_brackets(self):
        # Only stage-direction keywords are stripped, not every bracket.
        assert "(2026)" in clean_spoken_text("The budget (2026) passed.")


class TestScriptFromPayload:
    payload = {
        "title": "Budget night",
        "description": "What the council did.",
        "tags": ["springfield", "budget"],
        "segments": [
            {
                "id": "cold_open",
                "name": "Cold Open",
                "blocks": [
                    {"text": "**The vote** came late.", "source_urls": ["https://a.com/1"]},
                    {"text": "   ", "source_urls": []},
                ],
            },
            {
                "id": "main",
                "name": "Main",
                "blocks": [{"text": "Here is what happened.", "visual": {"type": "lower_third", "text": "Budget"}}],
            },
        ],
    }

    def test_builds_segments_and_blocks(self):
        script = script_from_payload("2026-08-08", self.payload)
        assert len(script.segments) == 2
        assert script.title == "Budget night"

    def test_empty_blocks_dropped(self):
        script = script_from_payload("2026-08-08", self.payload)
        assert len(script.segments[0].blocks) == 1

    def test_block_ids_are_stable_and_scoped(self):
        script = script_from_payload("2026-08-08", self.payload)
        assert script.segments[0].blocks[0].id == "cold_open-1"
        assert script.segments[1].blocks[0].id == "main-1"

    def test_text_is_cleaned(self):
        script = script_from_payload("2026-08-08", self.payload)
        assert script.segments[0].blocks[0].text == "The vote came late."

    def test_visual_cue_preserved(self):
        script = script_from_payload("2026-08-08", self.payload)
        assert script.segments[1].blocks[0].visual["type"] == "lower_third"

    def test_round_trips_through_json(self, tmp_path):
        script = script_from_payload("2026-08-08", self.payload)
        path = tmp_path / "script.json"
        script.save(path)
        reloaded = Script.load(path)
        assert reloaded.title == script.title
        assert [b.id for b in reloaded.blocks()] == [b.id for b in script.blocks()]


class TestTimeline:
    def test_blocks_are_laid_end_to_end(self):
        blocks = [Block(id="a", text="x", duration=2.0), Block(id="b", text="y", duration=3.0)]
        total = assign_timeline(blocks)
        assert blocks[0].start == 0.0
        assert blocks[1].start == 2.0
        assert total == 5.0

    def test_gap_is_inserted_between_but_not_before(self):
        blocks = [Block(id="a", text="x", duration=1.0), Block(id="b", text="y", duration=1.0)]
        total = assign_timeline(blocks, gap=0.5)
        assert blocks[0].start == 0.0
        assert blocks[1].start == 1.5
        assert total == 2.5

    def test_block_end_property(self):
        block = Block(id="a", text="x", duration=2.0, start=3.0)
        assert block.end == 5.0

    def test_empty_list(self):
        assert assign_timeline([]) == 0.0


class TestValidation:
    def _config(self, target_minutes=18):
        from podcastpipe.config import (
            AudioConfig, AvatarConfig, Config, GpuConfig, LlmConfig,
            PublishConfig, ShowConfig, TtsConfig, VideoConfig,
        )
        from pathlib import Path

        return Config(
            root=Path("."), show=ShowConfig(target_minutes=target_minutes),
            gpu=GpuConfig(), llm=LlmConfig(), tts=TtsConfig(), avatar=AvatarConfig(),
            video=VideoConfig(), audio=AudioConfig(), publish=PublishConfig(),
            work_dir=Path("work"), output_dir=Path("output"),
        )

    def _script(self, text, title="T", sources=("https://a.com/1",)):
        """Build a script whose text is spread over realistically sized blocks,
        so length warnings fire on the script rather than on one huge block."""
        words = text.split()
        chunks = [" ".join(words[i : i + 120]) for i in range(0, len(words), 120)] or [""]
        blocks = [
            Block(id=f"s-{i + 1}", text=chunk, source_urls=list(sources))
            for i, chunk in enumerate(chunks)
        ]
        return Script(
            episode_id="2026-08-08",
            title=title,
            segments=[Segment(id="s", name="S", blocks=blocks)],
        )

    def test_missing_title_warns(self):
        warnings = validate_script(self._script("a " * 2700, title=""), self._config())
        assert any("no title" in w for w in warnings)

    def test_short_script_warns(self):
        warnings = validate_script(self._script("Just a few words here."), self._config())
        assert any("short" in w for w in warnings)

    def test_long_script_warns(self):
        warnings = validate_script(self._script("word " * 5000), self._config())
        assert any("long" in w for w in warnings)

    def test_vague_attribution_warns(self):
        script = self._script("Reports say the budget passed. " + "word " * 2700)
        assert any("vague attribution" in w for w in validate_script(script, self._config()))

    def test_raw_url_warns(self):
        script = self._script("Go to https://example.com now. " + "word " * 2700)
        assert any("raw URL" in w for w in validate_script(script, self._config()))

    def test_unsourced_blocks_warn(self):
        script = self._script("word " * 2700, sources=())
        assert any("cite no source" in w for w in validate_script(script, self._config()))

    def test_clean_script_has_no_warnings(self):
        script = self._script("The council voted, according to the agenda. " + "word " * 2600)
        assert validate_script(script, self._config()) == []


class TestJsonParsing:
    def test_plain_json(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_preamble(self):
        assert parse_json_response('Sure, here you go:\n{"a": 1}') == {"a": 1}

    def test_json_array(self):
        assert parse_json_response("[1, 2, 3]") == [1, 2, 3]

    def test_nested_braces_survive(self):
        payload = {"segments": [{"blocks": [{"text": "hi"}]}]}
        assert parse_json_response(f"Here:\n{json.dumps(payload)}\nDone.") == payload

    def test_no_json_raises(self):
        with pytest.raises(LlmError):
            parse_json_response("there is no json here at all")


class TestScriptMetrics:
    def test_word_count_ignores_non_speech(self):
        script = Script(
            episode_id="x", title="T",
            segments=[Segment(id="s", name="S", blocks=[
                Block(id="s-1", text="one two three"),
                Block(id="s-2", text="ignored", kind="music"),
            ])],
        )
        assert script.word_count() == 3
        assert len(script.speech_blocks()) == 1

    def test_estimated_minutes(self):
        script = Script(
            episode_id="x", title="T",
            segments=[Segment(id="s", name="S", blocks=[Block(id="s-1", text="word " * 300)])],
        )
        assert script.estimated_minutes(150) == pytest.approx(2.0)
