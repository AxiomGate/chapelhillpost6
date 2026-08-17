"""Tests for the media-adjacent pure logic: caption formatting, filtergraph
construction, chunk planning and feed generation. None of these need ffmpeg or a
GPU, which is exactly why they are worth having -- a filtergraph typo otherwise
surfaces forty minutes into a render."""

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from podcastpipe.config import (
    AudioConfig, AvatarConfig, Config, GpuConfig, LlmConfig,
    PublishConfig, ShowConfig, TtsConfig, VideoConfig,
)
from podcastpipe.proc import build_concat_command, build_loudnorm_command
from podcastpipe.stages.assemble import (
    Overlay, build_video_filtergraph, keyword_score,
)
from podcastpipe.stages.avatar import plan_chunks
from podcastpipe.stages.captions import (
    Cue, Word, format_ass_timestamp, format_timestamp, group_words, to_ass, to_srt,
)
from podcastpipe.stages.publish import (
    build_feed, build_youtube_description, format_duration, format_rfc2822,
)


def make_config(**overrides):
    base = dict(
        root=Path("."), show=ShowConfig(name="Test Show", host="Host", author_email="a@b.c"),
        gpu=GpuConfig(), llm=LlmConfig(), tts=TtsConfig(), avatar=AvatarConfig(),
        video=VideoConfig(), audio=AudioConfig(), publish=PublishConfig(),
        work_dir=Path("work"), output_dir=Path("output"),
    )
    base.update(overrides)
    return Config(**base)


class TestTimestamps:
    def test_srt_format(self):
        assert format_timestamp(0) == "00:00:00,000"
        assert format_timestamp(3661.5) == "01:01:01,500"

    def test_negative_clamps_to_zero(self):
        # ffmpeg rejects negative timestamps outright.
        assert format_timestamp(-5) == "00:00:00,000"

    def test_ass_format_uses_centiseconds(self):
        assert format_ass_timestamp(0) == "0:00:00.00"
        assert format_ass_timestamp(3661.5) == "1:01:01.50"

    def test_duration_format(self):
        assert format_duration(0) == "00:00:00"
        assert format_duration(1265) == "00:21:05"


class TestCaptionGrouping:
    def words(self, texts, step=0.4):
        return [Word(t, i * step, i * step + step * 0.9) for i, t in enumerate(texts)]

    def test_breaks_on_line_length(self):
        cues = group_words(self.words(["averylongword"] * 10), max_chars=30)
        assert len(cues) > 1

    def test_breaks_on_sentence_end(self):
        cues = group_words(self.words(["one", "two.", "three", "four"]))
        assert len(cues) == 2
        assert cues[0].text == "one two."

    def test_breaks_on_long_pause(self):
        words = [Word("a", 0, 0.3), Word("b", 5.0, 5.3)]
        assert len(group_words(words, max_gap=0.7)) == 2

    def test_breaks_on_max_duration(self):
        words = [Word(f"w{i}", i * 1.0, i * 1.0 + 0.9) for i in range(10)]
        cues = group_words(words, max_chars=999, max_duration=3.0)
        assert len(cues) > 1

    def test_cue_bounds_follow_words(self):
        cues = group_words(self.words(["a", "b."]))
        assert cues[0].start == 0.0
        assert cues[0].end == pytest.approx(0.76)

    def test_all_words_preserved(self):
        texts = [f"w{i}" for i in range(25)]
        cues = group_words(self.words(texts), max_chars=20)
        assert " ".join(c.text for c in cues).split() == texts

    def test_empty_input(self):
        assert group_words([]) == []


class TestCaptionOutput:
    cues = [Cue(0.0, 1.5, "Hello there", [Word("Hello", 0.0, 0.7), Word("there", 0.8, 1.5)])]

    def test_srt_structure(self):
        srt = to_srt(self.cues)
        assert srt.startswith("1\n")
        assert "00:00:00,000 --> 00:00:01,500" in srt
        assert "Hello there" in srt

    def test_zero_length_cue_gets_minimum_duration(self):
        srt = to_srt([Cue(1.0, 1.0, "x")])
        assert "00:00:01,000 --> 00:00:01,100" in srt

    def test_ass_has_header_and_dialogue(self):
        ass = to_ass(self.cues)
        assert "[V4+ Styles]" in ass
        assert "Dialogue: 0,0:00:00.00" in ass

    def test_ass_karaoke_tags_per_word(self):
        ass = to_ass(self.cues, karaoke=True)
        assert ass.count("\\k") == 2

    def test_ass_without_karaoke_is_plain(self):
        assert "\\k" not in to_ass(self.cues, karaoke=False)


class TestFiltergraph:
    def test_minimal_graph(self):
        graph = build_video_filtergraph(
            1920, 1080, avatar_input=1, background_input=0, overlays=[],
            overlay_input_start=3, avatar_box=(100, 50, 800, 900),
        )
        assert "scale=1920:1080" in graph
        assert "crop=800:900" in graph
        assert graph.endswith("[vout]")

    def test_avatar_covers_box_without_distortion(self):
        graph = build_video_filtergraph(
            1920, 1080, 1, 0, [], 3, (0, 0, 640, 480),
        )
        assert "force_original_aspect_ratio=increase" in graph

    def test_overlay_inputs_are_indexed_from_start(self):
        overlays = [
            Overlay("a.png", 1.0, 4.0, "10", "20"),
            Overlay("b.png", 5.0, 8.0, "10", "20"),
        ]
        graph = build_video_filtergraph(1920, 1080, 1, 0, overlays, 3, (0, 0, 100, 100))
        assert "[3:v]" in graph
        assert "[4:v]" in graph

    def test_overlay_enable_windows(self):
        overlays = [Overlay("a.png", 1.5, 4.25)]
        graph = build_video_filtergraph(1920, 1080, 1, 0, overlays, 3, (0, 0, 100, 100))
        assert "between(t,1.500,4.250)" in graph

    def test_overlay_chain_labels_are_sequential(self):
        overlays = [Overlay(f"{i}.png", i, i + 1) for i in range(3)]
        graph = build_video_filtergraph(1920, 1080, 1, 0, overlays, 3, (0, 0, 100, 100))
        assert "[v3]" in graph and "[v4]" not in graph

    def test_ass_path_colons_are_escaped(self):
        graph = build_video_filtergraph(
            1920, 1080, 1, 0, [], 3, (0, 0, 100, 100), ass_path="C:/x/captions.ass"
        )
        assert r"C\:/x/captions.ass" in graph

    def test_no_captions_ends_with_null(self):
        graph = build_video_filtergraph(1920, 1080, 1, 0, [], 3, (0, 0, 100, 100))
        assert "null[vout]" in graph


class TestAvatarChunkPlanning:
    def test_exact_multiple(self):
        assert plan_chunks(240.0, 120) == [(0.0, 120.0), (120.0, 120.0)]

    def test_short_audio_is_one_chunk(self):
        assert plan_chunks(30.0, 120) == [(0.0, 30.0)]

    def test_tiny_remainder_folds_into_previous(self):
        # A 5s tail would be almost entirely model warm-up wobble.
        chunks = plan_chunks(245.0, 120)
        assert len(chunks) == 2
        assert chunks[-1] == (120.0, 125.0)

    def test_substantial_remainder_stays_separate(self):
        chunks = plan_chunks(300.0, 120)
        assert len(chunks) == 3
        assert chunks[-1] == (240.0, 60.0)

    def test_chunks_cover_full_duration(self):
        for duration in (30.0, 245.0, 300.0, 1500.0):
            chunks = plan_chunks(duration, 120)
            assert sum(length for _, length in chunks) == pytest.approx(duration)
            assert chunks[0][0] == 0.0

    def test_zero_duration(self):
        assert plan_chunks(0, 120) == []


class TestBrollMatching:
    def test_overlap_counted(self):
        assert keyword_score("town council meeting", "town-council-chamber") == 2

    def test_stopwords_ignored(self):
        assert keyword_score("the flag of the post", "flag") == 1

    def test_no_overlap(self):
        assert keyword_score("residents parade", "franklin-street-traffic") == 0


class TestFfmpegCommands:
    def test_concat_without_crossfade(self):
        command = build_concat_command(["a.wav", "b.wav"], "out.wav", 0, 24000)
        assert "concat=n=2:v=0:a=1[out]" in " ".join(command)

    def test_concat_with_crossfade_chains(self):
        command = build_concat_command(["a.wav", "b.wav", "c.wav"], "out.wav", 40, 24000)
        graph = " ".join(command)
        assert graph.count("acrossfade") == 2
        assert "d=0.04" in graph

    def test_crossfade_final_label_is_out(self):
        command = build_concat_command(["a.wav", "b.wav"], "out.wav", 40, 24000)
        assert "[out]" in " ".join(command)

    def test_loudnorm_targets(self):
        command = build_loudnorm_command("in.wav", "out.wav", -16.0, -1.5, 7.0, "pcm_s16le", 48000)
        assert "loudnorm=I=-16.0:TP=-1.5:LRA=7.0" in " ".join(command)


class TestFeed:
    episodes = [
        {
            "id": "2026-08-08", "date": "2026-08-08", "number": 12,
            "title": "Budget night", "description": "What happened.",
            "duration": 1265, "audio_url": "https://m.example.com/2026-08-08.mp3",
            "audio_bytes": 12345678,
        }
    ]

    def test_feed_is_valid_xml(self):
        root = ET.fromstring(build_feed(make_config(), self.episodes))
        assert root.tag == "rss"

    def test_channel_metadata(self):
        root = ET.fromstring(build_feed(make_config(), self.episodes))
        assert root.find("channel/title").text == "Test Show"

    def test_item_enclosure(self):
        root = ET.fromstring(build_feed(make_config(), self.episodes))
        enclosure = root.find("channel/item/enclosure")
        assert enclosure.get("type") == "audio/mpeg"
        assert enclosure.get("length") == "12345678"

    def test_itunes_duration(self):
        feed = build_feed(make_config(), self.episodes)
        assert "00:21:05" in feed

    def test_episodes_without_audio_are_skipped(self):
        broken = [{"id": "x", "date": "2026-08-08", "title": "no audio"}]
        root = ET.fromstring(build_feed(make_config(), broken))
        assert root.find("channel/item") is None

    def test_rfc2822_dates(self):
        assert format_rfc2822("2026-08-08").startswith("Sat, 08 Aug 2026")

    def test_bad_date_falls_back_to_now(self):
        assert format_rfc2822("not a date")  # does not raise


class TestYoutubeDescription:
    def test_includes_sources(self):
        text = build_youtube_description("Body.", ["https://a.com/1"], "https://site")
        assert "Sources:" in text
        assert "https://a.com/1" in text

    def test_caps_source_list(self):
        urls = [f"https://a.com/{i}" for i in range(50)]
        text = build_youtube_description("Body.", urls, "https://site", max_sources=5)
        assert text.count("https://a.com/") == 5

    def test_no_sources_block_when_empty(self):
        assert "Sources:" not in build_youtube_description("Body.", [], "https://site")


class TestBumperJoin:
    """A client with no intro/outro still has to end up with a finished episode
    at the expected path. Returning the intermediate under its own name made a
    successful encode report "assembly produced no output".
    """

    def _config(self, tmp_path):
        from podcastpipe.config import load_config

        (tmp_path / "config").mkdir(parents=True, exist_ok=True)
        (tmp_path / "config" / "show.yaml").write_text(
            "show:\n  name: Bumper Test\n", encoding="utf-8"
        )
        return load_config(tmp_path / "config" / "show.yaml")

    def test_no_bumpers_still_produces_the_output_path(self, tmp_path, monkeypatch):
        from podcastpipe.stages.assemble import concat_with_bumpers

        monkeypatch.delenv("PODCASTPIPE_CLIENT", raising=False)
        monkeypatch.delenv("PODCASTPIPE_CONFIG", raising=False)
        config = self._config(tmp_path)

        body = tmp_path / "body.mp4"
        body.write_bytes(b"rendered episode")
        final = tmp_path / "episode.mp4"

        result = concat_with_bumpers(body, final, config, None, None)

        assert Path(result) == final
        assert final.exists(), "the deliverable must exist at the requested path"
        assert final.read_bytes() == b"rendered episode"
        assert not body.exists(), "the intermediate should not be left behind"

    def test_missing_bumper_files_are_ignored(self, tmp_path, monkeypatch):
        from podcastpipe.stages.assemble import concat_with_bumpers

        monkeypatch.delenv("PODCASTPIPE_CLIENT", raising=False)
        monkeypatch.delenv("PODCASTPIPE_CONFIG", raising=False)
        config = self._config(tmp_path)

        body = tmp_path / "body.mp4"
        body.write_bytes(b"x")
        final = tmp_path / "episode.mp4"

        # Paths that do not exist must not trigger a concat -- ffmpeg is never
        # invoked here, so a wrong branch would fail the test outright.
        result = concat_with_bumpers(
            body, final, config, tmp_path / "nope_intro.mp4", tmp_path / "nope_outro.mp4"
        )
        assert Path(result) == final and final.exists()


class TestAvatarBatching:
    """One MuseTalk invocation per worker instead of one per window.

    scripts.inference reloads every model on each start, so the old
    one-window-per-request design paid that cost for every chunk of every
    episode. Batching pays it once per worker. The split has to keep every
    window exactly once -- a dropped window is a silent gap in the episode.
    """

    def _jobs(self, count):
        return [{"id": f"chunk_{i:03d}"} for i in range(count)]

    def test_every_window_appears_exactly_once(self):
        from podcastpipe.stages.avatar import split_batches

        for count in (1, 2, 3, 4, 7, 15, 40):
            for width in (1, 2, 3, 5):
                batches = split_batches(self._jobs(count), width)
                flat = [j["id"] for b in batches for j in b]
                assert sorted(flat) == sorted(j["id"] for j in self._jobs(count)), (
                    f"{count} windows across {width} workers lost or duplicated one"
                )

    def test_uses_every_worker_when_there_is_work_for_them(self):
        from podcastpipe.stages.avatar import split_batches

        assert len(split_batches(self._jobs(15), 2)) == 2
        assert len(split_batches(self._jobs(4), 4)) == 4

    def test_never_makes_more_batches_than_windows(self):
        from podcastpipe.stages.avatar import split_batches

        # Three idle workers and one window is one batch, not one batch and two
        # empty requests that would each pay a model load for nothing.
        assert len(split_batches(self._jobs(1), 3)) == 1
        assert len(split_batches(self._jobs(2), 5)) == 2

    def test_batches_are_balanced(self):
        from podcastpipe.stages.avatar import split_batches

        sizes = [len(b) for b in split_batches(self._jobs(15), 2)]
        assert max(sizes) - min(sizes) <= 1, f"lopsided split {sizes}"

    def test_no_windows_is_no_batches(self):
        from podcastpipe.stages.avatar import split_batches

        assert split_batches([], 2) == []
