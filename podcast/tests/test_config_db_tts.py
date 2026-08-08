"""Config loading, episode state, and the TTS planning/caching logic.

The TTS tests matter because the cache key is what decides whether editing one
paragraph re-synthesizes one paragraph or the whole episode."""

import os
from pathlib import Path

import pytest
import yaml

from podcastpipe.config import ConfigError, expand_env, load_config
from podcastpipe.db import Database
from podcastpipe.models import Block, Episode, Script, Segment
from podcastpipe.stages.tts import chunk_hash, pending_jobs, plan_chunks


@pytest.fixture
def project(tmp_path):
    """A minimal project tree with a valid show.yaml."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "show.yaml").write_text(
        yaml.safe_dump(
            {
                "show": {"name": "Test Show", "target_minutes": 10},
                "tts": {"max_chars_per_chunk": 100, "reference_audio": "assets/ref.wav"},
                "gpu": {"avatar": 0, "tts": 1, "encode": 2},
                "pronunciations": {"Post 6": "Post Six"},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class TestConfigLoading:
    def test_loads_sections(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert config.show.name == "Test Show"
        assert config.tts.max_chars_per_chunk == 100

    def test_defaults_fill_missing_sections(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert config.video.width == 1920
        assert config.audio.podcast_lufs == -16.0

    def test_root_is_project_directory(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert config.root == project.resolve()

    def test_relative_paths_resolve_against_root(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert config.path("assets/ref.wav") == project.resolve() / "assets/ref.wav"

    def test_absolute_paths_pass_through(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert config.path("/tmp/x.wav") == Path("/tmp/x.wav")

    def test_extra_config_survives_in_raw(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert config.raw["pronunciations"]["Post 6"] == "Post Six"

    def test_unknown_key_is_rejected(self, project):
        path = project / "config" / "show.yaml"
        path.write_text(yaml.safe_dump({"show": {"nmae": "typo"}}), encoding="utf-8")
        with pytest.raises(ConfigError, match="nmae"):
            load_config(path)

    def test_env_expansion(self):
        os.environ["PODCASTPIPE_TEST_VAR"] = "secret"
        try:
            assert expand_env({"k": "${PODCASTPIPE_TEST_VAR}"}) == {"k": "secret"}
            assert expand_env(["${PODCASTPIPE_TEST_VAR}"]) == ["secret"]
        finally:
            del os.environ["PODCASTPIPE_TEST_VAR"]

    def test_unset_env_becomes_empty(self):
        assert expand_env("${DEFINITELY_NOT_SET_12345}") == ""


class TestGpuAssignment:
    def test_env_overlay_pins_single_device(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert config.gpu.env_for("avatar") == {"CUDA_VISIBLE_DEVICES": "0"}
        assert config.gpu.env_for("tts") == {"CUDA_VISIBLE_DEVICES": "1"}
        assert config.gpu.env_for("encode") == {"CUDA_VISIBLE_DEVICES": "2"}

    def test_unknown_stage_raises(self, project):
        config = load_config(project / "config" / "show.yaml")
        with pytest.raises(ConfigError):
            config.gpu.env_for("nonexistent")


class TestDatabase:
    def test_episode_round_trip(self, tmp_path):
        with Database(tmp_path / "p.db") as db:
            db.upsert_episode(Episode(id="2026-08-08", date="2026-08-08", title="T"))
            loaded = db.get_episode("2026-08-08")
            assert loaded.title == "T"
            assert loaded.number == 1

    def test_episode_numbers_increment(self, tmp_path):
        with Database(tmp_path / "p.db") as db:
            db.upsert_episode(Episode(id="2026-08-08", date="2026-08-08"))
            db.upsert_episode(Episode(id="2026-08-09", date="2026-08-09"))
            assert db.get_episode("2026-08-09").number == 2

    def test_upsert_preserves_number(self, tmp_path):
        with Database(tmp_path / "p.db") as db:
            db.upsert_episode(Episode(id="x", date="2026-08-08"))
            db.upsert_episode(Episode(id="x", date="2026-08-08", title="Updated", number=1))
            assert db.get_episode("x").number == 1
            assert db.get_episode("x").title == "Updated"

    def test_artifacts_accumulate(self, tmp_path):
        with Database(tmp_path / "p.db") as db:
            db.upsert_episode(Episode(id="x", date="2026-08-08"))
            db.set_artifact("x", "mp3", "/a.mp3")
            db.set_artifact("x", "video", "/a.mp4")
            assert db.get_episode("x").artifacts == {"mp3": "/a.mp3", "video": "/a.mp4"}

    def test_artifact_on_unknown_episode_raises(self, tmp_path):
        with Database(tmp_path / "p.db") as db:
            with pytest.raises(KeyError):
                db.set_artifact("nope", "k", "v")

    def test_seen_urls_within_window(self, tmp_path):
        from podcastpipe.models import Story

        with Database(tmp_path / "p.db") as db:
            db.upsert_episode(Episode(id="2026-08-08", date="2026-08-08"))
            db.add_stories("2026-08-08", [Story(title="A", url="https://a.com/1", source="s")])
            assert "https://a.com/1" in db.seen_urls(within_days=3650)

    def test_seen_urls_excludes_old_episodes(self, tmp_path):
        from podcastpipe.models import Story

        with Database(tmp_path / "p.db") as db:
            db.upsert_episode(Episode(id="2001-01-01", date="2001-01-01"))
            db.add_stories("2001-01-01", [Story(title="A", url="https://a.com/1", source="s")])
            assert db.seen_urls(within_days=21) == set()

    def test_events_logged_in_order(self, tmp_path):
        with Database(tmp_path / "p.db") as db:
            db.upsert_episode(Episode(id="x", date="2026-08-08"))
            db.log("x", "research", "ok")
            db.log("x", "script", "ok")
            assert [e["stage"] for e in db.events("x")] == ["research", "script"]


def make_script(texts):
    return Script(
        episode_id="2026-08-08",
        title="T",
        segments=[
            Segment(
                id="s",
                name="S",
                blocks=[Block(id=f"s-{i + 1}", text=t) for i, t in enumerate(texts)],
            )
        ],
    )


class TestTtsPlanning:
    def test_blocks_expand_into_chunks(self, project, tmp_path):
        config = load_config(project / "config" / "show.yaml")
        script = make_script([" ".join(f"Sentence number {i}." for i in range(10))])
        jobs = plan_chunks(script, config, tmp_path)
        assert len(jobs) > 1
        assert all(job.block_id == "s-1" for job in jobs)

    def test_chunk_ids_carry_block_and_index(self, project, tmp_path):
        config = load_config(project / "config" / "show.yaml")
        jobs = plan_chunks(make_script(["Hello there."]), config, tmp_path)
        assert jobs[0].id.startswith("s-1-000-")

    def test_pronunciations_applied(self, project, tmp_path):
        config = load_config(project / "config" / "show.yaml")
        jobs = plan_chunks(make_script(["Post 6 meets tonight."]), config, tmp_path)
        assert "Post Six" in jobs[0].text

    def test_identical_text_shares_a_cache_path(self, project, tmp_path):
        config = load_config(project / "config" / "show.yaml")
        jobs = plan_chunks(make_script(["Same line.", "Same line."]), config, tmp_path)
        assert jobs[0].out_path == jobs[1].out_path

    def test_repeated_text_is_synthesized_once(self, project, tmp_path):
        config = load_config(project / "config" / "show.yaml")
        jobs = plan_chunks(make_script(["Same line.", "Same line."]), config, tmp_path)
        assert len(pending_jobs(jobs)) == 1

    def test_cached_chunks_are_skipped(self, project, tmp_path):
        config = load_config(project / "config" / "show.yaml")
        jobs = plan_chunks(make_script(["One line.", "Another line."]), config, tmp_path)
        Path(jobs[0].out_path).write_bytes(b"fake wav")
        assert len(pending_jobs(jobs)) == len(jobs) - 1

    def test_editing_one_block_leaves_others_cached(self, project, tmp_path):
        config = load_config(project / "config" / "show.yaml")
        before = plan_chunks(make_script(["Line one.", "Line two."]), config, tmp_path)
        after = plan_chunks(make_script(["Line one.", "Line two, edited."]), config, tmp_path)
        assert before[0].out_path == after[0].out_path   # untouched block reuses audio
        assert before[1].out_path != after[1].out_path   # edited block does not


class TestChunkHash:
    def test_same_text_same_hash(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert chunk_hash("hello", config) == chunk_hash("hello", config)

    def test_different_text_different_hash(self, project):
        config = load_config(project / "config" / "show.yaml")
        assert chunk_hash("hello", config) != chunk_hash("hello!", config)

    def test_voice_parameters_invalidate_cache(self, project):
        config = load_config(project / "config" / "show.yaml")
        original = chunk_hash("hello", config)
        config.tts.exaggeration = 0.9
        assert chunk_hash("hello", config) != original

    def test_reference_audio_invalidates_cache(self, project):
        config = load_config(project / "config" / "show.yaml")
        original = chunk_hash("hello", config)
        config.tts.reference_audio = "assets/other.wav"
        assert chunk_hash("hello", config) != original
