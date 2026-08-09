"""Multi-tenant config: profile merging, path isolation, tenant separation.

This pipeline runs many clients off one codebase, so the failure that matters
most is cross-tenant leakage — one client's voice reference, feeds, YouTube
credentials or episode files reaching another's show. These tests pin the
isolation boundaries.
"""

import pytest
import yaml

from podcastpipe.config import ConfigError, deep_merge, list_clients, load_config


@pytest.fixture
def project(tmp_path):
    """A project with a neutral base config and two client profiles."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "show.yaml").write_text(
        yaml.safe_dump(
            {
                "show": {"name": "Untitled Show", "target_minutes": 18, "style_guide": ""},
                "video": {"encoder": "h264_nvenc", "accent": "#2F6F7E"},
                "audio": {"podcast_lufs": -16.0},
                "tts": {"engine": "chatterbox", "exaggeration": 0.45},
                "publish": {"youtube_privacy": "private"},
                "pronunciations": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config" / "sources.yaml").write_text(
        yaml.safe_dump({"feeds": [], "keywords": {}}), encoding="utf-8"
    )

    for name, spec in {
        "acme": {
            "show": {"name": "Acme Daily", "target_minutes": 12},
            "video": {"accent": "#FF6600"},
            "pronunciations": {"Acme": "ACK-me"},
        },
        "globex": {
            "show": {"name": "Globex Report"},
            "tts": {"exaggeration": 0.7},
        },
    }.items():
        client_dir = tmp_path / "clients" / name
        client_dir.mkdir(parents=True)
        (client_dir / "show.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")

    (tmp_path / "clients" / "acme" / "sources.yaml").write_text(
        yaml.safe_dump({"feeds": [{"name": "Acme Wire", "url": "https://acme.test/rss"}]}),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def base_config(project):
    return project / "config" / "show.yaml"


class TestDeepMerge:
    def test_override_wins(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_base_keys_survive(self):
        assert deep_merge({"a": 1, "b": 2}, {"a": 9}) == {"a": 9, "b": 2}

    def test_nested_maps_merge(self):
        result = deep_merge({"s": {"x": 1, "y": 2}}, {"s": {"y": 9}})
        assert result == {"s": {"x": 1, "y": 9}}

    def test_lists_replace_not_append(self):
        # A client redefining segments means "this is my lineup", not "append
        # mine to the default one" — appending would give them both.
        result = deep_merge({"segs": [1, 2, 3]}, {"segs": [9]})
        assert result == {"segs": [9]}

    def test_deeply_nested(self):
        result = deep_merge({"a": {"b": {"c": 1, "d": 2}}}, {"a": {"b": {"c": 9}}})
        assert result["a"]["b"] == {"c": 9, "d": 2}

    def test_empty_override_is_identity(self):
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_none_override_tolerated(self):
        assert deep_merge({"a": 1}, None) == {"a": 1}

    def test_base_is_not_mutated(self):
        base = {"s": {"x": 1}}
        deep_merge(base, {"s": {"x": 2}})
        assert base == {"s": {"x": 1}}

    def test_type_change_replaces(self):
        assert deep_merge({"a": {"x": 1}}, {"a": "scalar"}) == {"a": "scalar"}


class TestClientDiscovery:
    def test_lists_profiles(self, project):
        assert list_clients(project) == ["acme", "globex"]

    def test_ignores_dirs_without_show_yaml(self, project):
        (project / "clients" / "not-a-client").mkdir()
        assert "not-a-client" not in list_clients(project)

    def test_no_clients_dir(self, tmp_path):
        assert list_clients(tmp_path) == []


class TestProfileLoading:
    def test_base_alone_is_unbranded(self, base_config):
        config = load_config(base_config)
        assert config.show.name == "Untitled Show"
        assert config.show.style_guide == ""
        assert config.client == ""

    def test_client_overrides_apply(self, base_config):
        config = load_config(base_config, "acme")
        assert config.show.name == "Acme Daily"
        assert config.show.target_minutes == 12

    def test_unstated_settings_inherit(self, base_config):
        config = load_config(base_config, "acme")
        assert config.video.encoder == "h264_nvenc"
        assert config.audio.podcast_lufs == -16.0
        assert config.tts.engine == "chatterbox"

    def test_clients_do_not_see_each_other(self, base_config):
        acme = load_config(base_config, "acme")
        globex = load_config(base_config, "globex")
        assert acme.video.accent == "#FF6600"
        assert globex.video.accent == "#2F6F7E"      # base default, not Acme's
        assert globex.show.target_minutes == 18      # base default, not Acme's 12
        assert acme.tts.exaggeration == 0.45         # base, not Globex's 0.7

    def test_unknown_client_lists_available(self, base_config):
        with pytest.raises(ConfigError, match="acme, globex"):
            load_config(base_config, "nonexistent")

    def test_client_from_environment(self, base_config, monkeypatch):
        monkeypatch.setenv("PODCASTPIPE_CLIENT", "acme")
        assert load_config(base_config).show.name == "Acme Daily"

    def test_argument_beats_environment(self, base_config, monkeypatch):
        monkeypatch.setenv("PODCASTPIPE_CLIENT", "acme")
        assert load_config(base_config, "globex").show.name == "Globex Report"


class TestIsolation:
    def test_work_dirs_are_namespaced(self, base_config):
        acme = load_config(base_config, "acme")
        globex = load_config(base_config, "globex")
        assert acme.work_dir != globex.work_dir
        assert acme.work_dir.name == "acme"

    def test_episodes_cannot_collide_across_clients(self, base_config):
        acme = load_config(base_config, "acme")
        globex = load_config(base_config, "globex")
        assert acme.episode_dir("2026-08-09") != globex.episode_dir("2026-08-09")

    def test_output_dirs_are_namespaced(self, base_config):
        assert load_config(base_config, "acme").output_dir.name == "acme"

    def test_base_config_is_not_namespaced(self, base_config):
        assert load_config(base_config).work_dir.name == "work"

    def test_client_assets_shadow_base(self, project, base_config):
        shared = project / "assets" / "voice"
        shared.mkdir(parents=True)
        (shared / "reference.wav").write_bytes(b"base")

        client_asset = project / "clients" / "acme" / "assets" / "voice"
        client_asset.mkdir(parents=True)
        (client_asset / "reference.wav").write_bytes(b"acme")

        resolved = load_config(base_config, "acme").path("assets/voice/reference.wav")
        assert resolved.read_bytes() == b"acme"

    def test_missing_client_asset_falls_back_to_shared(self, project, base_config):
        shared = project / "assets" / "brand"
        shared.mkdir(parents=True)
        (shared / "background.png").write_bytes(b"shared")

        resolved = load_config(base_config, "acme").path("assets/brand/background.png")
        assert resolved.read_bytes() == b"shared"

    def test_client_without_own_asset_never_gets_another_clients(
        self, project, base_config
    ):
        # The failure that matters: globex must not pick up acme's voice.
        acme_voice = project / "clients" / "acme" / "assets" / "voice"
        acme_voice.mkdir(parents=True)
        (acme_voice / "reference.wav").write_bytes(b"acme")

        resolved = load_config(base_config, "globex").path("assets/voice/reference.wav")
        assert not resolved.exists()
        assert "acme" not in str(resolved)

    def test_client_path_always_writes_inside_the_client_dir(self, base_config):
        config = load_config(base_config, "acme")
        assert config.client_path("youtube_token.json").parent.name == "acme"

    def test_sources_resolve_per_client(self, base_config):
        acme = load_config(base_config, "acme")
        assert acme.sources_path().parent.name == "acme"

    def test_client_without_sources_falls_back_to_base_template(self, base_config):
        globex = load_config(base_config, "globex")
        assert globex.sources_path().name == "sources.yaml"
        assert globex.sources_path().parent.name == "config"

    def test_label_names_the_client(self, base_config):
        assert load_config(base_config, "acme").label() == "Acme Daily [acme]"
        assert load_config(base_config).label() == "Untitled Show"


class TestShippedDefaults:
    """The repo's own base config must stay unbranded — a client name or domain
    leaking into it would apply to every tenant."""

    def test_base_config_carries_no_client_branding(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        text = (root / "config" / "show.yaml").read_text(encoding="utf-8").lower()
        for term in ("alpost6", "chapelhillpost6", "post 6", "american legion", "axiom"):
            assert term not in text, f"{term!r} must not appear in the base config"

    def test_base_sources_ship_no_feeds(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        data = yaml.safe_load((root / "config" / "sources.yaml").read_text()) or {}
        assert not data.get("feeds"), "base sources.yaml must ship zero feeds"
