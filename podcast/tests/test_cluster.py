"""Tests for cluster dispatch and the streaming scheduler.

Both are exercised with injected transports and fake stage functions, so the
distribution logic is verified without a network or a GPU. This is the layer
where a bug costs the most: a silently misrouted job wastes 40 minutes of render
time on the wrong card, and a scheduling bug produces a video with a gap in it.
"""

import threading
import time

import pytest

from podcastpipe.cluster import (
    ClusterClient,
    ClusterError,
    Node,
    NodeState,
    NoHealthyNode,
    load_nodes,
    select_node,
    translate_path,
    translate_payload,
)
from podcastpipe.scheduler import StreamingScheduler, Window, build_windows


class TestPathTranslation:
    def test_local_path_becomes_container_path(self):
        assert translate_path("/mnt/user/podcast/work/a.wav", "/mnt/user/podcast") == \
            "/pipeline/work/a.wav"

    def test_custom_container_root(self):
        assert translate_path("/srv/pod/x.wav", "/srv/pod", "/data") == "/data/x.wav"

    def test_already_translated_passes_through(self):
        # Safe to apply twice — the orchestrator and workers share a root.
        assert translate_path("/pipeline/work/a.wav", "/mnt/user/podcast") == \
            "/pipeline/work/a.wav"

    def test_root_itself(self):
        assert translate_path("/mnt/user/podcast", "/mnt/user/podcast") == "/pipeline"

    def test_trailing_slashes_tolerated(self):
        assert translate_path("/mnt/user/podcast/a", "/mnt/user/podcast/") == "/pipeline/a"

    def test_path_outside_share_is_rejected(self):
        # A worker on another host cannot reach /home/you/x.wav. Failing here
        # beats a FileNotFoundError 40 minutes into an episode.
        with pytest.raises(ClusterError, match="not under the shared root"):
            translate_path("/home/you/x.wav", "/mnt/user/podcast")

    def test_prefix_collision_is_not_a_match(self):
        with pytest.raises(ClusterError):
            translate_path("/mnt/user/podcast-other/x", "/mnt/user/podcast")


class TestPayloadTranslation:
    root = "/mnt/user/podcast"

    def test_known_path_keys_translated(self):
        payload = {"audio": f"{self.root}/a.wav", "out_path": f"{self.root}/b.mp4"}
        result = translate_payload(payload, self.root)
        assert result == {"audio": "/pipeline/a.wav", "out_path": "/pipeline/b.mp4"}

    def test_suffix_convention_translated(self):
        result = translate_payload({"base_path": f"{self.root}/x"}, self.root)
        assert result["base_path"] == "/pipeline/x"

    def test_non_path_values_untouched(self):
        payload = {"text": "Visit /mnt/user/podcast today", "fps": 25, "id": "w1"}
        assert translate_payload(payload, self.root) == payload

    def test_urls_not_mangled(self):
        payload = {"source_url": "https://chapelboro.com/x"}
        assert translate_payload(payload, self.root) == payload

    def test_nested_structures(self):
        payload = {"jobs": [{"audio": f"{self.root}/a.wav"}]}
        result = translate_payload(payload, self.root)
        assert result["jobs"][0]["audio"] == "/pipeline/a.wav"

    def test_empty_path_left_alone(self):
        assert translate_payload({"out_path": ""}, self.root) == {"out_path": ""}


def make_nodes():
    return [
        Node("avatar-a", "http://10.10.5.15:8081", "avatar"),
        Node("avatar-c", "http://10.10.5.17:8081", "avatar"),
        Node("tts-b", "http://10.10.5.16:8080", "tts"),
    ]


class TestNodeSelection:
    def test_picks_idle_node(self):
        nodes = make_nodes()[:2]
        state = {"avatar-a": NodeState(inflight=1), "avatar-c": NodeState(inflight=0)}
        assert select_node(nodes, state).name == "avatar-c"

    def test_skips_unhealthy(self):
        nodes = make_nodes()[:2]
        state = {"avatar-a": NodeState(healthy=False), "avatar-c": NodeState()}
        assert select_node(nodes, state).name == "avatar-c"

    def test_skips_disabled(self):
        nodes = make_nodes()[:2]
        nodes[0].enabled = False
        assert select_node(nodes, {}).name == "avatar-c"

    def test_returns_none_when_all_busy(self):
        nodes = make_nodes()[:2]
        state = {n.name: NodeState(inflight=1) for n in nodes}
        assert select_node(nodes, state) is None

    def test_respects_max_concurrent(self):
        node = Node("a", "http://x", "avatar", max_concurrent=2)
        assert select_node([node], {"a": NodeState(inflight=1)}) is not None
        assert select_node([node], {"a": NodeState(inflight=2)}) is None

    def test_selection_is_deterministic(self):
        nodes = make_nodes()[:2]
        picks = {select_node(nodes, {}).name for _ in range(20)}
        assert len(picks) == 1

    def test_weight_breaks_ties(self):
        nodes = [
            Node("a", "http://a", "avatar", weight=1),
            Node("b", "http://b", "avatar", weight=5),
        ]
        assert select_node(nodes, {}).name == "b"


class FakeTransport:
    """Records posts and replays scripted outcomes."""

    def __init__(self, outcomes=None, delay=0.0):
        self.calls = []
        self.outcomes = outcomes or {}
        self.delay = delay
        self.lock = threading.Lock()

    def post(self, url, payload, timeout):
        with self.lock:
            self.calls.append((url, payload))
        if self.delay:
            time.sleep(self.delay)
        outcome = self.outcomes.get(url)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if outcome is not None else {"ok": True}

    def get(self, url, timeout):
        outcome = self.outcomes.get(url)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if outcome is not None else {"ok": True, "model_loaded": True}


def make_client(nodes=None, transport=None, **kwargs):
    transport = transport or FakeTransport()
    return ClusterClient(
        nodes or make_nodes(),
        shared_root_local="/mnt/user/podcast",
        poster=transport.post,
        getter=transport.get,
        **kwargs,
    ), transport


class TestDispatch:
    def test_job_reaches_a_node_of_the_right_role(self):
        client, transport = make_client()
        client.submit("tts", {"text": "hi"})
        assert transport.calls[0][0] == "http://10.10.5.16:8080/run"

    def test_payload_paths_are_translated_before_sending(self):
        client, transport = make_client()
        client.submit("avatar", {"audio": "/mnt/user/podcast/w.wav"})
        assert transport.calls[0][1]["audio"] == "/pipeline/w.wav"

    def test_result_records_which_node_ran_it(self):
        client, _ = make_client()
        assert client.submit("tts", {"text": "hi"})["node"] == "tts-b"

    def test_unknown_role_raises(self):
        client, _ = make_client()
        with pytest.raises(NoHealthyNode, match="no nodes configured"):
            client.submit("nonexistent", {})

    def test_failure_retries_on_another_node(self):
        transport = FakeTransport({
            "http://10.10.5.15:8081/run": RuntimeError("card fell over"),
        })
        client, _ = make_client(transport=transport)
        result = client.submit("avatar", {}, attempts=2)
        assert result["node"] == "avatar-c"

    def test_all_nodes_failing_raises(self):
        transport = FakeTransport({
            "http://10.10.5.15:8081/run": RuntimeError("boom"),
            "http://10.10.5.17:8081/run": RuntimeError("boom"),
        })
        client, _ = make_client(transport=transport)
        with pytest.raises(ClusterError, match="failed on all attempts"):
            client.submit("avatar", {}, attempts=2)

    def test_repeated_failures_mark_a_node_unhealthy(self):
        transport = FakeTransport({"http://10.10.5.16:8080/run": RuntimeError("boom")})
        client, _ = make_client(transport=transport, max_failures=2)
        for _ in range(2):
            with pytest.raises(ClusterError):
                client.submit("tts", {}, attempts=1)
        assert client.state["tts-b"].healthy is False

    def test_no_healthy_node_raises_immediately(self):
        client, _ = make_client()
        client.state["tts-b"].healthy = False
        with pytest.raises(NoHealthyNode, match="no healthy node"):
            client.submit("tts", {}, max_wait=0.1)

    def test_inflight_released_after_success(self):
        client, _ = make_client()
        client.submit("tts", {"text": "hi"})
        assert client.state["tts-b"].inflight == 0

    def test_inflight_released_after_failure(self):
        transport = FakeTransport({"http://10.10.5.16:8080/run": RuntimeError("boom")})
        client, _ = make_client(transport=transport)
        with pytest.raises(ClusterError):
            client.submit("tts", {}, attempts=1)
        assert client.state["tts-b"].inflight == 0

    def test_concurrent_submits_spread_across_nodes(self):
        transport = FakeTransport(delay=0.05)
        client, _ = make_client(transport=transport)
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(client.submit("avatar", {})))
            for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert {r["node"] for r in results} == {"avatar-a", "avatar-c"}


class TestHealth:
    def test_healthy_nodes_reported(self):
        client, _ = make_client()
        report = client.check_health()
        assert all(entry["ok"] for entry in report.values())

    def test_unreachable_node_reported_with_reason(self):
        transport = FakeTransport({
            "http://10.10.5.16:8080/health": OSError("connection refused"),
        })
        client, _ = make_client(transport=transport)
        report = client.check_health()
        assert report["tts-b"]["ok"] is False
        assert "connection refused" in report["tts-b"]["detail"]

    def test_reachable_but_still_loading_is_not_ok(self):
        transport = FakeTransport({
            "http://10.10.5.16:8080/health": {"ok": False, "model_loaded": False},
        })
        client, _ = make_client(transport=transport)
        assert client.check_health()["tts-b"]["ok"] is False

    def test_disabled_node_reported_without_probing(self):
        nodes = make_nodes()
        nodes[0].enabled = False
        client, transport = make_client(nodes=nodes)
        report = client.check_health()
        assert report["avatar-a"]["detail"] == "disabled in config"

    def test_capacity_counts_only_healthy(self):
        client, _ = make_client()
        assert client.capacity("avatar") == 2
        client.state["avatar-a"].healthy = False
        assert client.capacity("avatar") == 1


class TestLoadNodes:
    def test_builds_nodes(self):
        nodes = load_nodes({"a": {"url": "http://x", "role": "tts", "gpu": "3090"}})
        assert nodes[0].name == "a" and nodes[0].role == "tts"

    def test_defaults_applied(self):
        node = load_nodes({"a": {"url": "http://x", "role": "tts"}})[0]
        assert node.max_concurrent == 1 and node.enabled is True

    def test_missing_field_raises(self):
        with pytest.raises(ClusterError, match="missing role"):
            load_nodes({"a": {"url": "http://x"}})

    def test_empty_config(self):
        assert load_nodes({}) == []


class TestWindowBuilding:
    def test_groups_chunks_to_window_length(self):
        windows = build_windows([20.0] * 12, window_seconds=120)
        assert len(windows) == 2
        assert len(windows[0].chunk_indices) == 6

    def test_chunks_are_never_split(self):
        windows = build_windows([50.0, 50.0, 50.0], window_seconds=120)
        all_indices = [i for w in windows for i in w.chunk_indices]
        assert all_indices == [0, 1, 2]

    def test_short_tail_merged_into_previous(self):
        windows = build_windows([20.0] * 6 + [5.0], window_seconds=120)
        assert len(windows) == 1
        assert len(windows[0].chunk_indices) == 7

    def test_substantial_tail_kept_separate(self):
        windows = build_windows([20.0] * 6 + [60.0], window_seconds=120)
        assert len(windows) == 2

    def test_windows_are_contiguous_in_time(self):
        windows = build_windows([20.0] * 12, window_seconds=120)
        assert windows[0].start == 0.0
        assert windows[1].start == pytest.approx(windows[0].end)

    def test_total_duration_preserved(self):
        durations = [17.0, 22.0, 19.0, 31.0, 14.0, 28.0, 25.0]
        windows = build_windows(durations, window_seconds=60)
        assert sum(w.duration for w in windows) == pytest.approx(sum(durations))

    def test_single_short_chunk(self):
        windows = build_windows([5.0], window_seconds=120)
        assert len(windows) == 1

    def test_empty(self):
        assert build_windows([], window_seconds=120) == []


class TestStreamingScheduler:
    def _scheduler(self, chunk_duration=20.0, **kwargs):
        order = []

        def synthesize(index):
            order.append(("voice", index))
            return chunk_duration

        def render(window):
            order.append(("render", window.index))
            return f"/pipeline/avatar_{window.index:03d}.mp4"

        scheduler = StreamingScheduler(
            synthesize=synthesize, render=render, window_seconds=120.0, **kwargs
        )
        return scheduler, order

    def test_returns_rendered_paths_in_order(self):
        scheduler, _ = self._scheduler()
        paths = scheduler.run(12)
        assert paths == [
            "/pipeline/avatar_000.mp4",
            "/pipeline/avatar_001.mp4",
        ]

    def test_render_starts_before_all_voice_finishes(self):
        # The entire point of the two-machine split: the first window renders
        # while later chunks are still being synthesized.
        scheduler, order = self._scheduler()
        scheduler.run(12)
        first_render = next(i for i, e in enumerate(order) if e[0] == "render")
        last_voice = max(i for i, e in enumerate(order) if e[0] == "voice")
        assert first_render < last_voice

    def test_every_chunk_lands_in_exactly_one_window(self):
        scheduler, _ = self._scheduler()
        scheduler.run(13)
        covered = sorted(i for w in scheduler.windows for i in w.chunk_indices)
        assert covered == list(range(13))

    def test_windows_cover_the_full_timeline(self):
        scheduler, _ = self._scheduler()
        scheduler.run(12)
        assert scheduler.windows[0].start == 0.0
        assert scheduler.windows[1].start == pytest.approx(scheduler.windows[0].end)

    def test_short_episode_makes_one_window(self):
        scheduler, _ = self._scheduler()
        assert len(scheduler.run(2)) == 1

    def test_zero_chunks(self):
        scheduler, _ = self._scheduler()
        assert scheduler.run(0) == []

    def test_voice_failure_surfaces(self):
        def synthesize(index):
            if index == 3:
                raise RuntimeError("chatterbox died")
            return 20.0

        scheduler = StreamingScheduler(
            synthesize=synthesize, render=lambda w: "x", window_seconds=120.0
        )
        with pytest.raises(RuntimeError, match="voice chunk 3"):
            scheduler.run(8)

    def test_render_failure_surfaces(self):
        def render(window):
            raise RuntimeError("musetalk died")

        scheduler = StreamingScheduler(
            synthesize=lambda i: 20.0, render=render, window_seconds=120.0
        )
        with pytest.raises(RuntimeError, match="avatar window 0"):
            scheduler.run(12)

    def test_parallel_avatar_workers_overlap(self):
        started = []
        lock = threading.Lock()

        def render(window):
            with lock:
                started.append(window.index)
            time.sleep(0.15)
            return f"/x/{window.index}.mp4"

        scheduler = StreamingScheduler(
            synthesize=lambda i: 20.0, render=render,
            window_seconds=120.0, avatar_workers=2,
        )
        began = time.monotonic()
        scheduler.run(18)  # 3 windows
        elapsed = time.monotonic() - began
        # Serial would be ~0.45s; with two workers it should be well under.
        assert elapsed < 0.40
        assert len(started) == 3

    def test_events_are_emitted(self):
        events = []
        scheduler = StreamingScheduler(
            synthesize=lambda i: 20.0,
            render=lambda w: "/x.mp4",
            window_seconds=120.0,
            on_event=lambda kind, data: events.append(kind),
        )
        scheduler.run(12)
        assert "voice_chunk" in events
        assert "window_ready" in events
        assert "window_rendered" in events


class TestConcurrentDispatch:
    """The avatar stage submits from several threads so both render nodes stay
    busy. Selecting a node and claiming its slot has to be one atomic step --
    otherwise two threads see the same idle node, and half the cluster sits out
    the longest stage of the episode.

    The timing tests below verify that concurrent dispatch works end to end.
    They do NOT reliably catch the select-then-claim race: that window is
    microseconds wide, far narrower than thread-startup jitter, so it survives
    mutation. ``test_await_node_claims_the_slot_it_returns`` is the one that
    actually pins the invariant, deterministically and without threads.
    """

    def test_await_node_claims_the_slot_it_returns(self):
        client, _ = make_client()

        first = client._await_node("avatar", 0.01, 1.0)
        assert client.state[first.name].inflight == 1, (
            "_await_node must claim the slot before returning; a caller that "
            "increments afterwards leaves a window for a second thread to pick "
            "the same node"
        )

        # The claim is what makes the second caller pick the other node.
        second = client._await_node("avatar", 0.01, 1.0)
        assert second.name != first.name
        assert client.state[second.name].inflight == 1

        # Both slots are now taken, so a third caller waits and then gives up
        # rather than double-booking a node that declared max_concurrent: 1.
        with pytest.raises(ClusterError):
            client._await_node("avatar", 0.01, 0.05)

    def test_two_threads_land_on_different_nodes(self):
        # Both avatar nodes declare max_concurrent=1. A job that is still in
        # flight must make its node ineligible for the other thread.
        transport = FakeTransport(delay=0.15)
        client, _ = make_client(transport=transport)

        urls = []
        lock = threading.Lock()

        def dispatch(index):
            result = client.submit("avatar", {"id": index}, timeout=5)
            with lock:
                urls.append(result["node"])

        threads = [threading.Thread(target=dispatch, args=(i,)) for i in range(2)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        elapsed = time.monotonic() - started

        assert sorted(urls) == ["avatar-a", "avatar-c"]
        # Ran concurrently rather than queued: two 0.15s jobs in series would
        # take 0.30s, and the loser would also have paid a 2s poll interval.
        assert elapsed < 0.30

    def test_inflight_returns_to_zero_after_concurrent_jobs(self):
        transport = FakeTransport(delay=0.02)
        client, _ = make_client(transport=transport)

        threads = [
            threading.Thread(target=lambda: client.submit("avatar", {}, timeout=5))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert [s.inflight for s in client.state.values()] == [0, 0, 0]

    def test_capacity_counts_slots_for_the_role(self):
        client, _ = make_client()
        assert client.capacity("avatar") == 2
        assert client.capacity("tts") == 1
        assert client.capacity("media") == 0
