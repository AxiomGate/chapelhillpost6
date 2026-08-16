"""Cluster client: dispatch work to always-warm GPU workers over HTTP.

The single-box design loaded a model, did one episode's work, and unloaded it.
Across dedicated machines that is wasteful — MuseTalk and Chatterbox each take
30-90 seconds to load, every episode, forever. Instead each node runs a long-
lived container that loads its model once at startup and holds it in VRAM. The
orchestrator sends jobs; the model never reloads.

Two constraints drive the rest of the design:

**Paths must resolve identically everywhere.** A job payload names files, not
bytes. Every container mounts the shared Unraid export at the same path,
``/pipeline``, regardless of where it lives on each host — so ``/pipeline/work/
2026-08-08/audio/x.wav`` means the same file on every node.

**1 GbE is the budget.** Only small files cross the wire: audio chunks are ~2 MB,
job payloads are bytes. The driving video, which is gigabytes, is never
transmitted — each avatar worker builds its own from a locally cached copy of the
base loop.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence

DEFAULT_SHARED_ROOT = "/pipeline"


class ClusterError(RuntimeError):
    pass


class NoHealthyNode(ClusterError):
    """Raised when no worker can serve a role. Fail loudly rather than silently
    falling back to the local machine — a 40-minute render on the wrong box is
    worse than an error."""


@dataclass
class Node:
    """One worker container."""

    name: str
    url: str
    role: str  # tts | avatar | media
    gpu: str = ""
    max_concurrent: int = 1
    enabled: bool = True
    weight: int = 1

    def endpoint(self, path: str) -> str:
        return f"{self.url.rstrip('/')}/{path.lstrip('/')}"


@dataclass
class NodeState:
    inflight: int = 0
    failures: int = 0
    last_error: str = ""
    healthy: bool = True


def translate_path(
    path: str | Path, from_root: str | Path, to_root: str = DEFAULT_SHARED_ROOT
) -> str:
    """Rewrite an orchestrator-local path into its in-container equivalent.

    The orchestrator sees the share at its Unraid path (``/mnt/user/podcast``);
    every worker container sees the identical files at ``/pipeline``. A path
    already rooted at the destination passes through unchanged, so this is safe
    to apply twice.
    """
    text = str(path)
    from_text = str(from_root).rstrip("/")
    to_text = str(to_root).rstrip("/")

    if text.startswith(to_text + "/") or text == to_text:
        return text
    if from_text and (text == from_text or text.startswith(from_text + "/")):
        remainder = text[len(from_text) :].lstrip("/")
        return str(PurePosixPath(to_text) / remainder) if remainder else to_text

    raise ClusterError(
        f"path {text!r} is not under the shared root {from_text!r}; workers on other "
        "hosts cannot reach it. Put episode files under the shared export."
    )


def translate_payload(
    payload: Any, from_root: str | Path, to_root: str = DEFAULT_SHARED_ROOT
) -> Any:
    """Recursively translate every path-bearing value in a job payload.

    Keys ending in ``_path``, ``_dir``, or named ``path``/``video``/``audio`` are
    treated as paths. Being explicit about which keys are paths avoids mangling a
    field that merely looks path-like, such as a URL or a caption line.
    """
    path_keys = {
        "path",
        "video",
        "audio",
        "out_path",
        "output",
        "reference_audio",
        "base_loop",
    }

    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            is_path = key in path_keys or key.endswith(("_path", "_dir", "_file"))
            if is_path and isinstance(value, str) and value:
                result[key] = translate_path(value, from_root, to_root)
            else:
                result[key] = translate_payload(value, from_root, to_root)
        return result
    if isinstance(payload, list):
        return [translate_payload(item, from_root, to_root) for item in payload]
    return payload


def select_node(
    nodes: Sequence[Node],
    state: dict[str, NodeState],
    exclude: Iterable[str] = (),
) -> Node | None:
    """Pick the least-loaded healthy node with a free slot.

    Least-loaded rather than round-robin because avatar chunks vary in length —
    round-robin would keep handing work to a node still chewing on a long chunk.
    Ties break on declared weight, then name, so selection is deterministic and
    testable.

    ``exclude`` skips nodes already tried for the job in hand. Without it a retry
    re-picks the node that just failed, because releasing its in-flight slot
    restores it to being the least loaded.
    """
    skip = set(exclude)
    candidates = []
    for node in nodes:
        if not node.enabled or node.name in skip:
            continue
        node_state = state.get(node.name, NodeState())
        if not node_state.healthy:
            continue
        if node_state.inflight >= node.max_concurrent:
            continue
        load = node_state.inflight / max(node.max_concurrent, 1)
        candidates.append((load, -node.weight, node.name, node))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:3])[3]


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _get_json(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class ClusterClient:
    """Dispatches jobs to workers, tracks load, and retries on another node."""

    def __init__(
        self,
        nodes: Iterable[Node],
        shared_root_local: str | Path,
        shared_root_container: str = DEFAULT_SHARED_ROOT,
        max_failures: int = 3,
        poster: Callable[[str, dict, int], dict] | None = None,
        getter: Callable[[str, int], dict] | None = None,
    ):
        self.nodes = list(nodes)
        self.shared_root_local = str(shared_root_local)
        self.shared_root_container = shared_root_container
        self.max_failures = max_failures
        self.state: dict[str, NodeState] = {n.name: NodeState() for n in self.nodes}
        # Guards every read-modify-write of `state`. The avatar stage dispatches
        # from several threads at once so both render nodes stay busy, and
        # "pick a free node" then "mark it busy" has to be one atomic step --
        # otherwise two threads select the same idle node and one of them
        # queues behind the other on a box that declared max_concurrent: 1.
        self._lock = threading.Lock()
        # Injectable transport so the dispatch logic is testable without a network.
        self._post = poster or _post_json
        self._get = getter or _get_json

    def nodes_for(self, role: str) -> list[Node]:
        return [n for n in self.nodes if n.role == role]

    def capacity(self, role: str) -> int:
        """Total concurrent slots for a role across healthy nodes."""
        return sum(
            n.max_concurrent
            for n in self.nodes_for(role)
            if n.enabled and self.state.get(n.name, NodeState()).healthy
        )

    # ---- health ---------------------------------------------------------

    def check_health(self, timeout: int = 5) -> dict[str, dict]:
        """Probe every node. Returns a report; also updates internal health."""
        report: dict[str, dict] = {}
        for node in self.nodes:
            if not node.enabled:
                report[node.name] = {"ok": False, "detail": "disabled in config"}
                continue
            try:
                body = self._get(node.endpoint("health"), timeout)
                ready = bool(body.get("model_loaded", body.get("ok", False)))
                self.state[node.name].healthy = ready
                self.state[node.name].failures = 0
                report[node.name] = {
                    "ok": ready,
                    "role": node.role,
                    "gpu": body.get("gpu", node.gpu),
                    "vram_free_mb": body.get("vram_free_mb"),
                    "model": body.get("model"),
                    "detail": "" if ready else "reachable but model not loaded",
                }
            except Exception as exc:
                self.state[node.name].healthy = False
                self.state[node.name].last_error = str(exc)
                report[node.name] = {"ok": False, "role": node.role, "detail": str(exc)}
        return report

    # ---- dispatch -------------------------------------------------------

    def submit(
        self,
        role: str,
        payload: dict,
        timeout: int = 3600,
        attempts: int = 2,
        wait_interval: float = 2.0,
        max_wait: float = 900.0,
    ) -> dict:
        """Run one job on some healthy node of ``role``.

        Blocks while every node is busy rather than queueing locally — the
        orchestrator drives concurrency itself, so a blocked submit means the
        cluster is genuinely saturated.

        Safe to call from several threads at once. ``_await_node`` claims the
        node it returns, so the caller must not increment ``inflight`` again.
        """
        candidates = self.nodes_for(role)
        if not candidates:
            raise NoHealthyNode(f"no nodes configured for role {role!r}")

        wire_payload = translate_payload(
            payload, self.shared_root_local, self.shared_root_container
        )

        last_error = ""
        tried: set[str] = set()
        for attempt in range(1, attempts + 1):
            # Prefer a node we have not already tried for this job; fall back to
            # retrying the same one when it is the only node for the role.
            # Returns a node with its in-flight slot already claimed.
            node = self._await_node(role, wait_interval, max_wait, exclude=tried)
            tried.add(node.name)
            try:
                result = self._post(node.endpoint("run"), wire_payload, timeout)
                with self._lock:
                    self.state[node.name].failures = 0
                result.setdefault("node", node.name)
                return result
            except Exception as exc:
                last_error = f"{node.name}: {exc}"
                with self._lock:
                    node_state = self.state[node.name]
                    node_state.failures += 1
                    node_state.last_error = str(exc)
                    if node_state.failures >= self.max_failures:
                        # Stop sending work to a node that keeps failing; the
                        # health probe can bring it back.
                        node_state.healthy = False
                if attempt == attempts:
                    break
            finally:
                with self._lock:
                    self.state[node.name].inflight -= 1

        raise ClusterError(f"job failed on all attempts for role {role!r}: {last_error}")

    def _await_node(
        self, role: str, interval: float, max_wait: float, exclude: Iterable[str] = ()
    ) -> Node:
        """Block until a node of ``role`` is free, then claim a slot on it.

        Selection and the claim happen together under the lock. Splitting them
        would let two concurrent callers both see the same node as idle.
        """
        deadline = time.monotonic() + max_wait
        exclude = set(exclude)
        while True:
            with self._lock:
                node = select_node(self.nodes_for(role), self.state, exclude)
                if node is None and exclude:
                    # Every untried node is busy or gone. Retrying the same node
                    # is still better than failing outright -- a transient error
                    # on the only avatar box should not abandon the episode.
                    node = select_node(self.nodes_for(role), self.state)
                if node is not None:
                    self.state[node.name].inflight += 1
                    return node
                any_healthy = any(
                    self.state.get(n.name, NodeState()).healthy and n.enabled
                    for n in self.nodes_for(role)
                )
            if not any_healthy:
                raise NoHealthyNode(
                    f"no healthy node for role {role!r}. Run 'podcastpipe cluster status'."
                )
            if time.monotonic() >= deadline:
                raise ClusterError(f"timed out waiting for a free {role!r} node")
            time.sleep(interval)


def load_nodes(raw: dict[str, Any]) -> list[Node]:
    """Build Node objects from the ``nodes`` section of cluster.yaml."""
    nodes: list[Node] = []
    for name, spec in (raw or {}).items():
        if not isinstance(spec, dict):
            raise ClusterError(f"node {name!r} must be a mapping")
        missing = {"url", "role"} - set(spec)
        if missing:
            raise ClusterError(f"node {name!r} is missing {', '.join(sorted(missing))}")
        nodes.append(
            Node(
                name=name,
                url=spec["url"],
                role=spec["role"],
                gpu=spec.get("gpu", ""),
                max_concurrent=int(spec.get("max_concurrent", 1)),
                enabled=bool(spec.get("enabled", True)),
                weight=int(spec.get("weight", 1)),
            )
        )
    return nodes
