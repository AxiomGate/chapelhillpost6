"""Streaming scheduler: overlap voice synthesis and avatar rendering.

On a single box these stages competed for one machine and ran back to back. With
a dedicated node each, they can run at the same time — but only if the avatar
worker can start before the whole voice track exists.

The two stages work at different granularities. Voice is synthesized in ~300-
character chunks, about 15-20 seconds of audio each. The avatar renders in
windows of about two minutes, because per-render warm-up makes short renders
inefficient. So the scheduler accumulates finished voice chunks until they fill
an avatar window, then dispatches that window immediately while voice synthesis
continues on the next chunks.

The practical effect on a 25-minute episode: avatar rendering starts about 90
seconds in instead of 6-8 minutes in, and the voice node finishes its work while
the avatar node is only a fifth of the way through its own.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class Window:
    """A contiguous span of voice chunks that the avatar renders as one job."""

    index: int
    chunk_indices: list[int]
    duration: float
    start: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


def build_windows(
    durations: Sequence[float],
    window_seconds: float,
    min_tail_ratio: float = 0.25,
) -> list[Window]:
    """Group voice chunks into avatar render windows.

    A window closes once it reaches ``window_seconds``. Chunks are never split,
    because a window boundary mid-word puts a render seam in the middle of a
    syllable. A trailing window shorter than ``min_tail_ratio`` of a full window
    is merged into the previous one — the models produce a visible warm-up wobble
    in their first frames, and a five-second window is almost entirely warm-up.
    """
    windows: list[Window] = []
    current: list[int] = []
    current_duration = 0.0

    for index, duration in enumerate(durations):
        current.append(index)
        current_duration += duration
        if current_duration >= window_seconds:
            windows.append(Window(len(windows), current, round(current_duration, 3)))
            current, current_duration = [], 0.0

    if current:
        windows.append(Window(len(windows), current, round(current_duration, 3)))

    if len(windows) > 1 and windows[-1].duration < window_seconds * min_tail_ratio:
        tail = windows.pop()
        previous = windows[-1]
        previous.chunk_indices.extend(tail.chunk_indices)
        previous.duration = round(previous.duration + tail.duration, 3)

    cursor = 0.0
    for window in windows:
        window.start = round(cursor, 3)
        cursor += window.duration
    return windows


class StreamingScheduler:
    """Runs voice synthesis and avatar rendering concurrently across nodes.

    ``synthesize`` and ``render`` are injected, so the scheduling logic is
    testable without a cluster. Both are called from worker threads and must be
    safe to call concurrently.
    """

    def __init__(
        self,
        synthesize: Callable[[int], float],
        render: Callable[[Window], str],
        window_seconds: float = 120.0,
        voice_workers: int = 1,
        avatar_workers: int = 1,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.synthesize = synthesize
        self.render = render
        self.window_seconds = window_seconds
        self.voice_workers = max(1, voice_workers)
        self.avatar_workers = max(1, avatar_workers)
        self.on_event = on_event or (lambda kind, data: None)

        self._lock = threading.Lock()
        self.durations: dict[int, float] = {}
        self.windows: list[Window] = []
        self.rendered: dict[int, str] = {}
        self.errors: list[str] = []

    def run(self, chunk_count: int) -> list[str]:
        """Synthesize ``chunk_count`` voice chunks and render every window.

        Returns rendered window paths in timeline order. Raises if any stage
        failed — a missing window means a silent gap in the finished video, so
        partial success is not success.
        """
        if chunk_count <= 0:
            return []

        # Voice chunks are synthesized in order so windows close in order; the
        # avatar side is where parallelism pays, since it is 5-8x slower.
        next_chunk = 0
        pending_indices: list[int] = []
        pending_duration = 0.0
        dispatched: list[Future] = []

        with ThreadPoolExecutor(max_workers=self.voice_workers, thread_name_prefix="voice") as voice_pool, \
             ThreadPoolExecutor(max_workers=self.avatar_workers, thread_name_prefix="avatar") as avatar_pool:

            voice_futures: dict[int, Future] = {}
            for _ in range(min(self.voice_workers, chunk_count)):
                voice_futures[next_chunk] = voice_pool.submit(self._synthesize_one, next_chunk)
                next_chunk += 1

            completed = 0
            while completed < chunk_count:
                index = min(voice_futures)  # consume in order
                future = voice_futures.pop(index)
                duration = future.result()
                completed += 1

                if next_chunk < chunk_count:
                    voice_futures[next_chunk] = voice_pool.submit(
                        self._synthesize_one, next_chunk
                    )
                    next_chunk += 1

                pending_indices.append(index)
                pending_duration += duration

                is_last = completed == chunk_count
                if pending_duration >= self.window_seconds or is_last:
                    window = self._close_window(pending_indices, pending_duration, is_last)
                    pending_indices, pending_duration = [], 0.0
                    if window is not None:
                        dispatched.append(avatar_pool.submit(self._render_one, window))

            for future in dispatched:
                future.result()

        if self.errors:
            raise RuntimeError(
                f"{len(self.errors)} stage failure(s); first: {self.errors[0]}"
            )

        return [self.rendered[w.index] for w in sorted(self.windows, key=lambda w: w.index)]

    # ---- internals ------------------------------------------------------

    def _synthesize_one(self, index: int) -> float:
        try:
            duration = float(self.synthesize(index))
        except Exception as exc:
            with self._lock:
                self.errors.append(f"voice chunk {index}: {exc}")
            return 0.0
        with self._lock:
            self.durations[index] = duration
        self.on_event("voice_chunk", {"index": index, "duration": duration})
        return duration

    def _close_window(
        self, indices: list[int], duration: float, is_last: bool
    ) -> Window | None:
        """Seal a window and hand it to the avatar side.

        Unlike ``build_windows``, a short final window is *not* merged into its
        predecessor here. In streaming mode the predecessor was dispatched the
        moment it closed and is very likely already rendering on a worker, so
        there is nothing left to merge into. Rendering a short tail on its own
        costs a few seconds of warm-up; reaching into an in-flight job would cost
        correctness.
        """
        if not indices:
            return None

        with self._lock:
            start = sum(w.duration for w in self.windows)
            window = Window(
                index=len(self.windows),
                chunk_indices=list(indices),
                duration=round(duration, 3),
                start=round(start, 3),
            )
            self.windows.append(window)

        self.on_event(
            "window_ready",
            {"index": window.index, "duration": window.duration, "chunks": len(indices)},
        )
        return window

    def _render_one(self, window: Window) -> None:
        try:
            path = self.render(window)
        except Exception as exc:
            with self._lock:
                self.errors.append(f"avatar window {window.index}: {exc}")
            return
        with self._lock:
            self.rendered[window.index] = path
        self.on_event("window_rendered", {"index": window.index, "path": path})
