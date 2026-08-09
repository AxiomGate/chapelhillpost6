"""Shared scaffolding for the warm GPU workers.

Every worker is the same shape: load a model once at container start, hold it in
VRAM forever, and answer job requests over HTTP. This module supplies the parts
that do not differ — the health endpoint, VRAM accounting, the job envelope, and
serialized access to a model that is not thread-safe.

Each worker is a separate container with its own dependencies because MuseTalk,
Chatterbox and the caption stack pin conflicting versions of torch and
transformers. That was already true on one box; across nodes it is free.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


def gpu_info() -> dict[str, Any]:
    """Report the visible GPU. Returns placeholders rather than raising when
    torch is missing, so /health still answers during a broken build."""
    try:
        import torch
    except ImportError:
        return {"gpu": "torch not installed", "vram_total_mb": 0, "vram_free_mb": 0}

    if not torch.cuda.is_available():
        return {"gpu": "no CUDA device visible", "vram_total_mb": 0, "vram_free_mb": 0}

    free, total = torch.cuda.mem_get_info()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_mb": total // (1024 * 1024),
        "vram_free_mb": free // (1024 * 1024),
    }


def require_vram(needed_mb: int) -> None:
    """Refuse to start if another container has taken the VRAM we need.

    Worth failing loudly at boot: these servers already run other AI containers,
    and an out-of-memory error forty minutes into a render is far more expensive
    to diagnose than a container that refuses to start with a clear reason.
    """
    info = gpu_info()
    free = info.get("vram_free_mb", 0)
    if info.get("vram_total_mb", 0) == 0:
        raise RuntimeError(
            f"no CUDA device visible ({info.get('gpu')}). Check the Unraid Nvidia "
            "driver plugin and that NVIDIA_VISIBLE_DEVICES names this card's UUID."
        )
    if free < needed_mb:
        raise RuntimeError(
            f"need {needed_mb} MB of VRAM, only {free} MB free on {info['gpu']}. "
            "Another container is likely holding it — check what else is pinned "
            "to this GPU, or lower the model size."
        )


@dataclass
class WorkerStats:
    started_at: float = field(default_factory=time.time)
    jobs_done: int = 0
    jobs_failed: int = 0
    last_job_seconds: float = 0.0
    total_job_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        uptime = time.time() - self.started_at
        return {
            "uptime_seconds": round(uptime, 1),
            "jobs_done": self.jobs_done,
            "jobs_failed": self.jobs_failed,
            "last_job_seconds": round(self.last_job_seconds, 2),
            "mean_job_seconds": round(
                self.total_job_seconds / self.jobs_done, 2
            ) if self.jobs_done else 0.0,
        }


def build_app(
    name: str,
    role: str,
    loader: Callable[[], Any],
    handler: Callable[[Any, dict], dict],
    required_vram_mb: int = 0,
):
    """Build the FastAPI app for a worker.

    ``loader`` runs once at startup and returns the model handle. ``handler``
    receives that handle plus the job payload. Jobs are serialized behind a lock:
    these models are not thread-safe, and one GPU cannot usefully run two
    diffusion jobs at once anyway — concurrency belongs at the cluster level,
    across nodes, not inside one worker.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    class Job(BaseModel):
        model_config = {"extra": "allow"}

    app = FastAPI(title=f"podcastpipe {name}")
    state: dict[str, Any] = {"model": None, "error": "", "stats": WorkerStats()}
    lock = threading.Lock()

    @app.on_event("startup")
    def _startup() -> None:
        try:
            if required_vram_mb:
                require_vram(required_vram_mb)
            print(f"[{name}] loading model ...", flush=True)
            started = time.time()
            state["model"] = loader()
            print(f"[{name}] ready in {time.time() - started:.1f}s", flush=True)
        except Exception as exc:
            # Stay up with the error visible on /health. A container that exits
            # disappears into Unraid's restart loop with the reason buried in
            # docker logs; one that answers "here is why I am not ready" does not.
            state["error"] = f"{exc}"
            print(f"[{name}] FAILED TO LOAD: {exc}", flush=True)
            traceback.print_exc()

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": state["model"] is not None,
            "model_loaded": state["model"] is not None,
            "worker": name,
            "role": role,
            "model": os.environ.get("MODEL_NAME", name),
            "error": state["error"],
            "busy": lock.locked(),
            **gpu_info(),
            **state["stats"].as_dict(),
        }

    @app.post("/run")
    def run(job: Job) -> JSONResponse:
        if state["model"] is None:
            return JSONResponse(
                status_code=503,
                content={"error": state["error"] or "model still loading"},
            )

        payload = job.model_dump()
        started = time.time()
        with lock:
            try:
                result = handler(state["model"], payload)
            except Exception as exc:
                state["stats"].jobs_failed += 1
                traceback.print_exc()
                return JSONResponse(
                    status_code=500, content={"error": str(exc), "worker": name}
                )

        elapsed = time.time() - started
        stats = state["stats"]
        stats.jobs_done += 1
        stats.last_job_seconds = elapsed
        stats.total_job_seconds += elapsed
        result.setdefault("seconds", round(elapsed, 2))
        result.setdefault("worker", name)
        return JSONResponse(content=result)

    return app


def serve(app, default_port: int) -> None:
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104 — container-internal; exposure is Docker's job
        port=int(os.environ.get("PORT", default_port)),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
