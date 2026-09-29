"""In-memory background job manager for reconstruction runs (one at a time — they use every core)."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from . import tools
from .pipeline import STAGES, Reconstruction, Settings


@dataclass
class Job:
    id: str
    settings: Settings
    status: str = "queued"  # queued | running | done | error | cancelled
    percent: float = 0.0
    stage: str = "queued"
    stages: list = field(default_factory=list)
    detail: str = ""
    error: Optional[str] = None
    result: Optional[dict] = None
    workspace: Optional[str] = None
    recon: Optional[Reconstruction] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    updated_at: float = field(default_factory=time.time)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def public_dict(self, log_lines: int = 0) -> dict:
        out = {
            "id": self.id,
            "name": self.settings.name,
            "status": self.status,
            "percent": round(self.percent, 2),
            "stage": self.stage,
            "stage_label": STAGES[self.stage][0] if self.stage in STAGES else self.stage,
            "stages": [{"key": s, "label": STAGES[s][0]} for s in self.stages],
            "detail": self.detail,
            "error": self.error,
            "result": self.result,
            "workspace": self.workspace,
            "elapsed": round((self.updated_at if self.status in ("done", "error", "cancelled") else time.time())
                             - (self.started_at or self.created_at), 1),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if log_lines and self.recon:
            out["log"] = self.recon.log_lines[-log_lines:]
        return out


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    def create(self, settings: Settings) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], settings=settings)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status not in ("queued", "running"):
            return False
        job.cancel_event.set()
        return True

    async def run(self, job_id: str) -> None:
        job = self._jobs[job_id]

        def on_update() -> None:
            p = job.recon.progress
            job.stage = p.stage
            job.stages = p.stages
            job.detail = p.detail
            job.percent = min(99.9, p.percent)
            job.updated_at = time.time()

        async with self._lock:
            if job.cancel_event.is_set():
                job.status = job.stage = "cancelled"
                return
            job.status = "running"
            job.started_at = time.time()
            try:
                job.recon = Reconstruction(job.id, job.settings, on_update, job.cancel_event)
                job.workspace = str(job.recon.workspace)
                job.stages = job.recon.progress.stages
                job.result = await job.recon.run()
                job.status = job.stage = "done"
                job.percent = 100.0
            except tools.Cancelled:
                job.status = job.stage = "cancelled"
            except tools.ToolError as exc:
                job.status = "error"
                job.error = str(exc)
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.updated_at = time.time()


job_manager = JobManager()
