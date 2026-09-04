from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_usage() -> Dict[str, Dict[str, int]]:
    return {}


@dataclass
class ResearchJobRecord:
    job_id: str
    topic: str
    status: str = "queued"
    current_step: str = "init"
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    revision_count: int = 0
    review_score: int = 0
    review_passed: bool = False
    draft: str = ""
    route_reason: str = ""
    review_feedback: str = ""
    error_message: str = ""
    token_usage: Dict[str, Dict[str, int]] = field(default_factory=_empty_usage)
    plan: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    findings: List[dict] = field(default_factory=list)
    evidence_pack: List[dict] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)
    search_results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._jobs: dict[str, ResearchJobRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._cancelled: set[str] = set()

    def create_job(self, topic: str) -> ResearchJobRecord:
        job = ResearchJobRecord(job_id=str(uuid4()), topic=topic)
        with self._lock:
            self._jobs[job.job_id] = job
            self._events[job.job_id] = []
        self.append_event(
            job.job_id,
            "job.created",
            message="Job created",
            payload={"topic": topic},
        )
        return deepcopy(job)

    def get_job(self, job_id: str) -> Optional[ResearchJobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job else None

    def update_job(self, job_id: str, **fields: Any) -> Optional[ResearchJobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            return deepcopy(job)

    def append_event(
        self,
        job_id: str,
        event_type: str,
        message: str = "",
        step: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> dict:
        event = {
            "seq": 0,
            "type": event_type,
            "message": message,
            "step": step,
            "timestamp": _now_iso(),
            "payload": payload or {},
        }
        with self._lock:
            events = self._events.setdefault(job_id, [])
            event["seq"] = len(events) + 1
            events.append(event)
        return deepcopy(event)

    def get_events(self, job_id: str) -> list[dict]:
        with self._lock:
            return deepcopy(self._events.get(job_id, []))

    def get_events_after(self, job_id: str, after: int = 0) -> list[dict]:
        with self._lock:
            events = self._events.get(job_id, [])
            return deepcopy(events[after:])

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            self._cancelled.add(job_id)
            job.status = "cancelling"
            self.append_event(
                job_id,
                "job.cancelling",
                message="Cancellation requested",
                step=job.current_step,
            )
            return True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = job.started_at or _now_iso()

    def mark_completed(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "completed"
            job.finished_at = _now_iso()
            self._cancelled.discard(job_id)

    def mark_cancelled(self, job_id: str, message: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "cancelled"
            job.finished_at = _now_iso()
            job.error_message = message
            self._cancelled.discard(job_id)

    def mark_failed(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.finished_at = _now_iso()
            job.error_message = message
            self._cancelled.discard(job_id)


job_store = JobStore()
