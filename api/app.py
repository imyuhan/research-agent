from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

from api.schemas import ResearchCreateRequest, ResearchJobResponse
from infrastructure.config import settings
from service.job_store import job_store
from service.research_jobs import run_research_job


def _job_response(job) -> ResearchJobResponse:
    return ResearchJobResponse(**job.to_dict())


def _sse(event: dict) -> str:
    payload = json.dumps(event, ensure_ascii=False)
    return f"data: {payload}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Research Agent API", version="0.1.0", lifespan=lifespan)

origins = [origin.strip() for origin in settings.FRONTEND_ORIGINS.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.post("/api/research", response_model=ResearchJobResponse, status_code=202)
def create_research(payload: ResearchCreateRequest, background_tasks: BackgroundTasks):
    job = job_store.create_job(payload.topic)
    background_tasks.add_task(run_research_job, job.job_id, payload.topic, job_store)
    return _job_response(job)


@app.get("/api/research/{job_id}", response_model=ResearchJobResponse)
def get_research(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_response(job)


@app.post("/api/research/{job_id}/cancel", response_model=ResearchJobResponse)
def cancel_research(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job_store.request_cancel(job_id)
    refreshed = job_store.get_job(job_id)
    return _job_response(refreshed)


@app.get("/api/research/{job_id}/report")
def get_report(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "topic": job.topic,
        "status": job.status,
        "review_score": job.review_score,
        "review_passed": job.review_passed,
        "draft": job.draft,
        "review_feedback": job.review_feedback,
        "citations": job.citations,
        "evidence_pack": job.evidence_pack,
        "findings": job.findings,
        "plan": job.plan,
        "error_message": job.error_message,
    }


@app.get("/api/research/{job_id}/events/history")
def get_event_history(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "events": job_store.get_events(job_id),
    }


@app.get("/api/research/{job_id}/events")
async def stream_events(job_id: str, after: int = Query(0, ge=0)):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generator():
        cursor = after
        while True:
            events = job_store.get_events(job_id)
            while cursor < len(events):
                yield _sse(events[cursor])
                cursor += 1

            latest = job_store.get_job(job_id)
            if latest is None or latest.status in {"completed", "failed", "cancelled"}:
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
