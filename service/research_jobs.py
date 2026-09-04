from __future__ import annotations

import traceback
from typing import Any, Dict

from agents._token_tracker import merge_state_usage
from state import ResearchState
from workflow import graph
from .job_store import JobStore, job_store


def _initial_state(topic: str) -> ResearchState:
    return ResearchState(
        topic=topic,
        plan=[],
        search_queries=[],
        evidence_pack=[],
        findings=[],
        search_results=[],
        draft="",
        citations=[],
        review_feedback="",
        review_score=0,
        review_passed=False,
        needs_new_search=False,
        route_reason="",
        revision_count=0,
        current_step="init",
    )


def _snapshot_state(state: Dict[str, Any], token_usage: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    return {
        "current_step": state.get("current_step", "unknown"),
        "plan_count": len(state.get("plan", []) or []),
        "search_query_count": len(state.get("search_queries", []) or []),
        "finding_count": len(state.get("findings", []) or []),
        "evidence_count": len(state.get("evidence_pack", []) or []),
        "citation_count": len(state.get("citations", []) or []),
        "review_score": state.get("review_score", 0),
        "review_passed": state.get("review_passed", False),
        "revision_count": state.get("revision_count", 0),
        "route_reason": state.get("route_reason", ""),
        "draft_length": len(state.get("draft", "") or ""),
        "review_feedback_length": len(state.get("review_feedback", "") or ""),
        "token_usage": token_usage,
    }


def run_research_job(job_id: str, topic: str, store: JobStore = job_store) -> None:
    """
    Execute the LangGraph workflow in a background thread.
    The store keeps the live job record and event timeline for the API layer.
    """
    store.mark_running(job_id)
    store.append_event(job_id, "job.started", message="Research started", payload={"topic": topic})

    total_usage: Dict[str, Dict[str, int]] = {}
    last_state: Dict[str, Any] | None = None

    try:
        for state in graph.stream(_initial_state(topic), stream_mode="values"):
            if store.is_cancelled(job_id):
                store.mark_cancelled(job_id, "Cancelled by user")
                store.append_event(job_id, "job.cancelled", message="Job cancelled by user")
                return

            current_step = state.get("current_step", "unknown")
            event_usage = state.get("token_usage") or {}
            if event_usage:
                total_usage = merge_state_usage(total_usage, event_usage)

            snapshot = _snapshot_state(state, total_usage)
            store.update_job(
                job_id,
                current_step=current_step,
                revision_count=state.get("revision_count", 0),
                review_score=state.get("review_score", 0),
                review_passed=state.get("review_passed", False),
                draft=state.get("draft", ""),
                route_reason=state.get("route_reason", ""),
                review_feedback=state.get("review_feedback", ""),
                token_usage=total_usage,
                plan=state.get("plan", []),
                search_queries=state.get("search_queries", []),
                findings=state.get("findings", []),
                evidence_pack=state.get("evidence_pack", []),
                citations=state.get("citations", []),
                search_results=state.get("search_results", []),
            )
            store.append_event(
                job_id,
                f"{current_step}.completed",
                message=f"{current_step} completed",
                step=current_step,
                payload=snapshot,
            )
            last_state = state

        if last_state is None:
            raise RuntimeError("Workflow finished without producing any state")

        store.mark_completed(job_id)
        store.update_job(
            job_id,
            current_step=last_state.get("current_step", "done"),
            revision_count=last_state.get("revision_count", 0),
            review_score=last_state.get("review_score", 0),
            review_passed=last_state.get("review_passed", False),
            draft=last_state.get("draft", ""),
            route_reason=last_state.get("route_reason", ""),
            review_feedback=last_state.get("review_feedback", ""),
            token_usage=total_usage,
            plan=last_state.get("plan", []),
            search_queries=last_state.get("search_queries", []),
            findings=last_state.get("findings", []),
            evidence_pack=last_state.get("evidence_pack", []),
            citations=last_state.get("citations", []),
            search_results=last_state.get("search_results", []),
        )
        store.append_event(
            job_id,
            "job.completed",
            message="Research completed",
            step=last_state.get("current_step", "done"),
            payload={
                "topic": topic,
                "review_passed": last_state.get("review_passed", False),
                "review_score": last_state.get("review_score", 0),
                "revision_count": last_state.get("revision_count", 0),
                "draft_length": len(last_state.get("draft", "") or ""),
                "review_feedback_length": len(last_state.get("review_feedback", "") or ""),
                "token_usage": total_usage,
            },
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        store.mark_failed(job_id, message)
        store.append_event(
            job_id,
            "job.failed",
            message=message,
            payload={"traceback": traceback.format_exc()},
        )
