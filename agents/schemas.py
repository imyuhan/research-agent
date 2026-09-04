from __future__ import annotations

from typing import Any, Dict, List, NotRequired, TypedDict


class EvidenceItem(TypedDict, total=False):
    evidence_id: str
    kind: str
    query: str
    text: str
    title: str
    url: str
    source: str
    doc_id: str
    chunk_id: str
    chunk_index: int
    score: float
    retrieval_score: float
    rerank_score: float
    metadata: Dict[str, Any]


class FindingItem(TypedDict, total=False):
    query: str
    content: str
    summary: str
    evidence_ids: List[str]
    gaps: List[str]
    confidence: float
    needs_more_search: bool


class ReviewDecision(TypedDict, total=False):
    score: int
    passed: bool
    feedback: str
    needs_new_search: bool
    route_reason: str
    suggested_queries: List[str]

