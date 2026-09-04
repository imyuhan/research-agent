from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResearchCreateRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)


class ResearchJobResponse(BaseModel):
    job_id: str
    topic: str
    status: str
    current_step: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    revision_count: int = 0
    review_score: int = 0
    review_passed: bool = False
    draft: str = ""
    route_reason: str = ""
    review_feedback: str = ""
    error_message: str = ""
    token_usage: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    plan: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_pack: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
