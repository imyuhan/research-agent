from typing import TypedDict, List

from agents.schemas import EvidenceItem, FindingItem, ReviewDecision


class ResearchState(TypedDict, total=False):
    """
    LangGraph 状态定义
    用 total=False 让所有字段都可选,新增字段不破坏老节点
    """
    # 输入
    topic: str
    # Planner 输出
    plan: List[str]
    search_queries: List[str]
    # Researcher / RAG 输出
    evidence_pack: List[EvidenceItem]
    findings: List[FindingItem]
    # Researcher 输出
    search_results: List[dict]
    kb_chunks: List[dict]   # RAG 召回的 KB 片段
    # Writer 输出
    draft: str
    citations: List[dict]
    # Reviewer 输出
    review_feedback: str
    review_score: int
    review_passed: bool
    needs_new_search: bool  # reviewer标记是否需要重新搜索
    route_reason: str
    # 控制流
    revision_count: int
    current_step: str
    # Token 用量统计:{node_name: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}}
    # 用 dict 包裹以兼容 TypedDict(total=False),key 不存在视为该节点没产生用量
    token_usage: dict
