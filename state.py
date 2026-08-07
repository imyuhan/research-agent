from typing import TypedDict, List


class ResearchState(TypedDict):
    # 输入
    topic: str

    # Planner 输出
    plan: List[str]
    search_queries: List[str]

    # Researcher 输出
    search_results: List[dict]

    # Writer 输出
    draft: str
    citations: List[dict]

    # Reviewer 输出
    review_feedback: str
    review_score: int
    review_passed: bool

    # 控制流
    revision_count: int
    current_step: str