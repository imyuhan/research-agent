from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    # 输入
    topic: str

    # 对话/执行历史（LangGraph 会自动合并）
    messages: Annotated[List[BaseMessage], add_messages]

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