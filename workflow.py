from langgraph.graph import StateGraph, START, END
from state import ResearchState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.reviewer import reviewer_node
from config import settings


def route_after_review(state: dict) -> str:
    """Reviewer 后的条件路由逻辑"""
    if state.get("review_passed", False):
        print("✅ 审核通过，流程结束")
        return "approved"

    # 超过最大回退次数，强制结束（防止死循环）
    if state.get("revision_count", 0) >= settings.MAX_REVISIONS:
        print("⚠️ 达到最大回退次数，强制结束")
        return "max_retries"

    print("🔄 审核未通过，回退至 Writer 重写")
    return "needs_rewrite"


def build_graph():
    """构建 LangGraph 工作流"""
    builder = StateGraph(ResearchState)

    # 注册节点
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    # 静态边：顺序执行
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "reviewer")

    # 条件边：审核后的分支（核心亮点）
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "approved": END,  # 通过 → 结束
            "max_retries": END,  # 超次 → 结束
            "needs_rewrite": "writer"  # 不通过 → 回退重写
        }
    )

    return builder.compile()


# 全局图实例（供 main.py 导入使用）
graph = build_graph()