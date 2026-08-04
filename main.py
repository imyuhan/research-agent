# @File     : main.py
# @Time     : 2026/8/4 11:48
from workflow import graph
from state import ResearchState
from config import settings


def run_research(topic: str):
    """执行完整的研究流程"""
    initial_state = ResearchState(
        topic=topic,
        messages=[],
        plan=[],
        search_queries=[],
        search_results=[],
        draft="",
        citations=[],
        review_feedback="",
        review_score=0,
        review_passed=False,
        revision_count=0,
        current_step="init"
    )

    print(f"🚀 开始研究主题: {topic}\n")

    # 只执行一次，stream_mode="values" 每次返回完整状态
    final_state = None
    for event in graph.stream(initial_state, stream_mode="values"):
        step = event.get("current_step", "unknown")

        if step == "planner":
            print(f"📋 规划完成，生成 {len(event.get('plan', []))} 个子问题")
        elif step == "researcher":
            print(f"🔍 研究完成，收集 {len(event.get('search_results', []))} 条资料")
        elif step == "writer":
            print(f"✍️ 写作完成，报告长度 {len(event.get('draft', ''))} 字符")
        elif step == "reviewer":
            score = event.get("review_score", 0)
            passed = event.get("review_passed", False)
            print(f"🔍 审核完成，评分 {score}/10，{'✅通过' if passed else '❌未通过'}")
            if not passed:
                print(f"   回退次数: {event.get('revision_count', 0)}/{settings.MAX_REVISIONS}")

        # 保留最后一个状态（即最终完整状态）
        final_state = event

    if not final_state:
        print("❌ 流程异常结束")
        return None

    draft = final_state.get("draft", "")

    print("\n" + "=" * 60)
    print("📄 最终研究报告")
    print("=" * 60)
    print(draft if draft else "❌ 未能生成报告内容")
    print("=" * 60)

    return draft


if __name__ == "__main__":
    topic = input("请输入研究主题：").strip()
    if topic:
        run_research(topic)
    else:
        print("主题不能为空")