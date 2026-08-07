import time
from collections import OrderedDict

from config import settings
from workflow import graph
from state import ResearchState


def run_research(topic: str):
    """执行完整的研究流程"""
    initial_state = ResearchState(
        topic=topic,
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

    start_total = time.time()
    step_times = OrderedDict()  # step -> 累计用时（秒）
    last_arrival = start_total
    last_step = None

    print(f"🚀 开始研究主题: {topic}\n")

    # 只执行一次，stream_mode="values" 每次返回完整状态
    final_state = None
    for event in graph.stream(initial_state, stream_mode="values"):
        step = event.get("current_step", "unknown")
        now = time.time()

        # 相邻两个事件的时间差 = 刚执行完的节点（当前 step）的耗时
        step_times[step] = step_times.get(step, 0.0) + (now - last_arrival)
        last_arrival = now
        last_step = step

        if step == "planner":
            print(f"📋 规划完成，生成 {len(event.get('plan', []))} 个子问题")
        elif step == "researcher":
            print(f"🔍 研究完成，收集 {len(event.get('search_results', []))} 条资料（搜索用时 {step_times[step]:.1f}s）")
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

    if last_step is not None:
        step_times[last_step] = step_times.get(last_step, 0.0) + (time.time() - last_arrival)
    total = time.time() - start_total

    if not final_state:
        print("❌ 流程异常结束")
        return None

    draft = final_state.get("draft", "")

    print("\n" + "=" * 60)
    print("📄 最终研究报告")
    print("=" * 60)
    print(draft if draft else "❌ 未能生成报告内容")
    print("=" * 60)
    print(f"\n⏱️ 各阶段耗时：")
    for s, t in step_times.items():
        print(f"   {s:<12} {t:6.1f}s")
    print(f"   {'总计':<12} {total:6.1f}s")

    return draft


if __name__ == "__main__":
    topic = input("请输入研究主题：").strip()
    if topic:
        run_research(topic)
    else:
        print("主题不能为空")