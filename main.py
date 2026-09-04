import time
from collections import OrderedDict
from pathlib import Path

from infrastructure.config import settings
from workflow import graph
from state import ResearchState
from service import get_rag
from agents._token_tracker import merge_state_usage, format_table


def _auto_ingest_new_documents() -> bool:
    """
    启动时自动检测新文件并入库
    返回:是否实际入库了新文件
    """
    docs_dir = Path(settings.DOCUMENTS_DIR)
    print("📂 正在检测本地知识库增量...")
    if not docs_dir.exists():
        print(f"   ⚠️ 文档目录不存在:{docs_dir.resolve()}")
        return False

    rag = get_rag()
    new_files = rag.find_new_files(docs_dir)

    if not new_files:
        print(f"   ✅ 无新增/变更文件({docs_dir.resolve()})")
        return False

    # 显示具体哪些是待同步文件(只显示相对路径,长路径截断)
    print(f"   🆕 检测到 {len(new_files)} 个待同步文件:")
    for fp in new_files:
        try:
            rel = Path(fp).relative_to(docs_dir.resolve())
        except ValueError:
            rel = Path(fp).name
        print(f"      + {rel}")

    # 增量入库(只入新文件)
    print("   ⏳ 正在入库/更新(增量)...")
    t0 = time.time()
    result = rag.ingest_files(new_files)
    elapsed = time.time() - t0
    print(
        f"   ✅ 入库完成:{result.get('docs_loaded', 0)} 文件 / "
        f"{result.get('chunks', 0)} chunks / 写入 {result.get('written', 0)} 条 / "
        f"删除旧块 {result.get('deleted', 0)} 条,"
        f"耗时 {elapsed:.1f}s\n"
    )
    return True


def run_research(topic: str):
    """执行完整的研究流程"""
    initial_state = ResearchState(
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
        current_step="init"
    )

    start_total = time.time()
    step_times = OrderedDict()  # step -> 累计用时（秒）
    last_arrival = start_total
    last_step = None
    # 累加所有节点的 token_usage(reviewer→writer 回退时 state.token_usage 会被新一轮覆盖,要累加而不是取最后)
    total_usage: dict = {}

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

        # 累加 token_usage:event 是 state 快照,新增/重入节点都正确合并
        event_usage = event.get("token_usage") or {}
        if event_usage:
            total_usage = merge_state_usage(total_usage, event_usage)

        if step == "planner":
            print(f"📋 规划完成,生成 {len(event.get('plan', []))} 个子问题")
        elif step == "researcher":
            print(
                f"🔍 研究完成,收集 {len(event.get('findings', []))} 条 findings / "
                f"{len(event.get('evidence_pack', []))} 条证据(搜索用时 {step_times[step]:.1f}s)"
            )
        elif step == "writer":
            print(f"✍️ 写作完成,报告长度 {len(event.get('draft', ''))} 字符")
        elif step == "reviewer":
            score = event.get("review_score", 0)
            passed = event.get("review_passed", False)
            print(f"🔍 审核完成,评分 {score}/10,{'✅通过' if passed else '❌未通过'}")
            if not passed:
                print(f"   回退次数: {event.get('revision_count', 0)}/{settings.MAX_REVISIONS}")
                if event.get("route_reason"):
                    print(f"   路由原因: {event.get('route_reason')}")

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
    print(f"\n⏱️ 各阶段耗时:")
    for s, t in step_times.items():
        print(f"   {s:<12} {t:6.1f}s")
    print(f"   {'总计':<12} {total:6.1f}s")
    print(f"\n📊 本次 Token 用量(单位:tokens):")
    print(format_table(total_usage))

    return draft


if __name__ == "__main__":
    # 1) 启动时自动检测新文件,有则入库
    _auto_ingest_new_documents()
    # 2) 入库完成后(无论是否有新文件)再问研究主题
    topic = input("请输入研究主题:").strip()
    if topic:
        run_research(topic)
    else:
        print("主题不能为空")
