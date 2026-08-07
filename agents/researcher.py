from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from config import settings
from tools.search import web_search, web_search_tool
from tools.date_utils import detect_date, to_chinese, date_window
import concurrent.futures

researcher_system = """你是一个研究智能体。针对给定的研究子问题，使用网络搜索收集资料：
- web_search_tool：进行网络搜索获取最新/外部信息（每个子问题最多调用 1-2 次）

要求：
- 每个子问题至少调用一次 web_search_tool
- 收集完成后，输出对该子问题的简短研究总结"""

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.3
)

researcher_llm = llm.bind_tools([web_search_tool])

MAX_TOOL_ROUNDS = 4


def _format_web(items: list[dict]) -> str:
    if not items:
        return "未找到相关搜索结果"
    return "\n\n".join(
        f"[{i + 1}] {r['title']}\n{r['content']}\n来源: {r['url']}"
        for i, r in enumerate(items)
    )

def _researcher_one_query(
        query: str,
        system_prompt:str,
        date_cn: str,
        window: tuple[str, str],
        cap: int,
) -> tuple[dict,list[dict]]:
    """处理单个子问题：LLM 工具协商 + 网络搜索，返回(result_item,local_citations)"""
    print(f"   研究：{query}")

    collected_web = []
    final_summary = ""
    search_count = 0

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"研究子问题：{query}"),
    ]
    for _ in range(MAX_TOOL_ROUNDS):
        response = researcher_llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            final_summary = response.content or ""
            break
        for tc in response.tool_calls:
            if tc["name"] == "web_search_tool":
                base_q = tc.get("args", {}).get("query",query)
                search_q = f"{base_q} {date_cn}".strip() if date_cn else base_q
                items = web_search(search_q, start_date=window[0], end_date=window[1])
                collected_web.extend(items)
                search_count += 1
                result_txt = _format_web(items)
            else:
                result_txt = f"该工具当前不可用：{tc['name']}，请改用 web_search_tool"
            messages.append(ToolMessage(content=result_txt, tool_call_id=tc["id"]))

    # 确定性组装：URL/片段去重，每个子问题引用数不超过上限
    parts = []
    seen_web = set()
    count = 0
    local_citations = []

    for r in collected_web:
        if count >= cap or r["url"] in seen_web:
            continue
        seen_web.add(r["url"])
        count += 1
        idx = len(local_citations) + 1
        local_citations.append({
            "query": query,
            "title": r["title"],
            "url": r["url"],
        })
        parts.append(f"[{idx}] {r['title']}\n{r['content']}\n来源： {r['url']}")

    content = "\n\n".join(parts) if parts else (final_summary or "未找到相关搜索结果")
    print(f"    ✅ 完成：{query}")
    print(f"       搜索 {search_count} 次 | 结果长度：{len(content)} 字符 | 引用：{count} 条")

    return {"query": query, "content": content}, local_citations
def researcher_node(state: dict) -> dict:
    """研究节点：智能体使用网络搜索收集资料，汇总结果与引用（去重 + 每子问题引用上限）"""
    queries = state.get("search_queries", [])

    if not queries:
        print("⚠️ 没有搜索查询")
        return {"search_results": [], "citations": [], "current_step": "researcher"}

    system_prompt = researcher_system
    cap = settings.MAX_CITATIONS_PER_QUERY

    date_iso = detect_date(" ".join([state.get("topic", ""), *queries]))
    date_cn = to_chinese(date_iso) if date_iso else ""
    window = date_window(date_iso, 1, 1) if date_iso else ("", "")
    if date_cn:
        system_prompt += (
            f"\n\n本次研究的指定日期为 {date_cn}。"
            "所有搜索收集的内容必须是该日期的信息。"
        )
        print(f"   📅 用户指定日期: {date_cn}，搜索窗口 {window[0]} ~ {window[1]}")

    print(f"🔍 Researcher 开始研究 {len(queries)} 个子问题:")

    all_results = []
    all_citations = []

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.MAX_SEARCH_CONCURRENCY
    ) as executor:
        futures = {
            executor.submit(
                _researcher_one_query, q, system_prompt, date_cn, window, cap
            ): q
            for q in queries
        }
        results_by_query = {}
        citations_by_query = {}
        for future in concurrent.futures.as_completed(futures):
            q = futures[future]
            result_item, local_citations = future.result()
            results_by_query[q] = result_item
            citations_by_query[q] = local_citations

    # 按原始顺序重组，保证引用编号确定性
    all_results = [results_by_query[q] for q in queries]
    all_citations = []
    for q in queries:
        for c in citations_by_query[q]:
            all_citations.append({
                "index": len(all_citations) + 1,
                "query": q,
                "title": c["title"],
                "url": c["url"],
            })

    print(f"✅ 研究完成，共收集 {len(all_results)} 条结果，{len(all_citations)} 条引用")

    return {
        "search_results": all_results,
        "citations": all_citations,
        "current_step": "researcher"
    }





