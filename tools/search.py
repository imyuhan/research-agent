from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from config import settings

search = TavilySearch(max_results=settings.MAX_SEARCH_RESULTS)


def web_search(query: str, start_date: str = "", end_date: str = "") -> list[dict]:
    """执行网络搜索，返回结构化结果列表 [{title, url, content}]，空结果或出错返回 []
    start_date/end_date 为 'YYYY-MM-DD'，用于将结果限定在指定日期范围"""
    try:
        params = {"query": query}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        response = search.invoke(params)
        items = response.get("results", [])

        results = []
        for r in items:
            results.append({
                "title": r.get("title", "无标题"),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:800],  # 直接截断，不用清洗
            })
        return results

    except Exception:
        return []


@tool
def web_search_tool(query: str) -> str:
    """网络搜索工具：输入查询词，返回网页搜索结果文本（标题/内容/来源URL）。"""
    items = web_search(query)
    if not items:
        return "未找到相关搜索结果"
    return "\n\n".join(
        f"[{i + 1}] {r['title']}\n{r['content']}\n来源: {r['url']}"
        for i, r in enumerate(items)
    )
