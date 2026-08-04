# @File     : search.py
# @Time     : 2026/8/4 11:47
from langchain_tavily import TavilySearch
from config import settings

search = TavilySearch(max_results=settings.MAX_SEARCH_RESULTS)


def web_search(query: str) -> str:
    """执行网络搜索，返回结构化结果"""
    try:
        # 返回的是 dict，不是 list
        response = search.invoke({"query": query})

        # 真正的搜索结果在 "results" 键下
        items = response.get("results", [])

        if not items:
            return "未找到相关搜索结果"

        formatted = []
        for i, r in enumerate(items, 1):
            title = r.get("title", "无标题")
            content = r.get("content", "无内容")
            url = r.get("url", "无来源")
            formatted.append(f"[{i}] {title}\n{content}\n来源: {url}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"搜索失败: {str(e)}"