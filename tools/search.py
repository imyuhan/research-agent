from langchain_tavily import TavilySearch
from config import settings

search = TavilySearch(max_results=settings.MAX_SEARCH_RESULTS)


def web_search(query: str) -> list[dict]:
    """执行网络搜索，返回结构化结果列表 [{title, url, content}]，空结果或出错返回 []"""
    try:
        response = search.invoke({"query": query})
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