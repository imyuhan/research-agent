# @File     : researcher.py
# @Time     : 2026/8/4 11:47
from tools.search import web_search


def researcher_node(state: dict) -> dict:
    """研究节点：并行执行搜索，汇总结果"""
    queries = state.get("search_queries", [])

    if not queries:
        print("⚠️ 没有搜索查询")
        return {"search_results": [], "current_step": "researcher"}

    print(f"🔍 Researcher 开始执行 {len(queries)} 个搜索:")

    all_results = []
    for query in queries:
        print(f"   搜索: {query}")
        result = web_search(query)
        all_results.append({
            "query": query,
            "content": result
        })
        print(f"   结果长度: {len(result)} 字符")

    print(f"✅ 研究完成，共收集 {len(all_results)} 条结果")

    return {
        "search_results": all_results,
        "current_step": "researcher"
    }