from tools.search import web_search


def researcher_node(state: dict) -> dict:
    """研究节点：并行执行搜索，汇总结果并收集引用"""
    queries = state.get("search_queries", [])

    if not queries:
        print("⚠️ 没有搜索查询")
        return {"search_results": [], "citations": [], "current_step": "researcher"}

    print(f"🔍 Researcher 开始执行 {len(queries)} 个搜索:")

    all_results = []
    all_citations = []
    for query in queries:
        print(f"   搜索: {query}")
        items = web_search(query)

        # 组装正文文本（编号与全局引用编号一致，供 Writer 正确对应参考资料）
        if not items:
            content = "未找到相关搜索结果"
        else:
            parts = []
            for r in items:
                all_citations.append({
                    "index": len(all_citations) + 1,
                    "query": query,
                    "title": r["title"],
                    "url": r["url"],
                })
                idx = all_citations[-1]["index"]
                parts.append(f"[{idx}] {r['title']}\n{r['content']}\n来源: {r['url']}")
            content = "\n\n".join(parts)

        all_results.append({
            "query": query,
            "content": content
        })

        print(f"   结果长度: {len(content)} 字符")

    print(f"✅ 研究完成，共收集 {len(all_results)} 条结果")

    return {
        "search_results": all_results,
        "citations": all_citations,
        "current_step": "researcher"
    }