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

        # 组装正文文本
        if not items:
            content = "未找到相关搜索结果"
        else:
            content = "\n\n".join([
                f"[{i}] {r['title']}\n{r['content']}\n来源: {r['url']}"
                for i, r in enumerate(items, 1)
            ])

        all_results.append({
            "query": query,
            "content": content
        })

        # 收集引用（全局编号，供 Writer 注入与报告参考资料使用）
        for r in items:
            all_citations.append({
                "index": len(all_citations) + 1,
                "query": query,
                "title": r["title"],
                "url": r["url"],
            })

        print(f"   结果长度: {len(content)} 字符")

    print(f"✅ 研究完成，共收集 {len(all_results)} 条结果")

    return {
        "search_results": all_results,
        "citations": all_citations,
        "current_step": "researcher"
    }