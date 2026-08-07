from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import settings

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的技术研究报告撰写者。请基于提供的研究资料，撰写一份结构清晰、内容详实的研究报告。

要求：
- 使用 Markdown 格式
- 包含以下结构：## 执行摘要、## 核心内容、## 结论
- 语言严谨、客观，适合技术读者
- 基于提供的资料撰写，不要编造未提及的信息
- 引用标注 [n] 必须对应"引用来源列表"中的编号，不要编造列表之外的编号
- 不要自行编写"参考资料"章节，该章节会由系统在报告末尾自动生成
- 若提供了"审核意见"，必须逐条响应：针对每条意见落实到正文中修改，如认为某条不适用则简要说明理由"""),
    ("human", """研究主题：{topic}

研究资料：
{sources}

引用来源列表（编号对应报告正文中的 [n]）：
{references}

审核意见（首次撰写时为"（无）"）：
{review_feedback}

请撰写报告：""")
])

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.7
)

writer_chain = writer_prompt | llm | StrOutputParser()


def _build_references(citations: list) -> str:
    """将引用列表格式化为注入 prompt 的编号文本"""
    if not citations:
        return "（无可用引用来源）"
    return "\n".join([
        f"[{c['index']}] {c['title']} — {c['url']}"
        for c in citations
    ])


def _append_references(draft: str, citations: list) -> str:
    """在草稿末尾追加参考资料章节（真实 URL，不依赖 LLM 输出）"""
    if not citations or "## 参考资料" in draft:
        return draft
    refs = "\n".join([
        f"[{c['index']}] {c['title']} — {c['url']}"
        for c in citations
    ])
    return f"{draft.rstrip()}\n\n## 参考资料\n{refs}"


def writer_node(state: dict) -> dict:
    """写作节点：基于研究资料生成报告草稿，并追加参考资料"""
    topic = state.get("topic", "")
    sources = state.get("search_results", [])
    citations = state.get("citations", [])
    review_feedback = state.get("review_feedback", "")

    if not sources:
        print("⚠️ 没有研究资料")
        return {"draft": "未能获取研究资料，无法生成报告。", "current_step": "writer"}

    # 格式化资料
    sources_text = "\n\n".join([
        f"--- 查询: {s['query']} ---\n{s['content']}"
        for s in sources
    ])

    references_text = _build_references(citations)

    print(f"✍️ Writer 正在撰写报告，基于 {len(sources)} 条研究资料...\n")

    draft = ""
    for text in writer_chain.stream({
        "topic": topic,
        "sources": sources_text,
        "references": references_text,
        "review_feedback": review_feedback or "（无）",
    }):
        draft += text
        print(text, end="", flush=True)
    print(f"\n\n   报告生成完成，长度: {len(draft)} 字符")

    draft = _append_references(draft, citations)

    return {
        "draft": draft,
        "current_step": "writer"
    }