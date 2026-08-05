from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import settings

writer_system_prompt = """你是一个专业的技术研究报告撰写者。请基于提供的研究资料，撰写一份结构清晰、内容详实的研究报告。

要求：
- 使用 Markdown 格式
- 包含以下结构：## 执行摘要、## 核心内容、## 结论
- 语言严谨、客观，适合技术读者
- 基于提供的资料撰写，不要编造未提及的信息
- 每个关键论点后可标注引用来源 [n]"""

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", writer_system_prompt),
    ("human", """研究主题：{topic}

研究资料：
{sources}

请撰写报告：""")
])

revision_prompt = ChatPromptTemplate.from_messages([
    ("system", writer_system_prompt + """

本次是修订任务：请针对审核修改意见逐条修订上一版草稿。
- 保留上一版中审核未指出的有价值内容，不要整篇推翻重写
- 修改意见可能涉及准确性、结构、覆盖度、深度，逐条回应
- 同样遵守原创性与引用规范"""),
    ("human", """研究主题：{topic}

研究资料：
{sources}

上一版草稿：
{previous_draft}

审核修改意见：
{feedback}

请根据修改意见修订报告：""")
])

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.7
)


def writer_node(state: dict) -> dict:
    """写作节点：基于研究资料生成报告草稿，如存在审核反馈则据此修订"""
    topic = state.get("topic", "")
    sources = state.get("search_results", [])
    feedback = state.get("review_feedback", "").strip()

    if not sources:
        print("⚠️ 没有研究资料")
        return {"draft": "未能获取研究资料，无法生成报告。", "current_step": "writer"}

    # 格式化资料
    sources_text = "\n\n".join([
        f"--- 查询: {s['query']} ---\n{s['content']}"
        for s in sources
    ])

    if feedback:
        # 修订模式：存在审核反馈时，结合上一版草稿针对性修订
        print(f"✍️ Writer 正在基于审核意见修订报告，基于 {len(sources)} 条研究资料...")
        chain = revision_prompt | llm
        response = chain.invoke({
            "topic": topic,
            "sources": sources_text,
            "previous_draft": state.get("draft", ""),
            "feedback": feedback,
        })
    else:
        print(f"✍️ Writer 正在撰写报告，基于 {len(sources)} 条研究资料...")
        chain = writer_prompt | llm
        response = chain.invoke({
            "topic": topic,
            "sources": sources_text,
        })

    draft = response.content
    print(f"   报告生成完成，长度: {len(draft)} 字符")

    return {
        "draft": draft,
        "current_step": "writer"
    }