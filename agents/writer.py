# @File     : writer.py
# @Time     : 2026/8/4 11:47
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import settings

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的技术研究报告撰写者。请基于提供的研究资料，撰写一份结构清晰、内容详实的研究报告。

要求：
- 使用 Markdown 格式
- 包含以下结构：## 执行摘要、## 核心内容、## 结论
- 语言严谨、客观，适合技术读者
- 基于提供的资料撰写，不要编造未提及的信息
- 每个关键论点后可标注引用来源 [n]"""),
    ("human", """研究主题：{topic}

研究资料：
{sources}

请撰写报告：""")
])

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.7
)


def writer_node(state: dict) -> dict:
    """写作节点：基于研究资料生成报告草稿"""
    topic = state.get("topic", "")
    sources = state.get("search_results", [])

    if not sources:
        print("⚠️ 没有研究资料")
        return {"draft": "未能获取研究资料，无法生成报告。", "current_step": "writer"}

    # 格式化资料
    sources_text = "\n\n".join([
        f"--- 查询: {s['query']} ---\n{s['content']}"
        for s in sources
    ])

    print(f"✍️ Writer 正在撰写报告，基于 {len(sources)} 条研究资料...")

    chain = writer_prompt | llm
    response = chain.invoke({
        "topic": topic,
        "sources": sources_text
    })

    draft = response.content
    print(f"   报告生成完成，长度: {len(draft)} 字符")

    return {
        "draft": draft,
        "current_step": "writer"
    }