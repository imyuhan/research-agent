# @File     : planner.py
# @Time     : 2026/8/4 11:46
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import settings

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个研究规划专家。请将用户的研究主题拆解为 3-5 个具体的子问题。
要求：
- 每个子问题相互独立，可并行研究
- 覆盖主题的核心维度
- 适合用搜索引擎回答
- 只输出子问题列表，每行一个，不要编号，不要多余解释"""),
    ("human", "研究主题：{topic}")
])

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.5
)


def planner_node(state: dict) -> dict:
    """规划节点：将主题拆解为研究子问题"""
    topic = state.get("topic", "")

    print(f"📋 Planner 正在规划主题: {topic}")

    chain = planner_prompt | llm
    response = chain.invoke({"topic": topic})

    # 解析子问题：按行分割，去空行
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    print(f"   生成 {len(queries)} 个子问题:")
    for q in queries:
        print(f"   • {q}")

    return {
        "plan": queries,
        "search_queries": queries,
        "current_step": "planner"
    }