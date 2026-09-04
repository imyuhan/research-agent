from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from infrastructure.config import settings
from agents.tools.date_utils import detect_date, to_chinese
from agents._token_tracker import add_usage, make_usage_callback

planner_system = """你是一个研究规划专家。请将用户的研究主题拆解为 1-5 个具体的子问题。

要求：
- 每个子问题相互独立，可并行研究
- 覆盖主题的核心维度
- 适合用搜索引擎检索回答
- 如果用户的问题本身已经是一个可以直接回答的简单问题，请只输出1个自问，直接复用用户原话，不要强行拆解
- 信息收集完成后，只输出最终的子问题列表，每行一个，不要编号，不要多余解释也不要输出思考过程"""

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.5,
    max_tokens=1000,  # 规划只需要 1-5 个短问题,不需要更长
)

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", planner_system),
    ("human", "{human_text}"),
])

planner_chain = planner_prompt | llm | StrOutputParser()


def planner_node(state: dict) -> dict:
    """规划节点：将主题拆解为研究子问题（简单问题可不拆）"""
    topic = state.get("topic", "")

    print(f"📋 Planner 正在规划主题: {topic}")

    date_iso = detect_date(topic)
    date_cn = to_chinese(date_iso) if date_iso else ""
    if date_cn:
        print(f"   📅 用户指定日期: {date_cn}")

    human_text = f"研究主题：{topic}"
    if date_cn:
        human_text += f"\n本次研究指定日期为 {date_cn}，拆解的子问题应围绕该日期展开。"

    # 收集本次节点的 token 用量,后续合并到 state
    node_usage: dict = {}
    text = planner_chain.invoke(
        {"human_text": human_text},
        config={"callbacks": [make_usage_callback("planner", node_usage)]},
    )
    add_usage(node_usage, "planner", text)  # 兜底:有的实现回调不到,从响应里再读一次

    queries = [q.strip() for q in text.strip().split("\n") if q.strip()]

    print(f"   生成 {len(queries)} 个子问题:")
    for q in queries:
        print(f"   • {q}")

    return {
        "plan": queries,
        "search_queries": queries,
        "current_step": "planner",
        "token_usage": node_usage,
    }
