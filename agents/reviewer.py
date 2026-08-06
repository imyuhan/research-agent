from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import settings
import re

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严格的研究质量审核员。请从以下维度评估报告：
1. 信息准确性（是否有明显事实错误）
2. 结构完整性（是否包含执行摘要、核心内容、结论）
3. 主题覆盖度（是否回答了研究主题的核心问题）
4. 内容深度（是否有实质性分析，而非泛泛而谈）

输出格式（严格按此格式）：
评分：1-10分
是否通过：是/否
修改意见：（如未通过，说明具体问题和修改建议）"""),
    ("human", "研究主题：{topic}\n\n报告草稿：\n{draft}")
])

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.3
)


def reviewer_node(state: dict) -> dict:
    """审核节点：评估报告质量，决定是否通过"""
    topic = state.get("topic", "")
    draft = state.get("draft", "")

    if not draft or len(draft) < 50:
        print("❌ Reviewer: 报告为空或太短")
        return {
            "review_feedback": "报告内容不足",
            "review_score": 0,
            "review_passed": False,
            "revision_count": state.get("revision_count", 0) + 1,
            "current_step": "reviewer"
        }

    print("🔍 Reviewer 正在审核报告...")

    chain = reviewer_prompt | llm
    response = chain.invoke({
        "topic": topic,
        "draft": draft
    })

    feedback = response.content

    # 解析评分（提取第一个数字）
    score = 5
    score_match = re.search(r'评分[：:]?\s*(\d+)', feedback)
    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(10, score))  # 限制在 1-10

    # 解析是否通过
    passed = False
    if re.search(r'是否通过[：:]?\s*是', feedback) or "通过：是" in feedback:
        passed = True
    # 评分 >= 8 默认通过（兜底逻辑）
    if score >= 8:
        passed = True

    print(f"   评分: {score}/10 | 是否通过: {'✅' if passed else '❌'}")
    if not passed:
        print(f"   修改意见: {feedback[:100]}...")

    return {
        "review_feedback": feedback,
        "review_score": score,
        "review_passed": passed,
        "revision_count": state.get("revision_count", 0) + 1 if not passed else state.get("revision_count", 0),
        "current_step": "reviewer"
    }