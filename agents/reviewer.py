from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import re

from infrastructure.config import settings
from agents._token_tracker import add_usage, make_usage_callback
from agents.evidence_utils import select_balanced_evidence
from agents.schemas import EvidenceItem, FindingItem

reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严格的研究质量审核员。请从以下维度评估报告：
1. 信息准确性（是否有明显事实错误）
2. 结构完整性（是否包含执行摘要、核心内容、结论）
3. 主题覆盖度（是否回答了研究主题的核心问题）
4. 内容深度（是否有实质性分析，而非泛泛而谈）
5. 证据严谨性（结论是否有 evidence_id / 引用支撑，编号是否一致）
- 审核证据时要同时检查 KB 与 Web,不要只看前面的 KB 证据

输出格式（严格按此格式）：
评分：1-10分
是否通过：是/否
修改意见：（如未通过，说明具体问题和修改建议）

如果问题是关于【资料/来源/引用】的（例如要求摒弃某类来源、补充权威文献），
在修改意见末尾另起一行写：
需要重新搜索：是
并简要说明需要重新搜索的关键词或方向。"""),
    ("human", """研究主题：{topic}
报告草稿：
{draft}

研究结论(findings):
{findings}

证据包(evidence_pack):
{evidence_pack}

本地知识库参考(用于事实核验,可能有助于你判断信息准确性):
{kb_context}
""")
])

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.3,
    max_tokens=800,  # 评分+简短意见
)

reviewer_chain = reviewer_prompt | llm | StrOutputParser()


def _format_kb_for_review(kb_chunks, max_chunks=3):
    if not kb_chunks:
        return "(本地知识库无相关命中)"
    top = kb_chunks[:max_chunks]
    return "\n\n".join([
        f"[KB-{i+1}] {hit.get('text', '')[:300]}"
        for i, hit in enumerate(top)
    ])


def _format_findings_for_review(findings: list[FindingItem], max_items: int = 6) -> str:
    if not findings:
        return "(无)"
    lines = []
    for i, item in enumerate(findings[:max_items], 1):
        evidence_ids = ", ".join(item.get("evidence_ids", [])) or "(无)"
        lines.append(
            f"[F{i}] {item.get('query', '')}\n"
            f"结论: {item.get('summary', item.get('content', ''))}\n"
            f"支持证据: {evidence_ids}\n"
            f"待确认: {'；'.join(item.get('gaps', [])) or '无'}"
        )
    return "\n\n".join(lines)


def _format_evidence_for_review(evidence_pack: list[EvidenceItem], max_items: int = 10) -> str:
    selected = select_balanced_evidence(evidence_pack, max_items=max_items)
    if not selected:
        return "(无)"
    lines = []
    for item in selected:
        if item.get("kind") == "kb":
            lines.append(
                f"[{item['evidence_id']}] KB | {item.get('title', '')} | "
                f"{item.get('source', '')} | score={item.get('score', 0.0):.2f}"
            )
        else:
            lines.append(
                f"[{item['evidence_id']}] WEB | {item.get('title', '')} | "
                f"{item.get('url', '')}"
            )
    return "\n\n".join(lines)


def _truncate_draft(draft: str, head: int = 6000, tail: int = 2000) -> str:
    """
    长 draft 截断:头 + 尾,中间省略。
    reviewer 看的是结构/逻辑,不需要完整每一段。
    大幅降低每次审核的 prompt token,回退多次时省得更多。
    """
    if not draft or len(draft) <= head + tail + 50:
        return draft
    return (
        draft[:head]
        + f"\n\n... [中间省略 {len(draft) - head - tail} 字符] ...\n\n"
        + draft[-tail:]
    )


def _extract_feedback_only(full_text: str) -> str:
    m = re.search(r"修改意见[：:]\s*(.*)", full_text, re.DOTALL)
    if not m:
        return full_text.strip()
    body = m.group(1).strip()
    # 抽掉结尾可能存在的"需要重新搜索"段,这个标记不进 prompt
    body = re.split(r"需要重新搜索[：:]", body, maxsplit=1)[0].strip()
    return body


def _detect_needs_new_search(full_text: str) -> bool:
    """
    检测 reviewer 反馈里是否要求重新搜索(资料/引用层面的问题)。
    用关键词兜底，避免LLM写一句"建议补充来源让论证更充分"就误判
    """
    # 1) 显式信号:LLM 主动说需要重新搜
    if re.search(r"需要重新搜索[：:]\s*是", full_text):
        return True
    # 2) 关键词兜底(收紧):只匹配"明确要换/换掉/摒弃某类来源"的强动作
    strong_keywords = [
        r"摒弃.{0,8}来源",          # "摒弃营销号来源"
        r"摒弃.{0,8}引用",
        r"替换.{0,8}引用",
        r"更换.{0,8}来源",
        r"删除.{0,8}引用",
        r"全部替换.{0,8}来源",
        r"重新搜索.{0,15}(关键词|方向|资料)",  # 明确说"重新搜索 关键词/方向"
    ]
    if any(re.search(k, full_text) for k in strong_keywords):
        return True
    # 3) 软建议("补充来源/更多引用/权威文献")不算 needs_new_search,
    #    留给 writer 在现有资料基础上改写就行,回退到 writer 成本更低。
    return False


def reviewer_node(state: dict) -> dict:
    """审核节点：评估报告质量，决定是否通过,并标记是否需要重新搜索"""
    topic = state.get("topic", "")
    draft = state.get("draft", "")

    kb_chunks = state.get("kb_chunks", [])
    findings = state.get("findings", state.get("search_results", []))
    evidence_pack = state.get("evidence_pack", [])
    kb_text = _format_kb_for_review(kb_chunks)
    findings_text = _format_findings_for_review(findings)
    evidence_text = _format_evidence_for_review(evidence_pack)

    if not draft or len(draft) < 50:
        print("❌ Reviewer: 报告为空或太短")
        return {
            "review_feedback": "报告内容不足",
            "review_score": 0,
            "review_passed": False,
            "revision_count": state.get("revision_count", 0) + 1,
            "needs_new_search": False,
            "route_reason": "needs_rewrite",
            "current_step": "reviewer",
            "token_usage": {},
        }

    full_output = ""
    node_usage: dict = {}
    cb = make_usage_callback("reviewer", node_usage)
    # 关键:长 draft 截断,避免每次回退都把全篇喂进 prompt
    draft_for_review = _truncate_draft(draft)
    for text in reviewer_chain.stream(
        {
            "topic": topic,
            "draft": draft_for_review,
            "findings": findings_text,
            "evidence_pack": evidence_text,
            "kb_context": kb_text,
        },
        config={"callbacks": [cb]},
    ):
        full_output += text

    # 解析评分(取第一个出现的数字)
    score = 5
    score_match = re.search(r'评分[：:]?\s*(\d+)', full_output)
    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(10, score))

    # 解析是否通过
    passed = False
    if re.search(r'是否通过[：:]?\s*是', full_output) or "通过：是" in full_output:
        passed = True
    # 评分 >= 8 默认通过(兜底)
    if score >= 8:
        passed = True

    # 只回写"修改意见"正文,去掉"评分/是否通过"模板行
    feedback_only = _extract_feedback_only(full_output)
    needs_new_search = _detect_needs_new_search(full_output)
    route_reason = "approved" if passed else ("needs_research" if needs_new_search else "needs_rewrite")

    return {
        "review_feedback": feedback_only,  # 纯反馈
        "review_score": score,
        "review_passed": passed,
        "needs_new_search": needs_new_search,  # workflow 据此路由
        "route_reason": route_reason,
        "revision_count": state.get("revision_count", 0) + 1 if not passed else state.get("revision_count", 0),
        "current_step": "reviewer",
        "token_usage": node_usage,
    }
