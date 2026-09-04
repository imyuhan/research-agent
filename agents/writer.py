from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from infrastructure.config import settings
from agents._token_tracker import make_usage_callback
from agents.evidence_utils import select_balanced_evidence
from agents.schemas import EvidenceItem, FindingItem

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的技术研究报告撰写者。请基于提供的研究资料，撰写一份结构清晰、内容详实的研究报告。

要求：
- 使用 Markdown 格式
- 包含以下结构:## 执行摘要、## 核心内容、## 结论
- 语言严谨、客观，适合技术读者
- 基于提供的 findings / evidence_pack 撰写，不要编造未提及的信息
- 证据包可能同时包含 KB 与 Web,两类证据都要参考,不要只盯着前面的 KB
- 引用标注 [n] 必须对应"引用来源列表"中的编号，不要编造列表之外的编号
- 优先使用 evidence_id 作为事实依据标记
- 不要自行编写"参考资料"章节，该章节会由系统在报告末尾自动生成
- 若提供了"审核意见"，必须逐条响应:针对每条意见落实到正文中修改，如认为某条不适用则简要说明理由
- 【本地知识库】部分为可选背景参考，若与研究资料冲突以研究资料为准"""),
    ("human", """研究主题:{topic}

研究结论(findings):
{findings}

证据包(evidence_pack):
{evidence_pack}

引用来源列表(编号对应报告正文中的 [n]):
{references}

审核意见(首次撰写时为"(无)"):
{review_feedback}

请撰写报告:""")
])

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.7,
    max_tokens=4000,  # 报告允许长一些,但要有个上限
)

writer_chain = writer_prompt | llm | StrOutputParser()


def _build_references(citations):
    """将引用列表格式化为注入 prompt 的编号文本"""
    if not citations:
        return "(无可用引用来源)"
    lines = []
    for c in citations:
        title = c.get("title", "") or "未命名来源"
        url = c.get("url", "")
        lines.append(f"[{c['index']}] {title} - {url}")
    return "\n".join(lines)


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    """按 URL / evidence_id 去重,保留顺序。"""
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in citations or []:
        key = item.get("url") or item.get("evidence_id") or f"{item.get('index', '')}:{item.get('title', '')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _format_findings(findings: list[FindingItem], max_items: int = 8) -> str:
    if not findings:
        return "(无)"
    lines = []
    for i, item in enumerate(findings[:max_items], 1):
        evidence_ids = ", ".join(item.get("evidence_ids", [])) or "(无)"
        gaps = "；".join(item.get("gaps", [])) or "无"
        lines.append(
            f"[F{i}] {item.get('query', '')}\n"
            f"结论: {item.get('summary', item.get('content', ''))}\n"
            f"支持证据: {evidence_ids}\n"
            f"待确认: {gaps}\n"
            f"置信度: {item.get('confidence', 0.0)}"
        )
    return "\n\n".join(lines)


def _format_evidence_pack(evidence_pack: list[EvidenceItem], max_items: int = 12) -> str:
    if not evidence_pack:
        return "(无)"
    lines = []
    for item in evidence_pack[:max_items]:
        kind = item.get("kind", "kb")
        if kind == "kb":
            lines.append(
                f"[{item['evidence_id']}] KB | {item.get('title', '')} | "
                f"source={item.get('source', '')} | score={item.get('score', 0.0):.2f}\n"
                f"{item.get('text', '')[:260]}"
            )
        else:
            lines.append(
                f"[{item['evidence_id']}] WEB | {item.get('title', '')} | url={item.get('url', '')}\n"
                f"{item.get('text', '')[:260]}"
            )
    return "\n\n".join(lines)


def _select_prompt_evidence(evidence_pack: list[EvidenceItem], max_items: int) -> list[EvidenceItem]:
    """优先交错保留 KB / Web 证据,避免前缀截断把 web 全挤掉。"""
    selected = select_balanced_evidence(evidence_pack, max_items=max_items)
    if selected:
        return selected
    return evidence_pack[:max_items]


def _append_references(draft, citations, evidence_pack=None):
    """在草稿末尾追加参考资料章节(真实 URL,不依赖 LLM 输出)
    末尾追加两个章节(如有内容):
      - ## 参考资料         (web 引用)
      - ## 证据追踪         (KB / Web 证据)
    """
    evidence_pack = evidence_pack or []
    blocks = []

    # 1) web 引用
    if citations and "## 参考资料" not in draft:
        web_lines = "\n".join(
            f"[{c['index']}] {c['title']} - {c['url']}"
            for c in citations
        )
        blocks.append(f"## 参考资料\n{web_lines}")

    # 2) 证据追踪
    if evidence_pack and "## 证据追踪" not in draft:
        evidence_lines = []
        for item in evidence_pack:
            if item.get("kind") == "kb":
                evidence_lines.append(
                    f"[{item['evidence_id']}] KB | {item.get('title', '')} | "
                    f"{item.get('source', '')}"
                )
            else:
                evidence_lines.append(
                    f"[{item['evidence_id']}] WEB | {item.get('title', '')} | "
                    f"{item.get('url', '')}"
                )
        blocks.append("## 证据追踪\n" + "\n".join(evidence_lines))

    if not blocks:
        return draft
    return f"{draft.rstrip()}\n\n" + "\n\n".join(blocks)


def _format_kb_context(kb_chunks, max_chunks=3):
    """把 KB 命中格式化,只取前 max_chunks 条,避免 prompt 过长"""
    if not kb_chunks:
        return "(本地知识库无相关命中)"
    top = kb_chunks[:max_chunks]
    parts = []
    for i, hit in enumerate(top, 1):
        text = hit.get("text", "")
        score = hit.get("score", 0.0)
        # 截断过长文本,避免 prompt 撑爆
        snippet = text[:300] + "..." if len(text) > 300 else text
        parts.append(f"[KB-{i}] (相关度:{score:.2f}) {snippet}")
    return "\n\n".join(parts)


def _truncate_sources(sources_text: str, max_chars: int = 12000) -> str:
    """
    sources 总量截断:超过 max_chars 时保留头部 + 尾部,中间省略。
    5 个 query × 4000 字符 ≈ 20000 token,超长会爆 writer prompt。
    """
    if not sources_text or len(sources_text) <= max_chars:
        return sources_text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return (
        sources_text[:head]
        + f"\n\n... [中间省略 {len(sources_text) - head - tail} 字符] ...\n\n"
        + sources_text[-tail:]
    )


def writer_node(state: dict) -> dict:
    """写作节点:基于 findings + evidence_pack 生成报告草稿,并追加参考资料"""
    topic = state.get("topic", "")
    findings = state.get("findings", state.get("search_results", []))
    citations = _dedupe_citations(state.get("citations", []))
    review_feedback = state.get("review_feedback", "")
    evidence_pack = state.get("evidence_pack", [])

    if not findings and not evidence_pack:
        print("⚠️ 没有研究资料")
        return {"draft": "未能获取研究资料,无法生成报告。", "current_step": "writer"}

    prompt_findings = findings[:settings.WRITER_MAX_FINDINGS_IN_PROMPT]
    prompt_evidence = _select_prompt_evidence(evidence_pack, settings.WRITER_MAX_EVIDENCE_IN_PROMPT)
    prompt_citations = citations[:settings.WRITER_MAX_CITATIONS_IN_PROMPT]

    findings_text = _format_findings(prompt_findings, max_items=settings.WRITER_MAX_FINDINGS_IN_PROMPT)
    evidence_text = _format_evidence_pack(prompt_evidence, max_items=settings.WRITER_MAX_EVIDENCE_IN_PROMPT)
    references_text = _build_references(prompt_citations)

    print(
        f"✍️ Writer 正在撰写报告,基于 {len(prompt_findings)} 条 findings + "
        f"{len(prompt_evidence)} 条证据 + {len(prompt_citations)} 条引用...\n"
    )

    draft = ""
    # 收集本次节点的 token 用量
    node_usage: dict = {}
    cb = make_usage_callback("writer", node_usage)
    for text in writer_chain.stream(
        {
            "topic": topic,
            "findings": findings_text,
            "evidence_pack": evidence_text,
            "references": references_text,
            "review_feedback": review_feedback or "(无)",
        },
        config={"callbacks": [cb]},
    ):
        draft += text
        print(text, end="", flush=True)
    print(f"\n\n   报告生成完成,长度: {len(draft)} 字符")

    draft = _append_references(draft, prompt_citations, evidence_pack=prompt_evidence)

    return {
        "draft": draft,
        "current_step": "writer",
        "token_usage": node_usage,
    }
