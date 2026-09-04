# -*- coding: utf-8 -*-
from __future__ import annotations

import concurrent.futures
import hashlib
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agents._token_tracker import make_usage_callback
from agents.schemas import EvidenceItem, FindingItem
from agents.tools.date_utils import date_window, detect_date, to_chinese
from agents.tools.search import web_search, web_search_tool
from infrastructure.config import settings
from service import get_rag

researcher_system = """你是一个研究智能体。针对给定的研究子问题,综合使用本地知识库和网络搜索收集资料。

要求:
- 优先参考【本地知识库】,如果 KB 已有答案,先用 KB
- KB 不足或时效性不足时,使用 web_search_tool 补充
- 收集完成后,输出一个简短结论,并明确列出支持证据 ID
- 不要编造证据 ID,只允许使用输入证据包中的 ID
"""

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0.3,
    max_tokens=1200,
)

researcher_llm = llm.bind_tools([web_search_tool])

MAX_TOOL_ROUNDS = 4


def _web_evidence_id(url: str) -> str:
    return "W-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def _format_web(items):
    if not items:
        return "未找到相关搜索结果"
    return "\n\n".join(
        f"[{i + 1}] {r['title']}\n{r['content']}\n来源: {r['url']}"
        for i, r in enumerate(items)
    )


def _dedupe_web_results(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items or []:
        key = item.get("url") or item.get("title") or item.get("content", "")[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _format_evidence_pack(evidence_pack, max_items: int = 8) -> str:
    if not evidence_pack:
        return "（当前没有可用证据）"
    lines = []
    for item in evidence_pack[:max_items]:
        kind = item.get("kind", "kb")
        title = item.get("title") or item.get("source") or "无标题"
        score = item.get("score", 0.0)
        snippet = (item.get("text", "") or "")[:260]
        if len(item.get("text", "") or "") > 260:
            snippet += "..."
        if kind == "kb":
            lines.append(
                f"[{item['evidence_id']}] KB | {title} | "
                f"source={item.get('source', '')} | score={score:.2f}\n{snippet}"
            )
        else:
            lines.append(
                f"[{item['evidence_id']}] WEB | {title} | "
                f"url={item.get('url', '')}\n{snippet}"
            )
    return "\n\n".join(lines)


def _extract_field(text: str, labels: tuple[str, ...]) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        for label in labels:
            if stripped.startswith(label):
                return stripped.split("：", 1)[-1].split(":", 1)[-1].strip()
    return ""


def _extract_support_ids(text: str) -> list[str]:
    value = _extract_field(text, ("支持证据", "支持证据ID", "支持证据 IDs"))
    if not value:
        return []
    return [p.strip() for p in re.split(r"[，,、\s]+", value) if p.strip()]


def _extract_gaps(text: str) -> list[str]:
    value = _extract_field(text, ("待确认", "不确定点", "缺口"))
    if not value:
        return []
    return [p.strip() for p in re.split(r"[；;，,]\s*", value) if p.strip()]


def _parse_finding(text: str, evidence_pack: list[EvidenceItem]) -> FindingItem:
    summary = _extract_field(text, ("结论", "摘要", "总结"))
    if not summary:
        summary = text.strip()
    support_ids = _extract_support_ids(text)
    gaps = _extract_gaps(text)
    if not support_ids and evidence_pack:
        support_ids = [item["evidence_id"] for item in evidence_pack[:2] if item.get("evidence_id")]
    confidence = 0.45 + min(len(support_ids), 4) * 0.12 - min(len(gaps), 3) * 0.05
    confidence = max(0.1, min(0.95, confidence))
    needs_more_search = bool(gaps) or len(support_ids) == 0
    return {
        "summary": summary,
        "content": summary,
        "evidence_ids": support_ids,
        "gaps": gaps,
        "confidence": round(confidence, 2),
        "needs_more_search": needs_more_search,
    }


def _researcher_one_query(
    query: str,
    system_prompt: str,
    date_cn: str,
    window,
    cap: int,
    node_usage: dict = None,
):
    """处理单个子问题:先检索,再基于证据包生成结构化 finding."""
    print(f"   研究:{query}")

    rag = get_rag()
    kb_hits = rag.query(query, top_k=3) or []
    print(
        f"    📚 KB 召回 {len(kb_hits)} 条 (相关度: "
        f"{[round(h.get('score', 0), 2) for h in kb_hits[:3]]})"
    )

    collected_web: list[dict] = []
    search_count = 0

    kb_evidence: list[EvidenceItem] = []
    for hit in kb_hits:
        evidence_id = hit.get("evidence_id") or hit.get("chunk_id") or hit.get("id") or ""
        kb_evidence.append({
            "evidence_id": evidence_id,
            "kind": "kb",
            "query": query,
            "text": hit.get("text", ""),
            "title": hit.get("title", ""),
            "source": hit.get("source", ""),
            "doc_id": hit.get("doc_id", ""),
            "chunk_id": hit.get("chunk_id", evidence_id),
            "chunk_index": hit.get("chunk_index", -1),
            "score": float(hit.get("score", 0.0) or 0.0),
            "retrieval_score": float(hit.get("retrieval_score", hit.get("score", 0.0)) or 0.0),
            "rerank_score": hit.get("rerank_score"),
            "metadata": hit.get("metadata", {}),
        })

    evidence_pack: list[EvidenceItem] = list(kb_evidence)

    messages = [
        SystemMessage(content=system_prompt + "\n\n【本地知识库参考】\n" + _format_evidence_pack(kb_evidence)),
        HumanMessage(content=f"研究子问题:{query}"),
    ]
    cb = make_usage_callback("researcher", node_usage) if node_usage is not None else None
    cfg = {"callbacks": [cb]} if cb is not None else None

    for _ in range(MAX_TOOL_ROUNDS):
        response = researcher_llm.invoke(messages, config=cfg)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            if tc["name"] == "web_search_tool":
                base_q = tc.get("args", {}).get("query", query)
                tc_start = tc.get("args", {}).get("start_date", "") or window[0]
                tc_end = tc.get("args", {}).get("end_date", "") or window[1]
                search_q = f"{base_q} {date_cn}".strip() if date_cn else base_q
                items = web_search(search_q, start_date=tc_start, end_date=tc_end)
                collected_web.extend(items)
                search_count += 1
                result_txt = _format_web(items)
            else:
                result_txt = f"该工具当前不可用:{tc['name']},请改用 web_search_tool"
            messages.append(ToolMessage(content=result_txt, tool_call_id=tc["id"]))

    collected_web = _dedupe_web_results(collected_web)[:cap]

    for i, r in enumerate(collected_web, 1):
        evidence_pack.append({
            "evidence_id": _web_evidence_id(r.get("url", "") + f"#{i}"),
            "kind": "web",
            "query": query,
            "title": r.get("title", "无标题"),
            "url": r.get("url", ""),
            "source": r.get("url", ""),
            "text": r.get("content", ""),
            "score": 1.0,
            "metadata": {"origin": "tavily"},
        })

    summary_prompt = (
        system_prompt
        + "\n\n【证据包】\n"
        + _format_evidence_pack(evidence_pack, max_items=max(6, cap))
        + "\n\n输出格式:\n"
        + "结论: 1-3 句简短结论\n"
        + "支持证据: E1, E2\n"
        + "待确认: 如有缺口再写,没有则写 空\n"
        + "要求: 不要编造证据 ID,只使用证据包里的 ID。"
    )
    summary_messages = [
        SystemMessage(content=summary_prompt),
        HumanMessage(content=f"研究子问题:{query}\n请只基于证据包总结。"),
    ]
    summary_response = llm.invoke(summary_messages, config=cfg)
    summary_text = summary_response.content or ""

    finding = _parse_finding(summary_text, evidence_pack)
    finding.update({
        "query": query,
        "content": summary_text.strip() or finding["content"],
    })

    local_citations = []
    for r in collected_web:
        local_citations.append({
            "query": query,
            "title": r["title"],
            "url": r["url"],
            "evidence_id": _web_evidence_id(r.get("url", "") + f"#{len(local_citations) + 1}"),
        })

    print(f"    ✅ 完成:{query}")
    print(
        f"       搜索 {search_count} 次 | KB 命中 {len(kb_hits)} 条 | "
        f"web 引用 {len(local_citations)} 条 | 长度 {len(summary_text)} 字符"
    )

    return finding, local_citations, kb_evidence, evidence_pack


def researcher_node(state: dict) -> dict:
    """
    研究节点:用 KB + 网络搜索收集资料,并输出结构化 findings / evidence_pack。
    """
    queries = state.get("search_queries", [])
    existing_kb = state.get("kb_chunks", [])
    review_feedback = state.get("review_feedback", "")

    if not queries:
        print("⚠️ 没有搜索查询")
        return {
            "search_results": [],
            "findings": [],
            "evidence_pack": [],
            "citations": [],
            "kb_chunks": existing_kb,
            "current_step": "researcher",
            "route_reason": "no_queries",
            "token_usage": {},
        }

    system_prompt = researcher_system
    cap = settings.MAX_CITATIONS_PER_QUERY

    date_iso = detect_date(" ".join([state.get("topic", ""), *queries]))
    date_cn = to_chinese(date_iso) if date_iso else ""
    window = date_window(date_iso, 1, 1) if date_iso else ("", "")
    if date_cn:
        system_prompt += (
            f"\n\n本次研究的指定日期为 {date_cn}。"
            "所有搜索收集的内容必须围绕该日期。"
        )
        print(f"   📅 用户指定日期:{date_cn},搜索窗口 {window[0]} ~ {window[1]}")

    if review_feedback and state.get("needs_new_search", False):
        system_prompt += (
            f"\n\n【上一轮审核反馈】\n{review_feedback}\n"
            "请针对以上反馈调整搜索方向,补充 reviewer 要求的资料或来源。"
        )
        print(f"   📝 注入 reviewer 反馈(长度 {len(review_feedback)} 字符)")

    print(f"🔍 Researcher 开始研究 {len(queries)} 个子问题:")

    node_usage: dict = {}
    results_by_query: dict[str, FindingItem] = {}
    citations_by_query: dict[str, list[dict]] = {}
    kb_by_query: dict[str, list[EvidenceItem]] = {}
    evidence_by_query: dict[str, list[EvidenceItem]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_SEARCH_CONCURRENCY) as executor:
        futures = {
            executor.submit(_researcher_one_query, q, system_prompt, date_cn, window, cap, node_usage): q
            for q in queries
        }
        for future in concurrent.futures.as_completed(futures):
            q = futures[future]
            finding, local_citations, kb_hits, evidence_pack = future.result()
            results_by_query[q] = finding
            citations_by_query[q] = local_citations
            kb_by_query[q] = kb_hits
            evidence_by_query[q] = evidence_pack

    all_results = [results_by_query[q] for q in queries]
    all_citations = []
    all_kb_hits = []
    all_evidence = []
    for q in queries:
        for c in citations_by_query[q]:
            all_citations.append({
                "index": len(all_citations) + 1,
                "query": q,
                "title": c["title"],
                "url": c["url"],
                "evidence_id": c.get("evidence_id", ""),
            })
        for hit in kb_by_query[q]:
            all_kb_hits.append({"query": q, **hit})
        for ev in evidence_by_query[q]:
            all_evidence.append({"query": q, **ev})

    print("\n✅ 研究完成:")
    print(f"   - 研究结果: {len(all_results)} 条")
    print(f"   - KB 命中合计: {len(all_kb_hits)} 条")
    print(f"   - 证据总数: {len(all_evidence)} 条")
    print(f"   - Web 引用: {len(all_citations)} 条")

    return {
        "search_results": all_results,
        "findings": all_results,
        "citations": all_citations,
        "evidence_pack": all_evidence,
        "kb_chunks": all_kb_hits,
        "current_step": "researcher",
        "route_reason": "research_complete",
        "token_usage": node_usage,
    }
