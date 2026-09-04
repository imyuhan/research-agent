"""
agents/_token_tracker.py
统一的 token 用量采集与聚合工具。
支持两种使用方式：
  1) make_usage_callback(node_name) -> 回调 handler,装到 ChatOpenAI(config={"callbacks": [...]})
  2) collect_from_response(response, accumulator) -> 从 AIMessage 读 usage_metadata 累加
"""
from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Optional

try:
    # langchain >=0.1 推荐路径
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # 旧版本兜底
    from langchain.callbacks.base import BaseCallbackHandler  # type: ignore


# ---------- 数据结构 ----------
_USAGE_LOCK = RLock()


def _empty_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _ensure_node(usage: Dict[str, Dict[str, int]], node: str) -> Dict[str, int]:
    """确保 usage[node] 存在,返回该节点的桶(原地写)"""
    if node not in usage:
        usage[node] = _empty_usage()
    return usage[node]


def _merge_into(usage: Dict[str, Dict[str, int]], node: str, info: Dict[str, int]) -> None:
    """把单次 LLM 调用的 token 累加到 node 桶"""
    with _USAGE_LOCK:
        bucket = _ensure_node(usage, node)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            bucket[k] += int(info.get(k, 0) or 0)


def add_usage(usage: Dict[str, Dict[str, int]], node: str, response: Any) -> None:
    """
    从 AIMessage / dict / None 中抽取 usage_metadata 并累加到 node 桶。
    无 usage 时静默跳过(部分兼容 OpenAI 协议的服务不返回 usage)。
    """
    info = _extract_usage(response)
    if not info:
        return
    _merge_into(usage, node, info)


def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
    """从 LLM 响应里取 usage_metadata,容错各种形态"""
    if response is None:
        return None

    # 1) AIMessage 对象的 .usage_metadata / .response_metadata
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return _normalize(usage)

    meta = getattr(response, "response_metadata", None) or {}
    if isinstance(meta, dict) and meta.get("usage"):
        return _normalize(meta["usage"])

    # 2) 已经是 dict
    if isinstance(response, dict):
        if "usage_metadata" in response and response["usage_metadata"]:
            return _normalize(response["usage_metadata"])
        if "token_usage" in response and response["token_usage"]:
            return _normalize(response["token_usage"])
        if "usage" in response and response["usage"]:
            return _normalize(response["usage"])
        if "response_metadata" in response and isinstance(response["response_metadata"], dict):
            inner = response["response_metadata"].get("usage")
            if inner:
                return _normalize(inner)

    return None


def _normalize(raw: Any) -> Dict[str, int]:
    """
    把 usage 字典统一成 {prompt_tokens, completion_tokens, total_tokens}。
    LangChain 标准 usage_metadata 长这样:
        {"input_tokens": N, "output_tokens": N, "total_tokens": N}
    也有可能是:
        {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
        {"prompt_tokens": N, "completion_tokens": N}
    """
    if not isinstance(raw, dict):
        return _empty_usage()

    pt = raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0
    ct = raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0
    tt = raw.get("total_tokens", 0) or 0
    if not tt:
        tt = int(pt) + int(ct)
    return {
        "prompt_tokens": int(pt),
        "completion_tokens": int(ct),
        "total_tokens": int(tt),
    }


# ---------- 回调 handler(给 config={"callbacks":[...]} 用) ----------
class _UsageCollectHandler(BaseCallbackHandler):
    """每次 LLM 结束时把 usage 写回共享 usage 字典。"""

    def __init__(self, usage: Dict[str, Dict[str, int]], node: str):
        self._usage = usage
        self._node = node

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):  # noqa: D401
        # 优先读 LLMResult.llm_output 里的原始 usage,比逐个 message 更稳
        try:
            llm_output = getattr(response, "llm_output", None)
            if llm_output:
                add_usage(self._usage, self._node, llm_output)
                return
        except Exception:
            pass

        # fallback: response 可能是 LLMResult,含 generations 列表,每条 .message 是 AIMessage
        try:
            results = getattr(response, "generations", None) or []
        except Exception:
            return
        for gen_list in results:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    add_usage(self._usage, self._node, msg)


def make_usage_callback(node: str, usage: Dict[str, Dict[str, int]]):
    """工厂:返回一个 callback handler,装到 ChatOpenAI(callbacks=[...])。"""
    return _UsageCollectHandler(usage=usage, node=node)


# ---------- 聚合 ----------
def total(usage: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """把每个节点的桶相加,返回全程总计"""
    out = _empty_usage()
    for bucket in usage.values():
        for k in out:
            out[k] += int(bucket.get(k, 0) or 0)
    return out


def merge_state_usage(prev: Optional[Dict[str, Dict[str, int]]],
                      new: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """
    合并两次统计(prev 是 state 里已有,new 是本次节点产生的)。
    节点重新进入时需要累加而不是覆盖。
    """
    out: Dict[str, Dict[str, int]] = {}
    for src in (prev or {}, new):
        for node, bucket in src.items():
            cur = _ensure_node(out, node)
            for k in cur:
                cur[k] += int(bucket.get(k, 0) or 0)
    return out


# ---------- 格式化输出 ----------
def format_table(usage: Dict[str, Dict[str, int]]) -> str:
    """格式化成 main.py 可直接打印的表格字符串"""
    if not usage:
        return "   (本次无 token 用量记录)"

    # 固定列宽,中文环境也能对齐
    header = f"   {'节点':<14} {'prompt':>10} {'completion':>12} {'total':>10}"
    sep = "   " + "-" * (len(header) - 3)
    lines = [header, sep]
    # 节点按固定顺序展示,缺则补 0
    for node in ("planner", "researcher", "writer", "reviewer"):
        b = usage.get(node, _empty_usage())
        lines.append(
            f"   {node:<14} {b['prompt_tokens']:>10} {b['completion_tokens']:>12} {b['total_tokens']:>10}"
        )
    # 其他不在固定顺序里的节点
    extra = [n for n in usage.keys() if n not in ("planner", "researcher", "writer", "reviewer")]
    for node in extra:
        b = usage[node]
        lines.append(
            f"   {node:<14} {b['prompt_tokens']:>10} {b['completion_tokens']:>12} {b['total_tokens']:>10}"
        )
    lines.append(sep)
    t = total(usage)
    lines.append(
        f"   {'总计':<14} {t['prompt_tokens']:>10} {t['completion_tokens']:>12} {t['total_tokens']:>10}"
    )
    return "\n".join(lines)
