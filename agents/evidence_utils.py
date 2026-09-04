from __future__ import annotations

from collections import deque
from typing import Any


def select_balanced_evidence(items: list[dict[str, Any]], max_items: int = 8) -> list[dict[str, Any]]:
    """
    轮流抽取不同 kind 的证据，避免前置排序把 web 或 KB 全部挤掉。
    保留同类内部原始顺序。
    """
    if not items or max_items <= 0:
        return []

    grouped: dict[str, deque[dict[str, Any]]] = {}
    order: list[str] = []

    for item in items:
        kind = str(item.get("kind", "kb") or "kb")
        if kind not in grouped:
            grouped[kind] = deque()
            order.append(kind)
        grouped[kind].append(item)

    selected: list[dict[str, Any]] = []
    while len(selected) < max_items and order:
        progressed = False
        for kind in order:
            queue = grouped.get(kind)
            if queue:
                selected.append(queue.popleft())
                progressed = True
                if len(selected) >= max_items:
                    break
        if not progressed:
            break

    return selected
