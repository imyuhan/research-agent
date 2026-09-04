# infrastructure/rerank_client.py
"""BGE Reranker 客户端封装（绝对路径版）

v2-m3 的 huggingface cache 路径解析对 cwd 敏感，
这里强制 resolve 成绝对路径，避免在不同 cwd 下出现 FileNotFoundError。
"""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from infrastructure.config import settings


class BGERRerankClient:
    """BGE Reranker v2-m3 客户端"""

    _instance: "BGERRerankClient | None" = None
    _lock = RLock()

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = settings.BGE_RERANK_MODEL_PATH
        # 关键：把相对路径 resolve 成绝对路径，绕开 cwd 依赖
        model_path = str(Path(model_path).resolve())
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_path, device=device)
        self.model_path = model_path

    @classmethod
    def get_instance(cls) -> "BGERRerankClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def score_documents(self, query: str, documents: List[str]) -> List[float]:
        """按输入顺序返回每个候选的重排分数。"""
        if not documents:
            return []
        pairs = [[query, d] for d in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(
        self, query: str, documents: List[str], top_k: int | None = None
    ) -> List[Tuple[str, float]]:
        if not documents:
            return []
        scores = self.score_documents(query, documents)
        ranked = sorted(
            zip(documents, scores), key=lambda x: float(x[1]), reverse=True
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
