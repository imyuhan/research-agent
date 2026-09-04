from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from infrastructure.rerank_client import BGERRerankClient
from infrastructure.embedding_client import LocalBGEEmbeddings, get_embeddings
from infrastructure.vector_stores.base import VectorStore
from infrastructure.vector_stores.chroma_store import ChromaStore
from infrastructure.config import settings

# L3 服务层 RAG 统一接口
# ======================
# 检索配置
# ======================
@dataclass
class RetrievalConfig:
    """检索参数封装:方便以后扩展 rerank/filter"""
    top_k: int = 5                    # 返回条数
    mode: str = "hybrid"              # vector | bm25 | hybrid
    alpha: float = 0.5                # hybrid 时向量权重
    min_score: float = settings.RAG_MIN_SCORE   # 过滤掉低分结果(可选)
    enable_rerank: bool = settings.RAG_ENABLE_RERANK    # 是否启用 rerank(预留)
    rerank_top_k: int = settings.RAG_RERANK_TOP_K   # rerank 后保留前3

# ======================
# 核心服务类
# ======================
class RAGService:
    """对 L4 暴露的 RAG 入口"""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        store: Optional[VectorStore] = None,
        embeddings: Optional[LocalBGEEmbeddings] = None,
    ):
        self.config = config or RetrievalConfig(
            top_k=settings.RAG_TOP_K,
            mode="hybrid" if settings.RAG_STORE_BACKEND == "opensearch" else "vector",
            alpha=settings.RAG_HYBRID_ALPHA,
        )
        # 关键:不自己 new,允许测试时注入 mock
        self.embeddings = embeddings or get_embeddings()
        self.store = store or self._create_default_store()

    def _create_default_store(self) -> VectorStore:
        """根据 settings 创建 store"""
        backend = settings.RAG_STORE_BACKEND
        if backend == "opensearch":
            from infrastructure.vector_stores.opensearch_store import OpenSearchStore
            # OpenSearchStore 其它参数(host/port/index/dim)从 settings 读
            return OpenSearchStore(embeddings=self.embeddings)
        elif backend == "chroma":
            return ChromaStore(embeddings=self.embeddings)
        else:
            raise ValueError(f"Unknown RAG store backend: {backend}")

    def _format_hit(self, hit: dict) -> dict:
        """统一检索结果字段,方便上层当作证据对象使用。"""
        meta = hit.get("metadata") or {}
        chunk_id = hit.get("chunk_id") or hit.get("id") or meta.get("chunk_id", "")
        doc_id = hit.get("doc_id") or meta.get("doc_id", "")
        evidence_id = chunk_id or doc_id or hit.get("id", "")
        return {
            "evidence_id": evidence_id,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source": hit.get("source") or meta.get("source", ""),
            "title": hit.get("title") or meta.get("title", ""),
            "chunk_index": hit.get("chunk_index", meta.get("chunk_index", -1)),
            "content_hash": hit.get("content_hash") or meta.get("content_hash", ""),
            "text": hit.get("text", ""),
            "score": float(hit.get("score", 0.0) or 0.0),
            "retrieval_score": float(hit.get("retrieval_score", hit.get("score", 0.0)) or 0.0),
            "rerank_score": (
                float(hit["rerank_score"])
                if hit.get("rerank_score") is not None
                else None
            ),
            "metadata": meta,
        }

    def _list_source_records(self) -> List[dict]:
        if not hasattr(self.store, "list_source_records"):
            return []
        try:
            return self.store.list_source_records() or []
        except Exception:
            return []

    def _records_by_source(self) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in self._list_source_records():
            source = record.get("source", "")
            if source:
                grouped[source].append(record)
        return grouped

    @staticmethod
    def _fingerprint_file(file_path: str | Path) -> str:
        from data_layer.indexer import fingerprint_file

        return fingerprint_file(file_path)

    def _classify_files(self, file_paths: List[str | Path]) -> tuple[List[str], List[str]]:
        """返回 (待同步文件, 需要删除的旧 chunk ids)。"""
        pending: List[str] = []
        stale_ids: List[str] = []
        indexed = self._records_by_source()

        for file_path in file_paths:
            path = Path(file_path).resolve()
            if not path.is_file():
                continue
            source = str(path)
            current_hash = self._fingerprint_file(path)
            records = indexed.get(source, [])
            if not records:
                pending.append(source)
                continue

            indexed_hashes = {
                str(r.get("document_hash") or r.get("content_hash") or "")
                for r in records
                if r.get("document_hash") or r.get("content_hash")
            }
            if current_hash and current_hash in indexed_hashes:
                continue

            pending.append(source)
            for record in records:
                chunk_id = record.get("chunk_id") or record.get("id")
                if chunk_id:
                    stale_ids.append(str(chunk_id))

        return pending, stale_ids

    # ======================
    # 核心方法 1: query
    # ======================
    def query(self, q: str, top_k: int | None = None) -> List[dict]:
        """
        L4 业务层检索入口:对调用方屏蔽 store backend 差异
        流程:embed → 召回(min_score 过滤)→ rerank(可选)→ 截断 top_k
        """
        q = (q or "").strip()
        if not q:
            # 嵌入失败直接返回空,不要让 RAG 故障拖垮主流程
            return []
        # 第 1 步:把 query 文本转成 embedding。BGE 查询侧需要专用前缀,不能走文档 embedding。
        query_embedding = self.embeddings.embed_query(q)
        # 第 2 步:向量库召回(粗排,多召 3 倍给 rerank 留余地)
        # 注意:store.search() 的统一签名是 (query, query_embedding, top_k, mode=...)
        # mode 在 RetrievalConfig 里取,默认 hybrid;Chroma 不支持 hybrid 会在 store 内部降级
        hits = self.store.search(
            query=q,
            query_embedding=query_embedding,
            top_k=self.config.rerank_top_k * 3,
            mode=self.config.mode,
        )
        for h in hits:
            h["retrieval_score"] = float(h.get("score", 0.0) or 0.0)
        # 第 3 步:相似度阈值过滤(第一道筛选)
        hits = [h for h in hits if h.get("score", 0) >= self.config.min_score]
        if not hits:
            return []
        # 第 4 步:Rerank 精排(第二道筛选,可选)
        if self.config.enable_rerank:
            try:
                reranker = BGERRerankClient.get_instance()
                docs = [h.get("text", "") for h in hits]
                scores = reranker.score_documents(q, docs)
                for h, rerank_score in zip(hits, scores):
                    h["rerank_score"] = float(rerank_score)
                    h["score"] = float(rerank_score)
                hits = sorted(
                    hits,
                    key=lambda h: float(h.get("rerank_score", h.get("score", 0.0)) or 0.0),
                    reverse=True,
                )[:self.config.rerank_top_k]
            except Exception as e:
                # Rerank 失败时降级到原顺序,不拖垮主流程
                import logging
                import traceback
                logging.error(
                    f"Rerank failed, fallback to original order.\n"
                    f"  error type: {type(e).__name__}\n"
                    f"  error msg : {e}\n"
                    f"  traceback :\n{traceback.format_exc()}"
                )
        # 第 5 步:按调用方 top_k 截断(没传就用 RetrievalConfig 里的)
        final_hits = hits[:top_k or self.config.top_k]
        return [self._format_hit(h) for h in final_hits]

    # ======================
    # 核心方法 2: ingest(走文件)
    # ======================
    def ingest(self, file_path: str) -> int:
        """
        在线入库入口(单文件)
        走同一套增量同步逻辑,保证变更文件会先清旧数据再写新数据
        """
        p = Path(file_path)
        if not p.is_file():
            return 0
        result = self.ingest_files([str(p)])
        return result.get("written", 0)

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        直接 ingest 一段文本(用于 researcher agent 实时保存片段)
        返回: doc id
        """
        if not text or not text.strip():
            return ""

        emb = self.embeddings.embed_documents([text])
        if not emb:
            return ""

        ids = self.store.add(
            texts=[text],
            embeddings=emb,
            metadatas=[metadata or {}],
        )
        return ids[0] if ids else ""

    # ======================
    # 核心方法 3: list_indexed_sources + find_new_files + ingest_files
    # 用于"启动 main 时自动检测新文件并入库"
    # ======================
    def list_indexed_sources(self) -> List[str]:
        """
        列出 KB 中已索引的文件 source 列表(去重)
        业务层入口,屏蔽后端差异
        失败时返回空列表(fail-open,绝不抛异常)
        """
        if not hasattr(self.store, "list_sources"):
            return []
        try:
            return self.store.list_sources() or []
        except Exception:
            return []

    def find_new_files(self, directory) -> List[str]:
        """
        对比文件系统和 KB,找出需要同步的文件。

        这里不仅会找"库里没有的新文件",也会找"同一路径但内容已变更的文件"。
        """
        directory = Path(directory)
        if not directory.exists():
            return []
        disk_files = [str(p.resolve()) for p in directory.rglob("*.txt") if p.is_file()]
        pending, _ = self._classify_files(disk_files)
        return pending

    def ingest_files(self, file_paths: List[str]) -> dict:
        """
        增量入库入口(文件列表)
        会先删除已存在但内容变更的旧 chunk,再写入新版本。
        """
        from data_layer.indexer import build_index_for_files
        if not file_paths:
            return {"docs_loaded": 0, "chunks": 0, "written": 0, "deleted": 0}

        pending_files, stale_ids = self._classify_files(file_paths)
        deleted = 0
        if stale_ids and hasattr(self.store, "delete"):
            try:
                deleted = self.store.delete(ids=stale_ids)
            except Exception:
                deleted = 0

        if not pending_files:
            return {"docs_loaded": 0, "chunks": 0, "written": 0, "deleted": deleted}

        result = build_index_for_files(pending_files, store=self.store)
        result["deleted"] = deleted
        result["pending_files"] = len(pending_files)
        return result

    # ======================
    # 核心方法 4: health_check
    # ======================
    def health_check(self) -> bool:
        """检查 store + embedding 都健康"""
        try:
            return bool(self.store.health_check())
        except Exception:
            return False


# ======================
# 全局单例(关键!)
# ======================
_service: Optional[RAGService] = None


def get_rag() -> RAGService:
    """L4 业务层唯一入口"""
    global _service
    if _service is None:
        _service = RAGService()
    return _service


def reset_rag() -> None:
    """测试时用 — 重置单例"""
    global _service
    _service = None
