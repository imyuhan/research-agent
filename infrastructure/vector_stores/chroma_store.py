from typing import List, Dict, Any
import uuid
from langchain_chroma import Chroma

from .base import VectorStore
from ..config import settings


class ChromaStore(VectorStore):
    def __init__(
            self,
            embeddings,  # ← 注入,LocalBGEEmbeddings 实例
            persist_directory: str = None,
            collection_name: str = "rag_docs",  # 关键:用 collection 区分多库
    ):
        self.embeddings = embeddings
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name

        # 关键点:Chroma 第一次建库时如果传 embeddings,会用它算向量
        # 后续检索 .similarity_search_by_vector(q_emb) 用的就是同一个 q_emb
        # 必须保证搜索时也用同一个 embeddings 实例,否则归一化/维度对不上
        self.db = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,  # ← LangChain 风格注入
        )

    def add(
            self,
            texts: List[str],
            embeddings: List[List[float]],
            metadatas: List[Dict[str, Any]] = None,
            ids: List[str] | None = None,
    ) -> List[str]:
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        self.db._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in texts],
        )
        return ids

    def search(
            self,
            query: str,
            query_embedding: List[float],
            top_k: int = None,
            mode: str = "vector",  # Chroma 只支持 vector,传 hybrid/bm25 给个 warning
    ) -> List[Dict[str, Any]]:
        top_k = top_k or settings.RAG_TOP_K

        if mode != "vector":
            # 关键点:Chroma 没有 BM25,降级为纯向量检索
            # 不要 raise,因为这是 fallback,挂了反而阻断主流程
            import warnings
            warnings.warn(f"ChromaStore only supports 'vector' mode, got '{mode}'")
            mode = "vector"

        # 关键点:LangChain Chroma 的检索有两种
        # 1) .similarity_search_by_vector(embedding) → 不返回 score
        # 2) .similarity_search_with_score(query, k) → 返回 (doc, score)
        # 我们要"按向量检索 + 拿到 score",用 .similarity_search_by_vector_with_score (LangChain 0.1+)
        # 或者 .similarity_search_by_vector + 后续自己算 score(复杂)
        results = self.db.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding,
            k=top_k,
        )
        # results: List[Tuple[Document, float]]
        # 关键点:Chroma 的 score 是 L2 距离(越小越相似),不是相似度
        # OpenSearch 的 score 是越大越相似,这里要反转一下,统一接口语义
        hits = []
        for doc, relevance in results:
            meta = doc.metadata or {}
            hits.append({
                "text": doc.page_content,
                "score": max(0.0, min(1.0, float(relevance))),  # 关键:距离→相似度的简单换算
                "metadata": meta,
                "id": meta.get("chunk_id") or meta.get("id") or meta.get("doc_id", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "doc_id": meta.get("doc_id", ""),
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "content_hash": meta.get("content_hash", ""),
            })
        return hits

    def delete(self, ids: List[str] = None, delete_all: bool = False) -> int:
        if delete_all:
            # 关键点:Chroma 没有"清空 collection"的方法,只有"删 collection 再重建"
            # .delete_collection() 是真的销毁
            # 删完要重新初始化一个空的,否则后续 add 报错
            self.db.delete_collection()
            self.db = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            return -1  # 跟 OpenSearchStore 保持一致:-1 表示全清

        if not ids:
            return 0

        # 关键点:.delete() 接 ids,不是接 query
        # 老的 LangChain Chroma 这里签名是 (ids: List[str])
        # 新的可能是 (ids: Optional[List[str]], where: Optional[Dict])
        self.db.delete(ids=ids)
        return len(ids)

    def health_check(self) -> bool:
        try:
            # 关键点:Chroma 没有 ping,试一下 _collection.count() 探活
            self.db._collection.count()
            return True
        except Exception:
            return False

    def list_sources(self) -> List[str]:
        """
        列出当前 collection 中所有 chunk 的 source 字段(去重)
        用于"启动时检测新文件"——对比文件系统和 KB 已入库文件
        空库时返回空列表
        """
        try:
            total = self.db._collection.count()
            if total == 0:
                return []
            data = self.db._collection.get(include=["metadatas"])
            sources = set()
            for m in data.get("metadatas", []) or []:
                if m and "source" in m:
                    sources.add(m["source"])
            return sorted(sources)
        except Exception:
            return []

    def list_source_records(self) -> List[Dict[str, Any]]:
        """列出库中所有 chunk 的来源指纹信息，供增量同步判断文件是否变更。"""
        try:
            total = self.db._collection.count()
            if total == 0:
                return []
            data = self.db._collection.get(include=["metadatas"])
            ids = data.get("ids", []) or []
            metadatas = data.get("metadatas", []) or []
            records: List[Dict[str, Any]] = []
            for idx, meta in enumerate(metadatas):
                meta = meta or {}
                records.append({
                    "id": ids[idx] if idx < len(ids) else meta.get("chunk_id", ""),
                    "chunk_id": meta.get("chunk_id", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "source": meta.get("source", ""),
                    "document_hash": meta.get("document_hash", ""),
                    "content_hash": meta.get("content_hash", ""),
                    "chunk_index": meta.get("chunk_index", -1),
                })
            return records
        except Exception:
            return []
