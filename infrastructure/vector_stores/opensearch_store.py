import uuid
from typing import Dict, List
from opensearchpy import OpenSearch, helpers
from .base import VectorStore
from ..config import settings


class OpenSearchStore(VectorStore):
    def __init__(
            self,
            embeddings,
            host: str = None,
            port: int = None,
            index_name: str = None,
            embedding_dim: int = None,
    ):
        self.client = OpenSearch(
            hosts=[{
                "host": host if host is not None else settings.OPENSEARCH_HOST,
                "port": port if port is not None else settings.OPENSEARCH_PORT,
            }],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            timeout=30,
        )
        self.index_name = index_name if index_name is not None else settings.OPENSEARCH_INDEX
        self.embedding_dim = embedding_dim if embedding_dim is not None else settings.EMBEDDING_DIM
        self.embeddings = embeddings
        self._ensure_index()

    def _ensure_index(self):
        if self.client.indices.exists(index=self.index_name):
            return
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True,
                "analysis": {
                    "analyzer": {
                        "ik_max_word": {"type": "custom", "tokenizer": "ik_max_word"}
                    }
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": "ik_max_word"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "parameters": {"ef_construction": 128, "m": 16}
                        }
                    }
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=index_settings)

    # 批量写入 Opensearch
    def add(self, texts, embeddings, metadatas, ids=None):
        metadatas = metadatas or [{} for _ in texts]
        out_ids = []
        actions = []
        input_ids = ids or [str(uuid.uuid4()) for _ in texts]
        for text, emb, meta, doc_id in zip(texts, embeddings, metadatas, input_ids):
            out_ids.append(doc_id)
            actions.append({
                "_op_type": "index",  # ← 关键:让 bulk 走 create-like 语义
                "_index": self.index_name,
                "_id": doc_id,
                "_source": {
                    "text": text,
                    "metadata": meta,
                    "embedding": emb,
                }
            })
        if actions:
            success, failed = helpers.bulk(self.client, actions)
            if failed:
                # 关键点:失败要给上层信号,别静默吞
                print(f"⚠️ 写入失败 {failed} 条")
        return out_ids

    def search(self, query, query_embedding, top_k, mode="hybrid", alpha=0.5):
        top_k = top_k or settings.RAG_TOP_K
        alpha = alpha if alpha is not None else settings.RAG_HYBRID_ALPHA
        if mode == "vector":
            return self._vector_search(query_embedding, top_k)
        elif mode == "bm25":
            return self._bm25_search(query, top_k)
        else:  # hybrid
            return self._hybrid_search(query, query_embedding, top_k, alpha)

    def _vector_search(self, query_embedding, top_k):
        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
            "_source": ["text", "metadata"],
        }
        response = self.client.search(index=self.index_name, body=body)
        return [
            {
                "text": h["_source"]["text"],
                "score": h["_score"],
                "metadata": h["_source"].get("metadata", {}),
                "id": h["_id"],
                "chunk_id": h["_id"],
                "doc_id": h["_source"].get("metadata", {}).get("doc_id", ""),
                "source": h["_source"].get("metadata", {}).get("source", ""),
                "title": h["_source"].get("metadata", {}).get("title", ""),
                "chunk_index": h["_source"].get("metadata", {}).get("chunk_index", -1),
                "content_hash": h["_source"].get("metadata", {}).get("content_hash", ""),
            }
            for h in response["hits"]["hits"]
        ]

    def _bm25_search(self, query, top_k):
        body = {
            "size": top_k,
            "query": {"match": {"text": query}},
            "_source": ["text", "metadata"],
        }
        response = self.client.search(index=self.index_name, body=body)
        return [  # 同样的格式化
            {
                "text": h["_source"]["text"],
                "score": h["_score"],
                "metadata": h["_source"].get("metadata", {}),
                "id": h["_id"],
                "chunk_id": h["_id"],
                "doc_id": h["_source"].get("metadata", {}).get("doc_id", ""),
                "source": h["_source"].get("metadata", {}).get("source", ""),
                "title": h["_source"].get("metadata", {}).get("title", ""),
                "chunk_index": h["_source"].get("metadata", {}).get("chunk_index", -1),
                "content_hash": h["_source"].get("metadata", {}).get("content_hash", ""),
            }
            for h in response["hits"]["hits"]
        ]

    def _hybrid_search(self, query, query_embedding, top_k, alpha):
        # 1. 拿 2 倍 top_k 的两种结果
        vec_hits = self._vector_search(query_embedding, top_k * 2)
        bm_hits = self._bm25_search(query, top_k * 2)

        # 2. 归一化(抽成辅助方法)
        vec_scores = {h["id"]: h["score"] for h in vec_hits}
        bm_scores = {h["id"]: h["score"] for h in bm_hits}
        norm_vec = self._normalize_scores(vec_scores)
        norm_bm = self._normalize_scores(bm_scores)

        # 3. 加权融合
        all_ids = set(norm_vec) | set(norm_bm)
        final = {
            i: alpha * norm_vec.get(i, 0) + (1 - alpha) * norm_bm.get(i, 0)
            for i in all_ids
        }

        # 4. 排序取 top_k
        sorted_ids = sorted(final, key=lambda i: final[i], reverse=True)[:top_k]

        # 5. 找回原文 → dict
        id2hit = {h["id"]: h for h in vec_hits + bm_hits}
        return [
            {
                "text": id2hit[i]["text"],
                "score": final[i],
                "metadata": id2hit[i]["metadata"],
                "id": i,
                "chunk_id": i,
                "doc_id": id2hit[i]["metadata"].get("doc_id", ""),
                "source": id2hit[i]["metadata"].get("source", ""),
                "title": id2hit[i]["metadata"].get("title", ""),
                "chunk_index": id2hit[i]["metadata"].get("chunk_index", -1),
                "content_hash": id2hit[i]["metadata"].get("content_hash", ""),
            }
            for i in sorted_ids
        ]

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-Max 归一化"""
        if not scores:
            return {}
        smin, smax = min(scores.values()), max(scores.values())
        span = smax - smin
        if span < 1e-8:  # 关键:防止除零
            return {k: 0.0 for k in scores}
        return {k: (v - smin) / span for k, v in scores.items()}

    def delete(self, ids: List[str] = None, delete_all: bool = False) -> int:
        if delete_all:
            self.client.indices.delete(index=self.index_name)
            self._ensure_index()  # 关键:删完重建空索引,否则后续 add 报错
            return -1  # 用 -1 表示"全清"
        if not ids:
            return 0

        body = {"query": {"ids": {"values": ids}}}
        response = self.client.delete_by_query(
            index=self.index_name, body=body, refresh=True
        )
        return response.get("deleted", 0)

    def health_check(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    def list_sources(self) -> List[str]:
        """
        列出 OpenSearch 中已索引的所有 source 字段(去重)
        不用 terms aggregation(依赖 keyword 子字段,mapping 不一定开)——
        直接 match_all + _source 拿 metadata.source,程序内去重
        受 size 限制(默认 10000 条,覆盖绝大多数 KB 场景)
        server 不可达或查询失败时返回空列表(fail-open)

        关键:先强制 refresh 一次,避免 OpenSearch 默认 1s refresh interval
        导致刚写入的数据 search 不到(增量检测会判为"新文件"重复入库)
        """
        try:
            # 强制刷新索引,让刚 bulk 的数据立刻可见
            try:
                self.client.indices.refresh(index=self.index_name)
            except Exception:
                pass
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": 10000,
                    "_source": ["metadata.source"],
                    "query": {"match_all": {}},
                },
            )
            sources = set()
            for hit in response.get("hits", {}).get("hits", []):
                src = hit.get("_source", {}).get("metadata", {}).get("source")
                if src:
                    sources.add(src)
            return sorted(sources)
        except Exception:
            return []

    def list_source_records(self) -> List[Dict[str, Any]]:
        """列出库中所有 chunk 的来源指纹信息，供增量同步判断文件是否变更。"""
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": 10000,
                    "_source": ["text", "metadata"],
                    "query": {"match_all": {}},
                },
            )
            records: List[Dict[str, Any]] = []
            for hit in response.get("hits", {}).get("hits", []):
                meta = hit.get("_source", {}).get("metadata", {}) or {}
                records.append({
                    "id": hit.get("_id", ""),
                    "chunk_id": hit.get("_id", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "source": meta.get("source", ""),
                    "document_hash": meta.get("document_hash", ""),
                    "content_hash": meta.get("content_hash", ""),
                    "chunk_index": meta.get("chunk_index", -1),
                })
            return records
        except Exception:
            return []
