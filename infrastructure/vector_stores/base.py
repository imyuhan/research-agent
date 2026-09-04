from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str] | None = None,
    ) -> List[str]:
        """写入文档,返回 ID 列表"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        mode: str = "vector",
    ) -> List[Dict[str, Any]]:
        """检索,返回 [{text, score, metadata}, ...]"""
        pass

    @abstractmethod
    def delete(self, ids: List[str] = None, delete_all: bool = False) -> int:
        """删除,返回删除条数"""
        pass
