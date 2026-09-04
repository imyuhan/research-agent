from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

from infrastructure.config import settings
from infrastructure.embedding_client import LocalBGEEmbeddings, get_embeddings
from infrastructure.vector_stores.base import VectorStore
from infrastructure.vector_stores.chroma_store import ChromaStore

from .loaders import load_directory, load_files
from .splitters import split_documents


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_source(source: str) -> str:
    return str(Path(source).resolve()) if source else ""


def _doc_id_for_source(source: str) -> str:
    source = _normalize_source(source)
    return f"doc-{_sha1(source)[:16]}" if source else ""


def _source_hashes(raw_docs: List[Document]) -> dict[str, str]:
    """按 source 汇总文档级哈希，用于增量同步判断文件是否变更。"""
    hashes: dict[str, str] = {}
    for doc in raw_docs:
        meta = dict(doc.metadata or {})
        source = _normalize_source(str(meta.get("source", "")))
        if not source:
            continue
        hashes[source] = _sha1(doc.page_content)
    return hashes


def fingerprint_file(file_path: str | Path) -> str:
    """计算单个文件的文档级指纹。"""
    path = Path(file_path)
    try:
        return _sha1(path.read_text(encoding="utf-8"))
    except Exception:
        return ""


def _prepare_chunks(chunks: List[Document], source_hashes: Optional[dict[str, str]] = None) -> List[Document]:
    """为 chunk 注入稳定 metadata,方便去重/更新/追踪."""
    counters: dict[str, int] = {}
    prepared: List[Document] = []
    for chunk in chunks:
        meta = dict(chunk.metadata or {})
        source = _normalize_source(str(meta.get("source", "")))
        if source:
            meta["source"] = source
            meta["title"] = meta.get("title") or Path(source).stem
            meta["file_name"] = Path(source).name
            meta["doc_id"] = meta.get("doc_id") or _doc_id_for_source(source)
            if source_hashes and source in source_hashes:
                meta["document_hash"] = source_hashes[source]
        else:
            meta["title"] = meta.get("title") or "unknown"
            meta["doc_id"] = meta.get("doc_id") or f"doc-{_sha1(chunk.page_content)[:16]}"

        chunk_index = counters.get(source, 0)
        counters[source] = chunk_index + 1
        content_hash = _sha1(chunk.page_content)
        meta["chunk_index"] = chunk_index
        meta["content_hash"] = content_hash
        meta["document_hash"] = meta.get("document_hash") or content_hash
        meta["chunk_id"] = meta.get("chunk_id") or f"{meta['doc_id']}::c{chunk_index:04d}::{content_hash[:12]}"
        prepared.append(Document(page_content=chunk.page_content, metadata=meta))
    return prepared


def build_chunks(
        directory: str | Path = None,
        glob_pattern: str = "**/*.txt",
) -> List[Document]:
    """
    纯数据准备:加载 + 切分,不涉及向量化/存储
    适合单元测试和复用
    """
    raw_docs = load_directory(directory, glob_pattern=glob_pattern)
    if not raw_docs:
        return []
    chunks = split_documents(raw_docs)
    return _prepare_chunks(chunks, _source_hashes(raw_docs))


def build_chunks_for_files(file_paths: List[str | Path]) -> List[Document]:
    """
    增量场景:只对给定的文件列表做 加载+切分
    """
    if not file_paths:
        return []
    raw_docs = load_files(file_paths)
    if not raw_docs:
        return []
    return _prepare_chunks(split_documents(raw_docs), _source_hashes(raw_docs))


def ingest_to_store(
        store: VectorStore,
        chunks: List[Document],
        embeddings: LocalBGEEmbeddings = None,
        batch_size: int = None,
) -> int:
    """
    把 chunks 写入 store
    store: 任何 VectorStore 子类(Chroma 或 OpenSearch)
    chunks: 已经切好的 Document 列表
    返回:写入条数
    """
    if not chunks:
        return 0

    embeddings = embeddings or get_embeddings()
    batch_size = batch_size or settings.RAG_EMBED_BATCH_SIZE

    total_written = 0
    # 关键:批量写入,避免一次 encode 太多 OOM
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c.page_content for c in batch]
        embs = embeddings.embed_documents(texts, batch_size=batch_size)
        metadatas = [c.metadata for c in batch]
        ids = [c.metadata.get("chunk_id", "") for c in batch]
        ids = ids if all(ids) and len(ids) == len(batch) else None
        written_ids = store.add(texts=texts, embeddings=embs, metadatas=metadatas, ids=ids)
        total_written += len(written_ids)

    return total_written


def create_store(backend: str = None) -> VectorStore:
    """
    工厂方法:根据 backend 创建对应的 store
    backend: "chroma" | "opensearch" | None(从 settings 读)
    """
    backend = backend or settings.RAG_STORE_BACKEND
    embeddings = get_embeddings()

    if backend == "chroma":
        return ChromaStore(embeddings=embeddings)
    elif backend == "opensearch":
        from infrastructure.vector_stores.opensearch_store import OpenSearchStore
        return OpenSearchStore(
            embeddings=embeddings,
            # 其它参数走 settings
        )
    else:
        raise ValueError(f"Unknown RAG store backend: {backend}")


def build_index(
        directory: str | Path = None,
        backend: str = None,
        store: VectorStore = None,
) -> dict:
    """
    完整流水线:加载 → 切分 → 向量化 → 入库
    返回:统计信息
    """
    directory = directory or settings.DOCUMENTS_DIR
    backend = backend or settings.RAG_STORE_BACKEND

    # 步骤 1:加载 + 切分
    chunks = build_chunks(directory)
    if not chunks:
        return {"backend": backend, "docs_loaded": 0, "chunks": 0, "written": 0}

    # 步骤 2:准备 store
    if store is None:
        store = create_store(backend)
        try:
            store.delete(delete_all=True)
        except Exception:
            pass

    # 步骤 3:入库
    written = ingest_to_store(store, chunks)

    return {
        "backend": backend,
        "docs_loaded": len({c.metadata.get("source", "") for c in chunks}),
        "chunks": len(chunks),
        "written": written,
    }


def build_index_for_files(
        file_paths: List[str | Path],
        store: VectorStore = None,
) -> dict:
    """
    增量入库:只对明确给定的文件做 加载→切分→向量化→入库
    用于"启动 main 时检测新文件,只入库新文件"的场景
    返回:统计信息
    """
    if not file_paths:
        return {"docs_loaded": 0, "chunks": 0, "written": 0}

    chunks = build_chunks_for_files(file_paths)
    if not chunks:
        return {"docs_loaded": 0, "chunks": 0, "written": 0}

    if store is None:
        store = create_store()

    written = ingest_to_store(store, chunks)

    return {
        "docs_loaded": len({c.metadata.get("source", "") for c in chunks}),
        "chunks": len(chunks),
        "written": written,
    }
