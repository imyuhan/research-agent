from pathlib import Path

from langchain_core.documents import Document

from data_layer.indexer import (
    _prepare_chunks,
    build_chunks_for_files,
    fingerprint_file,
    ingest_to_store,
)


def test_fingerprint_is_stable_and_changes_with_file_content(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("第一版内容", encoding="utf-8")

    first = fingerprint_file(file_path)
    assert first
    assert first == fingerprint_file(file_path)

    file_path.write_text("第二版内容", encoding="utf-8")
    assert fingerprint_file(file_path) != first


def test_build_chunks_for_files_adds_stable_tracking_metadata(tmp_path):
    file_path = tmp_path / "research.txt"
    file_path.write_text("标题\n\n这是第一段内容。这里还有一些补充内容。", encoding="utf-8")

    first = build_chunks_for_files([file_path])
    second = build_chunks_for_files([file_path])

    assert first
    assert len(first) == len(second)
    assert all(Path(chunk.metadata["source"]) == file_path.resolve() for chunk in first)
    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]
    assert all(chunk.metadata["doc_id"] for chunk in first)
    assert all(chunk.metadata["document_hash"] for chunk in first)
    assert [chunk.metadata["chunk_index"] for chunk in first] == list(range(len(first)))


def test_build_chunks_for_missing_file_returns_empty():
    assert build_chunks_for_files(["C:/path/that/does/not/exist.txt"]) == []


def test_prepare_chunks_assigns_independent_indexes_per_source():
    docs = [
        Document(page_content="a", metadata={"source": "a.txt"}),
        Document(page_content="b", metadata={"source": "a.txt"}),
        Document(page_content="c", metadata={"source": "b.txt"}),
    ]

    chunks = _prepare_chunks(docs)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 0]
    assert chunks[0].metadata["doc_id"] == chunks[1].metadata["doc_id"]
    assert chunks[0].metadata["doc_id"] != chunks[2].metadata["doc_id"]


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts, batch_size=32):
        self.calls.append((list(texts), batch_size))
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeStore:
    def __init__(self):
        self.calls = []

    def add(self, texts, embeddings, metadatas, ids=None):
        self.calls.append(
            {
                "texts": list(texts),
                "embeddings": embeddings,
                "metadatas": metadatas,
                "ids": ids,
            }
        )
        return ids or [f"generated-{index}" for index, _ in enumerate(texts)]


def test_ingest_to_store_batches_embeddings_and_preserves_chunk_ids():
    chunks = [
        Document(page_content="one", metadata={"chunk_id": "chunk-1"}),
        Document(page_content="two", metadata={"chunk_id": "chunk-2"}),
        Document(page_content="three", metadata={"chunk_id": "chunk-3"}),
    ]
    embeddings = FakeEmbeddings()
    store = FakeStore()

    written = ingest_to_store(store, chunks, embeddings=embeddings, batch_size=2)

    assert written == 3
    assert len(embeddings.calls) == 2
    assert [call[0] for call in embeddings.calls] == [
        ["one", "two"],
        ["three"],
    ]
    assert [call["ids"] for call in store.calls] == [
        ["chunk-1", "chunk-2"],
        ["chunk-3"],
    ]
