"""CLI 入口:从 data_layer/documents 重建索引"""
from data_layer.indexer import build_index
from infrastructure.config import settings

if __name__ == "__main__":
    result = build_index(settings.DOCUMENTS_DIR)
    print(
        f"✅ 索引完成: {result.get('docs_loaded', 0)} 文件 / "
        f"{result.get('chunks', 0)} chunks / 写入 {result.get('written', 0)} 条"
    )
