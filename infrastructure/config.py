from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 加载 .env 文件
load_dotenv()


class Settings(BaseSettings):
    # API 配置
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""

    # 通用模型
    MODEL_NAME: str = ""
    TAVILY_API_KEY: str = ""

    # 搜索配置
    MAX_SEARCH_RESULTS: int = 5
    MAX_REVISIONS: int = 3
    MAX_CITATIONS_PER_QUERY: int = 5
    MAX_SEARCH_CONCURRENCY: int = 5 #并行搜索的工程线程数

    # RAG配置
    BGE_MODEL_PATH: str = str(
        _PROJECT_ROOT / "local_models" / "models--BAAI--bge-base-zh-v1.5" / "snapshots" / "f03589ceff5aac7111bd60cfc7d497ca17ecac65")
    BGE_RERANK_MODEL_PATH: str = str(
        _PROJECT_ROOT / "local_models" / "models--BAAI--bge-reranker-v2-m3" / "snapshots" / "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_INDEX: str = "local_rag_docs"
    EMBEDDING_DIM: int = 768
    RAG_TOP_K: int = 5
    RAG_HYBRID_ALPHA: float = 0.5  # 向量权重
    RAG_ENABLE_RERANK: bool = True
    CHROMA_PERSIST_DIR: str = "./local_chroma_db"
    DOCUMENTS_DIR: str = "./data_layer/documents"

    # L2索引层
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
    RAG_EMBED_BATCH_SIZE: int = 32
    RAG_STORE_BACKEND: str = "chroma"
    RAG_RERANK_TOP_K: int = 5
    RAG_MIN_SCORE: float = 0.6
    FRONTEND_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    WRITER_MAX_CITATIONS_IN_PROMPT: int = 12
    WRITER_MAX_EVIDENCE_IN_PROMPT: int = 8
    WRITER_MAX_FINDINGS_IN_PROMPT: int = 8

settings = Settings()

# 启动时校验（防止运行时才发现没配 Key）
if not settings.OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 未设置，请在 .env 文件中配置")
