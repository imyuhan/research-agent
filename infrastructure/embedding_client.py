from threading import RLock

import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

from infrastructure.config import settings

# 解析项目根（基于本文件位置），避免 cwd 依赖
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MODELS_DIR = _PROJECT_ROOT / "local_models"


def _resolve_model_path(raw: str) -> str:
    """把 settings 里的模型路径解析成绝对路径。

    - 绝对路径：原样返回
    - 相对路径：相对项目根解析
    """
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((_PROJECT_ROOT / raw).resolve())


class LocalBGEEmbeddings:
    """自定义本地 BGE Embedding 类，适配 LangChain 接口"""

    def __init__(self, device: str | None = None, model_path: str | None = None):
        """
        :param device: 运行设备（cpu/cuda，有GPU建议用cuda加速）
        """
        self.model_path = _resolve_model_path(model_path or settings.BGE_MODEL_PATH)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 加载本地模型（首次运行会自动下载，后续直接加载本地文件）
        self.model = SentenceTransformer(
            model_name_or_path=self.model_path,
            device=device,
            # 模型本地缓存路径（基于项目根，避免依赖 cwd）
            cache_folder=str(_LOCAL_MODELS_DIR.resolve()),
        )
        # BGE 模型推荐的查询前缀（提升检索效果）
        self.query_prefix = "为这个句子生成表示以用于检索相关文档："

    def embed_query(self, text):
        """生成查询文本的向量（带BGE前缀）"""
        text = text.strip()
        # 拼接前缀，提升查询向量的检索精度
        embeddings = self.model.encode(
            self.query_prefix + text,
            normalize_embeddings=True,  # 归一化向量（必须，否则余弦相似度计算不准）
            batch_size=1,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_documents(self, texts, batch_size=32):
        """批量生成文档向量（核心优化：批量处理提升速度）"""
        # 清洗文本
        texts = [text.strip() for text in texts if text.strip()]
        if not texts:
            return []

        # 批量生成向量（tqdm 显示进度）
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,  # 批量大小：CPU建议16-32，GPU建议64-128
            show_progress_bar=True,
            convert_to_numpy=True  # 用numpy提升效率
        )
        return embeddings.tolist()


_EMBEDDINGS_LOCK = RLock()
_EMBEDDINGS_INSTANCE: LocalBGEEmbeddings | None = None


def get_embeddings() -> LocalBGEEmbeddings:
    """获取全局共享的 embedding 实例，避免重复加载模型。"""
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        with _EMBEDDINGS_LOCK:
            if _EMBEDDINGS_INSTANCE is None:
                _EMBEDDINGS_INSTANCE = LocalBGEEmbeddings()
    return _EMBEDDINGS_INSTANCE
