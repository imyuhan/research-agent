from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from infrastructure.config import settings


def split_documents(
        docs: List[Document],
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
) -> List[Document]:
    """
    把长文档切分成适合 embedding 的小块
    """
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
    # 关键:separators 给 None 时用默认中文分隔符
    separators = separators or ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        # 关键:keep_separator=False 会把分隔符去掉
        # 设为 True 可以保留段落分隔（可选,看你想要不想要）
        keep_separator=False,
        # 关键:length_function 默认是 len() (字符数)
        # 想按 token 数要换 tiktoken,这里默认字符数
    )
    return splitter.split_documents(docs)