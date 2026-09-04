from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from infrastructure.config import settings


def load_file(file_path: str | Path, encoding: str = "utf-8") -> List[Document]:
    """
    加载单个文件

    关键:统一用 resolve() 后的绝对路径作为 source
    避免全量 build_index(传相对 DOCUMENTS_DIR) 和增量 ingest(传绝对路径)
    两种 source 格式不一致,导致 find_new_files 永远匹配不上
    """
    loader = TextLoader(str(Path(file_path).resolve()), encoding=encoding)
    return loader.load()


def load_files(file_paths: List[str | Path], encoding: str = "utf-8") -> List[Document]:
    """
    加载一组明确指定的文件(不走 glob)
    用于"只入库新文件"这种增量场景
    加载失败的文件会被跳过
    """
    docs: List[Document] = []
    for fp in file_paths:
        try:
            docs.extend(load_file(fp, encoding=encoding))
        except Exception as e:
            print(f"   ⚠️ 跳过文件 {fp}: {e}")
    return docs


def load_directory(
    directory: str | Path = None,
    glob_pattern: str = "**/*.txt",
    encoding: str = "utf-8",
) -> List[Document]:
    """
    批量加载目录下所有匹配文件
    默认递归扫所有子目录的 .txt

    关键:用 Python rglob 自己扫,再走 load_file(load_file 内部会 resolve 成绝对路径)
    不用 DirectoryLoader——DirectoryLoader 内部直接 TextLoader(str(p)),
    其中 p 是它自己 rglob 出来的相对路径,不会 resolve,
    会导致全量 build_index 存相对路径,跟增量的绝对路径不匹配
    """
    directory = Path(directory or settings.DOCUMENTS_DIR)
    if not directory.exists():
        return []
    # 注意:glob_pattern 是 "**/*.txt" 这种,但 Path.rglob 不直接支持 glob string
    # 简化处理:提取扩展名,用 rglob("*.扩展名")
    # 够用就行,场景只有 .txt
    if glob_pattern == "**/*.txt":
        file_iter = directory.rglob("*.txt")
    else:
        # 兜底:还是用 DirectoryLoader(用户用了奇怪的 pattern 时)
        from langchain_community.document_loaders import DirectoryLoader as _DL
        loader = _DL(
            str(directory),
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": encoding},
            silent_errors=True,
        )
        return loader.load()

    docs: List[Document] = []
    for fp in file_iter:
        if fp.is_file():
            try:
                docs.extend(load_file(fp, encoding=encoding))
            except Exception as e:
                print(f"   ⚠️ 跳过文件 {fp}: {e}")
    return docs
