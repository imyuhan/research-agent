from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Settings(BaseSettings):
    # DeepSeek API 配置
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


settings = Settings()

# 启动时校验（防止运行时才发现没配 Key）
if not settings.OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 未设置，请在 .env 文件中配置")
