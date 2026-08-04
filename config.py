from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Settings(BaseSettings):
    # DeepSeek API 配置
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.deepseek.com/v1"

    # 通用模型
    MODEL_NAME: str = "deepseek-v4-pro"
    TAVILY_API_KEY: str = ""

    # 搜索配置
    MAX_SEARCH_RESULTS: int = 5
    MAX_REVISIONS: int = 3


settings = Settings()

# 启动时校验（防止运行时才发现没配 Key）
if not settings.OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 未设置，请在 .env 文件中配置")
