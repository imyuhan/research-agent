# @File     : config.py
# @Time     : 2026/8/4 11:48
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Settings(BaseSettings):
    # DeepSeek API 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_API_KEY", "https://api.deepseek.com/v1")

    # 通用模型
    MODEL_NAME: str = os.getenv("MODEL_NAME", "deepseek-v4-pro")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # 分角色模型配置（简历加分项：根据任务难度分配不同模型）
    MODEL_PLANNER: str = os.getenv("MODEL_PLANNER", "deepseek-v4-pro")
    MODEL_RESEARCHER: str = os.getenv("MODEL_RESEARCHER", "deepseek-v4-pro")
    MODEL_WRITER: str = os.getenv("MODEL_WRITER", "deepseek-v4-pro")
    MODEL_REVIEWER: str = os.getenv("MODEL_REVIEWER", "deepseek-v4-pro")

    # 搜索配置
    MAX_SEARCH_RESULTS: int = 5
    MAX_REVISIONS: int = 3


settings = Settings()

# 启动时校验（防止运行时才发现没配 Key）
if not settings.OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 未设置，请在 .env 文件中配置")