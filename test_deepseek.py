# @File     : test_deepseek.py
# @Time     : 2026/8/4 14:43
from langchain_openai import ChatOpenAI
from config import settings

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

response = llm.invoke("你好，请用一句话介绍自己")
print(response.content)