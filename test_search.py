# @File     : test_search.py
# @Time     : 2026/8/4 14:58
from tools.search import web_search

# 测试查询
query = "LangGraph 是什么 有什么优势"

print(f"🔍 测试搜索: {query}\n")
result = web_search(query)

print("=" * 50)
print("搜索结果:")
print("=" * 50)
print(result[:500] if len(result) > 500 else result)  # 只打印前500字符
print(f"\n✅ 总长度: {len(result)} 字符")