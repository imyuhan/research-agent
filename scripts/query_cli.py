"""CLI 入口:测试检索效果"""
from service.rag_service import get_rag
if __name__ == "__main__":
    rag = get_rag()
    while True:
        q = input("query> ").strip()
        if not q: break
        for i, hit in enumerate(rag.query(q, top_k=3), 1):
            print(f"[{i}] score={hit['score']:.3f} | {hit['text'][:80]}...")