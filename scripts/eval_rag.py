"""RAG 检索质量评估脚本
- 优先加载 data_layer/gold_set.json（手写 gold set，质量可控）
- 缺失时回退到 build_gold_set() 自动生成（仅供快速 sanity check）
- 评估指标：precision@5 / recall@5 / mrr
"""
from __future__ import annotations

import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

# 注意：get_rag 在 main() 里延迟导入，方便 build_gold_set 独立运行
DOCS_DIR = Path(__file__).resolve().parent.parent / "data_layer" / "documents"
GOLD_SET_PATH = Path(__file__).resolve().parent.parent / "data_layer" / "gold_set.json"


# ===========================
# Query 改写模板（仅 build_gold_set 自动生成用）
# ===========================
# 同一事实用不同问法包装，模拟真实用户问法多样性
QUERY_TEMPLATES = [
    "什么是 {keyword}？",
    "请介绍一下 {keyword}",
    "{keyword} 有什么特点？",
    "{keyword} 相关的内容",
    "关于 {keyword} 的描述",
    "{keyword} 的定义是什么",
    "能否解释一下 {keyword}",
    "我想了解 {keyword}",
    "{keyword} 的主要用途",
    "简单说说 {keyword}",
]


def split_sentences(text: str) -> List[str]:
    """按中英文标点切句"""
    text = text.strip()
    if not text:
        return []
    # 中英文句号 / 感叹号 / 问号 / 分号
    parts = re.split(r"[。！？；\n\r]+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_keywords(sentence: str) -> List[str]:
    """从句子中抽取关键词（用于相关性判定）
    策略：先按中英文边界切块，再分别抽取 n-gram，避免跨语言混在一起
    """
    keywords = set()
    # 1) 英文块：连续 [A-Za-z]+
    for m in re.finditer(r"[A-Za-z]{2,}", sentence):
        keywords.add(m.group())
    # 2) 中文块：连续 [\u4e00-\u9fff]+，去掉标点
    cn_block = re.sub(r"[^\u4e00-\u9fff]", " ", sentence)
    for seg in cn_block.split():
        if len(seg) < 2:
            continue
        # n-gram (2~5)
        for n in range(2, min(6, len(seg) + 1)):
            for i in range(len(seg) - n + 1):
                keywords.add(seg[i : i + n])
    # 转 list + 排序（长优先）+ 截断
    result = sorted(keywords, key=len, reverse=True)
    return result[:8]


def pick_focus_keyword(sentence: str) -> str:
    """挑出句子里最核心的关键词作为 query 主语
    优先选英文专有名词（Transformer / LangChain / FAISS 这种）
    """
    # 1) 英文专有名词：首字母大写 / 全大写
    camel = re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", sentence)
    if camel:
        return camel[0]
    # 2) 连续大写缩写（>=2 个字母）
    upper = re.findall(r"\b[A-Z]{2,}\b", sentence)
    if upper:
        return upper[0]
    # 3) 中文 2~5 字
    kws = extract_keywords(sentence)
    for k in kws:
        if 2 <= len(k) <= 5 and re.search(r"[\u4e00-\u9fff]", k):
            return k
    return kws[0] if kws else ""


def is_relevant(
    doc: str,
    keywords: List[str],
    min_hit: int = 2,
) -> bool:
    """相关性判定：doc 中至少出现 min_hit 个关键词才算相关
    - min_hit=1：宽松，容错高
    - min_hit=2：默认，平衡 precision/recall
    - min_hit=3：严格，倾向于"明确命中主题"的文档
    """
    if not keywords:
        return False
    hits = sum(1 for kw in keywords if kw in doc)
    return hits >= min_hit


def load_gold_set(path: Optional[Path] = None) -> List[Dict]:
    """加载手写 gold set。优先 data_layer/gold_set.json，不存在则回退到 build_gold_set()。

    字段约定：
      - query: 用户真实问法
      - source_file: 期望命中的源文档（相对 data_layer/documents/）
      - source_sentence: 期望命中的原句（人工标注，便于排查失败 case）
      - relevant_keywords: 真正的实体词/数字/专有名词（不再用 n-gram 碎片）
    """
    p = path or GOLD_SET_PATH
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            gold = json.load(f)
        print(f"✅ 加载手写 gold set: {p} ({len(gold)} 条)")
        return gold
    print(f"⚠️ 未找到 {p}，回退到 build_gold_set() 自动生成（质量较差，建议手写）")
    return build_gold_set()


def build_gold_set(seed: int = 42) -> List[Dict]:
    """扫描 documents/ 目录，对每条句子生成 query + 关键词标签
    目标：生成 ~50 条 gold query
    """
    random.seed(seed)
    gold: List[Dict] = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"文档目录不存在: {DOCS_DIR}")

    txt_files = sorted(DOCS_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"在 {DOCS_DIR} 下找不到任何 .txt 文件")

    print(f"扫描到 {len(txt_files)} 个文档: {[f.name for f in txt_files]}")

    for txt in txt_files:
        text = txt.read_text(encoding="utf-8")
        sentences = split_sentences(text)
        for sent in sentences:
            keywords = extract_keywords(sent)
            focus = pick_focus_keyword(sent)
            if not focus or not keywords:
                continue
            # 每条句子生成 2-3 个 query 变体（保证总条数 ~50）
            n_variants = random.choice([2, 2, 3])
            templates = random.sample(QUERY_TEMPLATES, n_variants)
            for tmpl in templates:
                query = tmpl.format(keyword=focus)
                gold.append({
                    "query": query,
                    "relevant_keywords": keywords,
                    "source_file": txt.name,
                    "source_sentence": sent,
                })
                if len(gold) >= 60:  # 留 10 条余量
                    break
            if len(gold) >= 60:
                break
        if len(gold) >= 60:
            break

    # 截断到 50
    gold = gold[:50]
    print(f"生成 gold set: {len(gold)} 条")
    return gold


# ===========================
# 评估指标
# ===========================
def precision_at_k(hits: List[dict], k: int, keywords: List[str]) -> float:
    if not hits:
        return 0.0
    top_k = hits[:k]
    relevant = sum(1 for h in top_k if is_relevant(h.get("text", ""), keywords))
    return relevant / k


def recall_at_k(hits: List[dict], keywords: List[str], min_hits: int = 1) -> float:
    """简化的 recall：top-5 中相关文档数 / 预期至少 1 条相关"""
    if not hits:
        return 0.0
    relevant = sum(1 for h in hits if is_relevant(h.get("text", ""), keywords))
    return min(relevant / min_hits, 1.0)


def mrr(hits: List[dict], keywords: List[str]) -> float:
    """Mean Reciprocal Rank：第一个相关文档的倒数排名"""
    for i, h in enumerate(hits, start=1):
        if is_relevant(h.get("text", ""), keywords):
            return 1.0 / i
    return 0.0


# ===========================
# 主流程
# ===========================
def main(eval_k: int = 5):
    print("=" * 60)
    print("RAG 检索质量评估")
    print("=" * 60)

    # 1. 构造 gold set
    gold = load_gold_set()
    print()

    # 2. 延迟导入 RAG（避免 sentence_transformers 缺失时 gold set 也跑不了）
    try:
        from service import get_rag
    except ImportError as e:
        print(f"❌ 无法加载 RAG 服务: {e}")
        print("   请先安装依赖: pip install sentence-transformers pydantic-settings")
        return
    rag = get_rag()

    # 3. 评估
    p5_list, r5_list, mrr_list = [], [], []
    for i, item in enumerate(gold, 1):
        try:
            hits = rag.query(item["query"], top_k=eval_k)
        except Exception as e:
            print(f"[{i:2d}] 查询失败: {e}")
            continue
        p = precision_at_k(hits, eval_k, item["relevant_keywords"])
        r = recall_at_k(hits, item["relevant_keywords"])
        m = mrr(hits, item["relevant_keywords"])
        p5_list.append(p)
        r5_list.append(r)
        mrr_list.append(m)
        if i <= 5 or i % 10 == 0:  # 打印前 5 条 + 每 10 条
            status = "✅" if p > 0 else "❌"
            print(f"[{i:2d}] {status} P@{eval_k}={p:.2f} | Q: {item['query']}")

    # 4. 汇总
    n = len(p5_list) or 1
    print()
    print("=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"样本数:     {len(p5_list)}")
    print(f"Precision@{eval_k}: {sum(p5_list) / n:.3f}")
    print(f"Recall@{eval_k}:    {sum(r5_list) / n:.3f}")
    print(f"MRR:             {sum(mrr_list) / n:.3f}")
    print()
    print(f"💡 切换评估模式请设置环境变量 RAG_ENABLE_RERANK=true/false")
    print(f"   当前: RAG_ENABLE_RERANK={os.getenv('RAG_ENABLE_RERANK', 'false')}")


if __name__ == "__main__":
    main()
