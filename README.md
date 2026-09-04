# Research Agent — 基于 LangGraph 的多 Agent 智能研究助手

## 项目简介

一个基于 LangGraph 构建的多 Agent 协作研究系统，按「**规划 → 研究 → 写作 → 审核**」四阶段闭环执行，
未通过审核会自动回退到写作节点重写，最多 3 次。Researcher 节点融合了**本地知识库（RAG）**和
**网络搜索（Tavily）**两种信息来源，Writer/Reviewer 也能参考 KB 命中做事实校准。

适合做"给定一个研究主题 → 自动产出带引用、可溯源 Markdown 报告"这类场景。

## 示例运行

终端输入主题后，系统会自动完成 规划 → 搜索 → 写作 → 审核 四个阶段。下面展示一个典型流程：

```text
$ python main.py
请输入研究主题：2026年8月6日 苹果公司发布了什么新产品

🚀 开始研究主题：2026年8月6日 苹果公司发布了什么新产品
   📅 用户指定日期：2026年8月6日

📋 Planner 正在规划主题：2026年8月6日 苹果公司发布了什么新产品
   生成 2 个子问题：
   • 2026年8月6日 苹果公司发布的新产品有哪些
   • 这些新产品的核心参数和定价
📋 规划完成，生成 2 个子问题

🔍 Researcher 开始研究 2 个子问题:
   📅 用户指定日期：2026年8月6日，搜索窗口 2026-08-05 ~ 2026-08-07
   研究：2026年8月6日 苹果公司发布的新产品有哪些
    📚 KB 召回 0 条 (相关度: [])
    ✅ 完成：2026年8月6日 苹果公司发布的新产品有哪些
       搜索 1 次 | KB 命中 0 条 | web 引用 3 条 | 长度 1856 字符
   ...
✅ 研究完成：
   - 研究结果：2 条
   - KB 命中合计：0 条
   - Web 引用：6 条

✍️ Writer 正在撰写报告，基于 2 条研究资料 + 0 条 KB 命中...
   报告生成完成，长度：3254 字符
✍️ 写作完成，报告长度 3254 字符

🔍 Reviewer 正在审核报告...
   评分: 9/10 | 是否通过: ✅
🔍 审核完成，评分 9/10，✅通过
✅ 审核通过，流程结束

============================================================
📄 最终研究报告
============================================================
## 执行摘要
...
============================================================

⏱️ 各阶段耗时：
   planner         1.8s
   researcher      6.4s
   writer          5.2s
   reviewer        3.1s
   总计           16.5s
```

**关键点**：
- 4 个阶段串行执行，每完成一步立即打印进度
- 主题里出现的日期（"2026年8月6日" / "今天" / "2026-08-06"）会被自动识别，搜索结果会限定在该日期窗口内
- Researcher 节点**优先使用本地知识库**，KB 不足时再用 web 搜索补充
- 终端会同时打印 KB 命中数 / 搜索次数 / 各阶段耗时，便于排查
- 审核未通过时打印 `回退次数: N/3`，强制达到上限后流程结束

### 示例报告结构（Writer 输出节选）

```markdown
## 执行摘要
围绕主题的核心结论性陈述...

## 核心内容
### 1. 维度一
引用 [1][2] 支撑的论述...
### 2. 维度二
...

## 结论
总结性判断 + 展望

## 参考资料
[1] 标题 - URL
[2] 标题 - URL
```

> 引用编号 `[n]` 与报告末尾的「参考资料」一一对应，由系统根据检索顺序自动生成，
> Writer 不会自行编造参考列表。

## 技术栈

| 模块 | 选型 |
|------|------|
| Agent 编排 | LangGraph（StateGraph + 条件边） |
| LLM 调用 | LangChain + `langchain-openai`（兼容 OpenAI 协议，含 DeepSeek 等） |
| 网络搜索 | Tavily |
| 本地 Embedding | BAAI/bge-base-zh-v1.5（sentence-transformers，可 CPU/GPU） |
| 向量库 | Chroma（默认）/ OpenSearch（可选，需自配 ik 分词） |
| 配置 | pydantic-settings + .env |

## 核心特性

1. **闭环审核**：Reviewer 评分（1-10），未通过自动回退 writer 重写
2. **防死循环**：`MAX_REVISIONS` 限制最大回退次数
3. **KB + Web 融合检索**：Researcher 先查本地知识库，KB 不足时调用 Tavily，命中分别标记 `[来源:本地知识库]` / `[来源:网络]`
4. **日期感知**：自动识别主题里的日期（含「今天/昨天」相对词 + 中文/ISO 多种格式），搜索窗口围绕该日期展开
5. **可插拔向量库后端**：通过 `RAG_STORE_BACKEND` 在 Chroma / OpenSearch 间切换，业务代码无感
6. **Hybrid 检索**：OpenSearch 后端默认 `vector` + `bm25` 加权融合（`RAG_HYBRID_ALPHA` 控制权重）
7. **状态共享**：`ResearchState` TypedDict（`total=False`）让所有节点通过同一份状态协作

## 快速开始

### 1. 准备环境

```bash
# Python 3.10+（3.12 推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 然后编辑 .env，填入 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME / TAVILY_API_KEY
```

`.env` 关键字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ | LLM 鉴权 Key（启动时会校验，未配置直接报错） |
| `OPENAI_BASE_URL` | ✅ | OpenAI 兼容协议的 base URL（如 DeepSeek / 自建网关） |
| `MODEL_NAME` | ✅ | 调用的模型名 |
| `TAVILY_API_KEY` | ✅ | 网络搜索 Key |
| `BGE_MODEL_PATH` | ⛔ | 本地 BGE 模型快照路径，缺省走 HuggingFace 自动下载 |
| `RAG_STORE_BACKEND` | ⛔ | `chroma`（默认）/ `opensearch` |
| `OPENSEARCH_*` | ⛔ | 仅当后端选 `opensearch` 时需要 |
| `MAX_REVISIONS` | ⛔ | 审核不通过最大回退次数，默认 3 |
| `RAG_TOP_K` | ⛔ | KB 召回条数，默认 5 |
| `RAG_HYBRID_ALPHA` | ⛔ | hybrid 检索时向量权重，默认 0.5 |

### 3. （可选）构建本地知识库

把要作为底料的文档放到 `data_layer/documents/` 下（默认扫 `**/*.txt`），然后：

```bash
python -m scripts.build_index
# ✅ 索引完成，共 N 条
```

### 4. 跑主流程

```bash
python main.py
# 按提示输入研究主题，回车即可
```

### 5. 启动 FastAPI 后端

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

常用接口：

- `POST /api/research`
- `GET /api/research/{job_id}`
- `GET /api/research/{job_id}/events/history`
- `GET /api/research/{job_id}/events`
- `POST /api/research/{job_id}/cancel`
- `GET /api/research/{job_id}/report`
- `GET /health`

### 6. （可选）快速验证 KB 检索

```bash
python -m scripts.query_cli
# query> 你的问题
# [1] score=0.873 | ...
```

### 7. 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认前端会请求 `http://127.0.0.1:8000`（开发时走 vite 代理，生产时走 nginx 反代）。如需直连其他后端地址，就在 `frontend/.env` 里新建并设置 `VITE_API_BASE_URL`。

## 项目结构

```text
research-agent/
├── main.py                    # 入口：读主题 → 启动工作流 → 打印报告 + 耗时
├── workflow.py                # LangGraph 工作流（节点 + 条件路由 + 回退）
├── state.py                   # ResearchState TypedDict（total=False）
├── requirements               # pip 依赖清单
├── pytest.ini                 # pytest 配置
├── .env.example               # 环境变量模板
├── .dockerignore
├── Dockerfile.backend         # 后端镜像
├── docker-compose.yml         # 后端 + 前端编排
├── agents/
│   ├── planner.py             # 规划节点：主题拆解为 1-5 个子问题
│   ├── researcher.py          # 研究节点：KB 召回 + Web 搜索（多线程并发）
│   ├── writer.py              # 写作节点：基于资料 + KB 生成 Markdown 报告
│   ├── reviewer.py            # 审核节点：评分 + 通过判定
│   ├── schemas.py             # EvidenceItem / FindingItem / ReviewDecision
│   ├── evidence_utils.py      # 证据 / 引用工具函数
│   ├── _token_tracker.py      # Token 用量合并统计
│   └── tools/
│       ├── __init__.py
│       ├── search.py          # Tavily 搜索封装（含日期窗口）
│       └── date_utils.py      # 日期识别（今天/明天/中文/ISO）
├── api/                       # FastAPI 接口层
│   ├── app.py                 # 路由：创建 / 查询 / 取消 / 事件流
│   └── schemas.py             # 请求 / 响应模型
├── service/                   # L3 服务层
│   ├── rag_service.py         # RAG 统一入口（query/ingest/health_check）
│   ├── research_jobs.py       # 后台任务编排（跑工作流 + 落库）
│   └── job_store.py           # 内存 job / 事件存储
├── data_layer/                # L2 数据层
│   ├── indexer.py             # 加载 → 切分 → 向量化 → 入库 流水线
│   ├── loaders.py             # 文件加载（默认 .txt 递归）
│   ├── splitters.py           # 中文友好的递归切分
│   ├── gold_set.json          # RAG 评估手写 gold set
│   └── documents/             # 本地知识库原始文档（默认目录）
├── infrastructure/            # L1 基础设施
│   ├── config.py              # pydantic-settings 配置
│   ├── embedding_client.py    # LocalBGEEmbeddings（sentence-transformers）
│   ├── rerank_client.py       # BGE reranker 封装
│   └── vector_stores/         # 向量库抽象 + 实现
│       ├── base.py
│       ├── chroma_store.py    # Chroma 实现
│       └── opensearch_store.py# OpenSearch 实现（vector / bm25 / hybrid）
├── scripts/                   # CLI 工具
│   ├── build_index.py         # 重建知识库索引
│   ├── query_cli.py           # KB 检索交互测试
│   └── eval_rag.py            # RAG 检索质量评估（precision/recall/mrr）
├── tests/                     # pytest 测试
│   ├── conftest.py
│   └── test_*.py              # 各模块单元测试
├── frontend/                  # Vue3 + Vite 前端
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/                   # App.vue / main.js / lib/
├── local_chroma_db/           # Chroma 持久化目录（运行时生成，git 忽略）
├── local_models/              # 本地模型快照（首次启动自动下载，git 忽略）
└── README.md
```

### 架构分层

```
┌──────────────────────────────────────────────────────────┐
│ L4 业务层  agents/  workflow.py  main.py                │
│   (Planner / Researcher / Writer / Reviewer)            │
└──────────────┬───────────────────────────────────────────┘
               │  get_rag() 单例
               ▼
┌──────────────────────────────────────────────────────────┐
│ L3 服务层  service/rag_service.py                        │
│   (query / ingest / health_check)                       │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ L2 数据层  data_layer/  (loaders / splitters / indexer)  │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ L1 基础设施 infrastructure/                              │
│   (config / embedding_client / vector_stores)           │
└──────────────────────────────────────────────────────────┘
```

业务层只通过 `service.get_rag()` 拿 RAG，不直接 import L1；`RAGService` 根据
`RAG_STORE_BACKEND` 在 Chroma / OpenSearch 之间选择，**业务代码对后端无感**。
KB 检索失败时 RAGService 会 fail-open（返回空列表），不会拖垮主流程。

## 工作流

```text
                  ┌────────────────────────────┐
START → planner → researcher → writer → reviewer
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                       (通过)        (未通过+<3 次)    (未通过+≥3 次)
                         END       writer(回退)          END
```

### 各节点职责

| 节点 | 输入 | 输出 | 备注 |
|------|------|------|------|
| **planner** | `topic` | `plan`, `search_queries` | LLM 拆 1-5 个子问题，日期敏感 |
| **researcher** | `search_queries` | `search_results`, `citations`, `kb_chunks` | 多线程并发：先 KB 召回 top_k=3，再 Tavily 补充 |
| **writer** | `search_results`, `citations`, `kb_chunks`, `review_feedback` | `draft` | 按"执行摘要 / 核心内容 / 结论"结构，参考资料由系统追加 |
| **reviewer** | `draft`, `kb_chunks` | `review_score`, `review_passed`, `review_feedback` | 1-10 评分，≥8 或"通过：是"则放行 |

### 关键设计

- **状态共享**：`ResearchState` TypedDict（`total=False`），所有节点通过 `state.get(...)` 读、
  `return {...}` 增量写；新增字段不破坏老节点
- **条件路由**：`workflow.route_after_review` 四个分支（approved / needs_rewrite / needs_research / max_retries）
- **回退上限**：`MAX_REVISIONS` 防审核不通过时死循环
- **并发搜索**：Researcher 用 `ThreadPoolExecutor(MAX_SEARCH_CONCURRENCY)` 并行查每个子问题
- **KB 命中透传**：`kb_chunks` 字段在 researcher 收集后，会一并传给 writer / reviewer，
  让后续节点也能基于 KB 事实做校准
- **多模型预留**：每个 agent 各自 `ChatOpenAI(...)`，未来可独立配置 `MODEL_*`

## 常见操作

### 切换向量库后端

`.env` 里改：

```bash
RAG_STORE_BACKEND=opensearch   # 或 chroma
```

如果切到 OpenSearch，还需要：

1. `pip install opensearch-py`
2. OpenSearch 装好 ik 分词插件
3. 配置 `OPENSEARCH_HOST` / `OPENSEARCH_PORT` / `OPENSEARCH_INDEX` / `EMBEDDING_DIM`
4. 重新跑 `python -m scripts.build_index`

### 启用 Rerank

仓库已经下载了 `bge-reranker-v2-m3` 模型（`local_models/`），默认会启用 rerank；如需显式配置，把 `.env` 改成：

```bash
RAG_ENABLE_RERANK=true   # 或 false
```

### 让 KB 召回更多

```bash
RAG_TOP_K=8
RAG_CHUNK_SIZE=300   # 切更细
RAG_CHUNK_OVERLAP=80
```

### 强制走纯 Web 搜索（KB 暂时关掉）

不删配置也行——只要 `data_layer/documents/` 目录里没放文档，索引为空，KB 自然就走空了。

## 注意事项

- 启动时如果 `.env` 没配 `OPENAI_API_KEY`，会在 `infrastructure/config.py` 导入阶段直接 `ValueError`
- Researcher 单次最多调 4 轮 `web_search_tool`（`MAX_TOOL_ROUNDS`），超出即截断
- Writer 输出的引用编号必须落在 `citations` 列表里，Writer prompt 明确禁止编造编号
- Reviewer 评分靠正则从 LLM 输出里抽第一个数字，**模型回答格式漂移会直接影响评分**，
  调试时优先看 `review_feedback` 原文
- 切 OpenSearch 后端时，`metadata` 字段会原样存入；如需按来源过滤检索，可加 `terms` 子句
