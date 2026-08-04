# Research Agent — 基于 LangGraph 的智能信息研究助手

## 项目简介
基于 LangGraph 构建的多 Agent 协作研究系统，支持"规划-研究-写作-审核"闭环工作流，具备条件回退机制。

## 技术栈
- **LangGraph**: 状态图编排与条件路由
- **LangChain**: LLM 调用与工具封装
- **DeepSeek API**: 大语言模型推理
- **Tavily**: 网络搜索工具

## 核心特性
1. **闭环审核**：Reviewer Agent 评分，未达标自动回退重写
2. **防死循环**：最大回退次数限制，保证流程终止
3. **状态共享**：TypedDict 定义全局状态，各 Agent 通过状态对象协作

## 快速开始
1. 配置环境：`cp .env.example .env`（填入 API Key）
2. 安装依赖：`pip install -r requirements`
3. 运行：`python main.py`

## 项目结构
```
research-agent/
├── main.py              # 入口：读取主题 → 启动工作流 → 打印报告
├── workflow.py          # LangGraph 工作流定义（节点 + 条件边）
├── state.py             # 全局状态 TypedDict
├── config.py            # 配置加载（.env + pydantic-settings）
├── agents/
│   ├── planner.py       # 规划节点：拆解主题为 3-5 个子问题
│   ├── researcher.py    # 研究节点：调用 Tavily 搜索每个子问题
│   ├── writer.py        # 写作节点：基于资料生成 Markdown 报告
│   └── reviewer.py      # 审核节点：评分 + 通过/不通过
├── tools/
│   └── search.py        # Tavily 搜索封装
├── requirements        # pip 依赖清单
├── .env.example        # 环境变量模板
└── README.md
```

## 工作流
```
START → planner → researcher → writer → reviewer
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                       (通过)        (未通过+<3次)   (未通过+≥3次)
                         END       writer(回退)          END
```

- **planner**：LLM 拆题，输出 3-5 个子问题
- **researcher**：对每个子问题调一次 Tavily 搜索，汇总结果
- **writer**：把搜索资料拼成 Markdown 报告
- **reviewer**：LLM 评分（1-10），≥8 或明确"通过：是"则放行
- **回退机制**：不通过就回 writer 重写，最多 3 次（`MAX_REVISIONS`）

## 关键设计
- **状态共享**：`ResearchState` TypedDict，所有节点通过它交换数据
- **条件路由**：`route_after_review` 三个分支，审过 / 重写 / 强制终止
- **回退上限**：`MAX_REVISIONS` 防止审核不通过时死循环
- **多模型预留**：每个 agent 可独立配置 `MODEL_*`（当前统一用 `MODEL_NAME`）