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
2. 安装依赖：`pip install -r requirements.txt`
3. 运行：`python main.py`

## 项目结构