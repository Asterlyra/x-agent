"""把工具计数、错误和系统信息注入模型调用末尾的状态栏。"""

import os
import platform
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent.tools.todo_tool import todo_snapshot


def next_tool_call_number(messages: list[BaseMessage]) -> int:
    """统计已出现的模型工具调用数量。"""
    return sum(len(message.tool_calls or []) for message in messages if isinstance(message, AIMessage))


def todo_summary() -> str | None:
    """返回未完成待办概览。"""
    todos = todo_snapshot()
    pending = [item["task"] for item in todos if not item["done"]]
    if not pending:
        return None
    return f"{len(pending)} 项未完成（共 {len(todos)} 项）: {'、'.join(pending[:2])}{'…' if len(pending) > 2 else ''}"


def last_error_summary(messages: list[BaseMessage]) -> str | None:
    """提取最近一个工具错误，帮助模型选择修复策略。"""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and getattr(message, "status", None) == "error":
            return str(message.content or "")[:240]
    return None


def system_info(now: datetime | None = None) -> str:
    """返回当前时间、工作目录和运行环境。"""
    now = now or datetime.now()
    return f"时间 {now:%Y-%m-%d %H:%M}；工作目录 {os.getcwd()}；环境 Python {platform.python_version()} / {platform.system()}"


def build_status(state: dict, now: datetime | None = None) -> str:
    """构造只在本次模型请求中出现的 `<agent_status>` 内容。"""
    messages = state.get("messages", [])
    lines = [f"- 工具调用计数: 已执行 {next_tool_call_number(messages)} 次"]
    if (summary := todo_summary()):
        lines.append(f"- 待办: {summary}")
    if (error := last_error_summary(messages)):
        lines.append(f"- 最近工具错误: {error}")
    lines.append(f"- 系统状态: {system_info(now)}")
    return "\n".join(lines)
