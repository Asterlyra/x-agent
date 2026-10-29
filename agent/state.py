"""一次 Web Turn 内由 LangGraph 管理的消息和控制标志。"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """messages 是完整 LangChain 历史；动态提示词不写入这里。"""

    messages: Annotated[list[BaseMessage], add_messages]
    thinking_level: str  # Phase 13.1 固定为 off，保留字段只为状态结构兼容。
    tool_terminate: bool
    compact_needed: bool
