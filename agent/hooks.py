"""工具调用前后钩子。"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class BeforeToolCallResult:
    """before 钩子返回值；allow=False 时合成错误结果而不执行工具。"""

    allow: bool = True
    reason: str = ""

    @classmethod
    def block(cls, reason: str) -> "BeforeToolCallResult":
        """构造一个拦截结果。"""
        return cls(allow=False, reason=reason)


@dataclass(frozen=True)
class AfterToolCallResult:
    """after 钩子可修改展示内容、错误状态或结束当前工具批次。"""

    content: str | None = None
    is_error: bool | None = None
    terminate: bool | None = None


def _default_before(name: str, args: dict) -> BeforeToolCallResult:
    """默认放行。"""
    del name, args
    return BeforeToolCallResult()


def _default_after(name: str, args: dict, content: str, is_error: bool) -> AfterToolCallResult:
    """默认不修改工具结果。"""
    del name, args, content, is_error
    return AfterToolCallResult()


@dataclass
class ToolHooks:
    """一组可注入的工具前后钩子。"""

    before_tool_call: Callable[[str, dict], BeforeToolCallResult] = field(default=_default_before)
    after_tool_call: Callable[[str, dict, str, bool], AfterToolCallResult] = field(default=_default_after)


def make_task_complete_hooks() -> ToolHooks:
    """提供一个待办全部完成后提前收尾的示例钩子。"""
    from agent.tools.todo_tool import _todos

    def after_tool_call(name: str, args: dict, content: str, is_error: bool) -> AfterToolCallResult:
        del args, content
        if name == "todo_done" and not is_error and _todos and all(item["done"] for item in _todos):
            return AfterToolCallResult(terminate=True)
        return AfterToolCallResult()

    return ToolHooks(after_tool_call=after_tool_call)
