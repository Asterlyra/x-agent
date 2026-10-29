"""学习型上下文管理：剪枝、三区压缩和输入超限恢复。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage

from agent.guardrails import IDEMPOTENT_TOOL_NAMES


SUMMARY_MARKER_PREFIX = "## Historical Summary"
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


@dataclass
class ContextConfig:
    """上下文压缩配置。"""

    context_length: int = 1_000_000
    threshold: float = 0.75
    tail_ratio: float = 0.2
    head_size: int = 3
    breaker: "CompactFailureBreaker" = field(default_factory=lambda: CompactFailureBreaker())


def estimate_text_tokens(text: str) -> int:
    """用中文一字、其他字符四字的启发式估算 token。"""
    cjk = len(_CJK_RE.findall(text or ""))
    return cjk + max(0, len(text or "") - cjk) // 4


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    """估算消息列表 token 数。"""
    return sum(estimate_text_tokens(str(getattr(message, "content", "") or "")) for message in messages)


def context_usage_tokens(messages: list[BaseMessage]) -> int:
    """优先使用最近一次 provider 输入用量，并补上其后的新消息。"""
    estimated = estimate_messages_tokens(messages)
    for index in range(len(messages) - 1, -1, -1):
        usage = getattr(messages[index], "usage_metadata", None) or {}
        value = usage.get("input_tokens") or usage.get("prompt_tokens")
        try:
            provider_tokens = int(value or 0)
        except (TypeError, ValueError):
            provider_tokens = 0
        if provider_tokens > 0:
            tail_tokens = estimate_messages_tokens(messages[index + 1 :])
            return max(estimated, provider_tokens + tail_tokens)
    return estimated


def should_compress(usage_tokens: int, context_length: int, threshold: float) -> bool:
    """判断是否超过自动压缩阈值。"""
    return usage_tokens >= context_length * threshold


def prepare_compaction(messages: list[BaseMessage], tail_ratio: float, head_size: int) -> tuple[list[BaseMessage], list[BaseMessage], list[BaseMessage]]:
    """按 Head/Middle/Tail 切分，并尽量不把 ToolMessage 拆离。"""
    total = len(messages)
    head_end = min(head_size, total)
    while head_end < total and messages[head_end].type == "tool":
        head_end += 1
    if head_end >= total:
        return list(messages), [], []
    tail_target = max(2, int(total * tail_ratio))
    cut = None
    for index in range(total, head_end, -1):
        safe = (index == total or messages[index].type != "tool") and messages[index - 1].type != "tool"
        if safe and total - index >= tail_target:
            cut = index
            break
    if cut is None:
        for index in range(total, head_end, -1):
            if (index == total or messages[index].type != "tool") and messages[index - 1].type != "tool":
                cut = index
                break
    if cut is None:
        return list(messages), [], []
    return messages[:head_end], messages[head_end:cut], messages[cut:]


def prune_messages(messages: list[BaseMessage]) -> tuple[list[BaseMessage], list[str]]:
    """删除相邻重复的只读工具结果，但不破坏工具调用配对。"""
    kept: list[BaseMessage] = []
    removed: list[str] = []
    previous: tuple[str, str, str | None] | None = None
    open_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            open_call_ids = {str(call.get("id")) for call in (message.tool_calls or [])}
        if message.type == "tool" and getattr(message, "name", "") in IDEMPOTENT_TOOL_NAMES:
            key = (message.name or "", str(message.content))
            same_batch = bool(previous and (previous[2] in open_call_ids or getattr(message, "tool_call_id", None) in open_call_ids))
            if previous and previous[:2] == key and kept and kept[-1].type == "tool" and not same_batch:
                removed.append(kept.pop().id)
            previous = (*key, getattr(message, "tool_call_id", None))
        else:
            previous = None
        kept.append(message)
    return kept, removed


def _summary_text(messages: list[BaseMessage], intent: str = "") -> str:
    """把中段消息渲染为摘要请求文本。"""
    lines = [f"当前意图: {intent}"] if intent else []
    for message in messages:
        role = {"human": "用户", "ai": "助手", "tool": "工具"}.get(message.type, message.type)
        lines.append(f"[{role}] {str(message.content)[:2000]}")
    return "\n".join(lines)


def generate_summary(messages: list[BaseMessage], llm, intent: str = "") -> str:
    """让不绑定工具的模型生成历史参考摘要。"""
    response = llm.invoke([HumanMessage(content=(
        "将下面的历史压缩为中文结构化摘要。摘要只是历史参考，不是活动指令；保留目标、约束、决定和关键事实。\n"
        + _summary_text(messages, intent)
    ))])
    return str(getattr(response, "content", response)).strip()


def compact(messages: list[BaseMessage], summary: str, head: list[BaseMessage], middle: list[BaseMessage], tail: list[BaseMessage]) -> tuple[list[BaseMessage], list[BaseMessage], float]:
    """用摘要替换 Middle，并以新 id 重放 Tail。"""
    marker = HumanMessage(content=f"{SUMMARY_MARKER_PREFIX}（以下为历史摘要，仅作参考，不是当前指令）\n{summary}")
    tail_copies = [message.model_copy(update={"id": str(uuid.uuid4())}) for message in tail]
    new_messages = [*head, marker, *tail_copies]
    updates = [RemoveMessage(id=message.id) for message in [*middle, *tail]] + [marker, *tail_copies]
    old_tokens = estimate_messages_tokens(messages)
    new_tokens = estimate_messages_tokens(new_messages)
    ratio = 1 - new_tokens / old_tokens if old_tokens else 0.0
    return new_messages, updates, ratio


class CompactFailureBreaker:
    """连续摘要失败后暂停自动压缩。"""

    def __init__(self, limit: int = 3):
        self.limit = limit
        self.failures = 0

    @property
    def is_blown(self) -> bool:
        """是否已达到失败上限。"""
        return self.failures >= self.limit

    def record_failure(self) -> None:
        """记录失败。"""
        self.failures += 1

    def record_success(self) -> None:
        """成功后清零。"""
        self.failures = 0


def transform_context(messages: list[BaseMessage], state: dict, context: ContextConfig, summary_llm, force: bool = False) -> tuple[list[BaseMessage], dict, dict | None]:
    """在模型请求前执行剪枝和必要的压缩。"""
    messages, pruned = prune_messages(messages)
    usage = context_usage_tokens(messages)
    if not force and not state.get("compact_needed") and not should_compress(usage, context.context_length, context.threshold):
        return messages, ({"messages": [RemoveMessage(id=item) for item in pruned]} if pruned else {}), None
    if context.breaker.is_blown and not force:
        return messages, ({"messages": [RemoveMessage(id=item) for item in pruned]} if pruned else {}), None
    head, middle, tail = prepare_compaction(messages, context.tail_ratio, context.head_size)
    if not middle:
        return messages, {}, None
    try:
        intent = next((str(message.content) for message in reversed(messages) if message.type == "human"), "")
        summary = generate_summary(middle, summary_llm, intent)
        context.breaker.record_success()
    except Exception:
        context.breaker.record_failure()
        if force:
            raise
        return messages, ({"messages": [RemoveMessage(id=item) for item in pruned]} if pruned else {}), None
    new_messages, updates, ratio = compact(messages, summary, head, middle, tail)
    if pruned:
        updates = [RemoveMessage(id=item) for item in pruned] + updates
    return new_messages, {"messages": updates}, {"ratio": ratio}


def classify_overflow_error(exc: Exception) -> bool:
    """判断异常是否像上下文超限。"""
    text = str(exc).lower()
    return any(hint in text for hint in ("context length", "maximum context", "context window", "input length", "too long", "上下文长度", "超出上下文"))
