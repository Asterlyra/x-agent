"""为一次 Agent 工作流提供步骤级预算、取消和用量统计。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock
from time import monotonic
from typing import Any, Callable


class RunStopped(RuntimeError):
    """表示运行在调用下一步前被确定性控制面停止。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class RunCancelled(RunStopped):
    """表示用户已经请求取消当前运行。"""


class RunBudgetExceeded(RunStopped):
    """表示当前运行已经没有足够预算执行下一步。"""


@dataclass(frozen=True)
class RunBudget:
    """定义一次自主协作运行允许消耗的上限。"""

    role_invocations: int = 12
    model_requests: int = 80
    tool_calls: int = 120
    input_tokens: int = 1_000_000
    output_tokens: int = 300_000
    cache_read_tokens: int = 1_000_000
    delegation_calls: int = 9
    elapsed_seconds: float = 300.0
    per_request_output_tokens: int = 8_192

    def __post_init__(self) -> None:
        values = (
            self.role_invocations,
            self.model_requests,
            self.tool_calls,
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.delegation_calls,
            self.elapsed_seconds,
            self.per_request_output_tokens,
        )
        if any(value <= 0 for value in values):
            raise ValueError("run budget 必须全部大于 0")


@dataclass(frozen=True)
class RunUsage:
    """记录一次自主协作运行已经实际发生的消耗。"""

    role_invocations: int = 0
    model_requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    delegation_calls: int = 0
    elapsed_seconds: float = 0.0


class RunController:
    """在每个角色、模型和工具步骤前执行预算及取消检查。"""

    def __init__(
        self,
        budget: RunBudget | None = None,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ):
        self.budget = budget or RunBudget()
        self._cancel_requested = cancel_requested or (lambda: False)
        self._started_at = monotonic()
        self._usage = RunUsage()
        self._role_calls: dict[str, int] = {}
        self._exhausted_reason = ""
        self._lock = Lock()

    def _elapsed(self) -> float:
        return max(0.0, monotonic() - self._started_at)

    def _stop_reason_locked(self) -> str:
        if self._exhausted_reason:
            return self._exhausted_reason
        if self._elapsed() >= self.budget.elapsed_seconds:
            return "elapsed_time_limit"
        return ""

    def checkpoint(self) -> None:
        """在不消耗计数的边界检查取消、时间和请求后 Token 超限。"""
        if self._cancel_requested():
            raise RunCancelled("cancelled")
        with self._lock:
            reason = self._stop_reason_locked()
        if reason:
            raise RunBudgetExceeded(reason)

    def _reserve(self, field_name: str, limit: int, reason: str) -> None:
        if self._cancel_requested():
            raise RunCancelled("cancelled")
        with self._lock:
            stop_reason = self._stop_reason_locked()
            if not stop_reason:
                current = int(getattr(self._usage, field_name))
                if current >= limit:
                    self._exhausted_reason = reason
                    stop_reason = reason
                else:
                    self._usage = replace(self._usage, **{field_name: current + 1})
        if stop_reason:
            raise RunBudgetExceeded(stop_reason)

    def before_role(self, role: str) -> None:
        """在角色图创建前预留一次角色调用。"""
        self._reserve("role_invocations", self.budget.role_invocations, "role_invocation_limit")
        with self._lock:
            self._role_calls[role] = self._role_calls.get(role, 0) + 1

    def before_model_request(self) -> None:
        """在真实模型请求前预留一次调用。"""
        self._reserve("model_requests", self.budget.model_requests, "model_request_limit")

    def before_tool_call(self) -> None:
        """在真实工具执行前预留一次调用。"""
        self._reserve("tool_calls", self.budget.tool_calls, "tool_call_limit")

    def before_delegation(self) -> None:
        """在创建一级子 Agent 前预留一次委派。"""
        self._reserve("delegation_calls", self.budget.delegation_calls, "delegation_call_limit")

    def output_token_limit(self) -> int:
        """返回下一次请求可声明的最大输出 Token 数。"""
        with self._lock:
            remaining = self.budget.output_tokens - self._usage.output_tokens
        if remaining <= 0:
            raise RunBudgetExceeded("output_token_limit")
        return min(remaining, self.budget.per_request_output_tokens)

    def record_model_response(self, response: Any) -> None:
        """请求完成后按 Provider usage_metadata 准确累计 Token。"""
        usage = getattr(response, "usage_metadata", None) or {}
        details = usage.get("input_token_details", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_tokens = int(
            usage.get("cache_read_input_tokens", details.get("cache_read", 0)) or 0
        )
        with self._lock:
            self._usage = replace(
                self._usage,
                input_tokens=self._usage.input_tokens + input_tokens,
                output_tokens=self._usage.output_tokens + output_tokens,
                cache_read_tokens=self._usage.cache_read_tokens + cache_tokens,
            )
            checks = (
                ("input_token_limit", self._usage.input_tokens, self.budget.input_tokens),
                ("output_token_limit", self._usage.output_tokens, self.budget.output_tokens),
                ("cache_token_limit", self._usage.cache_read_tokens, self.budget.cache_read_tokens),
            )
            for reason, actual, limit in checks:
                if actual >= limit:
                    self._exhausted_reason = reason
                    break

    def usage(self) -> RunUsage:
        """返回包含当前墙钟耗时的不可变用量快照。"""
        with self._lock:
            return replace(self._usage, elapsed_seconds=self._elapsed())

    def role_calls(self) -> dict[str, int]:
        """返回每种角色的调用次数。"""
        with self._lock:
            return dict(self._role_calls)


__all__ = [
    "RunBudget",
    "RunBudgetExceeded",
    "RunCancelled",
    "RunController",
    "RunStopped",
    "RunUsage",
]
