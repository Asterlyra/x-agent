"""工具调用护栏：检测重复失败和只读工具原地打转。"""

import hashlib
import json
from dataclasses import dataclass, field


IDEMPOTENT_TOOL_NAMES = frozenset({
    "read_file",
    "list_repo_files",
    "read_repo_file",
    "todo_list",
    "web_search",
    "fetch_url",
})


@dataclass(frozen=True)
class ToolGuardrailConfig:
    """护栏阈值。"""

    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)


@dataclass(frozen=True)
class ToolCallSignature:
    """工具名和规范化参数的短哈希。"""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: dict | None) -> "ToolCallSignature":
        normalized = json.dumps(args or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return cls(tool_name, hashlib.sha256(normalized.encode()).hexdigest()[:16])


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """护栏动作：allow、warn、block 或 halt。"""

    action: str = "allow"
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0

    @property
    def allows_execution(self) -> bool:
        """warn 继续执行，block/halt 拦截。"""
        return self.action in {"allow", "warn"}


class ToolCallGuardrailController:
    """一轮内记录工具调用结果并作出下一步决策。"""

    def __init__(self, config: ToolGuardrailConfig | None = None):
        self.config = config or ToolGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        """清空本轮计数。"""
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        """返回本轮是否已经要求 halt。"""
        return self._halt_decision

    def before_call(self, tool_name: str, args: dict | None) -> ToolGuardrailDecision:
        """执行前拦截达到阈值的重复调用。"""
        signature = ToolCallSignature.from_call(tool_name, args)
        exact = self._exact_failure_counts.get(signature, 0)
        if exact >= self.config.exact_failure_block_after:
            decision = ToolGuardrailDecision("block", "repeated_exact_failure_block", f"拦截 {tool_name}: 相同调用已连续失败 {exact} 次。", tool_name, exact)
            self._halt_decision = decision
            return decision
        record = self._no_progress.get(signature)
        if tool_name in self.config.idempotent_tools and record and record[1] >= self.config.no_progress_block_after:
            decision = ToolGuardrailDecision("block", "idempotent_no_progress_block", f"拦截 {tool_name}: 相同只读结果已重复 {record[1]} 次。", tool_name, record[1])
            self._halt_decision = decision
            return decision
        return ToolGuardrailDecision(tool_name=tool_name)

    def after_call(self, tool_name: str, args: dict | None, content: str, is_error: bool) -> ToolGuardrailDecision:
        """执行后记录失败或重复结果。"""
        signature = ToolCallSignature.from_call(tool_name, args)
        if is_error:
            exact = self._exact_failure_counts.get(signature, 0) + 1
            same = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._exact_failure_counts[signature] = exact
            self._same_tool_failure_counts[tool_name] = same
            if same >= self.config.same_tool_failure_halt_after:
                decision = ToolGuardrailDecision("halt", "same_tool_failure_halt", f"停止 {tool_name}: 本轮连续失败 {same} 次。", tool_name, same)
                self._halt_decision = decision
                return decision
            if exact >= self.config.exact_failure_warn_after or same >= self.config.same_tool_failure_warn_after:
                return ToolGuardrailDecision("warn", "repeated_tool_failure", f"{tool_name} 已连续失败，检查错误并换策略。", tool_name, max(exact, same))
            return ToolGuardrailDecision(tool_name=tool_name, count=exact)

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)
        if tool_name not in self.config.idempotent_tools:
            return ToolGuardrailDecision(tool_name=tool_name)
        digest = hashlib.sha256((content or "").encode()).hexdigest()[:16]
        previous = self._no_progress.get(signature)
        repeat = previous[1] + 1 if previous and previous[0] == digest else 1
        self._no_progress[signature] = (digest, repeat)
        if repeat >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision("warn", "idempotent_no_progress_warning", f"{tool_name} 已重复返回相同结果 {repeat} 次。", tool_name, repeat)
        return ToolGuardrailDecision(tool_name=tool_name)
