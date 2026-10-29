"""单一 Agent 配置，替代旧的 Profile/AgentSpec 双层配置。"""

from dataclasses import dataclass, field
from typing import Callable

from agent.context import ContextConfig
from agent.guardrails import ToolCallGuardrailController
from agent.hooks import ToolHooks
from agent.memory import Memory
from agent.run_control import RunController
from agent.status import build_status
from agent.tools import TOOL_REGISTRY


@dataclass(frozen=True)
class AgentSettings:
    """描述本地 Web Agent 的单一依赖和行为配置。"""

    persona: str | None = None
    tools: tuple = field(default_factory=lambda: tuple(TOOL_REGISTRY))
    skills: tuple = ()
    memory: Memory | None = None
    history_search: Callable | None = None
    context: ContextConfig | None = field(default_factory=ContextConfig)
    status_builder: Callable | None = build_status
    hooks: ToolHooks = field(default_factory=ToolHooks)
    guardrails: ToolCallGuardrailController = field(default_factory=ToolCallGuardrailController)
    run_control: RunController | None = None
