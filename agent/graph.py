"""构建一个 Web Turn 内可反复执行 Step 的 LangGraph。"""

from functools import partial

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agent.config import AgentSettings
from agent.memory_tools import make_memory_tools
from agent.nodes import call_model, execute_tools, route, route_after_tools
from agent.skills.skill_tools import make_skill_tools
from agent.state import AgentState
from agent.system_prompt import build_system_prompt


class GraphFactory:
    """从外部注入的模型和单一配置构建无持久状态的 Agent 图。"""

    def __init__(self, llm: BaseChatModel, settings: AgentSettings | None = None):
        self.llm = llm
        self.settings = settings or AgentSettings()

    def build(self):
        """编译 START → 模型 → 工具 → 模型/END 的最小图。"""
        tools = list(self.settings.tools)
        skills = list(self.settings.skills)
        if skills:
            tools.extend(make_skill_tools(skills))
        if self.settings.memory is not None:
            if self.settings.history_search is None:
                raise ValueError("启用长期记忆时必须提供 history_search")
            tools.extend(make_memory_tools(self.settings.memory, self.settings.history_search))
        memory = self.settings.memory

        def system_prompt_builder(prompt_tools):
            """读取当前会话冻结快照；只有会话边界 refreeze 后才变化。"""
            return build_system_prompt(
                prompt_tools,
                persona=self.settings.persona,
                skills=skills,
                memory=(memory.frozen_snapshot() if memory is not None else None),
            )
        llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm
        call_model_node = partial(
            call_model,
            llm=llm_with_tools,
            tools=tools,
            system_prompt_builder=system_prompt_builder,
            context=self.settings.context,
            summary_llm=self.llm.bind_tools([]) if self.settings.context is not None and tools else self.llm,
            status_builder=self.settings.status_builder,
            run_control=self.settings.run_control,
        )

        graph = StateGraph(AgentState)
        graph.add_node("call_model", call_model_node)
        graph.add_node(
            "execute_tools",
            partial(
                execute_tools,
                tools=tools,
                hooks=self.settings.hooks,
                guardrails=self.settings.guardrails,
                run_control=self.settings.run_control,
            ),
        )
        graph.add_edge(START, "call_model")
        graph.add_conditional_edges(
            "call_model",
            route,
            {"continue": "execute_tools", "end": END},
        )
        graph.add_conditional_edges(
            "execute_tools",
            route_after_tools,
            {"continue": "call_model", "end": END},
        )
        return graph.compile()
