from langchain_core.messages import AIMessage, HumanMessage

from agent.context import ContextConfig, context_usage_tokens, prepare_compaction, transform_context
from agent.guardrails import ToolGuardrailConfig, ToolCallGuardrailController
from agent.hooks import BeforeToolCallResult, ToolHooks
from agent.memory import Memory
from agent.nodes import execute_tools
from agent.tools import calculate


def test_context_compaction_keeps_head_and_tail():
    messages = [HumanMessage(content=f"消息 {index}", id=f"m{index}") for index in range(10)]
    head, middle, tail = prepare_compaction(messages, tail_ratio=0.2, head_size=2)
    assert [message.content for message in head] == ["消息 0", "消息 1"]
    assert middle
    assert [message.content for message in tail][-1] == "消息 9"


def test_context_transform_replaces_middle_with_historical_summary():
    class SummaryModel:
        def invoke(self, messages):
            return AIMessage(content="保留的历史摘要")

    messages = [HumanMessage(content="中文" * 10, id=f"m{index}") for index in range(8)]
    new_messages, updates, info = transform_context(
        messages,
        {},
        ContextConfig(context_length=1, threshold=0.1, head_size=1, tail_ratio=0.2),
        SummaryModel(),
    )
    assert info and info["ratio"] > 0
    assert any("Historical Summary" in str(message.content) for message in new_messages)
    assert updates["messages"]


def test_tool_before_hook_can_block_execution():
    state = {
        "messages": [AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"expression": "2+3"}, "id": "c1", "type": "tool_call"}])]
    }
    result = execute_tools(
        state,
        tools=[calculate],
        hooks=ToolHooks(before_tool_call=lambda name, args: BeforeToolCallResult.block("教学测试拦截")),
    )
    assert result["messages"][0].status == "error"
    assert "教学测试拦截" in result["messages"][0].content


def test_guardrail_halts_repeated_failures():
    controller = ToolCallGuardrailController(ToolGuardrailConfig(same_tool_failure_halt_after=2))
    controller.after_call("read_file", {"path": "missing"}, "Error", True)
    decision = controller.after_call("read_file", {"path": "missing"}, "Error", True)
    assert decision.action == "halt"
    assert controller.halt_decision is not None


def test_memory_write_does_not_change_frozen_snapshot(tmp_path):
    memory = Memory(tmp_path / "MEMORY.md")
    assert memory.frozen_snapshot() == ""
    assert memory.remember("用户偏好中文")['success'] is True
    assert memory.frozen_snapshot() == ""
    memory.refreeze()
    assert "用户偏好中文" in memory.frozen_snapshot()


def test_context_usage_prefers_provider_metadata_and_adds_new_tail():
    messages = [
        HumanMessage(content="旧问题", id="u1"),
        AIMessage(
            content="旧回答",
            id="a1",
            usage_metadata={"input_tokens": 900, "output_tokens": 20, "total_tokens": 920},
        ),
        HumanMessage(content="新问题" * 20, id="u2"),
    ]

    assert context_usage_tokens(messages) > 900


def test_guardrail_block_sets_halt_decision():
    controller = ToolCallGuardrailController(ToolGuardrailConfig(exact_failure_block_after=1))
    controller.after_call("read_file", {"path": "missing"}, "Error", True)

    decision = controller.before_call("read_file", {"path": "missing"})

    assert decision.action == "block"
    assert controller.halt_decision == decision
