import json
from pathlib import Path
import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import agent.llm as llm_module
import agent.tools.shell_tool as shell_tool_module
from agent.config import AgentSettings
from agent.llm import create_llm
from agent.nodes import execute_tools
from agent.tools import TOOL_REGISTRY
from agent.tools.shell_tool import run_command
from backend.chat_service import ChatService, TurnBusyError


class BlockingGraph:
    """让测试可以把一个流式 Turn 固定在图节点内部。"""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def stream(self, state, config=None, stream_mode=None):
        del state, config, stream_mode
        self.started.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("测试未释放 BlockingGraph")
        yield {"model": {"messages": [AIMessage(content="第一轮完成", id="a1")]}}


def test_single_flight_rejects_overlap_and_reset_without_losing_turn(tmp_path):
    graph = BlockingGraph()
    service = ChatService(
        transcript_root=tmp_path / "history",
        graph=graph,
        settings=AgentSettings(tools=()),
    )
    completed: list[dict] = []

    worker = threading.Thread(target=lambda: completed.extend(service.stream_send("第一轮")))
    worker.start()
    assert graph.started.wait(timeout=1)

    overlap = list(service.stream_send("第二轮"))
    assert len(overlap) == 1
    assert overlap[0]["type"] == "error"
    assert overlap[0]["messages"] == []
    assert "已有回合" in overlap[0]["detail"]
    with pytest.raises(TurnBusyError):
        service.reset()

    graph.release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert completed[-1]["type"] == "done"
    assert [item["content"] for item in service.get_history()] == ["第一轮", "第一轮完成"]


def test_closing_stream_releases_single_flight(tmp_path):
    graph = BlockingGraph()
    service = ChatService(
        transcript_root=tmp_path / "history",
        graph=graph,
        settings=AgentSettings(tools=()),
    )
    stream = service.stream_send("未完成")
    assert next(stream)["type"] == "message"
    stream.close()

    assert service.reset() == []


def test_default_registry_does_not_expose_raw_shell():
    assert "run_command" not in {item.name for item in TOOL_REGISTRY}
    assert "list_files" in {item.name for item in TOOL_REGISTRY}


def test_list_files_stays_inside_workspace_and_caps_results(tmp_path: Path, monkeypatch):
    from agent.tools import file_tools

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "outside-link").symlink_to(outside)
    monkeypatch.setattr(file_tools, "WORKSPACE_ROOT", workspace)

    result = file_tools.list_files.invoke({"directory": ".", "recursive": True, "limit": 1})

    assert "secret.txt" not in result
    assert "最多显示 1 条" in result


def test_explicit_shell_nonzero_exit_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(shell_tool_module, "WORKSPACE_ROOT", tmp_path / "workspace")

    with pytest.raises(RuntimeError, match="退出码 7"):
        run_command.invoke({"command": "exit 7"})


def test_deepseek_factory_explicitly_disables_thinking():
    llm = create_llm(api_key="test-key", reasoning_effort="off")

    assert llm.extra_body == {"thinking": {"type": "disabled"}}
    assert llm.reasoning_effort is None
    with pytest.raises(ValueError, match="只支持关闭"):
        create_llm(api_key="test-key", reasoning_effort="high")


def test_environment_key_overrides_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=dotenv-key\n", encoding="utf-8")
    monkeypatch.setattr(llm_module, "ENV_FILE", env_file)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")

    llm = create_llm()

    assert llm.openai_api_key.get_secret_value() == "environment-key"


def test_structured_tool_failure_becomes_error_message():
    @tool
    def failing_tool() -> str:
        """返回结构化失败。"""
        return json.dumps({"success": False, "error": "失败"}, ensure_ascii=False)

    state = {
        "messages": [AIMessage(
            content="",
            tool_calls=[{"name": "failing_tool", "args": {}, "id": "call-1", "type": "tool_call"}],
        )],
    }

    result = execute_tools(state, tools=[failing_tool])

    assert result["messages"][0].status == "error"
