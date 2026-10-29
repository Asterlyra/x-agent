from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from fastapi.testclient import TestClient

from agent.config import AgentSettings
from agent.collaboration import ArtifactEnvelope, WorkflowCoordinator
from agent.role_runner import StructuredRoleOutputError
from agent.context import ContextConfig
from agent.skills.loader import Skill
from agent.transcripts import TranscriptStore
from agent.workflows import ReviewVerdict
from backend.chat_service import ChatService
from backend.app import app, get_chat_service
from conftest import FakeMessagesListChatModelWithTools


def _events(service: ChatService, command: str) -> list[dict]:
    return list(service.stream_command(command))


def _passing_workflow() -> WorkflowCoordinator:
    return WorkflowCoordinator(
        lambda package, control: ArtifactEnvelope(
            "协作生成的最终报告", version=package.attempt
        ),
        lambda package, control: ReviewVerdict.passed("审核通过"),
    )


def test_command_layer_help_tools_thinking_and_clear(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModelWithTools(responses=[AIMessage(content="reply")]),
        settings=AgentSettings(tools=()),
    )

    help_events = _events(service, "/help")
    assert help_events[0]["type"] == "command"
    assert "/compact" in help_events[0]["detail"]
    assert "/workflow" in help_events[0]["detail"]
    assert "可用工具" in _events(service, "/tools")[0]["detail"]
    thinking_events = _events(service, "/thinking high")
    assert service.thinking_level == "off"
    assert "固定关闭" in thinking_events[0]["detail"]

    service.send("保留这条")
    _events(service, "/clear")
    assert service.get_history() == []


def test_session_new_switch_search_fork_and_stats(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="reply")]),
        settings=AgentSettings(tools=()),
    )
    service.send("查找 LangGraph")
    first_id = service.session_id

    new_events = _events(service, "/new")
    second_id = service.session_id
    assert second_id != first_id
    assert any(item["id"] == first_id and not item["active"] for item in new_events[-1]["sessions"])

    _events(service, "/switch " + first_id)
    assert service.session_id == first_id
    assert any("LangGraph" in hit["content"] for hit in service.search_sessions("langgraph"))

    fork_events = _events(service, "/fork")
    assert service.session_id not in {first_id, second_id}
    assert fork_events[-1]["type"] == "done"
    assert len(service.session_stats()) >= 2


def test_session_store_lists_current_and_archive(tmp_path: Path):
    store = TranscriptStore(tmp_path / "history")
    store.save_current([])
    store.set_current_session_id("session-a")
    store.save_archive("session-b", [])
    sessions = store.list_sessions()
    assert {item.id for item in sessions} == {"session-a", "session-b"}
    assert next(item for item in sessions if item.id == "session-a").active


def test_interrupt_stops_between_graph_updates_without_commit(tmp_path: Path):
    class TwoStepGraph:
        def stream(self, state, config=None, stream_mode=None):
            del state, config, stream_mode
            yield {"step": {"messages": [AIMessage(content="第一步", id="a1")]}}
            yield {"step": {"messages": [AIMessage(content="第二步", id="a2")]}}

    service = ChatService(transcript_root=tmp_path / "history", graph=TwoStepGraph(), settings=AgentSettings(tools=()))
    stream = service.stream_send("hello")
    assert next(stream)["type"] == "message"
    assert next(stream)["type"] == "message"
    assert service.request_interrupt() is True
    interrupted = next(stream)
    assert interrupted["type"] == "interrupted"
    assert service.messages == []


def test_retry_replaces_last_turn_without_duplicating_history(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="第一次"), AIMessage(content="重跑后")]),
        settings=AgentSettings(tools=()),
    )
    service.send("原问题")

    events = _events(service, "/retry")

    assert any(event["type"] == "replace" for event in events)
    assert events[-1]["type"] == "done"
    assert [message["content"] for message in service.get_history()] == ["原问题", "重跑后"]
    persisted = service.transcripts.load_current()
    assert [message.content for message in persisted if isinstance(message, HumanMessage)] == ["原问题"]


def test_redirect_uses_new_text_and_replaces_last_turn(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="旧答案"), AIMessage(content="新答案")]),
        settings=AgentSettings(tools=()),
    )
    service.send("旧问题")

    events = _events(service, "/redirect 新问题")

    assert events[-1]["type"] == "done"
    assert [message["content"] for message in service.get_history()] == ["新问题", "新答案"]


def test_skill_command_persists_explicit_skill_injection(tmp_path: Path):
    skill = Skill(
        name="demo",
        description="测试技能",
        content="# 测试技能\n先确认输入。",
        location="skills/demo/SKILL.md",
    )
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModelWithTools(responses=[AIMessage(content="reply")]),
        settings=AgentSettings(tools=(), skills=(skill,)),
    )

    events = _events(service, "/skill demo")

    assert "技能 demo 已注入" in events[0]["detail"]
    persisted = service.transcripts.load_current()
    assert len(persisted) == 1
    assert isinstance(persisted[0], HumanMessage)
    assert persisted[0].additional_kwargs["x_agent_kind"] == "skill"
    assert "先确认输入" in persisted[0].content


def test_compact_command_saves_summary_and_keeps_head_tail(tmp_path: Path):
    class SummaryModel:
        def invoke(self, messages):
            del messages
            return AIMessage(content="压缩后的历史摘要")

    context = ContextConfig(context_length=1, threshold=0.1, head_size=1, tail_ratio=0.2)
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=SummaryModel(),
        settings=AgentSettings(tools=(), context=context),
    )
    service.messages = [HumanMessage(content=f"消息 {index}", id=f"m{index}") for index in range(8)]
    service.transcripts.save_current(service.messages)

    events = _events(service, "/compact")

    assert "当前会话已压缩" in events[0]["detail"]
    assert len(service.messages) < 8
    assert any("Historical Summary" in str(message.content) for message in service.messages)
    assert [message.content for message in service.messages[:1]] == ["消息 0"]
    assert service.transcripts.load_current() == service.messages


def test_chat_service_restart_recovers_current_transcript(tmp_path: Path):
    root = tmp_path / "history"
    first = ChatService(
        transcript_root=root,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="持久化答案")]),
        settings=AgentSettings(tools=()),
    )
    first.send("重启恢复")

    restarted = ChatService(
        transcript_root=root,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="下一轮")]),
        settings=AgentSettings(tools=()),
    )

    assert restarted.session_id == first.session_id
    assert restarted.get_history() == first.get_history()


def test_workflow_command_streams_public_progress_and_commits_only_final_turn(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        graph=object(),
        workflow=_passing_workflow(),
        settings=AgentSettings(tools=()),
    )

    events = _events(service, "/workflow 研究当前项目")

    assert events[0]["type"] == "command"
    assert [event["type"] for event in events].count("workflow") == 5
    assert events[-1]["type"] == "done"
    assert events[-1]["workflow"]["status"] == "passed"
    assert events[-1]["workflow"]["artifact"]["content"] == "协作生成的最终报告"
    assert events[-1]["workflow"]["usage"]["roleInvocations"] == 2
    persisted = service.transcripts.load_current()
    assert [message.type for message in persisted] == ["human", "ai"]
    assert persisted[0].additional_kwargs["x_agent_kind"] == "workflow"
    assert persisted[1].additional_kwargs["workflow_status"] == "passed"


def test_command_and_session_http_api(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="reply")]),
        settings=AgentSettings(tools=()),
    )
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        assert client.get("/api/sessions").json()["currentSessionId"] == service.session_id
        response = client.post("/api/command", json={"text": "/help"})
        assert response.status_code == 200
        assert "可用命令" in response.text
        created = client.post("/api/sessions").json()
        assert created["session"]["id"] == service.session_id
        assert client.get("/api/sessions/search", params={"q": "不存在"}).json() == {"hits": []}
        assert "stats" in client.get("/api/sessions/stats").json()
    finally:
        app.dependency_overrides.clear()


def test_workflow_command_http_api_uses_workflow_sse_event(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        graph=object(),
        workflow=_passing_workflow(),
        settings=AgentSettings(tools=()),
    )
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        response = TestClient(app).post("/api/command", json={"text": "/workflow 生成报告"})
        assert response.status_code == 200
        assert "event: workflow" in response.text
        assert '"status":"passed"' in response.text
        assert "协作生成的最终报告" in response.text
    finally:
        app.dependency_overrides.clear()


def test_workflow_command_exposes_safe_structured_output_failure(tmp_path: Path):
    workflow = WorkflowCoordinator(
        lambda package, control: (_ for _ in ()).throw(StructuredRoleOutputError(
            "角色 producer 输出校验失败：evidence 必须是对象数组", role="producer"
        )),
        lambda package, control: ReviewVerdict.passed("不会执行"),
    )
    service = ChatService(
        transcript_root=tmp_path / "history",
        graph=object(),
        workflow=workflow,
        settings=AgentSettings(tools=()),
    )

    events = _events(service, "/workflow 研究项目")

    assert events[-1]["type"] == "error"
    assert events[-1]["detail"] == "角色 producer 输出校验失败：evidence 必须是对象数组"
