import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent.config import AgentSettings
from agent.tools import calculate
from backend.app import app, get_chat_service
from backend.chat_service import ChatService
from conftest import FakeMessagesListChatModelWithTools


def _sse_events(response):
    """解析测试中的 SSE 帧。"""
    events = []
    for frame in response.text.strip().split("\n\n"):
        event_type = "message"
        data = ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").lstrip()
        if data:
            events.append({"type": event_type, **json.loads(data)})
    return events


def test_chat_api_history_chat_and_reset(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="fake reply")]),
        settings=AgentSettings(tools=()),
    )
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        assert client.get("/api/history").json() == {"messages": []}
        response = client.post("/api/chat", json={"text": "hello"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(response)
        assert [event["type"] for event in events] == ["message", "message", "done"]
        assert [event["message"]["role"] for event in events[:2]] == ["user", "assistant"]
        assert [item["role"] for item in events[-1]["messages"]] == ["user", "assistant"]
        assert client.post("/api/reset").json() == {"messages": []}
    finally:
        app.dependency_overrides.clear()


def test_chat_api_does_not_commit_failed_turn(tmp_path: Path):
    class BrokenGraph:
        def stream(self, state, stream_mode=None):
            del state, stream_mode
            raise RuntimeError("model unavailable")

    service = ChatService(transcript_root=tmp_path / "history", graph=BrokenGraph())
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        response = client.post("/api/chat", json={"text": "hello"})
        assert response.status_code == 200
        assert _sse_events(response)[-1]["type"] == "error"
        assert service.get_history() == []
    finally:
        app.dependency_overrides.clear()


def test_chat_service_persists_tool_call_pair_but_projects_tool_card(tmp_path: Path):
    model = FakeMessagesListChatModelWithTools(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"expression": "2+3"}, "id": "call-1", "type": "tool_call"}]),
            AIMessage(content="答案是 5"),
        ]
    )
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=model,
        settings=AgentSettings(tools=(calculate,)),
    )
    views = service.send("2+3 等于几？")
    assert [item["role"] for item in views] == ["user", "tool", "tool", "assistant"]
    persisted = service.transcripts.load_current()
    assert persisted[1].tool_calls[0]["name"] == "calculate"
    assert persisted[2].tool_call_id == "call-1"


def test_chat_api_streams_tool_start_and_result(tmp_path: Path):
    model = FakeMessagesListChatModelWithTools(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"expression": "2+3"}, "id": "call-1", "type": "tool_call"}]),
            AIMessage(content="答案是 5"),
        ]
    )
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=model,
        settings=AgentSettings(tools=(calculate,)),
    )
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        response = TestClient(app).post("/api/chat", json={"text": "2+3 等于几？"})
        events = _sse_events(response)
        messages = [event["message"] for event in events if event["type"] == "message"]
        tool_messages = [message for message in messages if message["role"] == "tool"]
        assert [message["toolStatus"] for message in tool_messages] == ["running", "success"]
        assert [message["toolName"] for message in tool_messages] == ["calculate", "calculate"]
        assert events[-1]["type"] == "done"
        persisted_tool_views = [message for message in events[-1]["messages"] if message["role"] == "tool"]
        assert persisted_tool_views[0]["toolStatus"] == "success"
    finally:
        app.dependency_overrides.clear()


def test_session_api_reports_missing_and_busy_states(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="reply")]),
        settings=AgentSettings(tools=()),
    )
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        assert client.post("/api/sessions/missing-session/switch").status_code == 404

        service._acquire_operation(turn=True)
        try:
            response = client.post("/api/reset")
            assert response.status_code == 409
            assert "已有回合" in response.json()["detail"]
        finally:
            service._release_operation(turn=True)
    finally:
        app.dependency_overrides.clear()


def test_session_search_uses_documented_q_parameter(tmp_path: Path):
    service = ChatService(
        transcript_root=tmp_path / "history",
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="reply")]),
        settings=AgentSettings(tools=()),
    )
    service.send("查找这个关键词")
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        response = TestClient(app).get("/api/sessions/search", params={"q": "关键词"})
        assert response.status_code == 200
        assert response.json()["hits"][0]["sessionId"] == service.session_id
    finally:
        app.dependency_overrides.clear()
