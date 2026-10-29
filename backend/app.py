"""FastAPI 网页入口。"""

from __future__ import annotations

import json

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from backend.chat_service import ChatService, ChatStreamEvent, MessageView, TurnBusyError


class ChatRequest(BaseModel):
    """发送消息请求。"""

    text: str = Field(min_length=1)


class CommandRequest(BaseModel):
    """发送斜杠命令请求。"""

    text: str = Field(min_length=1)


class ForkRequest(BaseModel):
    """会话 fork 请求；不传位置时复制全部历史。"""

    beforeMessage: int | None = None


class MessageResponse(BaseModel):
    """前端消费的消息结构。"""

    id: str
    role: str
    content: str
    toolName: str | None = None
    toolStatus: str | None = None


def _message_responses(messages: list[MessageView]) -> list[MessageResponse]:
    """把服务层的消息字典转换为 FastAPI 响应模型。"""
    return [MessageResponse(**message) for message in messages]


class ChatResponse(BaseModel):
    """历史或发送接口的响应。"""

    messages: list[MessageResponse]


class SessionsResponse(BaseModel):
    """会话列表响应。"""

    currentSessionId: str
    sessions: list[dict]


class SessionActionResponse(BaseModel):
    """会话切换或创建响应。"""

    session: dict
    sessions: list[dict]
    messages: list[MessageResponse]


class SearchResponse(BaseModel):
    """会话搜索响应。"""

    hits: list[dict]


class StatsResponse(BaseModel):
    """会话用量统计响应。"""

    stats: list[dict]


def _encode_sse(event: ChatStreamEvent) -> str:
    """把聊天事件编码为明确 UTF-8 的 SSE 帧。"""
    event_type = event.get("type", "message")
    payload = {key: value for key, value in event.items() if key != "type"}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {data}\n\n"


app = FastAPI(title="x-agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """按需创建服务，避免导入模块时强制读取 API key。"""
    global _service
    if _service is None:
        _service = ChatService()
    return _service


@app.get("/api/history", response_model=ChatResponse)
def get_history(service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    """返回全部消息。"""
    return ChatResponse(messages=_message_responses(service.get_history()))


@app.post("/api/chat")
def chat(request: ChatRequest, service: ChatService = Depends(get_chat_service)) -> StreamingResponse:
    """以 UTF-8 SSE 按消息和工具步骤返回一轮对话。"""
    return StreamingResponse(
        (_encode_sse(event) for event in service.stream_send(request.text)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/command")
def command(request: CommandRequest, service: ChatService = Depends(get_chat_service)) -> StreamingResponse:
    """以 UTF-8 SSE 执行斜杠命令或运行控制命令。"""
    return StreamingResponse(
        (_encode_sse(event) for event in service.stream_command(request.text)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions", response_model=SessionsResponse)
def list_sessions(service: ChatService = Depends(get_chat_service)) -> SessionsResponse:
    """返回当前和归档会话。"""
    return SessionsResponse(currentSessionId=service.session_id, sessions=service.list_sessions())


@app.post("/api/sessions", response_model=SessionActionResponse)
def create_session(service: ChatService = Depends(get_chat_service)) -> SessionActionResponse:
    """新建并切换当前会话。"""
    try:
        session = service.new_session()
    except TurnBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SessionActionResponse(
        session=session,
        sessions=service.list_sessions(),
        messages=_message_responses(service.get_history()),
    )


@app.get("/api/sessions/search", response_model=SearchResponse)
def search_sessions(
    query: str = Query("", alias="q"),
    service: ChatService = Depends(get_chat_service),
) -> SearchResponse:
    """搜索当前和归档会话。"""
    return SearchResponse(hits=service.search_sessions(query))


@app.get("/api/sessions/stats", response_model=StatsResponse)
def session_stats(service: ChatService = Depends(get_chat_service)) -> StatsResponse:
    """返回会话用量统计。"""
    return StatsResponse(stats=service.session_stats())


@app.post("/api/sessions/{session_id}/switch", response_model=SessionActionResponse)
def switch_session(session_id: str, service: ChatService = Depends(get_chat_service)) -> SessionActionResponse:
    """切换到指定归档会话。"""
    try:
        session = service.switch_session(session_id)
    except TurnBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail=f"会话不存在或不可读取: {session_id}") from exc
    return SessionActionResponse(
        session=session,
        sessions=service.list_sessions(),
        messages=_message_responses(service.get_history()),
    )


@app.post("/api/sessions/{session_id}/fork", response_model=SessionActionResponse)
def fork_session(
    session_id: str,
    request: ForkRequest | None = None,
    service: ChatService = Depends(get_chat_service),
) -> SessionActionResponse:
    """从当前会话分叉并切换；路径 id 必须是当前会话。"""
    try:
        session = service.fork_session(
            request.beforeMessage if request else None,
            source_session_id=session_id,
        )
    except TurnBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SessionActionResponse(
        session=session,
        sessions=service.list_sessions(),
        messages=_message_responses(service.get_history()),
    )


@app.post("/api/reset", response_model=ChatResponse)
def reset(service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    """清空消息历史。"""
    try:
        messages = service.reset()
    except TurnBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ChatResponse(messages=_message_responses(messages))
