"""单用户 ChatService：把 Web Turn、命令和轻量会话接到 LangGraph。"""

from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path
import threading
from typing import Any, NotRequired, TypedDict, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from agent.config import AgentSettings
from agent.collaboration import WorkflowCoordinator, WorkflowEvent, WorkflowResult
from agent.context import estimate_messages_tokens, transform_context
from agent.graph import GraphFactory
from agent.llm import PROJECT_ROOT, THINKING_LEVELS, create_llm
from agent.memory import Memory
from agent.role_runner import RoleInvocationError, StructuredRoleOutputError, build_collaborative_workflow
from agent.skills.format import format_skill_content
from agent.skills.loader import load_skills
from agent.transcripts import TranscriptFileError, TranscriptStore


GRAPH_RECURSION_LIMIT = 50
logger = logging.getLogger(__name__)


class TurnBusyError(RuntimeError):
    """表示当前已有 Turn 或会话写操作占用 single-flight。"""


class MessageView(TypedDict):
    """浏览器消费的消息投影，不是 durable transcript 格式。"""

    id: str
    role: str
    content: str
    toolName: NotRequired[str]
    toolStatus: NotRequired[str]


class ChatStreamEvent(TypedDict, total=False):
    """一次聊天或命令 SSE 事件的内部结构。"""

    type: str
    message: MessageView
    messages: list[MessageView]
    detail: str
    command: str
    sessions: list[dict]
    session: dict
    hits: list[dict]
    stats: list[dict]
    workflow: dict


def _content_text(content: Any) -> str:
    """把 LangChain 内容块收敛为浏览器可显示文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif block:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


def _view_from_message(message: BaseMessage) -> MessageView | None:
    """将完整消息投影成稳定 id 的前端视图。"""
    message_id = str(getattr(message, "id", ""))
    if isinstance(message, HumanMessage):
        human_view: MessageView = {
            "id": message_id,
            "role": "user",
            "content": _content_text(message.content),
        }
        return human_view
    if isinstance(message, ToolMessage):
        tool_view: MessageView = {
            "id": message_id,
            "role": "tool",
            "content": _content_text(message.content),
            "toolName": message.name or "tool",
            "toolStatus": "error" if message.status == "error" else "success",
        }
        return tool_view
    if isinstance(message, AIMessage):
        if message.tool_calls and not _content_text(message.content):
            tool_call_view: MessageView = {
                "id": message_id,
                "role": "tool",
                "content": "正在调用工具",
                "toolName": message.tool_calls[0]["name"],
                "toolStatus": "success",
            }
            return tool_call_view
        assistant_view: MessageView = {
            "id": message_id,
            "role": "assistant",
            "content": _content_text(message.content),
        }
        return assistant_view
    return None


def _project(messages: list[BaseMessage]) -> list[MessageView]:
    """过滤系统消息并生成前端投影。"""
    return [view for message in messages if (view := _view_from_message(message)) is not None]


def _ensure_message_ids(messages: list[BaseMessage]) -> list[BaseMessage]:
    """为 fake model 等没有 id 的消息补稳定 id。"""
    result: list[BaseMessage] = []
    seen: set[str] = set()
    for message in messages:
        message_id = str(getattr(message, "id", "") or "")
        if not message_id or message_id in seen:
            message = message.model_copy(update={"id": f"msg-{uuid4().hex}"})
            message_id = str(message.id)
        seen.add(message_id)
        result.append(message)
    return result


def _workflow_event_view(event: WorkflowEvent) -> dict:
    """把内部协作事件收敛成前端可展示、无隐藏推理的投影。"""
    usage = event.usage
    view: dict[str, Any] = {
        "phase": event.phase,
        "status": event.status,
        "detail": event.detail,
        "usage": {
            "roleInvocations": usage.role_invocations,
            "modelRequests": usage.model_requests,
            "toolCalls": usage.tool_calls,
            "inputTokens": usage.input_tokens,
            "outputTokens": usage.output_tokens,
            "cacheReadTokens": usage.cache_read_tokens,
            "delegationCalls": usage.delegation_calls,
            "elapsedSeconds": round(usage.elapsed_seconds, 6),
        },
    }
    if event.role:
        view["role"] = event.role
    if event.delegation_id:
        view["delegationId"] = event.delegation_id
    if event.attempt:
        view["attempt"] = event.attempt
    if event.delegation is not None:
        item = event.delegation
        view["delegation"] = {
            "delegationId": item.delegation_id,
            "parentRole": item.parent_role,
            "role": item.request.role,
            "task": item.request.task,
            "expectedOutput": item.request.expected_output,
            "status": item.status,
            "summary": item.summary,
            "errorType": item.error_type,
        }
    if event.result is not None:
        result = event.result
        result_view: dict[str, Any] = {
            "stopReason": result.stop_reason,
            "roleCalls": dict(result.role_calls),
            "delegations": [
                {
                    "delegationId": item.delegation_id,
                    "parentRole": item.parent_role,
                    "role": item.request.role,
                    "task": item.request.task,
                    "expectedOutput": item.request.expected_output,
                    "status": item.status,
                    "summary": item.summary,
                    "errorType": item.error_type,
                }
                for item in result.delegations
            ],
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "message": issue.message,
                    "evidence": issue.evidence,
                    "suggestion": issue.suggestion,
                    "owner": issue.owner,
                }
                for issue in (result.last_verdict.issues if result.last_verdict is not None else ())
            ],
        }
        if result.artifact is not None:
            result_view["artifact"] = {
                "content": result.artifact.content,
                "version": result.artifact.version,
                "evidenceRefs": list(result.artifact.evidence_refs),
                "assumptions": list(result.artifact.assumptions),
                "unresolvedQuestions": list(result.artifact.unresolved_questions),
            }
        view.update(result_view)
    return view


def _workflow_answer(result: WorkflowResult) -> str:
    """把最终 artifact 和未通过边界整理成持久化助手正文。"""
    if result.status == "passed" and result.artifact is not None:
        return result.artifact.content
    lines = ["[协作工作流未通过最终审核]"]
    if result.artifact is not None:
        lines.append(result.artifact.content)
    else:
        lines.append("工作流在形成可验证产物前停止。")
    if result.last_verdict is not None and result.last_verdict.summary:
        lines.append(f"审核摘要：{result.last_verdict.summary}")
    if result.last_verdict is not None and result.last_verdict.issues:
        lines.append("待处理问题：")
        lines.extend(
            f"- [{issue.owner}] {issue.message}"
            for issue in result.last_verdict.issues
        )
    return "\n\n".join(lines)


class ChatService:
    """单进程、单用户的 Turn 外壳；命令和会话都在此处顺序执行。"""

    def __init__(
        self,
        *,
        transcript_root: str | Path | None = None,
        llm=None,
        settings: AgentSettings | None = None,
        graph=None,
        workflow: WorkflowCoordinator | None = None,
    ):
        root = Path(transcript_root or PROJECT_ROOT / "data" / "history")
        self.transcripts = TranscriptStore(root)
        self.messages = self.transcripts.load_current()
        self.session_id = self.transcripts.current_session_id()
        self.transcripts.set_current_session_id(self.session_id)
        self.settings = settings or self._default_settings()
        self.llm = llm
        self.thinking_level = "off"
        self._interrupt_requested = threading.Event()
        self._operation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._turn_active = False
        self._workflow = workflow
        if graph is not None:
            self.graph = graph
        else:
            self.llm = llm or create_llm()
            self.graph = GraphFactory(self.llm, self.settings).build()

    def _collaborative_workflow(self) -> WorkflowCoordinator:
        """按需构造真实角色图，避免普通聊天和 fake 测试提前绑定全部工具。"""
        if self._workflow is None:
            if self.llm is None:
                raise RuntimeError("当前 ChatService 没有可用于协作角色的 LLM")
            self._workflow = build_collaborative_workflow(self.llm)
        return self._workflow

    def _default_settings(self) -> AgentSettings:
        """加载本地 skills 和长期记忆，形成单一 AgentSettings。"""
        skills, _diagnostics = load_skills(PROJECT_ROOT / "skills")
        memory = Memory(PROJECT_ROOT / "memory" / "MEMORY.md")
        return AgentSettings(
            skills=tuple(skills),
            memory=memory,
            history_search=self.transcripts.search_archives,
        )

    def get_history(self) -> list[MessageView]:
        """返回当前完整 transcript 的浏览器投影。"""
        return _project(self.messages)

    def _acquire_operation(self, *, turn: bool = False) -> None:
        """非阻塞取得唯一写操作资格。"""
        if not self._operation_lock.acquire(blocking=False):
            raise TurnBusyError("当前已有回合正在执行，请先等待完成或使用 /interrupt")
        if turn:
            with self._state_lock:
                self._turn_active = True

    def _release_operation(self, *, turn: bool = False) -> None:
        """释放唯一写操作资格。"""
        if turn:
            with self._state_lock:
                self._turn_active = False
        self._operation_lock.release()

    def _reset_turn_scoped_state(self) -> None:
        """在新会话边界清空护栏和压缩失败计数。"""
        self.settings.guardrails.reset_for_turn()
        if self.settings.context is not None:
            self.settings.context.breaker.record_success()

    @staticmethod
    def parse_command(text: str) -> tuple[str, str] | None:
        """解析 `/name args`，普通消息返回 None。"""
        value = (text or "").strip()
        if not value.startswith("/"):
            return None
        parts = value[1:].split(maxsplit=1)
        name = parts[0].casefold() if parts else ""
        return name, parts[1].strip() if len(parts) > 1 else ""

    @staticmethod
    def command_help() -> str:
        """返回当前 Web 支持的命令说明。"""
        return "\n".join([
            "可用命令:",
            "  /help                 显示本帮助",
            "  /tools                列出可用工具",
            "  /thinking             查看当前固定的 non-thinking 模式",
            "  /compact              强制压缩当前会话历史",
            "  /skill <技能名>       注入技能全文",
            "  /clear                清空当前会话",
            "  /new                  新建并切换会话",
            "  /sessions             列出会话",
            "  /switch <会话 id>     切换会话",
            "  /search <关键词>      搜索会话",
            "  /fork [消息序号]      分叉并切换会话",
            "  /stats                查看会话用量",
            "  /retry                重跑最后一轮",
            "  /redirect <文本>      用修正文本重跑最后一轮",
            "  /workflow <任务>      运行自主 Producer/Reviewer 与可选子 Agent 协作",
            "  /interrupt             在下一个图步骤边界停止",
        ])

    def _tool_help(self) -> str:
        """列出基础工具和按配置注入的技能/记忆工具。"""
        tools = list(self.settings.tools)
        lines = [f"  {tool.name}: {(tool.description or '').splitlines()[0]}" for tool in tools]
        if self.settings.skills:
            lines.extend([
                "  skills_list: 列出可用技能索引",
                "  skill_view: 按名称加载技能全文",
            ])
        if self.settings.memory is not None:
            lines.extend([
                "  session_search: 搜索归档会话",
                "  remember: 写入长期记忆",
            ])
        return "可用工具:\n" + ("\n".join(lines) if lines else "  （无）")

    @staticmethod
    def _session_dict(summary) -> dict:
        """把 SessionSummary 投影成 API 字典。"""
        return {
            "id": summary.id,
            "createdAt": summary.created_at,
            "updatedAt": summary.updated_at,
            "messageCount": summary.message_count,
            "title": summary.title,
            "active": summary.active,
        }

    def list_sessions(self) -> list[dict]:
        """列出当前和归档会话。"""
        return [self._session_dict(item) for item in self.transcripts.list_sessions()]

    def _active_session(self) -> dict:
        """返回当前会话摘要。"""
        return next(item for item in self.list_sessions() if item["active"])

    def _refreeze_memory(self) -> None:
        """在会话边界刷新长期记忆快照。"""
        if self.settings.memory is not None:
            self.settings.memory.refreeze()

    def new_session(self) -> dict:
        """归档当前会话并创建一个空的当前会话。"""
        self._acquire_operation()
        try:
            if self.messages:
                self.transcripts.save_archive(self.session_id, self.messages)
            self.session_id = self.transcripts.new_session_id()
            self.messages = []
            self.transcripts.save_current([])
            self.transcripts.set_current_session_id(self.session_id)
            self._reset_turn_scoped_state()
            self._refreeze_memory()
            return self._active_session()
        finally:
            self._release_operation()

    def _switch_session_unlocked(self, session_id: str) -> dict:
        """在已持有 single-flight 时切换会话。"""
        target = self.transcripts._validate_session_id(session_id)
        if target == self.session_id:
            return self._active_session()
        messages = self.transcripts.load_session(target)
        if self.messages:
            self.transcripts.save_archive(self.session_id, self.messages)
        self.session_id = target
        self.messages = list(messages)
        self.transcripts.save_current(self.messages)
        self.transcripts.set_current_session_id(self.session_id)
        self._reset_turn_scoped_state()
        self._refreeze_memory()
        return self._active_session()

    def switch_session(self, session_id: str) -> dict:
        """保存当前工作副本后切换到一个归档会话。"""
        self._acquire_operation()
        try:
            return self._switch_session_unlocked(session_id)
        finally:
            self._release_operation()

    def fork_session(self, message_index: int | None = None, *, source_session_id: str | None = None) -> dict:
        """复制当前会话全部历史或复制到指定消息序号之前并切换。"""
        self._acquire_operation()
        try:
            source_id = self.session_id
            if source_session_id is not None:
                source_id = self.transcripts._validate_session_id(source_session_id)
            source = (
                list(self.messages)
                if source_id == self.session_id
                else list(self.transcripts.load_session(source_id))
            )
            if message_index is None:
                selected = source
            else:
                if message_index < 0 or message_index >= len(source):
                    raise ValueError("消息序号越界")
                if getattr(source[message_index], "type", "") != "human":
                    raise ValueError("分叉位置必须是用户消息")
                selected = source[:message_index]
            if self.messages:
                self.transcripts.save_archive(self.session_id, self.messages)
            self.session_id = self.transcripts.new_session_id()
            self.messages = list(selected)
            self.transcripts.save_archive(self.session_id, self.messages)
            self.transcripts.save_current(self.messages)
            self.transcripts.set_current_session_id(self.session_id)
            self._reset_turn_scoped_state()
            self._refreeze_memory()
            return self._active_session()
        finally:
            self._release_operation()

    def search_sessions(self, query: str) -> list[dict]:
        """在当前和归档会话中按大小写不敏感子串搜索。"""
        needle = (query or "").strip().casefold()
        if not needle:
            return []
        hits: list[dict] = []
        for summary in self.transcripts.list_sessions():
            try:
                messages = self.transcripts.load_session(summary.id)
            except (TranscriptFileError, FileNotFoundError):
                continue
            for index, message in enumerate(messages):
                content = str(message.content)
                if needle in content.casefold():
                    hits.append({"sessionId": summary.id, "index": index, "content": content})
        return hits

    def session_stats(self) -> list[dict]:
        """计算每个会话的消息数、估算 token 和 usage 汇总。"""
        result: list[dict] = []
        for summary in self.transcripts.list_sessions():
            try:
                messages = self.transcripts.load_session(summary.id)
            except (TranscriptFileError, FileNotFoundError):
                continue
            input_tokens = 0
            output_tokens = 0
            cache_read_tokens = 0
            for message in messages:
                usage = getattr(message, "usage_metadata", {}) or {}
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
                cache_read_tokens += int(usage.get("cache_read_input_tokens", 0) or 0)
            result.append({
                "sessionId": summary.id,
                "messageCount": len(messages),
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "cacheReadTokens": cache_read_tokens,
                "estimatedTokens": estimate_messages_tokens(messages),
            })
        return result

    def send(self, text: str) -> list[MessageView]:
        """执行一个 Turn；成功后才保存完整用户/AI/工具消息。"""
        text = text.strip()
        if not text:
            raise ValueError("消息不能为空")
        self._acquire_operation(turn=True)
        try:
            self.settings.guardrails.reset_for_turn()
            user_message = HumanMessage(content=text, id=f"msg-{uuid4().hex}")
            result = self.graph.invoke(
                {"messages": [*self.messages, user_message], "thinking_level": self.thinking_level},
                config={"recursion_limit": GRAPH_RECURSION_LIMIT},
            )
            result_messages = _ensure_message_ids(result.get("messages", []))
            if not result_messages or not any(
                message.id == user_message.id
                or (isinstance(message, HumanMessage) and message.content == text)
                for message in result_messages
            ):
                raise RuntimeError("Agent 没有返回完整消息历史")
            self.transcripts.save_current(result_messages)
            self.messages = list(result_messages)
            return self.get_history()
        finally:
            self._release_operation(turn=True)

    def stream_send(
        self,
        text: str,
        *,
        base_messages: list[BaseMessage] | None = None,
        replace_before_start: bool = False,
    ) -> Iterator[ChatStreamEvent]:
        """按图节点产生聊天事件，并在成功后一次性提交 transcript。"""
        text = text.strip()
        if not text:
            yield {"type": "error", "detail": "消息不能为空", "messages": self.get_history()}
            return

        try:
            self._acquire_operation(turn=True)
        except TurnBusyError as exc:
            yield {"type": "error", "detail": str(exc), "messages": self.get_history()}
            return

        try:
            if self._interrupt_requested.is_set():
                self._interrupt_requested.clear()
            self.settings.guardrails.reset_for_turn()
            user_message = HumanMessage(content=text, id=f"msg-{uuid4().hex}")
            initial_messages = list(self.messages if base_messages is None else base_messages)
            working_messages: list[BaseMessage] = [*initial_messages, user_message]
            emitted: dict[str, MessageView] = {
                view["id"]: view for view in _project(initial_messages) if view.get("id")
            }

            if replace_before_start:
                yield {"type": "replace", "messages": _project(initial_messages)}

            user_view = _view_from_message(user_message)
            if user_view is not None:
                emitted[user_view["id"]] = user_view
                yield {"type": "message", "message": user_view}

            for update in self.graph.stream(
                {"messages": working_messages, "thinking_level": self.thinking_level},
                config={"recursion_limit": GRAPH_RECURSION_LIMIT},
                stream_mode="updates",
            ):
                if self._interrupt_requested.is_set():
                    self._interrupt_requested.clear()
                    yield {
                        "type": "interrupted",
                        "detail": "本轮已在图步骤边界停止，未提交半成品历史",
                        "messages": self.get_history(),
                    }
                    return
                if not isinstance(update, dict):
                    continue
                for node_update in update.values():
                    if not isinstance(node_update, dict):
                        continue
                    node_messages = node_update.get("messages")
                    if not isinstance(node_messages, list):
                        continue
                    node_messages = cast(list[BaseMessage], node_messages)
                    # LangGraph 的 add_messages 类型签名还允许多种消息表示；此处运行时已确认是 BaseMessage。
                    working_messages = cast(
                        list[BaseMessage],
                        add_messages(cast(Any, working_messages), cast(Any, node_messages)),
                    )
                    working_messages = _ensure_message_ids(working_messages)
                    for node_message in node_messages:
                        view = _view_from_message(node_message)
                        if (
                            view is not None
                            and isinstance(node_message, AIMessage)
                            and node_message.tool_calls
                            and not _content_text(node_message.content)
                        ):
                            running_view: MessageView = {**view, "toolStatus": "running"}
                            view = running_view
                        if view is None:
                            continue
                        message_id = view.get("id")
                        if not message_id or emitted.get(message_id) == view:
                            continue
                        emitted[message_id] = view
                        yield {"type": "message", "message": view}

            result_messages = _ensure_message_ids(working_messages)
            if not result_messages or not any(
                message.id == user_message.id
                or (isinstance(message, HumanMessage) and message.content == text)
                for message in result_messages
            ):
                raise RuntimeError("Agent 没有返回完整消息历史")
            self.transcripts.save_current(result_messages)
            self.messages = list(result_messages)
            yield {"type": "done", "messages": self.get_history(), "sessions": self.list_sessions()}
        except Exception:
            # SSE 已经开始后不能再改变 HTTP 状态码，只发送不泄露内部细节的错误事件。
            yield {
                "type": "error",
                "detail": "Agent 调用失败，请查看本地服务状态",
                "messages": self.get_history(),
            }
        finally:
            self._release_operation(turn=True)

    def stream_workflow(self, task: str) -> Iterator[ChatStreamEvent]:
        """运行一次有界协作工作流，并只把最终用户/助手消息写入 transcript。"""
        task = task.strip()
        if not task:
            yield {
                "type": "command",
                "command": "workflow",
                "detail": "用法: /workflow <任务>",
                "messages": self.get_history(),
            }
            yield {"type": "done", "messages": self.get_history()}
            return
        try:
            self._acquire_operation(turn=True)
        except TurnBusyError as exc:
            yield {"type": "error", "detail": str(exc), "messages": self.get_history()}
            return

        try:
            if self._interrupt_requested.is_set():
                self._interrupt_requested.clear()
            yield {
                "type": "command",
                "command": "workflow",
                "detail": "有界协作工作流已启动",
                "messages": self.get_history(),
            }
            user_message = HumanMessage(
                content=task,
                id=f"msg-{uuid4().hex}",
                additional_kwargs={"x_agent_kind": "workflow"},
            )
            user_view = _view_from_message(user_message)
            if user_view is not None:
                yield {"type": "message", "message": user_view}

            final_result: WorkflowResult | None = None
            final_view: dict | None = None
            for event in self._collaborative_workflow().stream(
                task,
                cancel_requested=self._interrupt_requested.is_set,
            ):
                if self._interrupt_requested.is_set():
                    self._interrupt_requested.clear()
                    yield {
                        "type": "interrupted",
                        "detail": "协作工作流已在角色调用边界停止，未提交半成品历史",
                        "messages": self.get_history(),
                    }
                    return
                final_view = _workflow_event_view(event)
                yield {"type": "workflow", "workflow": final_view}
                if event.result is not None:
                    final_result = event.result
            if final_result is None or final_view is None:
                raise RuntimeError("协作工作流没有返回最终结果")

            assistant_message = AIMessage(
                content=_workflow_answer(final_result),
                id=f"msg-{uuid4().hex}",
                additional_kwargs={
                    "x_agent_kind": "workflow",
                    "workflow_status": final_result.status,
                },
            )
            committed = _ensure_message_ids([*self.messages, user_message, assistant_message])
            self.transcripts.save_current(committed)
            self.messages = committed
            assistant_view = _view_from_message(assistant_message)
            if assistant_view is not None:
                yield {"type": "message", "message": assistant_view}
            yield {
                "type": "done",
                "messages": self.get_history(),
                "sessions": self.list_sessions(),
                "workflow": final_view,
            }
        except StructuredRoleOutputError as exc:
            logger.warning("workflow structured output rejected role=%s", exc.role or "unknown")
            yield {
                "type": "error",
                "detail": str(exc),
                "messages": self.get_history(),
            }
        except RoleInvocationError as exc:
            logger.error("workflow role invocation failed role=%s error_type=%s", exc.role, exc.error_type)
            yield {
                "type": "error",
                "detail": f"角色 {exc.role} 调用失败（{exc.error_type}），请检查模型或工具服务",
                "messages": self.get_history(),
            }
        except (TypeError, ValueError) as exc:
            logger.warning("workflow validation failed error_type=%s", type(exc).__name__)
            yield {
                "type": "error",
                "detail": f"协作工作流数据校验失败：{exc}",
                "messages": self.get_history(),
            }
        except Exception as exc:
            logger.error("workflow execution failed error_type=%s", type(exc).__name__)
            yield {
                "type": "error",
                "detail": "协作工作流执行失败，请检查角色结构化输出或本地服务状态",
                "messages": self.get_history(),
            }
        finally:
            self._release_operation(turn=True)

    def reset(self) -> list[MessageView]:
        """清空当前 transcript；不自动归档。"""
        self._acquire_operation()
        try:
            self.transcripts.clear_current()
            self.transcripts.delete_archive(self.session_id)
            self.messages = []
            self._reset_turn_scoped_state()
            return []
        finally:
            self._release_operation()

    def archive(self) -> str | None:
        """归档当前会话内容，但保留当前会话 id。"""
        self._acquire_operation()
        try:
            if not self.messages:
                self.transcripts.clear_current()
                return None
            self.transcripts.save_archive(self.session_id, self.messages)
            self.transcripts.clear_current()
            self.messages = []
            return self.session_id
        finally:
            self._release_operation()

    def compact(self) -> list[MessageView]:
        """强制压缩当前历史并保存结果。"""
        self._acquire_operation()
        try:
            if self.settings.context is None:
                raise RuntimeError("当前 Agent 未启用上下文压缩")
            if self.llm is None:
                raise RuntimeError("当前图没有可用于摘要的模型")
            new_messages, _updates, info = transform_context(
                self.messages,
                {"messages": self.messages, "compact_needed": True},
                self.settings.context,
                self.llm,
                force=True,
            )
            if info is None:
                return self.get_history()
            self.messages = list(new_messages)
            self.transcripts.save_current(self.messages)
            return self.get_history()
        finally:
            self._release_operation()

    def request_interrupt(self) -> bool:
        """请求当前图在下一个更新边界停止。"""
        with self._state_lock:
            if not self._turn_active:
                return False
        self._interrupt_requested.set()
        return True

    def _last_retry(self) -> tuple[list[BaseMessage], str] | None:
        """返回最后一条用户消息之前的历史和该消息文本。"""
        for index in range(len(self.messages) - 1, -1, -1):
            message = self.messages[index]
            if isinstance(message, HumanMessage):
                content = str(message.content)
                if message.additional_kwargs.get("x_agent_kind") == "skill" or content.startswith("## Historical Summary"):
                    continue
                return list(self.messages[:index]), content
        return None

    def stream_command(self, text: str) -> Iterator[ChatStreamEvent]:
        """执行一条斜杠命令，并统一用 SSE 返回命令结果。"""
        parsed = self.parse_command(text)
        if parsed is None or not parsed[0]:
            yield {"type": "error", "detail": "命令必须以 / 开头", "messages": self.get_history()}
            return
        name, arg = parsed

        def done(
            detail: str,
            *,
            sessions: list[dict] | None = None,
            session: dict | None = None,
            hits: list[dict] | None = None,
            stats: list[dict] | None = None,
        ) -> Iterator[ChatStreamEvent]:
            command_event: ChatStreamEvent = {
                "type": "command",
                "command": name,
                "detail": detail,
                "messages": self.get_history(),
            }
            if sessions is not None:
                command_event["sessions"] = sessions
            if session is not None:
                command_event["session"] = session
            if hits is not None:
                command_event["hits"] = hits
            if stats is not None:
                command_event["stats"] = stats
            yield command_event
            done_event: ChatStreamEvent = {"type": "done", "messages": self.get_history()}
            if sessions is not None:
                done_event["sessions"] = sessions
            if session is not None:
                done_event["session"] = session
            if hits is not None:
                done_event["hits"] = hits
            if stats is not None:
                done_event["stats"] = stats
            yield done_event

        if name == "help":
            yield from done(self.command_help())
        elif name == "tools":
            yield from done(self._tool_help())
        elif name == "thinking":
            if arg and arg not in THINKING_LEVELS:
                yield from done("Phase 13.1 工具图固定关闭思考模式；当前只支持 off")
            else:
                yield from done("当前思考模式: off（显式 non-thinking）")
        elif name == "compact":
            try:
                self.compact()
                yield from done("当前会话已压缩")
            except Exception as exc:
                yield from done(f"压缩失败: {exc}")
        elif name == "skill":
            skill = next((item for item in self.settings.skills if item.name == arg), None)
            if skill is None:
                available = "、".join(item.name for item in self.settings.skills) or "（无）"
                yield from done(f"未找到技能 {arg or '（空）'}，可用: {available}")
            else:
                try:
                    self._acquire_operation()
                    try:
                        self.messages.append(HumanMessage(
                            content=format_skill_content(skill),
                            id=f"msg-{uuid4().hex}",
                            additional_kwargs={"x_agent_kind": "skill"},
                        ))
                        self.transcripts.save_current(self.messages)
                    finally:
                        self._release_operation()
                    yield from done(f"技能 {arg} 已注入，请继续提问")
                except TurnBusyError as exc:
                    yield from done(str(exc))
        elif name == "clear":
            try:
                self.reset()
                yield from done("当前会话已清空", sessions=self.list_sessions())
            except TurnBusyError as exc:
                yield from done(str(exc))
        elif name == "new":
            try:
                session = self.new_session()
                yield from done(f"已切换到新会话 {session['id']}", session=session, sessions=self.list_sessions())
            except TurnBusyError as exc:
                yield from done(str(exc), sessions=self.list_sessions())
        elif name == "sessions":
            yield from done("会话列表已更新", sessions=self.list_sessions())
        elif name == "switch":
            try:
                session = self.switch_session(arg)
                yield from done(f"已切换到会话 {session['id']}", session=session, sessions=self.list_sessions())
            except (ValueError, FileNotFoundError, TranscriptFileError, TurnBusyError) as exc:
                yield from done(f"切换失败: {exc}", sessions=self.list_sessions())
        elif name == "search":
            hits = self.search_sessions(arg)
            yield from done(f"找到 {len(hits)} 条结果", hits=hits)
        elif name == "fork":
            try:
                index = None if not arg else int(arg)
                session = self.fork_session(index)
                yield from done(f"已分叉到会话 {session['id']}", session=session, sessions=self.list_sessions())
            except (ValueError, TypeError, TurnBusyError) as exc:
                yield from done(f"分叉失败: {exc}", sessions=self.list_sessions())
        elif name == "stats":
            stats = self.session_stats()
            yield from done("会话统计已生成", stats=stats)
        elif name == "workflow":
            yield from self.stream_workflow(arg)
        elif name in {"retry", "redirect"}:
            retry = self._last_retry()
            if retry is None:
                yield from done("当前会话没有可重跑的用户消息")
                return
            base, previous_text = retry
            if name == "redirect":
                if not arg:
                    yield from done("用法: /redirect <修正文本>")
                    return
                previous_text = arg
            yield {
                "type": "command",
                "command": name,
                "detail": f"正在{'重跑' if name == 'retry' else '重建'}最后一轮",
                "messages": self.get_history(),
            }
            yield from self.stream_send(previous_text, base_messages=base, replace_before_start=True)
        elif name == "interrupt":
            requested = self.request_interrupt()
            yield from done("已请求在下一个图步骤边界停止" if requested else "当前没有正在运行的回合")
        else:
            yield from done(f"未知命令 /{name}，输入 /help 查看")
