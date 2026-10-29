"""Agent 图节点：Prompt Assembly、模型调用和顺序 Tool Pipeline。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.context import classify_overflow_error, transform_context
from agent.guardrails import ToolCallGuardrailController
from agent.hooks import ToolHooks
from agent.run_control import RunBudgetExceeded, RunCancelled, RunController
from agent.state import AgentState

_TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."


def _build_invoke_messages(messages, system_prompt_builder, tools, state, status_builder) -> list:
    """组装系统提示词、完整历史和末尾状态栏；动态段不写回历史。"""
    result = list(messages)
    if system_prompt_builder is not None:
        result.insert(0, SystemMessage(content=system_prompt_builder(tools)))
    if status_builder is not None:
        result.append(HumanMessage(content=f"<agent_status>\n{status_builder(state)}\n</agent_status>"))
    return result


def call_model(
    state: AgentState,
    llm,
    tools=None,
    system_prompt_builder=None,
    context=None,
    summary_llm=None,
    status_builder=None,
    run_control: RunController | None = None,
    config: RunnableConfig | None = None,
) -> dict:
    """执行一次模型 Step，并在输入超限时压缩后重试一次。"""
    del config
    messages = list(state.get("messages", []))
    updates: dict = {}
    if context is not None:
        messages, updates, _ = transform_context(messages, state, context, summary_llm or llm)
    invoke_messages = _build_invoke_messages(messages, system_prompt_builder, tools or [], state, status_builder)
    try:
        if run_control is not None:
            max_tokens = run_control.output_token_limit()
            run_control.before_model_request()
            response = llm.invoke(
                invoke_messages,
                max_tokens=max_tokens,
            )
            run_control.record_model_response(response)
        else:
            response = llm.invoke(invoke_messages)
    except Exception as exc:
        if context is None or not classify_overflow_error(exc):
            raise
        messages, retry_updates, info = transform_context(state.get("messages", []), state, context, summary_llm or llm, force=True)
        if info is None or info.get("ratio", 0) < 0.05:
            raise
        retry_messages = _build_invoke_messages(
            messages, system_prompt_builder, tools or [], state, status_builder
        )
        if run_control is not None:
            max_tokens = run_control.output_token_limit()
            run_control.before_model_request()
            response = llm.invoke(
                retry_messages,
                max_tokens=max_tokens,
            )
            run_control.record_model_response(response)
        else:
            response = llm.invoke(retry_messages)
        updates = retry_updates
    return {"messages": [*updates.get("messages", []), response], **{key: value for key, value in updates.items() if key != "messages"}}


def route(state: AgentState) -> str:
    """带工具调用时进入工具节点，否则结束本轮。"""
    last_message = state["messages"][-1]
    return "continue" if isinstance(last_message, AIMessage) and last_message.tool_calls else "end"


def route_after_tools(state: AgentState) -> str:
    """工具批次未终止时回到模型，否则结束本轮。"""
    return "end" if state.get("tool_terminate") else "continue"


def _structured_result_is_error(content: str) -> bool:
    """识别工具返回的 `success=false` 结构化失败。"""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("success") is False


def execute_tools(
    state: AgentState,
    config: RunnableConfig | None = None,
    tools=None,
    hooks: ToolHooks | None = None,
    guardrails: ToolCallGuardrailController | None = None,
    run_control: RunController | None = None,
) -> dict:
    """按 unknown → before → guardrail → execute → after 顺序执行工具。"""
    configurable = (config or {}).get("configurable", {})
    hooks = hooks or configurable.get("hooks") or ToolHooks()
    guardrails = guardrails or configurable.get("guardrails") or ToolCallGuardrailController()

    tool_map = {tool.name: tool for tool in (tools or [])}
    calls = list(state["messages"][-1].tool_calls)
    messages: list[ToolMessage] = []
    terminate_flags: list[bool] = []
    for call in calls:
        name, call_id = call["name"], call["id"]
        if guardrails.halt_decision is not None:
            messages.append(ToolMessage(
                guardrails.halt_decision.message,
                name=name,
                tool_call_id=call_id,
                status="error",
            ))
            terminate_flags.append(True)
            continue
        tool = tool_map.get(name)
        if tool is None:
            content = f"Error: {name} is not a valid tool."
            decision = guardrails.after_call(name, call.get("args") or {}, content, True)
            messages.append(ToolMessage(
                content,
                name=name,
                tool_call_id=call_id,
                status="error",
            ))
            terminate_flags.append(decision.action == "halt")
            continue
        decision = guardrails.before_call(name, call.get("args") or {})
        if not decision.allows_execution:
            messages.append(ToolMessage(decision.message, name=name, tool_call_id=call_id, status="error"))
            terminate_flags.append(decision.action in {"block", "halt"})
            continue
        before = hooks.before_tool_call(name, call.get("args") or {})
        if not before.allow:
            content = before.reason or "工具调用被拦截"
            decision = guardrails.after_call(name, call.get("args") or {}, content, True)
            messages.append(ToolMessage(content, name=name, tool_call_id=call_id, status="error"))
            terminate_flags.append(decision.action == "halt")
            continue
        if run_control is not None:
            try:
                run_control.before_tool_call()
            except (RunBudgetExceeded, RunCancelled) as exc:
                messages.append(ToolMessage(
                    f"Error: workflow stopped before tool execution ({exc.reason}).",
                    name=name,
                    tool_call_id=call_id,
                    status="error",
                ))
                terminate_flags.append(True)
                continue
        try:
            response = tool.invoke(call)
            if isinstance(response, ToolMessage):
                content = str(response.content)
                is_error = response.status == "error" or _structured_result_is_error(content)
            else:
                content = str(response)
                is_error = _structured_result_is_error(content)
        except Exception as exc:
            content, is_error = _TOOL_CALL_ERROR_TEMPLATE.format(error=repr(exc)), True
        decision = guardrails.after_call(name, call.get("args") or {}, content, is_error)
        if decision.action == "warn":
            content += f"\n[护栏提示] {decision.message}"
        elif decision.action == "halt":
            content += f"\n[护栏停止] {decision.message}"
        after = hooks.after_tool_call(name, call.get("args") or {}, content, is_error)
        if after.content is not None:
            content = after.content
        if after.is_error is not None:
            is_error = after.is_error
        messages.append(ToolMessage(content, name=name, tool_call_id=call_id, status="error" if is_error else "success"))
        terminate_flags.append(bool(after.terminate) or decision.action == "halt")
    terminate = bool(calls) and (guardrails.halt_decision is not None or any(terminate_flags))
    return {"messages": messages, "tool_terminate": terminate}
