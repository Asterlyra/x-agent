import pytest
from langchain_core.messages import AIMessage

from agent.collaboration import (
    ArtifactEnvelope,
    EvidenceRef,
    ProductionRequest,
    ReviewRequest,
    RunBudget,
    WorkflowCoordinator,
)
from agent.role_runner import build_collaborative_workflow, build_workflow
from agent.run_control import RunBudgetExceeded, RunController
from agent.workflows import ReviewIssue, ReviewVerdict, RubricItem
from conftest import FakeMessagesListChatModelWithTools


def _artifact(content: str, version: int = 1) -> ArtifactEnvelope:
    return ArtifactEnvelope(content, version=version)


def test_raw_task_is_preserved_and_zero_delegation_runs_both_roles():
    seen: dict[str, object] = {}

    def producer(package: ProductionRequest, control: RunController) -> ArtifactEnvelope:
        del control
        seen["producer_task"] = package.task
        return _artifact("直接完成的报告", package.attempt)

    def reviewer(package: ReviewRequest, control: RunController) -> ReviewVerdict:
        del control
        seen["reviewer_task"] = package.task
        return ReviewVerdict.passed("通过")

    events = list(WorkflowCoordinator(producer, reviewer).stream("  保留原始空格的任务  "))
    result = events[-1].result

    assert [event.phase for event in events] == [
        "preparing", "producing", "producing", "reviewing", "completed",
    ]
    assert result is not None and result.status == "passed"
    assert seen["producer_task"].raw_request == "  保留原始空格的任务  "
    assert seen["reviewer_task"] == seen["producer_task"]
    assert result.usage.role_invocations == 2
    assert result.delegations == ()


def test_reviewer_feedback_returns_only_to_producer():
    packages: list[ProductionRequest] = []

    def producer(package: ProductionRequest, control: RunController) -> ArtifactEnvelope:
        del control
        packages.append(package)
        return _artifact("初稿" if package.attempt == 1 else "终稿", package.attempt)

    def reviewer(package: ReviewRequest, control: RunController) -> ReviewVerdict:
        del control
        if package.attempt == 1:
            return ReviewVerdict.revise([
                ReviewIssue("blocking", "缺少结论", category="coverage", owner="producer"),
            ])
        return ReviewVerdict.passed()

    result = WorkflowCoordinator(producer, reviewer).run("写报告")

    assert result.status == "passed"
    assert result.attempts == 2
    assert packages[1].review_issues[0].owner == "producer"
    assert result.artifact is not None and result.artifact.version == 2


def test_needs_user_stops_without_extra_producer_round():
    producer_calls = 0

    def producer(package: ProductionRequest, control: RunController) -> ArtifactEnvelope:
        nonlocal producer_calls
        del package, control
        producer_calls += 1
        return _artifact("需要授权的草稿")

    def reviewer(package: ReviewRequest, control: RunController) -> ReviewVerdict:
        del package, control
        return ReviewVerdict.needs_user([
            ReviewIssue("blocking", "需要用户授权", category="safety", owner="user"),
        ])

    result = WorkflowCoordinator(producer, reviewer).run("处理受限资料")

    assert result.status == "not_passed"
    assert result.stop_reason == "needs_user_input"
    assert producer_calls == 1


def test_role_budget_is_checked_before_next_role():
    reviewer_called = False

    def reviewer(package: ReviewRequest, control: RunController) -> ReviewVerdict:
        nonlocal reviewer_called
        del package, control
        reviewer_called = True
        return ReviewVerdict.passed()

    result = WorkflowCoordinator(
        lambda package, control: _artifact("报告", package.attempt),
        reviewer,
        budget=RunBudget(role_invocations=1),
    ).run("任务")

    assert result.status == "not_passed"
    assert result.stop_reason == "role_invocation_limit"
    assert reviewer_called is False
    assert result.usage.role_invocations == 1


def test_tool_and_token_budgets_stop_before_the_next_step():
    control = RunController(RunBudget(
        tool_calls=1,
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=30,
    ))
    control.before_tool_call()
    with pytest.raises(RunBudgetExceeded, match="tool_call_limit"):
        control.before_tool_call()

    token_control = RunController(RunBudget(
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=30,
    ))
    token_control.before_model_request()
    token_control.record_model_response(AIMessage(
        content="完成",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
        },
    ))
    with pytest.raises(RunBudgetExceeded, match="input_token_limit"):
        token_control.checkpoint()
    assert token_control.usage().model_requests == 1
    assert token_control.usage().input_tokens == 10


def test_direct_tool_evidence_does_not_require_delegation_namespace():
    evidence = EvidenceRef("direct-1", "DESIGN.md", "960", origin="direct")
    result = WorkflowCoordinator(
        lambda package, control: ArtifactEnvelope("报告", evidence=(evidence,)),
        lambda package, control: ReviewVerdict.passed(),
    ).run("研究项目")

    assert result.status == "passed"
    assert result.artifact is not None
    assert result.artifact.evidence_refs == ("direct-1",)


def test_structured_output_is_repaired_in_same_role_context():
    llm = FakeMessagesListChatModelWithTools(responses=[
        AIMessage(content='{"content": 123, "evidence": []}'),
        AIMessage(content=(
            '{"content":"修复后的产物","evidence":[],"assumptions":[],'
            '"unresolved_questions":[]}'
        )),
    ])

    result = build_workflow(llm, with_reviewer=False, allow_delegation=False).run("写报告")

    assert result.status == "passed"
    assert result.artifact is not None and result.artifact.content == "修复后的产物"
    assert result.usage.role_invocations == 1
    assert result.usage.model_requests == 2


def test_producer_can_use_direct_tool_without_delegation():
    llm = FakeMessagesListChatModelWithTools(responses=[
        AIMessage(
            content="",
            tool_calls=[{
                "name": "calculate",
                "args": {"expression": "2+3"},
                "id": "calc-1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content=(
            '{"content":"结果是 5","evidence":[{"id":"calc","source":"calculate",'
            '"locator":"2+3","confidence":1.0}],"assumptions":[],'
            '"unresolved_questions":[]}'
        )),
    ])

    result = build_workflow(llm, with_reviewer=False, allow_delegation=True).run("计算 2+3")

    assert result.status == "passed"
    assert result.usage.model_requests == 2
    assert result.usage.tool_calls == 1
    assert result.usage.delegation_calls == 0


def test_producer_can_optionally_delegate_one_level_and_receive_result():
    llm = FakeMessagesListChatModelWithTools(responses=[
        AIMessage(
            content="",
            tool_calls=[{
                "name": "delegate_research",
                "args": {
                    "role": "repo_researcher",
                    "task": "读取真实入口调用链并给出行号",
                    "expected_output": "带路径和行号的摘要",
                },
                "id": "delegate-1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content=(
            '{"status":"completed","summary":"已读取入口","findings":["入口存在"],'
            '"evidence":[{"id":"e1","source":"backend/chat_service.py",'
            '"locator":"260","confidence":1.0}],"uncertainties":[]}'
        )),
        AIMessage(content=(
            '{"content":"带委派证据的报告","evidence":[{"id":"d1:e1",'
            '"source":"backend/chat_service.py","locator":"260","confidence":1.0}],'
            '"assumptions":[],"unresolved_questions":[]}'
        )),
        AIMessage(content='{"status":"pass","summary":"通过","issues":[]}'),
    ])

    result = build_collaborative_workflow(llm).run("研究当前项目")

    assert result.status == "passed"
    assert result.usage.role_invocations == 3
    assert result.usage.model_requests == 4
    assert result.usage.tool_calls == 1
    assert result.usage.delegation_calls == 1
    assert len(result.delegations) == 1
    assert result.delegations[0].parent_role == "producer"
    assert result.delegations[0].evidence[0].id == "d1:e1"
    assert result.artifact is not None
    assert result.artifact.evidence[0].origin == "delegated"


def test_reviewer_can_introduce_new_information_with_its_own_tool():
    llm = FakeMessagesListChatModelWithTools(responses=[
        AIMessage(content=(
            '{"content":"初稿","evidence":[],"assumptions":[],'
            '"unresolved_questions":[]}'
        )),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "calculate",
                "args": {"expression": "3*7"},
                "id": "review-calc",
                "type": "tool_call",
            }],
        ),
        AIMessage(content=(
            '{"status":"revise","summary":"独立计算发现遗漏","issues":['
            '{"severity":"blocking","category":"evidence","message":"应为 21",'
            '"evidence":"calculate: 3*7","suggestion":"修正数字",'
            '"owner":"producer"}]}'
        )),
        AIMessage(content=(
            '{"content":"修订稿：21","evidence":[{"id":"calc-21","source":"calculate",'
            '"locator":"3*7","confidence":1.0}],"assumptions":[],'
            '"unresolved_questions":[]}'
        )),
        AIMessage(content='{"status":"pass","summary":"通过","issues":[]}'),
    ])

    result = build_collaborative_workflow(llm).run("核对 3*7")

    assert result.status == "passed"
    assert result.attempts == 2
    assert result.review_trail[0].verdict.issues[0].evidence == "calculate: 3*7"
    assert result.usage.tool_calls == 1


def test_model_budget_stops_before_request_beyond_limit():
    llm = FakeMessagesListChatModelWithTools(responses=[
        AIMessage(content=(
            '{"content":"产物","evidence":[],"assumptions":[],'
            '"unresolved_questions":[]}'
        )),
    ])
    workflow = build_collaborative_workflow(llm)
    workflow.budget = RunBudget(model_requests=1)

    result = workflow.run("只允许一次模型请求")

    assert result.status == "not_passed"
    assert result.stop_reason == "model_request_limit"
    assert result.usage.model_requests == 1


def test_cancel_before_first_role_has_no_background_work():
    producer_called = False

    def producer(package: ProductionRequest, control: RunController) -> ArtifactEnvelope:
        nonlocal producer_called
        del package, control
        producer_called = True
        return _artifact("不应执行")

    events = list(WorkflowCoordinator(producer, None).stream(
        "取消任务",
        cancel_requested=lambda: True,
    ))

    assert events[-1].status == "cancelled"
    assert events[-1].result is not None
    assert events[-1].result.stop_reason == "cancelled"
    assert producer_called is False
