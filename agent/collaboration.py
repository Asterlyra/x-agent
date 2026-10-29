"""自主 Producer/Reviewer 与可选一级子 Agent 的确定性协调器。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from agent.run_control import (
    RunBudget,
    RunCancelled,
    RunController,
    RunStopped,
    RunUsage,
)
from agent.workflows import ReviewIssue, ReviewVerdict, RubricItem


SUBAGENT_ROLES = frozenset({
    "repo_researcher",
    "web_researcher",
    "fact_checker",
    "calculator",
    "document_analyst",
})
WorkflowStatus = Literal["passed", "not_passed"]
DelegationStatus = Literal["completed", "blocked", "failed"]


@dataclass(frozen=True)
class TaskInput:
    """原样保存用户任务和用户显式给出的审核标准。"""

    raw_request: str
    explicit_rubric: tuple[RubricItem, ...] = ()

    def __post_init__(self) -> None:
        if not self.raw_request.strip():
            raise ValueError("raw_request 不能为空")
        object.__setattr__(self, "explicit_rubric", tuple(self.explicit_rubric))
        if any(not isinstance(item, RubricItem) for item in self.explicit_rubric):
            raise TypeError("explicit_rubric 必须全部是 RubricItem")
        keys = [item.key for item in self.explicit_rubric]
        if len(keys) != len(set(keys)):
            raise ValueError("explicit_rubric key 不能重复")


@dataclass(frozen=True)
class EvidenceRef:
    """统一描述直接工具或子 Agent 返回的可追溯证据。"""

    id: str
    source: str
    locator: str = ""
    confidence: float = 1.0
    origin: str = "direct"

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.source.strip():
            raise ValueError("evidence id 和 source 不能为空")
        if not 0 <= self.confidence <= 1:
            raise ValueError("evidence confidence 必须在 0 到 1 之间")
        if self.origin not in {"direct", "delegated"}:
            raise ValueError("evidence origin 必须是 direct 或 delegated")


@dataclass(frozen=True)
class ArtifactEnvelope:
    """Producer 生成的文本产物、证据、假设和未决问题。"""

    content: str
    evidence: tuple[EvidenceRef, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("artifact content 不能为空")
        if self.version < 1:
            raise ValueError("artifact version 必须从 1 开始")
        for field_name in ("evidence", "assumptions", "unresolved_questions"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if any(not isinstance(item, EvidenceRef) for item in self.evidence):
            raise TypeError("artifact evidence 必须全部是 EvidenceRef")
        ids = [item.id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("artifact evidence id 不能重复")

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        """返回便于 UI 和旧结果读取的证据标识。"""
        return tuple(item.id for item in self.evidence)


@dataclass(frozen=True)
class DelegationRequest:
    """父 Agent 交给一级只读子 Agent 的局部、自包含任务。"""

    role: str
    task: str
    expected_output: str

    def __post_init__(self) -> None:
        if self.role not in SUBAGENT_ROLES:
            raise ValueError(f"不支持的子 Agent 角色: {self.role}")
        if not self.task.strip() or not self.expected_output.strip():
            raise ValueError("delegation task 和 expected_output 不能为空")


@dataclass(frozen=True)
class DelegationResult:
    """一级子 Agent 折叠回父 Agent 的结构化局部结果。"""

    delegation_id: str
    parent_role: str
    request: DelegationRequest
    status: DelegationStatus
    summary: str
    findings: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    uncertainties: tuple[str, ...] = ()
    error_type: str = ""

    def __post_init__(self) -> None:
        if not self.delegation_id.strip() or not self.parent_role.strip() or not self.summary.strip():
            raise ValueError("delegation id 和 summary 不能为空")
        if self.status not in {"completed", "blocked", "failed"}:
            raise ValueError(f"不支持的 delegation status: {self.status}")
        for field_name in ("findings", "evidence", "uncertainties"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if any(not isinstance(item, EvidenceRef) for item in self.evidence):
            raise TypeError("delegation evidence 必须全部是 EvidenceRef")


@dataclass(frozen=True)
class ProductionRequest:
    """自主 Producer 一轮调用可以看到的最小公开输入。"""

    task: TaskInput
    prior_artifact: ArtifactEnvelope | None = None
    review_issues: tuple[ReviewIssue, ...] = ()
    attempt: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_issues", tuple(self.review_issues))
        if self.attempt < 1:
            raise ValueError("production attempt 必须从 1 开始")


@dataclass(frozen=True)
class ReviewRequest:
    """自主 Reviewer 一轮调用可以看到的任务、产物和历史问题。"""

    task: TaskInput
    artifact: ArtifactEnvelope
    previous_issues: tuple[ReviewIssue, ...] = ()
    attempt: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "previous_issues", tuple(self.previous_issues))
        if self.attempt < 1:
            raise ValueError("review attempt 必须从 1 开始")


@dataclass(frozen=True)
class ReviewRecord:
    """保存一次生产、审核和当时已经完成的委派快照。"""

    attempt: int
    artifact: ArtifactEnvelope
    verdict: ReviewVerdict
    delegations: tuple[DelegationResult, ...] = ()


@dataclass(frozen=True)
class WorkflowResult:
    """表示自主协作工作流的最终产物、审核和停止原因。"""

    status: WorkflowStatus
    artifact: ArtifactEnvelope | None
    attempts: int
    usage: RunUsage
    role_calls: tuple[tuple[str, int], ...]
    delegations: tuple[DelegationResult, ...]
    review_trail: tuple[ReviewRecord, ...]
    last_verdict: ReviewVerdict | None
    stop_reason: str = ""


@dataclass(frozen=True)
class WorkflowEvent:
    """工作流在角色和委派边界返回的公开阶段事件。"""

    phase: str
    status: str
    detail: str
    role: str = ""
    delegation_id: str = ""
    attempt: int = 0
    usage: RunUsage = RunUsage()
    delegation: DelegationResult | None = None
    result: WorkflowResult | None = None


Producer = Callable[[ProductionRequest, RunController], ArtifactEnvelope]
Reviewer = Callable[[ReviewRequest, RunController], ReviewVerdict]
DelegationProvider = Callable[[RunController], tuple[DelegationResult, ...]]


def _default_rubric() -> tuple[RubricItem, ...]:
    """在用户没有显式 Rubric 时提供确定性的通用审核标准。"""
    return (
        RubricItem("coverage", "覆盖用户明确提出的任务和交付要求"),
        RubricItem("evidence", "关键事实有来源，推断和未知被明确标记"),
        RubricItem("consistency", "结论内部一致并披露未解决问题"),
    )


class WorkflowCoordinator:
    """只负责原始任务保真、角色顺序、预算、返工和停止。"""

    def __init__(
        self,
        producer: Producer,
        reviewer: Reviewer | None,
        *,
        budget: RunBudget | None = None,
        max_revisions: int = 2,
        delegation_provider: DelegationProvider | None = None,
    ):
        if max_revisions < 0:
            raise ValueError("max_revisions 不能小于 0")
        self._producer = producer
        self._reviewer = reviewer
        self.budget = budget or RunBudget()
        self.max_revisions = max_revisions
        self._delegation_provider = delegation_provider or (lambda control: ())

    def _delegations(self, control: RunController) -> tuple[DelegationResult, ...]:
        return deepcopy(tuple(self._delegation_provider(control)))

    @staticmethod
    def _new_delegation_events(
        delegations: tuple[DelegationResult, ...],
        emitted: int,
        *,
        attempt: int,
        usage: RunUsage,
    ) -> list[WorkflowEvent]:
        events: list[WorkflowEvent] = []
        for item in delegations[emitted:]:
            events.append(WorkflowEvent(
                "delegating",
                item.status,
                item.summary,
                role=item.request.role,
                delegation_id=item.delegation_id,
                attempt=attempt,
                usage=usage,
                delegation=item,
            ))
        return events

    def _result(
        self,
        status: WorkflowStatus,
        *,
        artifact: ArtifactEnvelope | None,
        control: RunController,
        trail: list[ReviewRecord],
        verdict: ReviewVerdict | None,
        stop_reason: str = "",
    ) -> WorkflowResult:
        return WorkflowResult(
            status=status,
            artifact=deepcopy(artifact),
            attempts=(artifact.version if artifact is not None else len(trail)),
            usage=control.usage(),
            role_calls=tuple(sorted(control.role_calls().items())),
            delegations=self._delegations(control),
            review_trail=deepcopy(tuple(trail)),
            last_verdict=deepcopy(verdict),
            stop_reason=stop_reason,
        )

    def stream(
        self,
        task: str,
        *,
        rubric: tuple[RubricItem, ...] | list[RubricItem] = (),
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Iterator[WorkflowEvent]:
        """执行自主协作，并在公开边界返回进度。"""
        explicit = tuple(rubric)
        task_input = TaskInput(task, explicit or _default_rubric())
        control = RunController(self.budget, cancel_requested=cancel_requested)
        artifact: ArtifactEnvelope | None = None
        review_issues: tuple[ReviewIssue, ...] = ()
        trail: list[ReviewRecord] = []
        last_verdict: ReviewVerdict | None = None
        emitted_delegations = 0

        yield WorkflowEvent(
            "preparing",
            "completed",
            "用户任务已按原文保存，确定性 Rubric 已就绪",
            usage=control.usage(),
        )

        try:
            for attempt in range(1, self.max_revisions + 2):
                control.checkpoint()
                yield WorkflowEvent(
                    "producing",
                    "running",
                    "Producer 正在自主研究并生成产物",
                    role="producer",
                    attempt=attempt,
                    usage=control.usage(),
                )
                control.before_role("producer")
                artifact = self._producer(ProductionRequest(
                    task=task_input,
                    prior_artifact=deepcopy(artifact),
                    review_issues=review_issues,
                    attempt=attempt,
                ), control)
                if not isinstance(artifact, ArtifactEnvelope):
                    raise TypeError("producer 必须返回 ArtifactEnvelope")
                if artifact.version != attempt:
                    raise ValueError("artifact version 必须等于当前 production attempt")
                delegations = self._delegations(control)
                for event in self._new_delegation_events(
                    delegations, emitted_delegations, attempt=attempt, usage=control.usage()
                ):
                    yield event
                emitted_delegations = len(delegations)
                control.checkpoint()
                yield WorkflowEvent(
                    "producing",
                    "completed",
                    f"Producer 已生成产物 v{artifact.version}",
                    role="producer",
                    attempt=attempt,
                    usage=control.usage(),
                )

                if self._reviewer is None:
                    result = self._result(
                        "passed",
                        artifact=artifact,
                        control=control,
                        trail=trail,
                        verdict=None,
                    )
                    yield WorkflowEvent(
                        "completed", "passed", "Producer 产物已完成",
                        attempt=attempt, usage=result.usage, result=result,
                    )
                    return

                yield WorkflowEvent(
                    "reviewing",
                    "running",
                    "Reviewer 正在独立研究并审核产物",
                    role="reviewer",
                    attempt=attempt,
                    usage=control.usage(),
                )
                control.before_role("reviewer")
                verdict = self._reviewer(ReviewRequest(
                    task=task_input,
                    artifact=deepcopy(artifact),
                    previous_issues=review_issues,
                    attempt=attempt,
                ), control)
                if not isinstance(verdict, ReviewVerdict):
                    raise TypeError("reviewer 必须返回 ReviewVerdict")
                last_verdict = verdict
                delegations = self._delegations(control)
                for event in self._new_delegation_events(
                    delegations, emitted_delegations, attempt=attempt, usage=control.usage()
                ):
                    yield event
                emitted_delegations = len(delegations)
                control.checkpoint()
                trail.append(ReviewRecord(
                    attempt=attempt,
                    artifact=deepcopy(artifact),
                    verdict=deepcopy(verdict),
                    delegations=delegations,
                ))

                if verdict.status == "pass":
                    result = self._result(
                        "passed", artifact=artifact, control=control,
                        trail=trail, verdict=verdict,
                    )
                    yield WorkflowEvent(
                        "completed", "passed", verdict.summary or "产物已通过独立审核",
                        attempt=attempt, usage=result.usage, result=result,
                    )
                    return
                if verdict.status == "needs_user":
                    result = self._result(
                        "not_passed", artifact=artifact, control=control,
                        trail=trail, verdict=verdict, stop_reason="needs_user_input",
                    )
                    yield WorkflowEvent(
                        "completed", "not_passed", "任务需要用户补充信息或授权",
                        attempt=attempt, usage=result.usage, result=result,
                    )
                    return
                review_issues = verdict.issues

            result = self._result(
                "not_passed", artifact=artifact, control=control,
                trail=trail, verdict=last_verdict, stop_reason="revision_limit",
            )
            yield WorkflowEvent(
                "completed", "not_passed", "已达到最大返工次数",
                attempt=len(trail), usage=result.usage, result=result,
            )
        except RunStopped as exc:
            result = self._result(
                "not_passed", artifact=artifact, control=control,
                trail=trail, verdict=last_verdict, stop_reason=exc.reason,
            )
            status = "cancelled" if isinstance(exc, RunCancelled) else "not_passed"
            detail = (
                "工作流已在步骤边界取消"
                if status == "cancelled"
                else f"工作流达到预算边界：{exc.reason}"
            )
            yield WorkflowEvent(
                "completed", status, detail,
                attempt=len(trail), usage=result.usage, result=result,
            )

    def run(
        self,
        task: str,
        *,
        rubric: tuple[RubricItem, ...] | list[RubricItem] = (),
    ) -> WorkflowResult:
        """执行完整工作流并返回最后一个公开结果。"""
        final: WorkflowResult | None = None
        for event in self.stream(task, rubric=rubric):
            if event.result is not None:
                final = event.result
        if final is None:
            raise RuntimeError("协作工作流没有返回结果")
        return final


__all__ = [
    "ArtifactEnvelope",
    "DelegationRequest",
    "DelegationResult",
    "EvidenceRef",
    "ProductionRequest",
    "ReviewRecord",
    "ReviewRequest",
    "RunBudget",
    "RunUsage",
    "SUBAGENT_ROLES",
    "TaskInput",
    "WorkflowEvent",
    "WorkflowCoordinator",
    "WorkflowResult",
]
