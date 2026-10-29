"""自主协作工作流共享的 Rubric 与审核结果类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


VerdictStatus = Literal["pass", "revise", "needs_user"]


@dataclass(frozen=True)
class RubricItem:
    """描述一条 Reviewer 必须检查的要求。"""

    key: str
    requirement: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("rubric key 不能为空")
        if not self.requirement.strip():
            raise ValueError("rubric requirement 不能为空")


@dataclass(frozen=True)
class ReviewIssue:
    """描述 Reviewer 发现的问题、证据和唯一合法接收者。"""

    severity: str
    message: str
    evidence: str = ""
    suggestion: str = ""
    category: str = "quality"
    owner: str = "producer"

    def __post_init__(self) -> None:
        if self.severity not in {"blocking", "warning"}:
            raise ValueError(f"不支持的 review issue severity: {self.severity}")
        if not self.message.strip():
            raise ValueError("review issue message 不能为空")
        if self.category not in {
            "quality", "coverage", "evidence", "consistency", "format", "safety",
        }:
            raise ValueError(f"不支持的 review issue category: {self.category}")
        if self.owner not in {"producer", "user"}:
            raise ValueError(f"不支持的 review issue owner: {self.owner}")


@dataclass(frozen=True)
class ReviewVerdict:
    """表示 Reviewer 通过、要求 Producer 修订或需要用户输入。"""

    status: VerdictStatus
    issues: tuple[ReviewIssue, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"pass", "revise", "needs_user"}:
            raise ValueError(f"不支持的 review verdict: {self.status!r}")
        object.__setattr__(self, "issues", tuple(self.issues))
        if any(not isinstance(issue, ReviewIssue) for issue in self.issues):
            raise TypeError("review issues 必须全部是 ReviewIssue")
        if self.status == "pass" and self.issues:
            raise ValueError("pass verdict 不能同时包含 issues")
        if self.status in {"revise", "needs_user"} and not self.issues:
            raise ValueError(f"{self.status} verdict 至少需要一个 issue")
        if self.status == "revise" and any(issue.owner != "producer" for issue in self.issues):
            raise ValueError("revise verdict 的 issue owner 必须是 producer")
        if self.status == "needs_user" and any(issue.owner != "user" for issue in self.issues):
            raise ValueError("needs_user verdict 的 issue owner 必须是 user")

    @classmethod
    def passed(cls, summary: str = "") -> "ReviewVerdict":
        """构造通过 verdict。"""
        return cls(status="pass", summary=summary)

    @classmethod
    def revise(
        cls,
        issues: tuple[ReviewIssue, ...] | list[ReviewIssue],
        summary: str = "",
    ) -> "ReviewVerdict":
        """构造返回 Producer 的修订 verdict。"""
        return cls(status="revise", issues=tuple(issues), summary=summary)

    @classmethod
    def needs_user(
        cls,
        issues: tuple[ReviewIssue, ...] | list[ReviewIssue],
        summary: str = "",
    ) -> "ReviewVerdict":
        """构造需要用户信息或授权的 verdict。"""
        return cls(status="needs_user", issues=tuple(issues), summary=summary)


__all__ = ["ReviewIssue", "ReviewVerdict", "RubricItem"]
