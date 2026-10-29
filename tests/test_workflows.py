import pytest

from agent.workflows import ReviewIssue, ReviewVerdict, RubricItem


def test_rubric_requires_nonempty_fields():
    assert RubricItem("coverage", "覆盖要求").required is True
    with pytest.raises(ValueError, match="key"):
        RubricItem("", "要求")
    with pytest.raises(ValueError, match="requirement"):
        RubricItem("coverage", "")


def test_review_issue_only_routes_to_producer_or_user():
    assert ReviewIssue("warning", "调整结构").owner == "producer"
    assert ReviewIssue("blocking", "需要授权", owner="user").owner == "user"
    with pytest.raises(ValueError, match="owner"):
        ReviewIssue("blocking", "补充证据", owner="subagent:repo", category="evidence")


def test_review_verdict_rejects_contradictory_status_and_issues():
    producer_issue = ReviewIssue("blocking", "仍有问题")
    user_issue = ReviewIssue("blocking", "需要确认", owner="user")
    with pytest.raises(ValueError, match="pass verdict"):
        ReviewVerdict(status="pass", issues=(producer_issue,))
    with pytest.raises(ValueError, match="至少需要"):
        ReviewVerdict(status="revise")
    with pytest.raises(ValueError, match="producer"):
        ReviewVerdict.revise([user_issue])
    with pytest.raises(ValueError, match="user"):
        ReviewVerdict.needs_user([producer_issue])


def test_review_verdict_factories_preserve_explicit_routes():
    revise = ReviewVerdict.revise([
        ReviewIssue("blocking", "补充结论", owner="producer"),
    ], "返工")
    needs_user = ReviewVerdict.needs_user([
        ReviewIssue("blocking", "需要授权", owner="user"),
    ], "等待用户")

    assert revise.status == "revise"
    assert revise.issues[0].owner == "producer"
    assert needs_user.status == "needs_user"
    assert needs_user.issues[0].owner == "user"
    assert ReviewVerdict.passed("通过").issues == ()
