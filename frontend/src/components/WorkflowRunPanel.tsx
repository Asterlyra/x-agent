import { useEffect, useState } from "react"
import type { WorkflowView } from "../api"
import type { WorkflowActivity } from "../state/conversation-timeline"

type Props = { activity: WorkflowActivity }

const PHASE_LABELS: Record<string, string> = {
  preparing: "任务准备",
  delegating: "子任务执行",
  producing: "成果生成",
  reviewing: "独立审核",
  completed: "运行完成",
}

const STATUS_LABELS: Record<string, string> = {
  running: "运行中",
  completed: "已完成",
  passed: "已通过",
  not_passed: "未通过",
  cancelled: "已停止",
  error: "运行异常",
}

function waitingDetail(workflow: WorkflowView): string {
  if (workflow.role === "reviewer") return "Reviewer 正在执行模型或工具调用，等待下一个公开步骤边界"
  if (workflow.role === "producer") return "Producer 正在执行模型或工具调用，等待下一个公开步骤边界"
  if (workflow.phase === "delegating") return "正在等待一级子 Agent 返回结构化结果"
  return "工作流仍在运行，等待下一个公开步骤边界"
}

function formatDuration(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds))
  if (safeSeconds < 60) return `${safeSeconds} 秒`
  const minutes = Math.floor(safeSeconds / 60)
  return `${minutes} 分 ${safeSeconds % 60} 秒`
}

export function WorkflowRunPanel({ activity }: Props) {
  const workflow = activity.current
  const final = workflow.status === "passed"
    || workflow.status === "not_passed"
    || workflow.status === "cancelled"
    || workflow.status === "error"
  const [now, setNow] = useState(() => Date.now())
  const [activePanel, setActivePanel] = useState<"execution" | "result">(
    () => workflow.artifact ? "result" : "execution",
  )

  useEffect(() => {
    setNow(Date.now())
    if (final) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [activity.updatedAt, final])

  useEffect(() => {
    if (final && workflow.artifact) setActivePanel("result")
  }, [final, workflow.artifact])

  const serverElapsed = workflow.usage?.elapsedSeconds ?? 0
  const liveElapsed = final
    ? serverElapsed
    : Math.max(serverElapsed, (now - activity.startedAt) / 1000)
  const secondsSinceUpdate = Math.max(0, (now - activity.updatedAt) / 1000)
  const waitingTooLong = !final && secondsSinceUpdate >= 15

  return (
    <section className={`workflow-panel ${workflow.status}`} aria-label="协作工作流">
      <header className="workflow-header">
        <div>
          <div className="workflow-eyebrow">有界多 Agent</div>
          <strong>{PHASE_LABELS[workflow.phase] ?? workflow.phase}</strong>
        </div>
        <span className="workflow-status">
          {!final && <span className="workflow-status-dot" aria-hidden="true" />}
          {final ? (STATUS_LABELS[workflow.status] ?? workflow.status) : "运行中"}
        </span>
      </header>
      <p className="workflow-detail">{workflow.detail}</p>
      {!final && (
        <div className="workflow-live-status">
          <span className="workflow-live-pulse" aria-hidden="true" />
          <div>
            <strong>{waitingDetail(workflow)}</strong>
            <p>
              已运行 {formatDuration(liveElapsed)} · 最后收到服务端更新 {formatDuration(secondsSinceUpdate)}前
            </p>
          </div>
        </div>
      )}
      {waitingTooLong && (
        <p className="workflow-wait-hint">
          角色内部统计会在模型或工具步骤结束后刷新；如需终止，可点击页面顶部“停止”。
        </p>
      )}
      <div className="workflow-meta">
        {workflow.role && <span>角色：{workflow.role}</span>}
        {workflow.delegationId && <span>委派：{workflow.delegationId}</span>}
        {!!workflow.attempt && <span>第 {workflow.attempt} 轮</span>}
        {workflow.usage && <>
          <span>角色调用：{workflow.usage.roleInvocations}</span>
          <span>模型请求：{workflow.usage.modelRequests}</span>
          <span>工具调用：{workflow.usage.toolCalls}</span>
          <span>子 Agent：{workflow.usage.delegationCalls}</span>
          <span>Token：{workflow.usage.inputTokens + workflow.usage.outputTokens}</span>
          <span>服务端统计耗时：{workflow.usage.elapsedSeconds.toFixed(2)}s</span>
        </>}
      </div>
      <div className="workflow-tabs" role="tablist" aria-label="工作流详情">
        <button
          type="button"
          role="tab"
          aria-selected={activePanel === "execution"}
          className={activePanel === "execution" ? "active" : ""}
          onClick={() => setActivePanel("execution")}
        >
          执行过程 <span>{activity.events.length}</span>
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activePanel === "result"}
          className={activePanel === "result" ? "active" : ""}
          disabled={!workflow.artifact && !final}
          onClick={() => setActivePanel("result")}
        >
          结果与审核
        </button>
      </div>
      {activePanel === "execution" && (
        <div className="workflow-tab-panel" role="tabpanel">
          <div className="workflow-section workflow-event-history">
            <h3>已收到的阶段事件</h3>
            <ol>
              {activity.events.map((event, index) => (
                <li className={event.workflow.status} key={`${event.receivedAt}-${index}`}>
                  <span className="workflow-event-marker" aria-hidden="true" />
                  <div>
                    <strong>{PHASE_LABELS[event.workflow.phase] ?? event.workflow.phase}</strong>
                    {event.workflow.role && <span> · {event.workflow.role}</span>}
                    <span> · {STATUS_LABELS[event.workflow.status] ?? event.workflow.status}</span>
                    <p>{event.workflow.detail}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
          {workflow.delegation && (
            <div className="workflow-current-run">
              <strong>{workflow.delegation.delegationId}</strong>
              <span>{workflow.delegation.parentRole} → {workflow.delegation.role} / {workflow.delegation.status}</span>
              <p>{workflow.delegation.task}</p>
              {workflow.delegation.summary && <p>{workflow.delegation.summary}</p>}
            </div>
          )}
          {workflow.delegations && workflow.delegations.length > 0 && (
            <div className="workflow-section">
              <h3>按需委派记录</h3>
              <ul>
                {workflow.delegations.map((item) => (
                  <li key={item.delegationId}>
                    <strong>{item.delegationId}</strong>
                    <span>{item.parentRole} → {item.role} / {item.status}</span>
                    <p>{item.task}</p>
                    {item.summary && <p>{item.summary}</p>}
                    {item.errorType && <p>错误类型：{item.errorType}</p>}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      {activePanel === "result" && (
        <div className="workflow-tab-panel" role="tabpanel">
          {workflow.issues && workflow.issues.length > 0 && (
            <div className="workflow-section issues">
              <h3>审核问题</h3>
              <ul>
                {workflow.issues.map((issue, index) => (
                  <li key={`${issue.owner}-${index}`}>
                    <strong>{issue.owner}</strong>
                    <span>{issue.category} / {issue.severity}</span>
                    <p>{issue.message}</p>
                    {issue.evidence && <p>依据：{issue.evidence}</p>}
                    {issue.suggestion && <p>建议：{issue.suggestion}</p>}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {workflow.artifact
            ? <div className="workflow-artifact">
                <h3>最终产物 v{workflow.artifact.version}</h3>
                <pre>{workflow.artifact.content}</pre>
              </div>
            : <p className="workflow-result-empty">工作流结束后将在这里展示最终产物和审核结果。</p>}
        </div>
      )}
    </section>
  )
}
