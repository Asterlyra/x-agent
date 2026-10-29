import type { MessageView, WorkflowView } from "../api"
import type { ConversationItem } from "../state/conversation-timeline"

type Props = { messages: MessageView[]; timeline: ConversationItem[]; pending: boolean }

const PHASE_LABELS: Record<string, string> = {
  preparing: "任务准备", delegating: "资料检索", producing: "成果生成", reviewing: "独立审核", completed: "运行完成",
}
const STATUS_LABELS: Record<string, string> = {
  running: "运行中", completed: "已完成", passed: "已通过", not_passed: "未通过", cancelled: "已停止", error: "运行异常",
}

function latestWorkflow(timeline: ConversationItem[]): { current: WorkflowView; events: WorkflowView[] } | null {
  const item = [...timeline].reverse().find((entry) => entry.type === "workflow")
  if (item?.type !== "workflow") return null
  return { current: item.activity.current, events: item.activity.events.map((event) => event.workflow) }
}

function uniqueToolNames(messages: MessageView[]): string[] {
  return [...new Set(messages.filter((message) => message.role === "tool").map((message) => message.toolName ?? "tool"))]
}

export function ResearchLedger({ messages, timeline, pending }: Props) {
  const workflow = latestWorkflow(timeline)
  const toolNames = uniqueToolNames(messages)
  const status = workflow?.current.status ?? (pending ? "running" : "completed")
  const final = ["passed", "not_passed", "cancelled", "error"].includes(status)
  const live = pending || Boolean(workflow && !final)
  const events = workflow?.events.slice(-5) ?? []

  return (
    <aside className="research-ledger" aria-label="证据与运行记录">
      <div className="ledger-title"><div className="ledger-eyebrow">Evidence ledger</div><h2>本轮材料</h2></div>
      <div className="ledger-status">
        <strong>{workflow ? `${workflow.current.role === "reviewer" ? "Reviewer" : "Producer"} 正在${final ? "完成" : "审核产物"}` : "普通对话记录"}</strong>
        <span className={live ? "status-live" : "status-final"}>{STATUS_LABELS[status] ?? status}</span>
      </div>
      {workflow && <section className="ledger-section"><h3>运行过程</h3><ol className="ledger-events">
        {events.map((event, index) => <li className={index === events.length - 1 && !final ? "live" : "done"} key={`${event.phase}-${index}`}><strong>{PHASE_LABELS[event.phase] ?? event.phase}</strong><span>{event.detail}</span></li>)}
      </ol></section>}
      <section className="ledger-section"><h3>工具记录</h3>
        {toolNames.length > 0
          ? <ul className="ledger-list">{toolNames.map((toolName) => <li key={toolName}><span>工具 · {toolName}</span><span>已投影</span></li>)}</ul>
          : <p className="ledger-empty">本轮还没有工具记录。</p>}
      </section>
      {workflow?.current.issues && workflow.current.issues.length > 0 && <section className="ledger-section"><h3>审核提示</h3><ul className="ledger-list">
        {workflow.current.issues.slice(0, 3).map((issue, index) => <li key={`${issue.owner}-${index}`}><span>{issue.message}</span></li>)}
      </ul></section>}
      <p className="ledger-note">这里展示的是公开消息与 SSE 边界投影，不保存第二份工作流日志。</p>
    </aside>
  )
}
