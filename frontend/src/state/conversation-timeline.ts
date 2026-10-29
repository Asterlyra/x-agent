import type { MessageView, WorkflowView } from "../api"

export type WorkflowActivityEvent = {
  workflow: WorkflowView
  receivedAt: number
}

export type WorkflowActivity = {
  current: WorkflowView
  events: WorkflowActivityEvent[]
  startedAt: number
  updatedAt: number
}

export type ConversationItem =
  | { type: "message"; message: MessageView }
  | { type: "command"; id: string; command: string; detail: string }
  | { type: "workflow"; id: string; activity: WorkflowActivity }

export function timelineFromMessages(messages: MessageView[]): ConversationItem[] {
  return messages.map((message) => ({ type: "message", message }))
}

export function upsertTimelineMessage(items: ConversationItem[], message: MessageView): ConversationItem[] {
  const index = items.findIndex((item) => item.type === "message" && item.message.id === message.id)
  if (index < 0) return [...items, { type: "message", message }]
  const next = [...items]
  next[index] = { type: "message", message }
  return next
}

export function appendCommand(items: ConversationItem[], command: string, detail: string): ConversationItem[] {
  return [...items, { type: "command", id: "activity-command", command, detail }]
}

export function upsertWorkflow(
  items: ConversationItem[],
  workflow: WorkflowView,
  receivedAt: number = Date.now(),
): ConversationItem[] {
  const index = items.findIndex((item) => item.type === "workflow" && item.id === "activity-workflow")
  const event = { workflow, receivedAt }
  if (index < 0) {
    return [...items, {
      type: "workflow",
      id: "activity-workflow",
      activity: {
        current: workflow,
        events: [event],
        startedAt: receivedAt,
        updatedAt: receivedAt,
      },
    }]
  }
  const previous = items[index]
  if (previous.type !== "workflow") return items
  const next = [...items]
  next[index] = {
    type: "workflow",
    id: "activity-workflow",
    activity: {
      current: workflow,
      events: [...previous.activity.events, event],
      startedAt: previous.activity.startedAt,
      updatedAt: receivedAt,
    },
  }
  return next
}

export function failWorkflowActivity(
  items: ConversationItem[],
  detail: string,
  receivedAt: number = Date.now(),
): ConversationItem[] {
  const active = items.find((item) => item.type === "workflow" && item.id === "activity-workflow")
  if (active?.type !== "workflow") return items
  if (["passed", "not_passed", "cancelled", "error"].includes(active.activity.current.status)) return items
  return upsertWorkflow(items, {
    phase: "completed",
    status: "error",
    detail,
    usage: active.activity.current.usage,
  }, receivedAt)
}

export function clearActivities(items: ConversationItem[]): ConversationItem[] {
  return items.filter((item) => item.type === "message")
}

export function reconcileTimelineMessages(items: ConversationItem[], messages: MessageView[]): ConversationItem[] {
  const byId = new Map(messages.map((message) => [message.id, message]))
  const next = items
    .filter((item) => item.type !== "message" || byId.has(item.message.id))
    .map((item) => item.type === "message" ? { type: "message" as const, message: byId.get(item.message.id)! } : item)
  const present = new Set(next.flatMap((item) => item.type === "message" ? [item.message.id] : []))
  return messages.reduce(
    (current, message) => present.has(message.id) ? current : upsertTimelineMessage(current, message),
    next,
  )
}
