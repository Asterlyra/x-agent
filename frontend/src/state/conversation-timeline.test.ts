import { describe, expect, it } from "vitest"
import {
  clearActivities,
  failWorkflowActivity,
  upsertWorkflow,
  type ConversationItem,
} from "./conversation-timeline"

describe("conversation timeline workflow activity", () => {
  it("keeps every public workflow boundary event", () => {
    const first = upsertWorkflow([], {
      phase: "producing",
      status: "running",
      detail: "Producer 开始生成",
    }, 1000)
    const second = upsertWorkflow(first, {
      phase: "reviewing",
      status: "running",
      detail: "Reviewer 开始审核",
    }, 2000)

    const activity = second.find((item) => item.type === "workflow")
    expect(activity?.type === "workflow" && activity.activity.events.map((event) => event.workflow.phase))
      .toEqual(["producing", "reviewing"])
    expect(activity?.type === "workflow" && activity.activity.startedAt).toBe(1000)
    expect(activity?.type === "workflow" && activity.activity.updatedAt).toBe(2000)
  })

  it("removes transient workflow progress when a new turn starts", () => {
    const items: ConversationItem[] = upsertWorkflow([
      { type: "message", message: { id: "u1", role: "user", content: "任务" } },
    ], {
      phase: "producing",
      status: "running",
      detail: "运行中",
    }, 1000)

    expect(clearActivities(items)).toEqual([
      { type: "message", message: { id: "u1", role: "user", content: "任务" } },
    ])
  })

  it("turns an active workflow into a final error event", () => {
    const running = upsertWorkflow([], {
      phase: "reviewing",
      status: "running",
      detail: "审核中",
    }, 1000)
    const failed = failWorkflowActivity(running, "SSE 连接中断", 2000)
    const activity = failed.find((item) => item.type === "workflow")

    expect(activity?.type === "workflow" && activity.activity.current.status).toBe("error")
    expect(activity?.type === "workflow" && activity.activity.events).toHaveLength(2)
    expect(activity?.type === "workflow" && activity.activity.current.detail).toBe("SSE 连接中断")
  })
})
