import { act, cleanup, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { WorkflowActivity } from "../state/conversation-timeline"
import { WorkflowRunPanel } from "./WorkflowRunPanel"

function runningActivity(): WorkflowActivity {
  const now = Date.now()
  const workflow = {
    phase: "reviewing",
    status: "running",
    detail: "Reviewer 正在独立研究并审核产物",
    role: "reviewer",
    usage: {
      roleInvocations: 1,
      modelRequests: 8,
      toolCalls: 11,
      inputTokens: 70000,
      outputTokens: 1856,
      cacheReadTokens: 0,
      delegationCalls: 0,
      elapsedSeconds: 5,
    },
  }
  return {
    current: workflow,
    events: [{ workflow, receivedAt: now - 1000 }],
    startedAt: now - 5000,
    updatedAt: now - 1000,
  }
}

describe("WorkflowRunPanel", () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it("keeps a live clock and explains a quiet role interval", () => {
    vi.useFakeTimers()
    vi.setSystemTime(10000)
    render(<WorkflowRunPanel activity={runningActivity()} />)

    expect(screen.getByText(/已运行 5 秒/)).toBeInTheDocument()
    expect(screen.queryByText(/如需终止/)).not.toBeInTheDocument()

    act(() => vi.advanceTimersByTime(15000))

    expect(screen.getByText(/已运行 20 秒/)).toBeInTheDocument()
    expect(screen.getByText(/如需终止/)).toBeInTheDocument()
    expect(screen.queryByText(/%/)).not.toBeInTheDocument()
  })

  it("treats cancelled as a final state", () => {
    const workflow = {
      phase: "completed",
      status: "cancelled",
      detail: "工作流已在步骤边界取消",
      usage: {
        roleInvocations: 1,
        modelRequests: 2,
        toolCalls: 1,
        inputTokens: 100,
        outputTokens: 20,
        cacheReadTokens: 0,
        delegationCalls: 0,
        elapsedSeconds: 8,
      },
    }
    render(<WorkflowRunPanel activity={{
      current: workflow,
      events: [{ workflow, receivedAt: 9000 }],
      startedAt: 1000,
      updatedAt: 9000,
    }} />)

    expect(screen.getByText("已停止")).toBeInTheDocument()
    expect(screen.queryByText(/最后收到服务端更新/)).not.toBeInTheDocument()
  })

  it("keeps the overall label running when only the current stage completed", () => {
    const activity = runningActivity()
    const workflow = { ...activity.current, status: "completed", detail: "Producer 已完成" }
    render(<WorkflowRunPanel activity={{
      ...activity,
      current: workflow,
      events: [{ workflow, receivedAt: activity.updatedAt }],
    }} />)

    expect(screen.getByText("运行中")).toBeInTheDocument()
    expect(screen.getAllByText(/已完成/).length).toBeGreaterThan(0)
  })

  it("separates the final result from the public execution trace", async () => {
    const user = userEvent.setup()
    const activity = runningActivity()
    const workflow = {
      ...activity.current,
      phase: "completed",
      status: "passed",
      detail: "审核通过",
      artifact: {
        content: "最终报告",
        version: 1,
        evidenceRefs: [],
        assumptions: [],
        unresolvedQuestions: [],
      },
    }
    render(<WorkflowRunPanel activity={{
      ...activity,
      current: workflow,
      events: [{ workflow, receivedAt: activity.updatedAt }],
    }} />)

    expect(screen.getByText("最终报告")).toBeInTheDocument()
    await user.click(screen.getByRole("tab", { name: /执行过程/ }))
    expect(screen.getByText("已收到的阶段事件")).toBeInTheDocument()
    expect(screen.queryByText("最终报告")).not.toBeInTheDocument()
  })
})
