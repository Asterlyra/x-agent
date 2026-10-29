import { cleanup, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { ConversationItem } from "../state/conversation-timeline"
import { ConversationPage } from "./ConversationPage"

describe("ConversationPage", () => {
  afterEach(() => cleanup())
  it("renders user, assistant and collapsed tool messages", () => {
    render(
      <ConversationPage
        messages={[
          { id: "u1", role: "user", content: "你好" },
          { id: "a1", role: "assistant", content: "你好！" },
          { id: "t1", role: "tool", content: "2", toolName: "calculate", toolStatus: "success" },
        ]}
        pending={false}
        onSend={vi.fn().mockResolvedValue(true)}
      />,
    )
    expect(screen.getByText("你好！")).toBeInTheDocument()
    expect(screen.getByText("calculate")).toBeInTheDocument()
    expect(screen.queryByText("2")).not.toBeVisible()
  })

  it("sends on Enter and clears the draft after success", async () => {
    const user = userEvent.setup()
    const onSend = vi.fn().mockResolvedValue(true)
    render(<ConversationPage messages={[]} pending={false} onSend={onSend} />)
    const input = screen.getByLabelText("输入消息")
    await user.type(input, "测试")
    await user.keyboard("{Enter}")
    expect(onSend).toHaveBeenCalledWith("测试")
    expect(input).toHaveValue("")
  })

  it("keeps the draft when the request fails", async () => {
    const user = userEvent.setup()
    const onSend = vi.fn().mockResolvedValue(false)
    render(<ConversationPage messages={[]} pending={false} onSend={onSend} />)
    const input = screen.getByLabelText("输入消息")
    await user.type(input, "失败后保留")
    await user.keyboard("{Enter}")
    expect(onSend).toHaveBeenCalledWith("失败后保留")
    expect(input).toHaveValue("失败后保留")
  })

  it("uses Shift+Enter for a newline", async () => {
    const user = userEvent.setup()
    render(<ConversationPage messages={[]} pending={false} onSend={vi.fn()} />)
    const input = screen.getByLabelText("输入消息")
    await user.type(input, "第一行")
    await user.keyboard("{Shift>}{Enter}{/Shift}")
    await user.type(input, "第二行")
    expect(input).toHaveValue("第一行\n第二行")
  })

  it("renders command, workflow progress and messages in stream order", () => {
    const receivedAt = Date.now()
    const workflowView = {
      phase: "reviewing",
      status: "running",
      detail: "审核中",
      role: "reviewer",
      usage: {
        roleInvocations: 4,
        modelRequests: 5,
        toolCalls: 2,
        inputTokens: 100,
        outputTokens: 20,
        cacheReadTokens: 0,
        delegationCalls: 1,
        elapsedSeconds: 1.25,
      },
    }
    const timeline: ConversationItem[] = [
      { type: "command", id: "command", command: "workflow", detail: "有界协作工作流已启动" },
      { type: "message", message: { id: "u1", role: "user", content: "研究项目" } },
      {
        type: "workflow",
        id: "workflow",
        activity: {
          current: workflowView,
          events: [{ workflow: workflowView, receivedAt }],
          startedAt: receivedAt,
          updatedAt: receivedAt,
        },
      },
      { type: "message", message: { id: "a1", role: "assistant", content: "最终报告" } },
    ]
    render(
      <ConversationPage
        messages={[]}
        pending={true}
        onSend={vi.fn()}
        timeline={timeline}
      />,
    )
    const command = screen.getByText("有界协作工作流已启动")
    const user = screen.getByText("研究项目")
    const workflow = screen.getByLabelText("协作工作流")
    const assistant = screen.getByText("最终报告")
    expect(command.compareDocumentPosition(user) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(user.compareDocumentPosition(workflow) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(workflow.compareDocumentPosition(assistant) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByText("模型请求：5")).toBeInTheDocument()
    expect(screen.queryByText(/reasoning/i)).not.toBeInTheDocument()
  })
})
