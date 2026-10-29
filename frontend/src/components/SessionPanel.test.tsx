import { cleanup, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { SessionPanel } from "./SessionPanel"

const sessions = [
  {
    id: "current",
    createdAt: "2026-08-25T10:00:00",
    updatedAt: "2026-08-25T10:00:00",
    messageCount: 2,
    title: "当前会话",
    active: true,
  },
  {
    id: "archive-1",
    createdAt: "2026-08-20T10:00:00",
    updatedAt: "2026-08-20T10:00:00",
    messageCount: 4,
    title: "世界杯研究",
    active: false,
  },
]

describe("SessionPanel workbench", () => {
  afterEach(() => cleanup())

  it("searches sessions and exposes matching content", async () => {
    const user = userEvent.setup()
    const onSearch = vi.fn().mockResolvedValue([
      { sessionId: "archive-1", index: 2, content: "世界杯比赛资料" },
    ])
    render(<SessionPanel
      sessions={sessions}
      currentSessionId="current"
      pending={false}
      onNew={vi.fn()}
      onSelect={vi.fn()}
      onFork={vi.fn().mockResolvedValue(true)}
      onSearch={onSearch}
      onLoadStats={vi.fn().mockResolvedValue([])}
    />)

    await user.type(screen.getByLabelText("搜索会话"), "世界杯")

    await waitFor(() => expect(onSearch).toHaveBeenCalledWith("世界杯"))
    expect(await screen.findByText(/#2 世界杯比赛资料/)).toBeInTheDocument()
    expect(screen.queryByText("当前会话")).not.toBeInTheDocument()
  })

  it("loads usage stats and can fork a session", async () => {
    const user = userEvent.setup()
    const onFork = vi.fn().mockResolvedValue(true)
    render(<SessionPanel
      sessions={sessions}
      currentSessionId="current"
      pending={false}
      onNew={vi.fn()}
      onSelect={vi.fn()}
      onFork={onFork}
      onSearch={vi.fn().mockResolvedValue([])}
      onLoadStats={vi.fn().mockResolvedValue([
        { sessionId: "current", messageCount: 2, inputTokens: 100, outputTokens: 20, cacheReadTokens: 0, estimatedTokens: 120 },
        { sessionId: "archive-1", messageCount: 4, inputTokens: 200, outputTokens: 30, cacheReadTokens: 0, estimatedTokens: 230 },
      ])}
    />)

    await user.click(screen.getByRole("button", { name: "统计" }))
    expect(await screen.findByText("估算 350 Token")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "分叉会话 世界杯研究" }))
    expect(onFork).toHaveBeenCalledWith("archive-1")
  })
})
