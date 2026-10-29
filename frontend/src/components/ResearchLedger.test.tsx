import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { ResearchLedger } from "./ResearchLedger"

describe("ResearchLedger", () => {
  afterEach(() => cleanup())

  it("projects workflow events and tool names without creating a second run log", () => {
    const workflow = {
      phase: "reviewing",
      status: "running",
      detail: "Reviewer 正在审核",
      role: "reviewer",
    }
    render(<ResearchLedger
      pending
      messages={[{ id: "tool-1", role: "tool", content: "结果", toolName: "read_file", toolStatus: "success" }]}
      timeline={[{
        type: "workflow",
        id: "workflow",
        activity: {
          current: workflow,
          events: [{ workflow, receivedAt: 1000 }],
          startedAt: 1000,
          updatedAt: 1000,
        },
      }]}
    />)

    expect(screen.getByRole("complementary", { name: "证据与运行记录" })).toBeInTheDocument()
    expect(screen.getByText("独立审核")).toBeInTheDocument()
    expect(screen.getByText("工具 · read_file")).toBeInTheDocument()
    expect(screen.getByText(/不保存第二份工作流日志/)).toBeInTheDocument()
  })
})
