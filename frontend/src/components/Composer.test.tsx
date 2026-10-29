import { cleanup, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { Composer } from "./Composer"

describe("Composer workbench", () => {
  afterEach(() => cleanup())

  it("turns workflow mode input into the existing workflow command", async () => {
    const user = userEvent.setup()
    const onSend = vi.fn().mockResolvedValue(true)
    render(<Composer pending={false} onSend={onSend} />)

    await user.click(screen.getByRole("button", { name: "多 Agent 工作流" }))
    await user.type(screen.getByLabelText("输入消息"), "研究当前项目")
    await user.click(screen.getByRole("button", { name: "启动工作流" }))

    expect(onSend).toHaveBeenCalledWith("/workflow 研究当前项目")
  })

  it("supports keyboard selection in the slash command palette", async () => {
    const user = userEvent.setup()
    const onSend = vi.fn().mockResolvedValue(true)
    render(<Composer pending={false} onSend={onSend} />)
    const input = screen.getByLabelText("输入消息")

    await user.type(input, "/")
    expect(screen.getByLabelText("命令建议")).toBeInTheDocument()
    await user.keyboard("{ArrowDown}{Enter}")
    expect(input).toHaveValue("/retry")
    await user.keyboard("{Enter}")

    expect(onSend).toHaveBeenCalledWith("/retry")
  })
})
