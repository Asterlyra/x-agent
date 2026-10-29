import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { AppShell } from "./AppShell"

describe("AppShell", () => {
  afterEach(() => cleanup())

  it("disables reset while a turn is pending", () => {
    render(
      <AppShell error={null} onReset={vi.fn()} onInterrupt={vi.fn()} pending>
        <div>内容</div>
      </AppShell>,
    )

    expect(screen.getByRole("button", { name: "清空对话" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "停止" })).toBeEnabled()
  })
})
