import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { ToolCard } from "./ToolCard"

describe("ToolCard resources", () => {
  afterEach(() => cleanup())

  it("renders public URLs as safe external links", () => {
    render(<ToolCard message={{
      id: "tool-1",
      role: "tool",
      toolName: "web_search",
      toolStatus: "success",
      content: "来源 https://example.com/report",
    }} />)

    const link = screen.getByRole("link", { name: "https://example.com/report" })
    expect(link).toHaveAttribute("href", "https://example.com/report")
    expect(link).toHaveAttribute("target", "_blank")
    expect(link).toHaveAttribute("rel", "noreferrer")
  })
})
