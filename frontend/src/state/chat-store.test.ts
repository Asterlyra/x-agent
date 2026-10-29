import { describe, expect, it } from "vitest"
import { chatReducer, initialChatState } from "./chat-store"

describe("chatReducer", () => {
  it("ends initial loading when history loading fails", () => {
    const next = chatReducer(initialChatState, { type: "failure", error: "加载失败" })

    expect(next.loading).toBe(false)
    expect(next.pending).toBe(false)
    expect(next.error).toBe("加载失败")
  })
})
