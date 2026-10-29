import { afterEach, describe, expect, it, vi } from "vitest"
import { ChatStreamError, sendCommand, sendMessage, type ChatStreamEvent } from "./api"

function fakeSseResponse(text: string, cutPoints: number[]) {
  const payload = new TextEncoder().encode(text)
  const chunks: Uint8Array[] = []
  let start = 0
  for (const point of cutPoints) {
    chunks.push(payload.slice(start, point))
    start = point
  }
  chunks.push(payload.slice(start))
  let index = 0
  const reader = {
    read: vi.fn(async () => index < chunks.length
      ? { done: false, value: chunks[index++] }
      : { done: true, value: undefined }),
    releaseLock: vi.fn(),
  }
  return {
    ok: true,
    headers: { get: () => "text/event-stream; charset=utf-8" },
    body: { getReader: () => reader },
  }
}

describe("SSE client", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("reassembles frames split across arbitrary UTF-8 chunks", async () => {
    const body = [
      'event: message\ndata: {"message":{"id":"u1","role":"user","content":"中文问题"}}\n\n',
      'event: done\ndata: {"messages":[{"id":"u1","role":"user","content":"中文问题"}]}\n\n',
    ].join("")
    vi.stubGlobal("fetch", vi.fn(async () => fakeSseResponse(body, [7, 23, 39, 58, 91])))
    const events: ChatStreamEvent[] = []

    const messages = await sendMessage("中文问题", (event) => events.push(event))

    expect(events.map((event) => event.type)).toEqual(["message", "done"])
    expect(messages[0].content).toBe("中文问题")
  })

  it("surfaces an SSE error with the last committed messages", async () => {
    const body = 'event: error\ndata: {"detail":"当前已有回合","messages":[{"id":"a1","role":"assistant","content":"旧回答"}]}\n\n'
    vi.stubGlobal("fetch", vi.fn(async () => fakeSseResponse(body, [5, 17, 41])))

    const error = await sendMessage("冲突请求").catch((caught: unknown) => caught)

    expect(error).toBeInstanceOf(ChatStreamError)
    expect((error as ChatStreamError).messages?.[0].content).toBe("旧回答")
  })

  it("delivers workflow progress and the final projection", async () => {
    const body = [
      'event: workflow\ndata: {"workflow":{"phase":"preparing","status":"completed","detail":"任务已保存","usage":{"roleInvocations":0,"modelRequests":0,"toolCalls":0,"inputTokens":0,"outputTokens":0,"cacheReadTokens":0,"delegationCalls":0,"elapsedSeconds":0}}}\n\n',
      'event: workflow\ndata: {"workflow":{"phase":"completed","status":"passed","detail":"通过","usage":{"roleInvocations":2,"modelRequests":2,"toolCalls":0,"inputTokens":100,"outputTokens":20,"cacheReadTokens":0,"delegationCalls":0,"elapsedSeconds":1},"artifact":{"content":"报告","version":1,"evidenceRefs":[],"assumptions":[],"unresolvedQuestions":[]}}}\n\n',
      'event: done\ndata: {"messages":[],"workflow":{"phase":"completed","status":"passed","detail":"通过"}}\n\n',
    ].join("")
    vi.stubGlobal("fetch", vi.fn(async () => fakeSseResponse(body, [11, 47, 116])))
    const events: ChatStreamEvent[] = []

    await sendCommand("/workflow 生成报告", (event) => events.push(event))

    expect(events.map((event) => event.type)).toEqual(["workflow", "workflow", "done"])
    const finalProgress = events[1]
    expect(finalProgress.type === "workflow" && finalProgress.workflow.artifact?.content).toBe("报告")
    expect(finalProgress.type === "workflow" && finalProgress.workflow.usage?.modelRequests).toBe(2)
  })
})
