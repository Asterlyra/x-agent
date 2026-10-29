export type MessageView = {
  id: string
  role: "user" | "assistant" | "tool"
  content: string
  toolName?: string
  toolStatus?: "running" | "success" | "error"
}

type ChatResponse = { messages: MessageView[] }

export type SessionSummary = {
  id: string
  createdAt: string
  updatedAt: string
  messageCount: number
  title: string
  active: boolean
}

export type SessionListResponse = {
  currentSessionId: string
  sessions: SessionSummary[]
}

export type SessionActionResponse = {
  session: SessionSummary
  sessions: SessionSummary[]
  messages: MessageView[]
}

export type SearchHit = { sessionId: string; index: number; content: string }
export type SessionStat = {
  sessionId: string
  messageCount: number
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  estimatedTokens: number
}

export type WorkflowIssueView = {
  severity: string
  category: string
  message: string
  evidence: string
  suggestion: string
  owner: string
}

export type WorkflowDelegationView = {
  delegationId: string
  parentRole: string
  role: string
  task: string
  expectedOutput: string
  status: string
  summary: string
  errorType: string
}

export type WorkflowActiveDelegationView = {
  delegationId: string
  parentRole: string
  role: string
  task: string
  expectedOutput: string
  status: string
  errorType: string
  summary: string
}

export type WorkflowUsageView = {
  roleInvocations: number
  modelRequests: number
  toolCalls: number
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  delegationCalls: number
  elapsedSeconds: number
}

export type WorkflowView = {
  phase: string
  status: string
  detail: string
  usage?: WorkflowUsageView
  role?: string
  delegationId?: string
  attempt?: number
  stopReason?: string
  roleCalls?: Record<string, number>
  delegation?: WorkflowActiveDelegationView
  artifact?: {
    content: string
    version: number
    evidenceRefs: string[]
    assumptions: string[]
    unresolvedQuestions: string[]
  }
  delegations?: WorkflowDelegationView[]
  issues?: WorkflowIssueView[]
}

export type ChatStreamEvent =
  | { type: "message"; message: MessageView }
  | { type: "replace"; messages: MessageView[] }
  | { type: "workflow"; workflow: WorkflowView }
  | { type: "command"; command: string; detail: string; messages: MessageView[]; sessions?: SessionSummary[]; hits?: SearchHit[]; stats?: SessionStat[] }
  | { type: "done"; messages: MessageView[]; sessions?: SessionSummary[]; workflow?: WorkflowView }
  | { type: "interrupted"; detail: string; messages: MessageView[] }
  | { type: "error"; detail: string; messages: MessageView[] }

export class ChatStreamError extends Error {
  readonly messages?: MessageView[]

  constructor(detail: string, messages?: MessageView[]) {
    super(detail)
    this.name = "ChatStreamError"
    this.messages = messages
  }
}

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `请求失败（${response.status}）`)
  }
  return response.json() as Promise<T>
}

export async function loadHistory(): Promise<MessageView[]> {
  const result = await request<ChatResponse>("/api/history")
  return result.messages
}

export async function loadSessions(): Promise<SessionListResponse> {
  return request<SessionListResponse>("/api/sessions")
}

export async function createSession(): Promise<SessionActionResponse> {
  return request<SessionActionResponse>("/api/sessions", { method: "POST" })
}

export async function switchSession(sessionId: string): Promise<SessionActionResponse> {
  return request<SessionActionResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/switch`, { method: "POST" })
}

export async function forkSession(sessionId: string, beforeMessage?: number): Promise<SessionActionResponse> {
  return request<SessionActionResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/fork`, {
    method: "POST",
    body: JSON.stringify(beforeMessage === undefined ? {} : { beforeMessage }),
  })
}

export async function searchSessions(query: string): Promise<SearchHit[]> {
  const result = await request<{ hits: SearchHit[] }>(`/api/sessions/search?q=${encodeURIComponent(query)}`)
  return result.hits
}

export async function loadSessionStats(): Promise<SessionStat[]> {
  const result = await request<{ stats: SessionStat[] }>("/api/sessions/stats")
  return result.stats
}

function parseSseFrame(frame: string): ChatStreamEvent | null {
  let eventType = "message"
  const dataLines: string[] = []
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) eventType = line.slice("event:".length).trim()
    if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trimStart())
  }
  if (dataLines.length === 0) return null
  const payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>
  return { type: eventType, ...payload } as ChatStreamEvent
}

function consumeSseFrame(
  frame: string,
  onEvent: (event: ChatStreamEvent) => void,
  state: { messages?: MessageView[] },
) {
  const event = parseSseFrame(frame)
  if (!event) return
  onEvent(event)
  if (event.type === "done") state.messages = event.messages
  if (event.type === "interrupted") state.messages = event.messages
  if (event.type === "error") throw new ChatStreamError(event.detail, event.messages)
}

async function streamRequest(
  path: string,
  text: string,
  onEvent: (event: ChatStreamEvent) => void = () => undefined,
): Promise<MessageView[]> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    method: "POST",
    body: JSON.stringify({ text }),
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `请求失败（${response.status}）`)
  }
  if (!response.body) throw new Error("服务器没有返回流式响应")
  const contentType = response.headers.get("content-type") ?? ""
  if (!contentType.toLowerCase().startsWith("text/event-stream")) {
    throw new Error("服务器返回了无法识别的聊天响应")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder("utf-8", { fatal: true })
  const state: { messages?: MessageView[] } = {}
  let buffer = ""
  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
      while (true) {
        const separator = /\r?\n\r?\n/.exec(buffer)
        if (!separator || separator.index === undefined) break
        const frame = buffer.slice(0, separator.index)
        buffer = buffer.slice(separator.index + separator[0].length)
        consumeSseFrame(frame, onEvent, state)
      }
      if (done) break
    }
    if (buffer.trim()) consumeSseFrame(buffer, onEvent, state)
  } finally {
    reader.releaseLock()
  }
  if (!state.messages) throw new Error("聊天流在完成前断开")
  return state.messages
}

export async function sendMessage(
  text: string,
  onEvent: (event: ChatStreamEvent) => void = () => undefined,
): Promise<MessageView[]> {
  return streamRequest("/api/chat", text, onEvent)
}

export async function sendCommand(
  text: string,
  onEvent: (event: ChatStreamEvent) => void = () => undefined,
): Promise<MessageView[]> {
  return streamRequest("/api/command", text, onEvent)
}

export async function resetChat(): Promise<MessageView[]> {
  const result = await request<ChatResponse>("/api/reset", { method: "POST" })
  return result.messages
}
