import { useEffect, useReducer, useState } from "react"
import {
  ChatStreamError,
  createSession,
  forkSession,
  loadHistory,
  loadSessionStats,
  loadSessions,
  resetChat,
  searchSessions,
  sendCommand,
  sendMessage,
  switchSession,
  type ChatStreamEvent,
  type SessionSummary,
} from "./api"
import { AppShell } from "./components/AppShell"
import { ConversationPage } from "./components/ConversationPage"
import { SessionPanel } from "./components/SessionPanel"
import { chatReducer, initialChatState } from "./state/chat-store"
import {
  appendCommand,
  clearActivities,
  failWorkflowActivity,
  reconcileTimelineMessages,
  timelineFromMessages,
  upsertTimelineMessage,
  upsertWorkflow,
  type ConversationItem,
} from "./state/conversation-timeline"

export default function App() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [currentSessionId, setCurrentSessionId] = useState("current")
  const [timeline, setTimeline] = useState<ConversationItem[]>([])

  useEffect(() => {
    Promise.all([loadHistory(), loadSessions()])
      .then(([messages, sessionState]) => {
        setSessions(sessionState.sessions)
        setCurrentSessionId(sessionState.currentSessionId)
        dispatch({ type: "loaded", messages })
        setTimeline(timelineFromMessages(messages))
      })
      .catch((error: unknown) => dispatch({ type: "failure", error: error instanceof Error ? error.message : "历史加载失败" }))
  }, [])

  function handleStreamEvent(event: ChatStreamEvent) {
    if (event.type === "message") {
      dispatch({ type: "stream_message", message: event.message })
      setTimeline((current) => upsertTimelineMessage(current, event.message))
    }
    if (event.type === "replace" || event.type === "interrupted") {
      dispatch({ type: "replace", messages: event.messages })
      setTimeline(timelineFromMessages(event.messages))
    }
    if (event.type === "workflow") {
      setTimeline((current) => upsertWorkflow(current, event.workflow))
    }
    if (event.type === "done" && event.sessions) {
      setSessions(event.sessions)
      const active = event.sessions.find((session) => session.active)
      if (active) setCurrentSessionId(active.id)
    }
    if (event.type === "command") {
      let output = event.detail
      if (event.hits) {
        output += event.hits.length ? `\n\n${event.hits.map((hit) => `[${hit.sessionId} #${hit.index}] ${hit.content}`).join("\n")}` : "\n\n没有命中。"
      }
      if (event.stats) {
        output += event.stats.length
          ? `\n\n${event.stats.map((item) => `${item.sessionId}: ${item.messageCount} 条，估算 ${item.estimatedTokens} tokens`).join("\n")}`
          : "\n\n还没有统计数据。"
      }
      setTimeline((current) => appendCommand(current, event.command, output))
      if (event.sessions) setSessions(event.sessions)
      if (event.sessions) {
        const active = event.sessions.find((session) => session.active)
        if (active) setCurrentSessionId(active.id)
      }
    }
  }

  async function handleSend(text: string): Promise<boolean> {
    dispatch({ type: "pending" })
    setTimeline((current) => clearActivities(current))
    try {
      const runner = /^\s*\//.test(text) ? sendCommand : sendMessage
      const messages = await runner(text, handleStreamEvent)
      setTimeline((current) => reconcileTimelineMessages(current, messages))
      dispatch({ type: "success", messages })
      return true
    } catch (error) {
      const detail = error instanceof Error ? error.message : "发送失败"
      setTimeline((current) => failWorkflowActivity(current, detail))
      dispatch({
        type: "failure",
        error: detail,
        messages: error instanceof ChatStreamError ? error.messages : undefined,
      })
      return false
    }
  }

  async function handleReset() {
    if (state.pending) return
    try {
      dispatch({ type: "reset", messages: await resetChat() })
      setTimeline([])
      const sessionState = await loadSessions()
      setSessions(sessionState.sessions)
      setCurrentSessionId(sessionState.currentSessionId)
    } catch (error) {
      dispatch({ type: "failure", error: error instanceof Error ? error.message : "清空失败" })
    }
  }

  async function handleInterrupt() {
    try {
      await sendCommand("/interrupt", handleStreamEvent)
    } catch (error) {
      dispatch({ type: "failure", error: error instanceof Error ? error.message : "停止失败" })
    }
  }

  async function handleNewSession() {
    try {
      const result = await createSession()
      dispatch({ type: "replace", messages: result.messages })
      setTimeline(timelineFromMessages(result.messages))
      setSessions(result.sessions)
      setCurrentSessionId(result.session.id)
    } catch (error) {
      dispatch({ type: "failure", error: error instanceof Error ? error.message : "新建会话失败" })
    }
  }

  async function handleSwitchSession(sessionId: string) {
    try {
      const result = await switchSession(sessionId)
      dispatch({ type: "replace", messages: result.messages })
      setTimeline(timelineFromMessages(result.messages))
      setSessions(result.sessions)
      setCurrentSessionId(result.session.id)
    } catch (error) {
      dispatch({ type: "failure", error: error instanceof Error ? error.message : "切换会话失败" })
    }
  }

  async function handleForkSession(sessionId: string): Promise<boolean> {
    if (state.pending) return false
    try {
      const result = await forkSession(sessionId)
      dispatch({ type: "replace", messages: result.messages })
      setTimeline(timelineFromMessages(result.messages))
      setSessions(result.sessions)
      setCurrentSessionId(result.session.id)
      return true
    } catch (error) {
      dispatch({ type: "failure", error: error instanceof Error ? error.message : "分叉会话失败" })
      return false
    }
  }

  return (
    <AppShell error={state.error} onReset={() => void handleReset()} onInterrupt={() => void handleInterrupt()} pending={state.pending}>
      {state.loading
        ? <div className="loading-state">正在加载历史…</div>
        : <div className="workspace-layout">
            <SessionPanel
              sessions={sessions}
              currentSessionId={currentSessionId}
              pending={state.pending}
              onNew={() => void handleNewSession()}
              onSelect={(sessionId) => void handleSwitchSession(sessionId)}
              onFork={handleForkSession}
              onSearch={searchSessions}
              onLoadStats={loadSessionStats}
            />
            <ConversationPage
              messages={state.messages}
              pending={state.pending}
              onSend={handleSend}
              timeline={timeline}
              sessionTitle={sessions.find((session) => session.id === currentSessionId)?.title}
            />
          </div>}
    </AppShell>
  )
}
