import type { MessageView } from "../api"
import { Composer } from "./Composer"
import { MessageList } from "./MessageList"
import { timelineFromMessages, type ConversationItem } from "../state/conversation-timeline"
import { ResearchLedger } from "./ResearchLedger"

type Props = {
  messages: MessageView[]
  pending: boolean
  onSend: (text: string) => Promise<boolean>
  timeline?: ConversationItem[]
  sessionTitle?: string
}

export function ConversationPage({ messages, pending, onSend, timeline, sessionTitle = "当前研究档案" }: Props) {
  const effectiveTimeline = timeline ?? timelineFromMessages(messages)
  return (
    <main className="conversation-page">
      <section className="conversation-body">
        <header className="document-header">
          <div className="document-crumb">研究档案 / 当前会话</div>
          <h1>{sessionTitle}</h1>
          <div className="document-meta"><span>{effectiveTimeline.some((item) => item.type === "workflow") ? "多 Agent 工作流" : "普通对话"}</span><span>{messages.length} 条消息</span><span>证据与过程随本轮运行更新</span></div>
        </header>
        <MessageList messages={messages} items={effectiveTimeline} />
      </section>
      <ResearchLedger messages={messages} timeline={effectiveTimeline} pending={pending} />
      <Composer pending={pending} onSend={onSend} />
    </main>
  )
}
