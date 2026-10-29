import { useEffect, useRef } from "react"
import type { MessageView } from "../api"
import { ConversationTimeline } from "./ConversationTimeline"
import {
  timelineFromMessages,
  type ConversationItem,
} from "../state/conversation-timeline"

type Props = { messages?: MessageView[]; items?: ConversationItem[] }

export function MessageList({ messages = [], items }: Props) {
  const bottom = useRef<HTMLDivElement>(null)
  const timeline = items ?? timelineFromMessages(messages)
  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth" })
  }, [timeline])

  if (timeline.length === 0) {
    return <div className="empty-state">输入一个问题，开始学习。</div>
  }
  return (
    <div>
      <ConversationTimeline items={timeline} />
      <div ref={bottom} />
    </div>
  )
}
