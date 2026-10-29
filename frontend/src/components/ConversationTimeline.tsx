import type { ConversationItem } from "../state/conversation-timeline"
import { MessageItem } from "./MessageItem"
import { WorkflowRunPanel } from "./WorkflowRunPanel"

type Props = { items: ConversationItem[] }

export function ConversationTimeline({ items }: Props) {
  return (
    <div className="message-list" aria-live="polite">
      {items.map((item) => {
        if (item.type === "message") return <MessageItem key={item.message.id} message={item.message} />
        if (item.type === "command") {
          return (
            <pre className="command-output" data-command={item.command} key={item.id}>
              {item.detail}
            </pre>
          )
        }
        return <WorkflowRunPanel key={item.id} activity={item.activity} />
      })}
    </div>
  )
}
