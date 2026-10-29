import type { MessageView } from "../api"

type Props = { message: MessageView }

function LinkifiedToolContent({ content }: { content: string }) {
  const parts = content.split(/(https?:\/\/[^\s<>"']+)/g)
  return (
    <pre>
      {parts.map((part, index) => part.startsWith("http://") || part.startsWith("https://")
        ? <a href={part} key={`${part}-${index}`} target="_blank" rel="noreferrer">{part}</a>
        : part)}
    </pre>
  )
}

export function ToolCard({ message }: Props) {
  const status = message.toolStatus ?? "success"
  const statusText = status === "running" ? "执行中" : status === "error" ? "失败" : "完成"
  return (
    <details className={`tool-card ${status}`}>
      <summary>
        <span className="tool-dot" aria-hidden="true" />
        <span>{message.toolName ?? "tool"}</span>
        <span className="tool-status">{statusText}</span>
      </summary>
      <LinkifiedToolContent content={message.content} />
    </details>
  )
}
