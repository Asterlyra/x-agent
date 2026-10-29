import type { MessageView } from "../api"
import { ToolCard } from "./ToolCard"

type Props = { message: MessageView }

function renderInline(text: string) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>
    if (part.startsWith("`") && part.endsWith("`")) return <code key={`${part}-${index}`}>{part.slice(1, -1)}</code>
    return part
  })
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="markdown-content">
      {content.split(/\r?\n/).map((line, index) => {
        if (/^\s*$/.test(line)) return <div className="markdown-spacer" key={`spacer-${index}`} />
        const heading = /^(#{1,6})\s+(.+)$/.exec(line)
        if (heading) {
          const text = renderInline(heading[2])
          if (heading[1].length === 1) return <h2 key={`heading-${index}`}>{text}</h2>
          if (heading[1].length === 2) return <h3 key={`heading-${index}`}>{text}</h3>
          return <h4 key={`heading-${index}`}>{text}</h4>
        }
        if (/^\s*([-*])\s+/.test(line)) return <div className="markdown-list-item" key={`list-${index}`}>• {renderInline(line.replace(/^\s*[-*]\s+/, ""))}</div>
        if (/^\s*---+\s*$/.test(line)) return <hr key={`rule-${index}`} />
        return <p key={`paragraph-${index}`}>{renderInline(line)}</p>
      })}
    </div>
  )
}

export function MessageItem({ message }: Props) {
  if (message.role === "tool") return <ToolCard message={message} />
  return (
    <article className={`message ${message.role}`} data-message-id={message.id}>
      <div className="message-label">{message.role === "user" ? "你的请求" : "当前成果"}</div>
      <div className="message-content"><MarkdownContent content={message.content} /></div>
    </article>
  )
}
