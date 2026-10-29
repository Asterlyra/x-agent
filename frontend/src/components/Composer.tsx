import { useMemo, useState } from "react"

type Props = {
  pending: boolean
  onSend: (text: string) => Promise<boolean>
}

type ComposerMode = "chat" | "workflow"

const COMMANDS = [
  { value: "/workflow ", label: "多 Agent 工作流", description: "让 Producer 与 Reviewer 协作完成任务" },
  { value: "/retry", label: "重试上一轮", description: "重新执行最后一个用户请求" },
  { value: "/redirect ", label: "修正后重跑", description: "用新的要求替换上一轮输入" },
  { value: "/stats", label: "会话统计", description: "查看消息和 Token 用量" },
  { value: "/tools", label: "工具列表", description: "查看当前 Agent 可以调用的工具" },
  { value: "/help", label: "命令帮助", description: "查看全部可用命令" },
]

export function Composer({ pending, onSend }: Props) {
  const [draft, setDraft] = useState("")
  const [mode, setMode] = useState<ComposerMode>("chat")
  const [activeCommand, setActiveCommand] = useState(0)
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(true)
  const commands = useMemo(() => {
    if (!commandPaletteOpen || !draft.startsWith("/")) return []
    const needle = draft.slice(1).trim().toLowerCase()
    return COMMANDS.filter((command) => (
      !needle
      || command.value.slice(1).trim().startsWith(needle)
      || command.label.toLowerCase().includes(needle)
    ))
  }, [commandPaletteOpen, draft])

  async function submit() {
    const text = draft.trim()
    if (!text || pending) return
    const payload = mode === "workflow" && !text.startsWith("/")
      ? `/workflow ${text}`
      : text
    if (await onSend(payload)) setDraft("")
  }

  function selectCommand(value: string) {
    setDraft(value)
    setActiveCommand(0)
    setCommandPaletteOpen(false)
  }

  return (
    <div className="composer-area">
      {commands.length > 0 && (
        <div className="command-palette" aria-label="命令建议">
          <div className="command-palette-title">斜杠命令</div>
          {commands.map((command, index) => (
            <button
              type="button"
              className={index === activeCommand ? "active" : ""}
              key={command.value}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => selectCommand(command.value)}
            >
              <code>{command.value.trim()}</code>
              <span>
                <strong>{command.label}</strong>
                <small>{command.description}</small>
              </span>
            </button>
          ))}
        </div>
      )}
      <div className="composer-mode" aria-label="运行模式">
        <button
          type="button"
          aria-pressed={mode === "chat"}
          className={mode === "chat" ? "active" : ""}
          disabled={pending}
          onClick={() => setMode("chat")}
        >
          普通对话
        </button>
        <button
          type="button"
          aria-pressed={mode === "workflow"}
          className={mode === "workflow" ? "active" : ""}
          disabled={pending}
          onClick={() => setMode("workflow")}
        >
          多 Agent 工作流
        </button>
        <span>{mode === "workflow" ? "自动使用 /workflow" : "输入 / 查看更多命令"}</span>
      </div>
      <div className="composer-shell">
        <textarea
          aria-label="输入消息"
          value={draft}
          disabled={pending}
          placeholder={pending
            ? "x-agent 正在思考…"
            : mode === "workflow"
              ? "描述需要研究、生成并审核的任务"
              : "输入消息，Enter 发送，Shift+Enter 换行"}
          onChange={(event) => {
            setDraft(event.target.value)
            setActiveCommand(0)
            setCommandPaletteOpen(true)
          }}
          onKeyDown={(event) => {
            if (commands.length > 0 && event.key === "ArrowDown") {
              event.preventDefault()
              setActiveCommand((current) => (current + 1) % commands.length)
              return
            }
            if (commands.length > 0 && event.key === "ArrowUp") {
              event.preventDefault()
              setActiveCommand((current) => (current - 1 + commands.length) % commands.length)
              return
            }
            if (commands.length > 0 && event.key === "Enter" && !event.shiftKey) {
              event.preventDefault()
              selectCommand(commands[activeCommand]?.value ?? commands[0].value)
              return
            }
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault()
              void submit()
            }
          }}
        />
        <button type="button" onClick={() => void submit()} disabled={pending || !draft.trim()}>
          {pending ? "处理中…" : mode === "workflow" ? "启动工作流" : "发送"}
        </button>
      </div>
    </div>
  )
}
