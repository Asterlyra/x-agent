import type { ReactNode } from "react"

type Props = {
  error: string | null
  onReset: () => void
  onInterrupt: () => void
  pending: boolean
  children: ReactNode
}

export function AppShell({ error, onReset, onInterrupt, pending, children }: Props) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <span className="app-mark">x</span>
          <span className="app-title">x-agent</span>
          <span className="app-subtitle">学习型 Web Agent</span>
        </div>
        <div className="header-actions">
          {pending && <button type="button" className="interrupt-button" onClick={onInterrupt}>停止</button>}
          <button type="button" className="reset-button" onClick={onReset} disabled={pending}>清空对话</button>
        </div>
      </header>
      {error && <div role="alert" className="error-banner">{error}</div>}
      {children}
    </div>
  )
}
