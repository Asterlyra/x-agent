import { useEffect, useMemo, useState } from "react"
import type { SearchHit, SessionStat, SessionSummary } from "../api"

type Props = {
  sessions: SessionSummary[]
  currentSessionId: string
  pending: boolean
  onNew: () => void
  onSelect: (sessionId: string) => void
  onFork: (sessionId: string) => Promise<boolean>
  onSearch: (query: string) => Promise<SearchHit[]>
  onLoadStats: () => Promise<SessionStat[]>
}

type SessionGroup = { label: string; sessions: SessionSummary[] }

function groupSessions(sessions: SessionSummary[], now: Date = new Date()): SessionGroup[] {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const groups: Record<string, SessionSummary[]> = { "今天": [], "最近 7 天": [], "更早": [] }
  for (const session of sessions) {
    const updatedAt = new Date(session.updatedAt)
    const updatedDay = new Date(updatedAt.getFullYear(), updatedAt.getMonth(), updatedAt.getDate()).getTime()
    const age = today - updatedDay
    const label = Number.isFinite(updatedDay) && age <= 0
      ? "今天"
      : Number.isFinite(updatedDay) && age <= 7 * 24 * 60 * 60 * 1000
        ? "最近 7 天"
        : "更早"
    groups[label].push(session)
  }
  return Object.entries(groups)
    .filter(([, items]) => items.length > 0)
    .map(([label, items]) => ({ label, sessions: items }))
}

export function SessionPanel({
  sessions,
  currentSessionId,
  pending,
  onNew,
  onSelect,
  onFork,
  onSearch,
  onLoadStats,
}: Props) {
  const [query, setQuery] = useState("")
  const [hits, setHits] = useState<SearchHit[]>([])
  const [searching, setSearching] = useState(false)
  const [panelError, setPanelError] = useState("")
  const [stats, setStats] = useState<SessionStat[] | null>(null)
  const [showStats, setShowStats] = useState(false)
  const [statsLoading, setStatsLoading] = useState(false)

  useEffect(() => {
    const value = query.trim()
    if (!value) {
      setHits([])
      setSearching(false)
      return
    }
    setSearching(true)
    let active = true
    const timer = window.setTimeout(() => {
      onSearch(value)
        .then((result) => {
          if (!active) return
          setHits(result)
          setPanelError("")
        })
        .catch((error: unknown) => {
          if (!active) return
          setHits([])
          setPanelError(error instanceof Error ? error.message : "搜索失败")
        })
        .finally(() => {
          if (active) setSearching(false)
        })
    }, 250)
    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [onSearch, query])

  const hitBySession = useMemo(() => {
    const result = new Map<string, SearchHit>()
    for (const hit of hits) if (!result.has(hit.sessionId)) result.set(hit.sessionId, hit)
    return result
  }, [hits])
  const visibleSessions = query.trim()
    ? sessions.filter((session) => hitBySession.has(session.id))
    : sessions
  const groups = query.trim()
    ? [{ label: "搜索结果", sessions: visibleSessions }]
    : groupSessions(visibleSessions)
  const totalTokens = stats?.reduce((sum, item) => sum + item.estimatedTokens, 0) ?? 0

  async function toggleStats() {
    if (showStats) {
      setShowStats(false)
      return
    }
    setShowStats(true)
    if (stats) return
    setStatsLoading(true)
    try {
      setStats(await onLoadStats())
      setPanelError("")
    } catch (error) {
      setPanelError(error instanceof Error ? error.message : "统计加载失败")
    } finally {
      setStatsLoading(false)
    }
  }

  return (
    <aside className="session-panel" aria-label="会话管理">
      <div className="session-panel-header">
        <span>会话</span>
        <button type="button" onClick={onNew} disabled={pending}>新建</button>
      </div>
      <div className="session-toolbar">
        <input
          aria-label="搜索会话"
          type="search"
          value={query}
          placeholder="搜索全部会话"
          onChange={(event) => setQuery(event.target.value)}
        />
        <button type="button" aria-expanded={showStats} onClick={() => void toggleStats()}>统计</button>
      </div>
      {showStats && (
        <div className="session-stats" aria-label="会话统计">
          {stats
            ? <><strong>{stats.length} 个会话</strong><span>估算 {totalTokens.toLocaleString()} Token</span></>
            : <span>{statsLoading ? "正在加载统计…" : "暂无统计数据"}</span>}
        </div>
      )}
      {panelError && <div className="session-panel-error" role="alert">{panelError}</div>}
      <div className="session-list">
        {sessions.length === 0 && <div className="session-empty">还没有会话</div>}
        {query.trim() && !searching && visibleSessions.length === 0 && (
          <div className="session-empty">没有匹配的会话</div>
        )}
        {searching && <div className="session-empty">正在搜索…</div>}
        {!searching && groups.map((group) => (
          <section className="session-group" key={group.label}>
            <h2>{group.label}</h2>
            {group.sessions.map((session) => {
              const hit = hitBySession.get(session.id)
              return (
                <div className={`session-row${session.id === currentSessionId ? " active" : ""}`} key={session.id}>
                  <button
                    type="button"
                    className="session-item"
                    disabled={pending || session.id === currentSessionId}
                    onClick={() => onSelect(session.id)}
                    title={session.id}
                  >
                    <span className="session-title">{session.title}</span>
                    <span className="session-meta">
                      {hit
                        ? `#${hit.index} ${hit.content.replace(/\s+/g, " ").slice(0, 42)}`
                        : `${session.messageCount} 条 · ${session.id.slice(0, 18)}`}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="session-fork"
                    aria-label={`分叉会话 ${session.title}`}
                    disabled={pending}
                    onClick={() => void onFork(session.id)}
                  >
                    分叉
                  </button>
                </div>
              )
            })}
          </section>
        ))}
      </div>
    </aside>
  )
}
