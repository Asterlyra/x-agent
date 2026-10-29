import type { MessageView } from "../api"

export type ChatState = {
  messages: MessageView[]
  pending: boolean
  error: string | null
  loading: boolean
}

export const initialChatState: ChatState = {
  messages: [],
  pending: false,
  error: null,
  loading: true,
}

export type ChatAction =
  | { type: "loaded"; messages: MessageView[] }
  | { type: "pending" }
  | { type: "stream_message"; message: MessageView }
  | { type: "replace"; messages: MessageView[] }
  | { type: "success"; messages: MessageView[] }
  | { type: "failure"; error: string; messages?: MessageView[] }
  | { type: "reset"; messages: MessageView[] }

function upsertMessage(messages: MessageView[], message: MessageView): MessageView[] {
  const index = messages.findIndex((item) => item.id === message.id)
  if (index < 0) return [...messages, message]
  const next = [...messages]
  next[index] = message
  return next
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "loaded":
      return { ...state, messages: action.messages, loading: false, error: null }
    case "pending":
      return { ...state, pending: true, error: null }
    case "stream_message":
      return { ...state, messages: upsertMessage(state.messages, action.message), error: null }
    case "replace":
      return { ...state, messages: action.messages, error: null }
    case "success":
      return { ...state, messages: action.messages, pending: false, error: null }
    case "failure":
      return {
        ...state,
        messages: action.messages ?? state.messages,
        pending: false,
        loading: false,
        error: action.error,
      }
    case "reset":
      return { ...state, messages: action.messages, pending: false, error: null }
  }
}
