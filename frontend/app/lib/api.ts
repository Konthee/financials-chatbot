import type { ChatMessage, SessionDetail, SessionSummary, UserProfile } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

type TokenResponse = { access_token: string };

type ChatRunBody = { session_id: string | null; messages: ChatMessage[] };

type ChatRunResult = {
  session_id: string;
  answer: string;
  grounded: boolean | null;
  usage: Record<string, number>;
  trace_id: string | null;
};

async function request<T>(path: string, token: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json", ...(init.headers as Record<string, string>) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = ((await response.json()) as { detail?: string }).detail ?? detail;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function login(email: string, password: string): Promise<TokenResponse> {
  return request<TokenResponse>("/api/v1/auth/login", "", { method: "POST", body: JSON.stringify({ email, password }) });
}

export function register(email: string, password: string): Promise<TokenResponse> {
  return request<TokenResponse>("/api/v1/auth/register", "", { method: "POST", body: JSON.stringify({ email, password }) });
}

export function getProfile(token: string): Promise<UserProfile> {
  return request<UserProfile>("/api/v1/users/me", token);
}

export function listSections(token: string): Promise<SessionSummary[]> {
  return request<SessionSummary[]>("/api/v1/sections", token);
}

export function getSection(token: string, id: string): Promise<SessionDetail> {
  return request<SessionDetail>(`/api/v1/sections/${id}`, token);
}

export function deleteSection(token: string, id: string): Promise<void> {
  return request<void>(`/api/v1/sections/${id}`, token, { method: "DELETE" });
}

/** Non-streaming chat: returns only the final answer. Exposed for integrations/tests. */
export function chatOnce(token: string, body: ChatRunBody): Promise<ChatRunResult> {
  return request<ChatRunResult>("/api/v1/chat/runs", token, { method: "POST", body: JSON.stringify(body) });
}

/** Streaming chat: returns the raw Response so the caller can read the NDJSON body. */
export function streamChat(token: string, body: ChatRunBody, signal: AbortSignal): Promise<Response> {
  return fetch(`${API_BASE}/api/v1/chat/runs/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
    signal,
  });
}
