"use client";

import { FormEvent, ReactNode, useEffect, useMemo, useReducer, useRef, useState } from "react";
import type { ChatMessage, EvidenceItem, StreamEvent, TimelineStep } from "./lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const TOKEN_KEY = "financial_qa_token";
const SECTIONS_KEY = "financial_qa_sections";

interface ChatState {
  messages: ChatMessage[];
  answerDraft: string;
  timeline: TimelineStep[];
  evidence: { source: string; items: EvidenceItem[] }[];
  traceId: string | null;
  isStreaming: boolean;
  error: string | null;
}

interface ChatSection {
  id: string;
  title: string;
  updatedAt: number;
  state: ChatState;
}

type Action =
  | { type: "load"; state: ChatState }
  | { type: "start"; question: string }
  | { type: "event"; event: StreamEvent }
  | { type: "error"; message: string }
  | { type: "finish" };

const initialState: ChatState = {
  messages: [],
  answerDraft: "",
  timeline: [],
  evidence: [],
  traceId: null,
  isStreaming: false,
  error: null,
};

function nodeLabel(node: string): string {
  const labels: Record<string, string> = {
    orchestrator: "เลือกเส้นทางการตอบ",
    agent: "ให้โมเดลวิเคราะห์และร่างคำตอบ",
    preflight: "ตรวจว่าคำถามต้องใช้ข้อมูลอะไร",
    tools: "เรียกใช้เครื่องมือค้นข้อมูล",
    validate: "ตรวจคำตอบกับหลักฐาน",
    chat_agent: "ตอบแบบสนทนาทั่วไป",
  };
  return labels[node] ?? `ทำขั้นตอน ${node.replaceAll("_", " ")}`;
}

function isHiddenNode(node: string): boolean {
  return ["get_history", "save_history"].includes(node);
}

function toolLabel(tool: string): string {
  const labels: Record<string, string> = {
    query_financials: "ดึงตัวเลขการเงินจากฐานข้อมูล",
    search_filings: "ค้นหาเอกสารรายงานประจำปี",
  };
  return labels[tool] ?? `ใช้เครื่องมือ ${tool.replaceAll("_", " ")}`;
}

function readableList(value: unknown): string {
  if (!Array.isArray(value)) return "";
  return value.filter((item) => typeof item === "string" || typeof item === "number").join(", ");
}

function toolDetail(tool: string, args: Record<string, unknown>): string {
  if (tool === "query_financials") {
    const companies = readableList(args.companies);
    const years = readableList(args.years);
    const metric = compactValue(args.metric).replaceAll("_", " ");
    return [companies ? `บริษัท ${companies}` : "", metric ? `ตัวชี้วัด ${metric}` : "", years ? `ปี ${years}` : ""]
      .filter(Boolean)
      .join(" · ");
  }

  if (tool === "search_filings") {
    const companies = readableList(args.companies);
    const query = compactValue(args.query);
    return [query ? `ค้นหาเรื่อง "${query}"` : "", companies ? `ในเอกสารของ ${companies}` : ""].filter(Boolean).join(" · ");
  }

  return "";
}

function humanizeReasoning(text: string): string {
  return text
    .replaceAll("_", " ")
    .replace(/search filings\(query='([^']+)', companies=\[([^\]]+)\], top k=(\d+)\)/gi, (_match, query, companies) => {
      return `ค้นหาเอกสารเรื่อง "${query}" สำหรับ ${companies.replaceAll("'", "")}`;
    })
    .replace(/query financials\(([^)]+)\)/gi, "ดึงตัวเลขการเงินที่เกี่ยวข้อง")
    .replace(/; /g, " และ ");
}

function eventLabel(event: StreamEvent): TimelineStep | null {
  switch (event.type) {
    case "run.started":
      return null;
    case "node.finished":
      if (isHiddenNode(event.node)) return null;
      return { seq: event.seq, kind: event.type, label: nodeLabel(event.node) };
    case "reasoning.delta": {
      const phaseLabels: Record<string, string> = {
        preflight: "ตรวจว่าคำถามต้องใช้ข้อมูลอะไร",
        orchestrator: "เลือกเส้นทางการตอบ",
        chat: "ตอบแบบสนทนาทั่วไป",
        agent: "ให้โมเดลวิเคราะห์และร่างคำตอบ",
      };
      return {
        seq: event.seq,
        kind: event.type,
        label: phaseLabels[event.phase ?? "agent"] ?? "ให้โมเดลวิเคราะห์และร่างคำตอบ",
        detail: humanizeReasoning(event.text),
      };
    }
    case "tool.selected":
      return { seq: event.seq, kind: event.type, label: toolLabel(event.tool), detail: toolDetail(event.tool, event.args) };
    case "sql.query":
      return { seq: event.seq, kind: event.type, label: "อ่านข้อมูลจากฐานข้อมูลแล้ว", detail: `พบ ${event.row_count} แถว` };
    case "vector.search":
      return { seq: event.seq, kind: event.type, label: "ค้นหาเอกสารประกอบ", detail: event.query };
    case "coverage.notice":
      return { seq: event.seq, kind: event.type, label: "พบข้อจำกัดของข้อมูล", detail: event.message };
    case "validation":
      return {
        seq: event.seq,
        kind: event.type,
        label: event.grounded ? "ตรวจแล้วว่าคำตอบมีหลักฐานรองรับ" : "พบข้อความที่ยังไม่มีหลักฐานพอ",
        detail: event.unsupported_claims.length ? event.unsupported_claims.join("; ") : "ไม่พบ claim ที่ขาดหลักฐาน",
      };
    case "run.finished":
      return null;
    case "error":
      return { seq: event.seq, kind: event.type, label: "เกิดข้อผิดพลาด", detail: event.message };
    default:
      return null;
  }
}

function reducer(state: ChatState, action: Action): ChatState {
  if (action.type === "load") {
    return action.state;
  }
  if (action.type === "start") {
    return {
      ...state,
      messages: [...state.messages, { role: "user", content: action.question }],
      answerDraft: "",
      timeline: [],
      evidence: [],
      traceId: null,
      isStreaming: true,
      error: null,
    };
  }
  if (action.type === "error") {
    return { ...state, isStreaming: false, error: action.message };
  }
  if (action.type === "finish") {
    const answer = state.answerDraft.trim();
    return {
      ...state,
      isStreaming: false,
      messages: answer ? [...state.messages, { role: "assistant", content: answer }] : state.messages,
      answerDraft: "",
    };
  }

  const event = action.event;
  const step = eventLabel(event);
  const existingStep = step ? state.timeline.find((item) => item.label === step.label) : undefined;
  const timeline =
    step && existingStep
      ? step.detail
        ? state.timeline.map((item) => (item.label === step.label ? { ...item, detail: step.detail } : item))
        : state.timeline
      : step
        ? [...state.timeline, step]
        : state.timeline;

  if (event.type === "run.started") {
    return { ...state, timeline, traceId: event.trace_id };
  }
  if (event.type === "answer.delta") {
    return { ...state, timeline, answerDraft: state.answerDraft + event.text };
  }
  if (event.type === "evidence") {
    return { ...state, timeline, evidence: [...state.evidence, { source: event.source, items: event.items }] };
  }
  if (event.type === "error") {
    return { ...state, timeline, isStreaming: false, error: event.message };
  }
  return { ...state, timeline };
}

function compactValue(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString("en-US") : value.toFixed(3);
  return String(value);
}

function columnLabel(key: string): string {
  const labels: Record<string, string> = {
    company: "Company",
    doc: "Document",
    gross_profit: "Gross profit",
    net_income: "Net income",
    operating_income: "Op. income",
    page: "Page",
    revenue: "Revenue",
    score: "Score",
    ticker: "Ticker",
    value: "Value",
    year: "Year",
  };
  return labels[key] ?? key.replaceAll("_", " ");
}

function sourceLabel(source: string): string {
  if (source === "sql") return "SQL reference";
  if (source === "vector") return "10-K reference";
  return `${source} reference`;
}

function referenceTitle(source: string, count: number): string {
  if (source === "sql") return `SQL reference (${count} rows)`;
  if (source === "vector") return `10-K reference (${count} chunks)`;
  return `${sourceLabel(source)} (${count} items)`;
}

function previewText(text: unknown): string {
  const value = String(text ?? "").replace(/\s+/g, " ").trim();
  return value.length > 420 ? `${value.slice(0, 420)}...` : value;
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text))) {
    if (match.index > cursor) {
      nodes.push(text.slice(cursor, match.index));
    }

    const token = match[0];
    if (token.startsWith("**")) {
      nodes.push(<strong key={`${match.index}-strong`}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      nodes.push(<code key={`${match.index}-code`}>{token.slice(1, -1)}</code>);
    } else {
      const labelEnd = token.indexOf("](");
      const label = token.slice(1, labelEnd);
      const href = token.slice(labelEnd + 2, -1);
      nodes.push(
        <a href={href} key={`${match.index}-link`} rel="noreferrer" target="_blank">
          {label}
        </a>,
      );
    }
    cursor = match.index + token.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes;
}

function renderHeading(level: number, content: ReactNode[], key: string): ReactNode {
  if (level === 1) return <h1 key={key}>{content}</h1>;
  if (level === 2) return <h2 key={key}>{content}</h2>;
  if (level === 3) return <h3 key={key}>{content}</h3>;
  if (level === 4) return <h4 key={key}>{content}</h4>;
  if (level === 5) return <h5 key={key}>{content}</h5>;
  return <h6 key={key}>{content}</h6>;
}

function MarkdownMessage({ content }: { content: string }) {
  const blocks: ReactNode[] = [];
  const lines = content.split("\n");
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    if (line.trim().startsWith("```")) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      blocks.push(
        <pre key={`code-${index}`}>
          <code>{codeLines.join("\n")}</code>
        </pre>,
      );
      index += 1;
      continue;
    }

    if (/^#{1,6}\s+/.test(line)) {
      const level = Math.min(line.match(/^#+/)?.[0].length ?? 3, 6);
      const text = line.replace(/^#{1,6}\s+/, "");
      blocks.push(renderHeading(level, renderInlineMarkdown(text), `h-${index}`));
      index += 1;
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        items.push(<li key={`ol-${index}`}>{renderInlineMarkdown(lines[index].replace(/^\s*\d+\.\s+/, ""))}</li>);
        index += 1;
      }
      blocks.push(<ol key={`ol-${index}`}>{items}</ol>);
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        items.push(<li key={`ul-${index}`}>{renderInlineMarkdown(lines[index].replace(/^\s*[-*]\s+/, ""))}</li>);
        index += 1;
      }
      blocks.push(<ul key={`ul-${index}`}>{items}</ul>);
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].trim().startsWith("```") &&
      !/^\s*\d+\.\s+/.test(lines[index]) &&
      !/^\s*[-*]\s+/.test(lines[index])
    ) {
      paragraphLines.push(lines[index]);
      index += 1;
    }
    blocks.push(<p key={`p-${index}`}>{renderInlineMarkdown(paragraphLines.join(" "))}</p>);
  }

  return <div className="markdownMessage">{blocks}</div>;
}

function ThoughtDisclosure({
  isLive,
  timeline,
}: {
  isLive: boolean;
  timeline: TimelineStep[];
}) {
  if (!timeline.length) return null;

  return (
    <details className="thoughtDisclosure" open={isLive ? true : undefined}>
      <summary>
        <span>Thought process</span>
        <span className="chevron">›</span>
      </summary>
      <div className="thoughtBody">
        {timeline.length ? (
          <ol>
            {timeline.map((step) => (
              <li data-kind={step.kind} key={`${step.seq}-${step.kind}`}>
                <strong>{step.label}</strong>
                {step.detail ? <p>{step.detail}</p> : null}
              </li>
            ))}
          </ol>
        ) : null}
      </div>
    </details>
  );
}

function SqlReferenceTable({ items }: { items: EvidenceItem[] }) {
  const preferredColumns = ["company", "ticker", "year", "revenue", "net_income", "operating_income", "gross_profit", "value"];
  const columns = preferredColumns.filter((column) => items.some((item) => Object.prototype.hasOwnProperty.call(item, column)));

  return (
    <div className="referenceTableWrap">
      <table className="referenceTable">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{columnLabel(column)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.slice(0, 12).map((item, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{displayValue(item[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VectorReferenceCards({ items }: { items: EvidenceItem[] }) {
  return (
    <div className="referenceCards">
      {items.slice(0, 8).map((item, index) => (
        <article className="referenceCard" key={index}>
          <strong>
            [{displayValue(item.company)} p.{displayValue(item.page)}]
            {typeof item.score === "number" ? ` · score ${item.score.toFixed(3)}` : ""}
          </strong>
          <p>{previewText(item.text)}</p>
          {item.doc ? <span>{displayValue(item.doc)}</span> : null}
        </article>
      ))}
    </div>
  );
}

function GenericReferenceList({ items }: { items: EvidenceItem[] }) {
  return (
    <div className="referenceCards">
      {items.slice(0, 8).map((item, index) => (
        <article className="referenceCard" key={index}>
          <p>{previewText(compactValue(item))}</p>
        </article>
      ))}
    </div>
  );
}

function ReferenceDisclosure({ evidence }: { evidence: { source: string; items: EvidenceItem[] }[] }) {
  const populatedEvidence = evidence.filter((block) => block.items.length);
  if (!populatedEvidence.length) return null;

  const count = populatedEvidence.reduce((total, block) => total + block.items.length, 0);

  return (
    <details className="referenceDisclosure">
      <summary>
        <span>Reference</span>
        <span>{count} items</span>
      </summary>
      <div className="referenceBody">
        {populatedEvidence.map((block, index) => (
          <section className="referenceSection" key={`${block.source}-${index}`}>
            <h3>{referenceTitle(block.source, block.items.length)}</h3>
            {block.source === "sql" ? <SqlReferenceTable items={block.items} /> : null}
            {block.source === "vector" ? <VectorReferenceCards items={block.items} /> : null}
            {block.source !== "sql" && block.source !== "vector" ? <GenericReferenceList items={block.items} /> : null}
          </section>
        ))}
      </div>
    </details>
  );
}

function blankSection(): ChatSection {
  const now = Date.now();
  return {
    id: String(now),
    title: "New section",
    updatedAt: now,
    state: initialState,
  };
}

function stateTitle(state: ChatState): string {
  const firstUserMessage = state.messages.find((message) => message.role === "user");
  if (!firstUserMessage) return "New section";
  return firstUserMessage.content.length > 54 ? `${firstUserMessage.content.slice(0, 54)}...` : firstUserMessage.content;
}

function safeSections(): ChatSection[] {
  try {
    const stored = localStorage.getItem(SECTIONS_KEY);
    if (!stored) return [];
    const parsed = JSON.parse(stored) as ChatSection[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export default function Home() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [email, setEmail] = useState("demo@example.com");
  const [password, setPassword] = useState("demo1234");
  const [token, setToken] = useState("");
  const [question, setQuestion] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);
  const [sections, setSections] = useState<ChatSection[]>([]);
  const [activeSectionId, setActiveSectionId] = useState("");
  const [view, setView] = useState<"chat" | "settings">("chat");
  const [searchQuery, setSearchQuery] = useState("");
  const [hydrated, setHydrated] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    setToken(localStorage.getItem(TOKEN_KEY) ?? "");
  }, []);

  useEffect(() => {
    if (!token) return;
    const storedSections = safeSections();
    const nextSections = storedSections.length ? storedSections : [blankSection()];
    setSections(nextSections);
    setActiveSectionId(nextSections[0].id);
    dispatch({ type: "load", state: nextSections[0].state });
    setHydrated(true);
  }, [token]);

  useEffect(() => {
    if (!hydrated || !activeSectionId) return;
    setSections((currentSections) => {
      const nextSections = currentSections
        .map((section) =>
          section.id === activeSectionId
            ? {
                ...section,
                title: stateTitle(state),
                updatedAt: Date.now(),
                state,
              }
            : section,
        )
        .sort((left, right) => right.updatedAt - left.updatedAt);
      localStorage.setItem(SECTIONS_KEY, JSON.stringify(nextSections.slice(0, 12)));
      return nextSections.slice(0, 12);
    });
  }, [activeSectionId, hydrated, state]);

  const visibleMessages = useMemo(() => {
    if (!state.answerDraft && state.isStreaming) {
      return [...state.messages, { role: "assistant" as const, content: "" }];
    }
    if (!state.answerDraft) return state.messages;
    return [...state.messages, { role: "assistant" as const, content: state.answerDraft }];
  }, [state.answerDraft, state.isStreaming, state.messages]);

  const activeSection = sections.find((section) => section.id === activeSectionId);
  const filteredSections = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return sections;
    return sections.filter((section) => section.title.toLowerCase().includes(query));
  }, [searchQuery, sections]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [question]);

  async function login(event: FormEvent) {
    event.preventDefault();
    setLoginError(null);
    const response = await fetch(`${API_BASE}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!response.ok) {
      setLoginError("Sign-in failed");
      return;
    }
    const body = (await response.json()) as { access_token: string };
    localStorage.setItem(TOKEN_KEY, body.access_token);
    setToken(body.access_token);
  }

  function newSection() {
    if (state.isStreaming) return;
    const section = blankSection();
    setSections((currentSections) => {
      const nextSections = [section, ...currentSections].slice(0, 12);
      localStorage.setItem(SECTIONS_KEY, JSON.stringify(nextSections));
      return nextSections;
    });
    setActiveSectionId(section.id);
    dispatch({ type: "load", state: section.state });
    setQuestion("");
    setView("chat");
  }

  function continueSection(section: ChatSection) {
    if (state.isStreaming) return;
    setActiveSectionId(section.id);
    dispatch({ type: "load", state: section.state });
    setView("chat");
  }

  function deleteSection(sectionId: string) {
    if (state.isStreaming) return;
    setSections((currentSections) => {
      const remainingSections = currentSections.filter((section) => section.id !== sectionId);
      const nextSections = remainingSections.length ? remainingSections : [blankSection()];
      const nextActiveSection = nextSections[0];
      localStorage.setItem(SECTIONS_KEY, JSON.stringify(nextSections));

      if (sectionId === activeSectionId) {
        setActiveSectionId(nextActiveSection.id);
        dispatch({ type: "load", state: nextActiveSection.state });
        setQuestion("");
      }

      return nextSections;
    });
  }

  async function ask(nextQuestion = question) {
    const trimmed = nextQuestion.trim();
    if (!trimmed || state.isStreaming) return;
    dispatch({ type: "start", question: trimmed });
    setQuestion("");
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/api/v1/chat/runs/stream`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: trimmed }] }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(`Request failed (${response.status})`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          dispatch({ type: "event", event: JSON.parse(line) as StreamEvent });
        }
      }
      if (buffer.trim()) {
        dispatch({ type: "event", event: JSON.parse(buffer) as StreamEvent });
      }
      dispatch({ type: "finish" });
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        dispatch({ type: "error", message: (error as Error).message });
      }
    } finally {
      abortRef.current = null;
    }
  }

  function signOut() {
    localStorage.removeItem(TOKEN_KEY);
    setToken("");
    setSections([]);
    setActiveSectionId("");
    setHydrated(false);
  }

  if (!token) {
    return (
      <main className="loginShell">
        <form className="loginPanel" onSubmit={login}>
          <div>
            <p className="eyebrow">Financial QA</p>
            <h1>Sign in</h1>
          </div>
          <label>
            Email
            <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" />
          </label>
          <label>
            Password
            <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" />
          </label>
          {loginError ? <p className="errorText">{loginError}</p> : null}
          <button type="submit">Sign in</button>
        </form>
      </main>
    );
  }

  return (
    <main className="appShell">
      <section className="workspace">
        <aside className="historyRail">
          <div className="brandBlock">
            <h1>Financial QA</h1>
            <button aria-label="Collapse sidebar" className="iconButton" type="button">
              ◐
            </button>
          </div>

          <div className="sidebarActions">
            <button className="sidebarAction" disabled={state.isStreaming} onClick={newSection} type="button">
              <span>✎</span> New chat
            </button>
            <label className="searchBox">
              <span>⌕</span>
              <input
                aria-label="Search chats"
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search chats"
                value={searchQuery}
              />
            </label>
          </div>

          <nav className="railNav" aria-label="Main views">
            <button className={view === "chat" ? "active" : ""} onClick={() => setView("chat")} type="button">
              Chats
            </button>
            <button className={view === "settings" ? "active" : ""} onClick={() => setView("settings")} type="button">
              Account
            </button>
          </nav>

          <div className="historyHeader">
            <h2>Chats</h2>
            <span>{sections.length}</span>
          </div>
          <div className="sectionList">
            {filteredSections.map((section) => (
              <div
                className={`sectionItem ${section.id === activeSectionId ? "active" : ""}`}
                key={section.id}
              >
                <button disabled={state.isStreaming} onClick={() => continueSection(section)} type="button">
                  <span>{section.title}</span>
                </button>
                <button
                  aria-label={`Delete ${section.title}`}
                  className="deleteSectionButton"
                  disabled={state.isStreaming}
                  onClick={() => deleteSection(section.id)}
                  type="button"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <div className="accountStrip">
            <div>
              <strong>{email}</strong>
              <span>Business</span>
            </div>
            <button aria-label="Account settings" className="iconButton" onClick={() => setView("settings")} type="button">
              ⚙
            </button>
          </div>
        </aside>

        {view === "settings" ? (
          <section className="settingsColumn">
            <div className="settingsHeader">
              <div>
                <p className="eyebrow">Account settings</p>
                <h2>Profile and connection</h2>
              </div>
              <button className="secondary" type="button" onClick={signOut}>
                Sign out
              </button>
            </div>
            <div className="settingsGrid">
              <div className="settingsField">
                <span>Email</span>
                <strong>{email}</strong>
              </div>
              <div className="settingsField">
                <span>API endpoint</span>
                <strong>{API_BASE}</strong>
              </div>
              <div className="settingsField">
                <span>Saved frontend sections</span>
                <strong>{sections.length}</strong>
              </div>
              <div className="settingsField">
                <span>Authentication</span>
                <strong>{token ? "Signed in" : "Signed out"}</strong>
              </div>
            </div>
          </section>
        ) : (
          <div className="chatColumn">
            <div className="transcript">
              {visibleMessages.length === 0 ? (
                <div className="emptyState">
                  <h2>{activeSection?.title ?? "Ask about financials"}</h2>
                  <p>Ask a question or continue a chat from the left sidebar.</p>
                </div>
              ) : (
                visibleMessages.map((message, index) => {
                  const isLatestAssistant = message.role === "assistant" && index === visibleMessages.length - 1;
                  return (
                    <article className={`messageRow ${message.role}`} key={`${message.role}-${index}`}>
                      <div className="messageContent">
                        {isLatestAssistant ? (
                          <ThoughtDisclosure isLive={state.isStreaming} timeline={state.timeline} />
                        ) : null}
                        {message.content.trim() ? <MarkdownMessage content={message.content} /> : null}
                        {isLatestAssistant ? <ReferenceDisclosure evidence={state.evidence} /> : null}
                      </div>
                    </article>
                  );
                })
              )}
            </div>

            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault();
                void ask();
              }}
            >
              <button aria-label="Add attachment" className="composerIcon" type="button">
                +
              </button>
              <textarea
                aria-label="Ask anything"
                onKeyDown={(event) => {
                  if (event.key !== "Enter" || event.shiftKey) return;
                  event.preventDefault();
                  void ask();
                }}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask anything"
                ref={textareaRef}
                rows={1}
                value={question}
              />
              <button className="composerIcon" disabled={state.isStreaming} type="submit">
                {state.isStreaming ? "…" : "↗"}
              </button>
              {state.error ? <p className="errorText">{state.error}</p> : null}
            </form>
          </div>
        )}
      </section>
    </main>
  );
}
