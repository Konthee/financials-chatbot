export type EvidenceItem = Record<string, unknown>;

export type StreamEvent =
  | { type: "run.started"; seq: number; run_id: string; trace_id: string | null; model: string }
  | { type: "node.finished"; seq: number; node: string }
  | { type: "reasoning.delta"; seq: number; text: string; phase?: string }
  | { type: "tool.selected"; seq: number; tool: string; args: Record<string, unknown> }
  | { type: "sql.query"; seq: number; sql: string; params: unknown; row_count: number }
  | { type: "vector.search"; seq: number; query: string; top_k: number; pool_k?: number }
  | { type: "evidence"; seq: number; source: "sql" | "vector"; items: EvidenceItem[] }
  | { type: "coverage.notice"; seq: number; message: string; missing: string[] }
  | { type: "answer.delta"; seq: number; text: string }
  | { type: "validation"; seq: number; grounded: boolean; unsupported_claims: string[]; action?: string }
  | {
      type: "run.finished";
      seq: number;
      run_id: string;
      trace_id: string | null;
      usage: Record<string, number>;
      finish_reason: string;
    }
  | { type: "error"; seq: number; message: string };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface TimelineStep {
  seq: number;
  kind: string;
  label: string;
  detail?: string;
}
