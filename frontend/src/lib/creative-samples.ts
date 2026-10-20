import type { XhsSearchNote } from "../types";
import type { CreativeRun } from "../types/creative";

function publicationDate(timestamp: XhsSearchNote["timestamp"]): Date | undefined {
  const numeric = Number(timestamp);
  if (!Number.isFinite(numeric) || numeric <= 0) return undefined;
  const date = new Date(numeric > 1e12 ? numeric : numeric * 1000);
  return Number.isFinite(date.getTime()) ? date : undefined;
}

export function creativeSamples(search: CreativeRun | null, collection: CreativeRun | null): XhsSearchNote[] {
  const details = new Map((collection?.payload.sources || []).map(note => [note.note_id, note]));
  const candidates = search?.payload.items || collection?.payload.selected_notes || collection?.payload.sources || [];
  return candidates.map(note => {
    const detail = details.get(note.note_id);
    if (!detail) return note;
    // Retain search order and metrics; only supplement the actual publication time.
    return {
      ...note, captured_at: detail.captured_at,
      timestamp: publicationDate(detail.timestamp) ? detail.timestamp : note.timestamp,
    };
  });
}

export function formatPublicationTime(note: XhsSearchNote): string {
  const date = publicationDate(note.timestamp);
  return date ? date.toLocaleDateString("zh-CN", { timeZone: "Asia/Shanghai" }) : note.captured_at ? "详情未提供时间" : "待采集补全";
}
