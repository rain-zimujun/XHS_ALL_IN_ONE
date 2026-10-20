import type { TaskRecord, XhsSearchNote } from "./index";

export type CreativeSearch = {
  account_id: number;
  keyword: string;
  limit: 20 | 30 | 50;
  sort_type_choice: number;
  note_time: number;
  note_type: number;
};

export type CreativeRun = Omit<TaskRecord, "payload"> & {
  payload: {
    search: CreativeSearch;
    items?: XhsSearchNote[];
    warnings?: string[];
    error?: string;
    stage?: string;
    sources?: XhsSearchNote[];
    failures?: { note_id: string; title: string; reason: string }[];
    analysis?: CreativeAnalysisOptions | null;
    result?: CreativeResult;
    report_files?: { file_name: string; download_url: string }[];
    report_directory?: string;
    export_error?: string | null;
    analysis_metadata?: { model_name: string; sample_count: number; excerpt_chars_per_note: number; disclaimer: string };
  };
};

export type CreativeAnalysisOptions = {
  model_id: number;
  idea_count: 1 | 3 | 5;
  audience: string;
  instruction: string;
};

export type CreativeResult = {
  summary: string;
  patterns: string[];
  ideas: {
    title: string;
    alternative_titles: string[];
    body: string;
    tags: string[];
    reason: string;
    source_note_ids: string[];
    cover_text: string;
    image_plan: { purpose: string; composition: string; prompt: string; negative_prompt: string }[];
  }[];
};

export type CreativeHistoryItem = {
  id: number;
  task_type: string;
  status: string;
  created_at: string;
  keyword: string;
};
