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
    selected_notes?: XhsSearchNote[];
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
    analysis_metadata?: { model_name: string; sample_count: number; excerpt_chars_per_note: number; excerpt_strategy?: "head_and_tail"; disclaimer: string };
  };
};

export type CreativeAnalysisOptions = {
  model_id: number;
  idea_count: 1 | 3 | 5;
  variant_count?: 2 | 3;
  audience: string;
  instruction: string;
};

export type CreativeImagePlan = { purpose: string; composition: string; prompt: string; negative_prompt: string };

export type CreativeVariant = {
  label: string;
  body: string;
  tags: string[];
  cover_text: string;
  image_plan: CreativeImagePlan[];
};

type CreativeIdeaBase = {
  title: string;
  reason: string;
  source_note_ids: string[];
};

// Old saved results keep their original single body. Never fabricate extra variants.
export type CreativeIdea = CreativeIdeaBase & (
  { variants: CreativeVariant[]; alternative_titles?: never } |
  (Omit<CreativeVariant, "label"> & { variants?: never; alternative_titles: string[] })
);

export type CreativeResult = {
  summary: string;
  patterns: string[];
  style_summary?: string;
  ideas: CreativeIdea[];
};

export type CreativeHistoryItem = {
  id: number;
  task_type: string;
  status: string;
  created_at: string;
  keyword: string;
};
