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
  resume_available?: boolean;
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
    pipeline?: CreativePipeline;
    result?: CreativeResult;
    report_files?: { file_name: string; download_url: string }[];
    report_directory?: string;
    export_error?: string | null;
    analysis_metadata?: { model_name: string; sample_count: number; excerpt_chars_per_note: number; excerpt_strategy?: "head_and_tail"; disclaimer: string };
  };
};

export type CreativeAnalysisOptions = {
  model_id: number;
  preprocess_model_id?: number;
  workflow?: "legacy" | "multi_agent" | "langgraph";
  idea_count: 1 | 3 | 5;
  variant_count?: 2 | 3;
  title_max_length?: number;
  audience: string;
  instruction: string;
};

export type CreativeStageStatus = "pending" | "running" | "completed" | "failed" | "skipped" | "revision_required" | "invalidated" | "cancelled";
export type CreativePipeline = {
  schema_version: number;
  status: "pending" | "running" | "completed" | "failed" | "cancelled" | "revision_required";
  current_stage: string | null;
  activity: string;
  request_count: number;
  max_requests: number;
  max_quality_revisions: number;
  error_code?: string | null;
  stages: Record<string, {
    stage: string;
    status: CreativeStageStatus;
    revision: number;
    quality_revisions: number;
    attempts: { attempt: number; status: string; started_at: string; finished_at?: string | null; error?: string | null }[];
    output: Record<string, unknown> | null;
    review_result: Record<string, unknown> | null;
    history: { role: string; revision: number; output: Record<string, unknown>; target_ids: string[]; finished_at: string }[];
    error?: string | null;
    metadata?: Record<string, unknown>;
  }>;
};

export type CreativeReviewIssue = {
  criterion: string;
  severity: "blocking" | "suggestion";
  field: string;
  quote: string;
  problem: string;
  revision_instruction: string;
  responsible_stage?: "topic" | "title" | "copy" | "image" | null;
};

export type CreativeReviewDecision = {
  itemId: string;
  issues: CreativeReviewIssue[];
  preserve: string[];
  comparison: string;
  quality: CreativeQualityReview;
};

export type CreativeImagePlan = { purpose: string; composition: string; prompt: string; negative_prompt: string };

export type CreativeVariant = {
  variant_id?: string;
  label: string;
  body: string;
  tags: string[];
  cover_text: string;
  image_plan: CreativeImagePlan[];
  quality?: { copy?: CreativeQualityReview; final?: CreativeQualityReview };
};

export type CreativeQualityReview = {
  kind: "topic" | "title" | "copy" | "final";
  scores: { criterion: string; score: number; comment: string }[];
  overall_score: number;
  passed: boolean;
  blocking_issues: string[];
  revision_instructions: string[];
  reviewer_notes?: string;
};

type CreativeIdeaBase = {
  idea_id?: string;
  candidate_id?: string;
  title: string;
  reason: string;
  source_note_ids: string[];
  quality?: { title?: CreativeQualityReview };
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

export type CreativeHistoryPage = {
  items: CreativeHistoryItem[];
  page: number;
  page_size: number;
  total: number;
  has_more: boolean;
};
