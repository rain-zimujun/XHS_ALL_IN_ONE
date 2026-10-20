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
  };
};

export type CreativeHistoryItem = {
  id: number;
  task_type: string;
  status: string;
  created_at: string;
  keyword: string;
};
