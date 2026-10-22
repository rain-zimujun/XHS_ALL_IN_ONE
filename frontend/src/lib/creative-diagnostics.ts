type Failure = { attempt: number; duration_ms: number; status: string; error: string; quote_diagnostic?: unknown };
type NetworkFailure = { attempt: number; max_retries: number; duration_ms: number; error: string;
  error_code: string; will_retry: boolean; wait_seconds: number; role?: string };

export function readNetworkAttempts(metadata: unknown): NetworkFailure[] {
  if (!metadata || typeof metadata !== "object") return [];
  const attempts = (metadata as Record<string, unknown>).network_attempts;
  if (!Array.isArray(attempts)) return [];
  return attempts.filter((value): value is NetworkFailure => Boolean(value &&
    typeof value.attempt === "number" && Number.isFinite(value.attempt) &&
    typeof value.max_retries === "number" && Number.isFinite(value.max_retries) &&
    typeof value.duration_ms === "number" && Number.isFinite(value.duration_ms) &&
    typeof value.wait_seconds === "number" && Number.isFinite(value.wait_seconds) &&
    typeof value.will_retry === "boolean" && typeof value.error_code === "string" && typeof value.error === "string"));
}

export function readFailedAttempts(metadata: unknown): Failure[] {
  if (!metadata || typeof metadata !== "object") return [];
  const attempts = (metadata as Record<string, unknown>).failed_attempts;
  if (!Array.isArray(attempts)) return [];
  return attempts.filter((value): value is Failure => Boolean(value &&
    typeof value.attempt === "number" && Number.isFinite(value.attempt) &&
    typeof value.duration_ms === "number" && Number.isFinite(value.duration_ms) &&
    typeof value.status === "string" && typeof value.error === "string"));
}

export function formatQuoteDiagnostic(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const detail = value as Record<string, unknown>;
  if (typeof detail.validated_quote !== "string" || !Array.isArray(detail.source_candidates)) return null;
  const lines = [
    `输入样本序号：${detail.source_position}`,
    `引用字段路径：${JSON.stringify(detail.quote_path)}`,
    `模型返回原句：${JSON.stringify(detail.model_quote)}`,
    `模型原句是否截断：${detail.model_quote_truncated ? "是" : "否"}`,
    `校验实际使用的句子：${JSON.stringify(detail.validated_quote)}`,
    `校验前后是否有变化：${detail.model_quote_changed_by_validation ? "是" : "否"}`,
  ];
  detail.source_candidates.forEach((value, index) => {
    if (!value || typeof value !== "object") return;
    const candidate = value as Record<string, unknown>;
    lines.push(
      `\n原文对照 ${index + 1}：${candidate.field === "title" ? "标题" : "正文片段"}（${candidate.allowed ? "允许引用的字段" : "其他字段，仅辅助排查"}）`,
      `原文窗口 [${candidate.window_start}, ${candidate.window_end})：${JSON.stringify(candidate.source_window)}`,
      `逐字匹配：${candidate.exact_match ? "是" : "否"}；忽略空白后匹配：${candidate.whitespace_match ? "是" : "否"}；兼容字符归一化并忽略空白后匹配：${candidate.nfkc_whitespace_match ? "是" : "否"}`,
    );
  });
  lines.push(`\n以下差异基于对照 ${Number(detail.closest_candidate_index) + 1} 的近似窗口，仅辅助定位，不证明模型原本想引用这段。`,
    "operation 表示从模型引句到原文的编辑操作；字符位置从 0 开始，end 不包含末尾。",
    JSON.stringify(detail.differences, null, 2),
    `差异列表是否截断：${detail.differences_truncated ? "是" : "否"}`);
  return lines.join("\n");
}
