import type { CreativePipeline, CreativeQualityReview, CreativeReviewDecision, CreativeReviewIssue } from "../types/creative";

const record = (value: unknown): Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const strings = (value: unknown): string[] => Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
const text = (value: unknown): string => typeof value === "string" ? value : "";
const isScore = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 10;

// Read-only display adapter: topic uses idea_id, title wraps review, other judges use item_id.
// Missing or malformed old records must never become fabricated passes or match by array index.
export function readReview(value: unknown) {
  const outer = record(value);
  const review = outer.review ? record(outer.review) : outer;
  const decisions: CreativeReviewDecision[] = [];
  for (const entry of Array.isArray(review.reviews) ? review.reviews : []) {
    const decision = record(entry), judgement = record(decision.judgement), quality = record(decision.quality);
    const itemId = text(judgement.item_id) || text(judgement.idea_id);
    if (!itemId || !isScore(quality.overall_score) || typeof quality.passed !== "boolean" || !["topic", "title", "copy", "final"].includes(text(quality.kind))) continue;
    const scores = (Array.isArray(quality.scores) ? quality.scores : []).flatMap(value => {
      const score = record(value);
      return isScore(score.score) && typeof score.criterion === "string" ? [{ criterion: score.criterion, score: score.score, comment: text(score.comment) }] : [];
    });
    const issues: CreativeReviewIssue[] = (Array.isArray(judgement.issues) ? judgement.issues : []).flatMap(value => {
      const issue = record(value);
      if (typeof issue.problem !== "string" || !["blocking", "suggestion"].includes(text(issue.severity))) return [];
      return [{ criterion: text(issue.criterion), severity: issue.severity as CreativeReviewIssue["severity"], field: text(issue.field), quote: text(issue.quote),
        problem: issue.problem, revision_instruction: text(issue.revision_instruction),
        responsible_stage: ["topic", "title", "copy", "image"].includes(text(issue.responsible_stage)) ? issue.responsible_stage as CreativeReviewIssue["responsible_stage"] : null }];
    });
    decisions.push({ itemId, issues, preserve: strings(judgement.preserve), comparison: text(judgement.comparison),
      quality: { kind: quality.kind as CreativeQualityReview["kind"], scores, overall_score: quality.overall_score, passed: quality.passed,
        blocking_issues: strings(quality.blocking_issues), revision_instructions: strings(quality.revision_instructions), reviewer_notes: text(quality.reviewer_notes) } });
  }
  return { summary: text(review.summary), limitations: strings(review.limitations), decisions };
}

export function currentDecision(pipeline: CreativePipeline | undefined, stage: string, itemId: string | undefined) {
  const state = pipeline?.stages[stage];
  // Historical/invalidated assessments belong in the audit panel, not beside a current finished draft.
  if (!itemId || state?.status !== "completed") return undefined;
  return readReview(state.review_result).decisions.find(item => item.itemId === itemId);
}

export const criterionNames: Record<string, string> = {
  reader_value: "读者价值", specificity: "具体程度", fit: "搜索与受众匹配", novelty: "差异化", evidence: "样本依据",
  hook: "点击吸引力", reader_benefit: "读者收益", relevance: "关键词与受众匹配", difference: "标题差异化", naturalness: "自然度",
  opening: "前两句吸引力", value: "具体信息与价值", audience_fit: "目标读者匹配", platform_voice: "小红书语气与可读性", save_value: "收藏/转发价值", publishability: "直接发布可用度",
  title_body_fit: "标题与正文一致性", promise_support: "标题承诺兑现", image_fit: "正文与配图匹配", tag_quality: "标签相关性与去重", safety: "事实、原创与素材边界",
};

export const reviewFieldNames: Record<string, string> = {
  topic: "选题", title: "标题", body: "正文", tags: "话题标签", cover_text: "封面文字", image_plan: "配图方案", material_notes: "素材说明",
  source_note_ids: "参考来源", core_angle: "核心切入点", reader_promise: "读者承诺", must_cover: "必须覆盖要点", boundaries: "事实边界",
};
