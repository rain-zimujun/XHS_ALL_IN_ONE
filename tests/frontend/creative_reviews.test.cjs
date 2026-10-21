const assert = require("node:assert/strict");
const { test } = require("node:test");
const { readReview, currentDecision } = require("../../frontend/src/lib/creative-reviews.ts");

const decision = (id, kind = "copy") => ({
  judgement: { [kind === "topic" ? "idea_id" : "item_id"]: id, issues: [], preserve: ["保留细节"], comparison: "不同角度" },
  quality: { kind, overall_score: 8, passed: true, scores: [{ criterion: "value", score: 8, comment: "具体" }], blocking_issues: [], revision_instructions: [] },
});

test("适配选题、嵌套标题与正文/终审结构", () => {
  for (const kind of ["topic", "title", "copy", "final"]) {
    const batch = { summary: "摘要", limitations: ["仅参考"], reviews: [decision("item_1", kind)] };
    const actual = readReview(kind === "title" ? { review: batch } : batch);
    assert.equal(actual.decisions[0].itemId, "item_1");
    assert.equal(actual.decisions[0].quality.kind, kind);
    assert.deepEqual(actual.limitations, ["仅参考"]);
  }
});
test("缺失与畸形旧数据不抛异常、不伪造通过", () => {
  for (const value of [null, {}, [], { review: {} }, { reviews: [null, {}, { quality: { overall_score: "8", passed: "true" } }] }]) {
    assert.deepEqual(readReview(value).decisions, []);
  }
  const bad = decision("x");
  bad.quality.overall_score = NaN;
  assert.equal(readReview({ reviews: [bad] }).decisions.length, 0);
});
test("按稳定 ID 匹配，未完成/失效阶段不关联当前稿件", () => {
  const pipeline = { stages: { copy: { status: "completed", review_result: { reviews: [decision("copy_b"), decision("copy_a")] } } } };
  assert.equal(currentDecision(pipeline, "copy", "copy_a").itemId, "copy_a");
  assert.equal(currentDecision(pipeline, "copy", "missing"), undefined);
  assert.equal(currentDecision(pipeline, "copy", undefined), undefined);
  for (const status of ["invalidated", "running", "failed", "cancelled", "revision_required"]) {
    pipeline.stages.copy.status = status;
    assert.equal(currentDecision(pipeline, "copy", "copy_a"), undefined);
  }
});
test("高分阻断照实显示，问题/修改建议不并入正文", () => {
  const entry = decision("copy_a", "final");
  entry.quality.passed = false;
  entry.judgement.issues = [{ severity: "blocking", field: "body", quote: "原文", problem: "缺少依据", revision_instruction: "删去无依据数据", responsible_stage: "copy" }];
  const snapshot = structuredClone(entry);
  const actual = readReview({ reviews: [entry] }).decisions[0];
  assert.equal(actual.quality.passed, false);
  assert.equal(actual.issues[0].responsible_stage, "copy");
  assert.equal(actual.issues[0].revision_instruction, "删去无依据数据");
  assert.deepEqual(entry, snapshot);
});
