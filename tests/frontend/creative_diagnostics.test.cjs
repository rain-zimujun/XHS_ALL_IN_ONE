const assert = require("node:assert/strict");
const { test } = require("node:test");
const { formatQuoteDiagnostic, readFailedAttempts, readNetworkAttempts } = require("../../frontend/src/lib/creative-diagnostics.ts");

test("读取新旧流程保存的失败原因、次数和耗时", () => {
  const attempt = { attempt: 2, duration_ms: 56791, status: "creative_handoff_invalid", error: "observations[1].quote 逐字匹配失败" };
  assert.deepEqual(readFailedAttempts({ failed_attempts: [attempt] }), [attempt]);
});

test("缺失与畸形历史诊断兼容", () => {
  for (const metadata of [null, {}, [], { failed_attempts: {} }, { failed_attempts: [null, {}, { attempt: NaN, duration_ms: 1, status: "failed", error: "原因" }] }]) {
    assert.deepEqual(readFailedAttempts(metadata), []);
  }
});

test("网络重试记录和结构修复分开显示，兼容无重试信息的旧任务", () => {
  const failure = { attempt: 4, max_retries: 3, duration_ms: 132000, error: "HTTP 524", error_code: "ai_http_error", will_retry: false, wait_seconds: 0 };
  assert.deepEqual(readNetworkAttempts({ network_attempts: [failure] }), [failure]);
  for (const metadata of [null, {}, { network_attempts: [null, {}, { ...failure, duration_ms: NaN }] }]) {
    assert.deepEqual(readNetworkAttempts(metadata), []);
  }
});

test("引用诊断显示模型原句、原文窗口和转义换行，保留旧任务兼容", () => {
  for (const detail of [null, undefined, {}, "旧数据"]) assert.equal(formatQuoteDiagnostic(detail), null);
  const detail = { source_position: 2, quote_path: ["observations", 1, "quote"], model_quote: "原句\n有空白",
    validated_quote: "原句\n有空白", source_candidates: [{ field: "body", allowed: true,
      window_start: 0, window_end: 6, source_window: "原句有空白", exact_match: false, whitespace_match: true }],
    closest_candidate_index: 0, differences: [] };
  const result = formatQuoteDiagnostic(detail);
  assert.match(result, /模型返回原句："原句\\n有空白"/);
  assert.match(result, /忽略空白后匹配：是/);
  assert.match(result, /仅辅助定位/);
  const attempt = { attempt: 1, duration_ms: 123, status: "creative_handoff_invalid", error: "引用失败", quote_diagnostic: detail };
  assert.deepEqual(readFailedAttempts({ failed_attempts: [attempt] })[0].quote_diagnostic, detail);
});
