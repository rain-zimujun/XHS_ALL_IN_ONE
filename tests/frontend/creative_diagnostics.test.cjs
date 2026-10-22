const assert = require("node:assert/strict");
const { test } = require("node:test");
const { readFailedAttempts } = require("../../frontend/src/lib/creative-diagnostics.ts");

test("读取新旧流程保存的失败原因、次数和耗时", () => {
  const attempt = { attempt: 2, duration_ms: 56791, status: "creative_handoff_invalid", error: "observations[1].quote 逐字匹配失败" };
  assert.deepEqual(readFailedAttempts({ failed_attempts: [attempt] }), [attempt]);
});

test("缺失与畸形历史诊断兼容", () => {
  for (const metadata of [null, {}, [], { failed_attempts: {} }, { failed_attempts: [null, {}, { attempt: NaN, duration_ms: 1, status: "failed", error: "原因" }] }]) {
    assert.deepEqual(readFailedAttempts(metadata), []);
  }
});
