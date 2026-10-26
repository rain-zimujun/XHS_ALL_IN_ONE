// Isolated Chrome with every API mocked; never uses the user's browser session.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

async function main() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(() => localStorage.setItem("spider_xhs_refresh_token", "mock-recovery"));
    const pipeline = {
      schema_version: 2, status: "failed", current_stage: "title", activity: "模型服务错误（HTTP 524）",
      error_code: "ai_http_error", request_count: 9, max_requests: 36, max_quality_revisions: 2,
      stages: Object.fromEntries(["sample_analysis", "audience", "topic", "title", "copy", "image", "final_review"].map((stage, i) => [stage, {
        stage, status: i < 3 ? "completed" : i === 3 ? "failed" : "pending", revision: i < 4 ? 1 : 0,
        quality_revisions: 0, attempts: [], output: null, review_result: null, history: [],
        metadata: i === 3 ? { network_attempts: [1, 2, 3, 4].map(attempt => ({ attempt, max_retries: 3,
          duration_ms: 132000, error: "模型服务错误（HTTP 524）", error_code: "ai_http_error",
          will_retry: attempt < 4, wait_seconds: attempt < 4 ? 15 : 0, role: "title_review" })) } : {},
      }])),
    };
    const failed = { id: 701, task_type: "creative_analysis", status: "failed", progress: 55, resume_available: true,
      payload: { search: { account_id: 1, keyword: "恢复测试", limit: 20, sort_type_choice: 2, note_time: 2, note_type: 2 },
        analysis: { model_id: 2, workflow: "langgraph", idea_count: 1, variant_count: 2, audience: "服装行业", instruction: "精简" },
        sources: [], error: "模型服务错误（HTTP 524）", pipeline }, created_at: "2026-10-22T15:00:00" };
    let resumeCalls = 0, unexpected = [];
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      const reply = json => route.fulfill({ json });
      if (path === "/api/auth/refresh") return reply({ access_token: "mock-access" });
      if (path === "/api/auth/me") return reply({ id: 1, username: "恢复测试" });
      if (path === "/api/accounts") return reply({ items: [{ id: 1, sub_type: "pc", platform: "xhs", nickname: "模拟账号", status: "active" }] });
      if (path === "/api/model-configs") return reply({ items: [{ id: 2, name: "模拟模型", model_name: "mock", has_api_key: true, base_url: "https://invalid.test", model_type: "text" }] });
      if (path === "/api/notifications") return reply({ items: [] });
      if (path === "/api/xhs/creative/runs") return reply({ items: [{ id: 701, task_type: failed.task_type, keyword: "恢复测试", status: "failed" }] });
      if (path === "/api/xhs/creative/runs/701") return reply(failed);
      if (path === "/api/xhs/creative/runs/701/resume") {
        assert.equal(route.request().method(), "POST");
        resumeCalls++;
        return reply({ ...failed, id: 702, parent_task_id: 701, status: "pending", resume_available: false,
          payload: { ...failed.payload, error: undefined, stage: "等待从失败阶段继续", pipeline: { ...pipeline, status: "pending" } } });
      }
      if (path === "/api/xhs/creative/runs/702") return reply({ ...failed, id: 702, status: "running", resume_available: false,
        payload: { ...failed.payload, error: undefined, pipeline: { ...pipeline, status: "running", activity: "标题评审 Agent · 正在进行网络重试 1/3" } } });
      unexpected.push(path);
      return route.fulfill({ status: 404, json: { detail: "unexpected mock request" } });
    });
    await page.goto(process.env.XHS_TEST_BASE_URL || "http://127.0.0.1:5178/platforms/xhs/hot-content");
    await page.getByRole("button", { name: /#701 恢复测试/ }).click();
    await page.getByText("标题与评审 · 失败详情", { exact: true }).click();
    await page.getByText("网络请求第 4/4 次", { exact: false }).waitFor();
    await page.getByRole("button", { name: "从失败阶段继续", exact: true }).click();
    await page.getByText("创作任务 #702 · 恢复测试", { exact: true }).waitFor();
    await page.getByText("标题评审 Agent · 正在进行网络重试 1/3", { exact: true }).waitFor();
    assert.equal(resumeCalls, 1);
    assert.deepEqual(unexpected, []);
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.getByTestId("creative-pipeline").isVisible());
    assert.deepEqual(errors, []);
    process.stdout.write("恢复入口、网络重试详情、接续轮询及窄屏验收通过（全部 API 模拟）\n");
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
