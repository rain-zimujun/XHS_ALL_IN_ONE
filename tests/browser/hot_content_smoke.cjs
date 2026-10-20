// Optional browser smoke test. All /api requests are mocked; no real login, crawl or model calls.
// Requires Playwright and Chrome. Start the frontend, then run this file with Node.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

async function main() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(() => localStorage.setItem("spider_xhs_refresh_token", "browser-test-only"));
    let searchBody, collectBody, searchRun, collectedRun, pollCount = 0;
    const makeRun = (id, task_type, payload, status = "completed") => ({
      id, platform: "xhs", task_type, status, progress: status === "completed" ? 100 : 0,
      payload, created_at: "2026-10-20T10:00:00", started_at: null, finished_at: null, parent_task_id: null,
    });
    await page.route("**/api/**", async route => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      const reply = body => route.fulfill({ json: body });
      if (path === "/api/auth/refresh") return reply({ access_token: "mock-access", token_type: "bearer" });
      if (path === "/api/auth/me") return reply({ id: 1, username: "浏览器测试" });
      if (path === "/api/accounts") return reply({ items: [{ id: 1, platform: "xhs", sub_type: "pc", nickname: "测试账号", status: "active" }] });
      if (path === "/api/model-configs") return reply({ items: [{ id: 2, name: "模拟文字模型", model_type: "text", model_name: "test-model", has_api_key: true, base_url: "https://example.invalid", is_default: true }] });
      if (path === "/api/notifications") return reply({ items: [] });
      if (path === "/api/xhs/creative/search") {
        searchBody = request.postDataJSON();
        const items = Array.from({ length: searchBody.limit }, (_, i) => ({
          note_id: `note-${i}`, note_url: `https://www.xiaohongshu.com/explore/note-${i}`,
          title: `服装设计样本 ${i + 1}`, content: "完整的测试正文", author_name: "模拟作者",
          cover_url: "", likes: 100 + i, collects: 50, comments: 10, timestamp: 0,
        }));
        searchRun = makeRun(101, "creative_search", { search: searchBody, items, warnings: [] });
        return reply(searchRun);
      }
      if (path === "/api/xhs/creative/collect") {
        collectBody = request.postDataJSON();
        collectedRun = makeRun(102, "creative_collect", { search: searchBody, analysis: collectBody.analysis, stage: "等待采集" }, "pending");
        return reply(collectedRun);
      }
      if (path === "/api/xhs/creative/runs/102") {
        pollCount++;
        const sources = searchRun.payload.items.filter(n => collectBody.note_ids.includes(n.note_id)).map((n, i) => ({
          ...n, timestamp: i === 2 ? 0 : i % 2 ? 1792306914 : 1792306914000, captured_at: "2026-10-20T10:00:00",
        }));
        collectedRun = makeRun(102, "creative_collect", {
          search: searchBody, analysis: collectBody.analysis, sources, stage: "创作方案已生成",
          result: { summary: "多篇样本关注 AI 服装设计流程", patterns: ["拆解步骤", "对比方案"],
            ideas: Array.from({ length: collectBody.analysis.idea_count }, (_, i) => ({
              title: `AI 服装创作方案 ${i + 1}`, alternative_titles: ["备选一", "备选二", "备选三"],
              body: "这是原创文案测试结果，图片由用户自行准备。", tags: ["AI服装", "服装设计"],
              reason: "基于多篇样本的共同特点", source_note_ids: sources.slice(0, 2).map(n => n.note_id),
              cover_text: "AI 设计流程", image_plan: [{ purpose: "封面", composition: "3:4 竖图，顶部留白", prompt: "fashion sketch, neutral background, 3:4", negative_prompt: "logo, watermark, text" }],
            })) },
          analysis_metadata: { model_name: "test-model", sample_count: sources.length, excerpt_chars_per_note: 800, disclaimer: "仅供创作参考，不保证流量。" },
          report_directory: "/mock/exports", report_files: [
            { file_name: "xhs-report-u1-test.html", download_url: "/api/files/exports/xhs-report-u1-test.html" },
            { file_name: "xhs-report-u1-test.json", download_url: "/api/files/exports/xhs-report-u1-test.json" },
          ],
        });
        return reply(collectedRun);
      }
      if (path === "/api/xhs/creative/runs/101") return reply(searchRun);
      if (path === "/api/xhs/creative/runs") return reply({ items: [collectedRun, searchRun].filter(Boolean).map(run => ({ ...run, keyword: run.payload.search.keyword })) });
      if (path === "/api/files/exports/xhs-report-u1-test.html") return route.fulfill({ contentType: "text/html", body: "<!doctype html><h1>模拟报告</h1>" });
      throw new Error(`未模拟的接口：${request.method()} ${path}`);
    });
    await page.goto(process.env.XHS_TEST_BASE_URL || "http://127.0.0.1:5173/platforms/xhs/hot-content");
    await page.getByRole("heading", { name: "热点创作", exact: true }).waitFor();
    await page.getByLabel("搜索关键词").fill("AI 模特穿搭");
    await page.getByLabel("目标数量").click();
    await page.getByText("30 篇", { exact: true }).last().click();
    await page.getByLabel("排序方式").click();
    await page.getByText("最多收藏", { exact: true }).last().click();
    await page.getByRole("button", { name: /搜索 30 篇内容/ }).click();
    await page.getByText("已选 30 / 30 篇", { exact: true }).waitFor();
    assert.ok((await page.locator('tr[data-row-key="note-1"]').innerText()).includes("待采集补全"));
    assert.equal(searchBody.sort_type_choice, 4);
    assert.equal(searchBody.keyword, "AI 模特穿搭");
    assert.equal(searchBody.limit, 30);
    await page.locator('tr[data-row-key="note-0"] .ant-checkbox-input').uncheck();
    await page.getByText("已选 29 / 30 篇", { exact: true }).waitFor();
    await page.getByLabel("创作要求").fill("偏向实操教程，不夸张承诺");
    await page.getByRole("button", { name: "采集并生成 3 套方案", exact: true }).click();
    await page.getByTestId("creative-results").waitFor();
    assert.equal(collectBody.note_ids.length, 29);
    assert.equal(collectBody.analysis.model_id, 2);
    assert.equal(collectBody.analysis.instruction, "偏向实操教程，不夸张承诺");
    assert.ok(pollCount >= 1);
    // The existing search table is hydrated during polling, before any reload.
    assert.ok((await page.locator('tr[data-row-key="note-1"]').innerText()).includes("2026/10/18"));
    assert.ok((await page.locator('tr[data-row-key="note-2"]').innerText()).includes("2026/10/18"));
    assert.ok((await page.locator('tr[data-row-key="note-3"]').innerText()).includes("详情未提供时间"));
    assert.ok((await page.locator('tr[data-row-key="note-0"]').innerText()).includes("待采集补全"));
    assert.equal(await page.getByText("生图提示词", { exact: true }).count(), 3);
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "下载 HTML 阅读版", exact: true }).click();
    assert.equal((await downloadPromise).suggestedFilename(), "xhs-report-u1-test.html");
    await page.reload();
    await page.getByRole("button", { name: /#102 AI 模特穿搭/ }).click();
    await page.getByTestId("creative-results").waitFor();
    assert.equal(await page.getByText("生图提示词", { exact: true }).count(), 3);
    assert.equal(await page.getByLabel("搜索关键词").inputValue(), "AI 模特穿搭");
    assert.equal(await page.getByLabel("创作要求").inputValue(), "偏向实操教程，不夸张承诺");
    assert.ok((await page.locator('tr[data-row-key="note-1"]').innerText()).includes("2026/10/18"));
    assert.equal(await page.locator("tbody .ant-checkbox-input").count(), 0);
    if (process.env.XHS_TEST_SCREENSHOT) {
      await page.getByTestId("creative-results").scrollIntoViewIfNeeded();
      await page.screenshot({ path: process.env.XHS_TEST_SCREENSHOT, fullPage: true });
    }
    assert.deepEqual(errors, []);
    console.log("浏览器验证通过：筛选 30 篇、取消勾选、一键生成、发布时间回填、秒/毫秒时间、报告下载、刷新恢复已采集样本。全部使用模拟接口。");
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
