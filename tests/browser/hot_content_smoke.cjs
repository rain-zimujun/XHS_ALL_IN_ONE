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
    await page.addInitScript(() => {
      localStorage.setItem("spider_xhs_refresh_token", "browser-test-only");
      Object.defineProperty(navigator, "clipboard", { value: { writeText: async text => {
        if (window.__rejectCopy) throw new Error("mock clipboard denied");
        window.__copiedText = text;
      } } });
    });
    const variantCount = Number(process.env.XHS_TEST_VARIANTS || 3);
    assert.ok([2, 3].includes(variantCount));
    let searchBody, collectBody, searchRun, collectedRun, legacyRun, pollCount = 0, collectCalls = 0;
    const makeRun = (id, task_type, payload, status = "completed") => ({
      id, platform: "xhs", task_type, status, progress: status === "completed" ? 100 : 0,
      payload, created_at: "2026-10-20T10:00:00", started_at: null, finished_at: null, parent_task_id: null,
    });
    const review = (kind, overall_score) => ({
      kind, scores: [], overall_score, passed: overall_score >= 7, blocking_issues: [], revision_instructions: [],
    });
    const makePipeline = (running = false) => ({
      schema_version: 2, status: running ? "running" : "completed", current_stage: running ? "copy" : null,
      activity: running ? "文案 Agent（第 1/2 次）" : "多 Agent 创作及评审完成", request_count: running ? 7 : 10,
      max_requests: 36, max_quality_revisions: 2,
      stages: Object.fromEntries(["sample_analysis", "audience", "topic", "title", "copy", "image", "final_review"].map((stage, i) => [stage, {
        stage, status: running && i >= 4 ? i === 4 ? "running" : "pending" : "completed", revision: running && i >= 4 ? 0 : 1,
        quality_revisions: 0, attempts: [], output: null, review_result: null, history: [],
      }])),
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
        collectCalls++;
        collectBody = request.postDataJSON();
        collectedRun = makeRun(102, "creative_collect", { search: searchBody, analysis: collectBody.analysis, stage: "等待采集" }, "pending");
        return reply(collectedRun);
      }
      if (path === "/api/xhs/creative/runs/102") {
        pollCount++;
        if (pollCount === 1) return reply(makeRun(102, "creative_collect", {
          ...collectedRun.payload, pipeline: makePipeline(true), stage: "文案 Agent（第 1/2 次）",
        }, "running"));
        const sources = searchRun.payload.items.filter(n => collectBody.note_ids.includes(n.note_id)).map((n, i) => ({
          ...n, timestamp: i === 2 ? 0 : i % 2 ? 1792306914 : 1792306914000, captured_at: "2026-10-20T10:00:00",
        }));
        collectedRun = makeRun(102, "creative_collect", {
          search: searchBody, analysis: collectBody.analysis, sources, stage: "创作方案已生成", pipeline: makePipeline(),
          result: { summary: "多篇样本关注 AI 服装设计流程", patterns: ["拆解步骤", "对比方案"], style_summary: "问题开头、口语短句、分段清晰，结尾邀请讨论。",
            ideas: Array.from({ length: collectBody.analysis.idea_count }, (_, i) => ({
              idea_id: `idea_${i + 1}`,
              title: `AI 服装创作方案 ${i + 1}`,
              reason: "基于多篇样本的共同特点", source_note_ids: sources.slice(0, 2).map(n => n.note_id),
              quality: { title: review("title", [7, 9, 8][i]) },
              variants: Array.from({ length: collectBody.analysis.variant_count }, (_, v) => ({
                label: ["问题拆解", "步骤清单", "对比讨论"][v], body: `标题 ${i + 1} 的文案 ${v + 1}：雾白的衣摆和轻盈的线条，有海风吹过的感觉。`,
                tags: [`标题${i + 1}`, `版本${v + 1}`], cover_text: `封面 ${i + 1}-${v + 1}`,
                quality: { copy: review("copy", [7, 8, 6][i]), final: review("final", [8, 6.5, 9][i]) },
                image_plan: [{ purpose: "封面", composition: `配图 ${i + 1}-${v + 1}：3:4 竖图，顶部留白`, prompt: `concept ${i + 1}, version ${v + 1}, fashion sketch, 3:4`, negative_prompt: "logo, watermark, text" }],
              })),
            })) },
          analysis_metadata: { model_name: "test-model", sample_count: sources.length, excerpt_chars_per_note: 800, excerpt_strategy: "head_and_tail", disclaimer: "仅供创作参考，不保证流量。" },
          report_directory: "/mock/exports", report_files: [
            { file_name: "xhs-report-u1-test.html", download_url: "/api/files/exports/xhs-report-u1-test.html" },
            { file_name: "xhs-report-u1-test.json", download_url: "/api/files/exports/xhs-report-u1-test.json" },
          ],
        });
        legacyRun = JSON.parse(JSON.stringify(collectedRun));
        legacyRun.id = 100;
        delete legacyRun.payload.pipeline;
        delete legacyRun.payload.analysis.workflow;
        delete legacyRun.payload.analysis.variant_count;
        delete legacyRun.payload.result.style_summary;
        legacyRun.payload.result.ideas = [{
          title: "旧版标题", reason: "旧版推荐理由", source_note_ids: sources.slice(0, 2).map(n => n.note_id),
          ...collectedRun.payload.result.ideas[0].variants[0], body: "旧版正文原样保留，不补造额外版本。", alternative_titles: ["旧备选一", "旧备选二", "旧备选三"],
        }];
        delete legacyRun.payload.result.ideas[0].idea_id;
        delete legacyRun.payload.result.ideas[0].quality;
        legacyRun.payload.result.ideas.push({ ...legacyRun.payload.result.ideas[0], title: "旧版第二个标题", body: "旧版第二篇正文，无评分且没有 idea_id。" });
        legacyRun.payload.result.ideas[0].quality = { title: review("title", 7) };
        return reply(collectedRun);
      }
      if (path === "/api/xhs/creative/runs/100") return reply(legacyRun);
      if (path === "/api/xhs/creative/runs/101") return reply(searchRun);
      if (path === "/api/xhs/creative/runs/102/export") return reply(collectedRun);
      if (path === "/api/xhs/creative/runs") return reply({ items: [collectedRun, searchRun, legacyRun].filter(Boolean).map(run => ({ ...run, keyword: run.payload.search.keyword })) });
      if (path === "/api/files/exports/xhs-report-u1-test.html") return route.fulfill({ contentType: "text/html", body: "<!doctype html><h1>模拟报告</h1>" });
      throw new Error(`未模拟的接口：${request.method()} ${path}`);
    });
    await page.goto(process.env.XHS_TEST_BASE_URL || "http://127.0.0.1:5173/platforms/xhs/hot-content");
    await page.getByRole("heading", { name: "热点创作", exact: true }).waitFor();
    assert.equal(await page.getByLabel("目标受众").inputValue(), "电商、服装行业、AI、时尚行业、营销行业、美工、后期从业者");
    assert.equal(await page.getByLabel("创作要求").inputValue(), "符合小红书爆款文案的调性，文案尽可能精简、抓眼球");
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
    if (variantCount === 2) {
      await page.getByLabel("每个标题的文案数").click();
      await page.getByText("2 个版本", { exact: true }).last().click();
    }
    await page.getByRole("button", { name: `采集并生成 3 个标题 × ${variantCount} 版文案`, exact: true }).click();
    await page.getByTestId("creative-stage-copy").filter({ hasText: "进行中" }).waitFor();
    await page.getByTestId("creative-results").waitFor();
    assert.ok((await page.getByTestId("quality-标题").first().innerText()).includes("标题 7.0 分 · 通过"));
    await page.getByLabel("评分排序").click();
    await page.getByText("标题评分从高到低", { exact: true }).last().click();
    assert.equal(await page.locator('button[data-testid^="creative-title-"]').first().getAttribute("data-testid"), "creative-title-1");
    await page.getByLabel("评分排序").click();
    await page.getByText("正文评分从高到低", { exact: true }).last().click();
    assert.equal(await page.locator('button[data-testid^="creative-title-"]').first().getAttribute("data-testid"), "creative-title-1");
    await page.getByLabel("评分排序").click();
    await page.getByText("终审评分从高到低", { exact: true }).last().click();
    assert.equal(await page.locator('button[data-testid^="creative-title-"]').first().getAttribute("data-testid"), "creative-title-2");
    await page.getByLabel("评分筛选依据").click();
    await page.getByText("按标题评分筛选", { exact: true }).last().click();
    await page.getByLabel("最低评分").click();
    await page.getByText("至少 8 分", { exact: true }).last().click();
    assert.equal(await page.locator('button[data-testid^="creative-title-"]').count(), 2);
    assert.ok(await page.getByTestId("creative-title-1").count());
    assert.ok(await page.getByTestId("creative-title-2").count());
    await page.getByLabel("评分筛选依据").click();
    await page.getByText("按正文评分筛选", { exact: true }).last().click();
    assert.equal(await page.locator('button[data-testid^="creative-title-"]').count(), 1);
    assert.ok(await page.getByTestId("creative-title-1").count());
    await page.getByLabel("评分筛选依据").click();
    await page.getByText("按终审评分筛选", { exact: true }).last().click();
    assert.equal(await page.locator('button[data-testid^="creative-title-"]').count(), 2);
    assert.ok(await page.getByTestId("creative-title-0").count());
    assert.ok(await page.getByTestId("creative-title-2").count());
    await page.getByLabel("最低评分").click();
    await page.getByText("显示全部评分", { exact: true }).last().click();
    await page.getByLabel("评分排序").click();
    await page.getByText("按原顺序", { exact: true }).last().click();
    assert.equal(collectBody.analysis.workflow, "multi_agent");
    assert.ok((await page.getByTestId("creative-pipeline").innerText()).includes("10 / 36"));
    assert.equal(collectBody.note_ids.length, 29);
    assert.equal(collectBody.analysis.model_id, 2);
    assert.equal(collectBody.analysis.instruction, "偏向实操教程，不夸张承诺");
    assert.equal(collectBody.analysis.audience, "电商、服装行业、AI、时尚行业、营销行业、美工、后期从业者");
    assert.equal(collectBody.analysis.variant_count, variantCount);
    assert.ok(pollCount >= 1);
    // The existing search table is hydrated during polling, before any reload.
    assert.ok((await page.locator('tr[data-row-key="note-1"]').innerText()).includes("2026/10/18"));
    assert.ok((await page.locator('tr[data-row-key="note-2"]').innerText()).includes("2026/10/18"));
    assert.ok((await page.locator('tr[data-row-key="note-3"]').innerText()).includes("详情未提供时间"));
    assert.ok((await page.locator('tr[data-row-key="note-0"]').innerText()).includes("待采集补全"));
    assert.equal(await page.getByText("生图提示词", { exact: true }).count(), 1);
    assert.equal(await page.getByRole("tab").count(), variantCount);
    // Every title/variant pair has its own matching body, image advice, prompt and copy payload.
    for (let i = 0; i < 3; i++) {
      await page.getByTestId(`creative-title-${i}`).click();
      assert.equal(await page.getByTestId(`creative-title-${i}`).getAttribute("aria-pressed"), "true");
      await page.getByTestId("creative-body").filter({ hasText: `标题 ${i + 1} 的文案 1` }).waitFor();
      for (let v = 0; v < variantCount; v++) {
        await page.getByRole("tab", { name: `文案 ${v + 1} · ${["问题拆解", "步骤清单", "对比讨论"][v]}`, exact: true }).click();
        const variant = collectedRun.payload.result.ideas[i].variants[v];
        await page.getByTestId("creative-body").filter({ hasText: variant.body }).waitFor();
        const images = await page.getByTestId("creative-images").innerText();
        assert.ok(images.includes(variant.image_plan[0].composition));
        assert.ok(images.includes(variant.image_plan[0].prompt));
        assert.equal((await page.getByTestId("creative-selected-title").innerText()).trim(), collectedRun.payload.result.ideas[i].title);
        await page.getByRole("button", { name: "复制正文", exact: true }).click();
        assert.equal(await page.evaluate(() => window.__copiedText), variant.body);
        await page.getByRole("button", { name: "复制完整笔记", exact: true }).click();
        assert.equal(await page.evaluate(() => window.__copiedText), [collectedRun.payload.result.ideas[i].title, variant.body, variant.tags.map(t => `#${t}`).join(" ")].join("\n\n"));
        const copied = await page.evaluate(() => window.__copiedText);
        assert.ok(!copied.includes("配图建议") && !copied.includes("生图提示词") && !copied.includes("推荐理由"));
        assert.ok(variant.tags.every(tag => copied.split(`#${tag}`).length === 2));
      }
    }
    // Re-export changes download state only, not the current selection.
    await page.getByRole("button", { name: "重新导出报告", exact: true }).click();
    assert.ok((await page.getByTestId("creative-body").innerText()).includes(`标题 3 的文案 ${variantCount}`));
    await page.getByText("样本分析与文案风格依据", { exact: true }).click();
    await page.getByText("问题开头、口语短句、分段清晰，结尾邀请讨论。", { exact: true }).waitFor();
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "下载 HTML 阅读版", exact: true }).click();
    assert.equal((await downloadPromise).suggestedFilename(), "xhs-report-u1-test.html");
    await page.reload();
    await page.getByRole("button", { name: /#102 AI 模特穿搭/ }).click();
    await page.getByTestId("creative-results").waitFor();
    assert.equal(await page.getByText("生图提示词", { exact: true }).count(), 1);
    assert.equal(await page.getByRole("tab").count(), variantCount);
    await page.getByText(`${variantCount} 个版本`, { exact: true }).waitFor();
    assert.equal(await page.getByLabel("搜索关键词").inputValue(), "AI 模特穿搭");
    assert.equal(await page.getByLabel("创作要求").inputValue(), "偏向实操教程，不夸张承诺");
    assert.ok((await page.locator('tr[data-row-key="note-1"]').innerText()).includes("2026/10/18"));
    assert.equal(await page.locator("tbody .ant-checkbox-input").count(), 0);
    if (process.env.XHS_TEST_SCREENSHOT) {
      await page.getByTestId("creative-results").scrollIntoViewIfNeeded();
      await page.screenshot({ path: process.env.XHS_TEST_SCREENSHOT, fullPage: true });
    }
    // Narrow screens and legacy saved data keep the real content without fake extra choices.
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByTestId("creative-title-1").click();
    await page.getByTestId("creative-body").filter({ hasText: "标题 2 的文案 1" }).waitFor();
    const bodyBox = await page.getByTestId("creative-body").boundingBox();
    const imageBox = await page.getByTestId("creative-images").boundingBox();
    assert.ok(bodyBox && imageBox && imageBox.y > bodyBox.y);
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.getByRole("button", { name: /#100 AI 模特穿搭/ }).click();
    await page.getByText("这是旧版结果，仅保留原有的 1 版文案和配图。", { exact: true }).waitFor();
    assert.equal(await page.getByRole("tab").count(), 1);
    assert.ok((await page.getByTestId("creative-body").innerText()).includes("旧版正文原样保留"));
    await page.getByLabel("评分筛选依据").click();
    await page.getByText("按标题评分筛选", { exact: true }).last().click();
    await page.getByLabel("最低评分").click();
    await page.getByText("至少 6.5 分", { exact: true }).last().click();
    assert.equal(await page.locator('button[data-testid^="creative-title-"]').count(), 1);
    assert.equal(await page.getByTestId("creative-title-1").count(), 0);
    await page.getByLabel("最低评分").click();
    await page.getByText("显示全部评分", { exact: true }).last().click();
    await page.getByTestId("creative-title-1").click();
    assert.ok((await page.getByTestId("creative-body").innerText()).includes("旧版第二篇正文，无评分且没有 idea_id。"));
    await page.evaluate(() => { window.__rejectCopy = true; });
    await page.getByRole("button", { name: "复制完整笔记", exact: true }).click();
    await page.getByText("浏览器未允许复制，请选中正文手动复制。", { exact: true }).waitFor();
    assert.equal(collectCalls, 1);
    assert.deepEqual(errors, []);
    console.log(`浏览器验证通过：筛选 30 篇、采集生成、3 个标题 × ${variantCount} 版的正文与配图联动、完整复制、风格依据、时间回填、报告、历史和旧版兼容、窄屏。全部使用模拟接口。`);
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
