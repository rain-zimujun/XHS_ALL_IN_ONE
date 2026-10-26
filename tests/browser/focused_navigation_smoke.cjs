// Start the frontend and provide Playwright via node_modules or NODE_PATH.
// Every API request is intercepted; this test never logs into XHS or runs a model.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

async function main() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const baseUrl = process.env.XHS_TEST_ORIGIN || "http://127.0.0.1:5178";
    await page.addInitScript(() => localStorage.setItem("spider_xhs_refresh_token", "mock-refresh"));
    let qrCalls = 0;
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      const reply = json => route.fulfill({ json });
      if (path === "/api/auth/refresh") return reply({ access_token: "mock-access", token_type: "bearer" });
      if (path === "/api/auth/me") return reply({ id: 1, username: "导航测试" });
      if (["/api/accounts", "/api/notifications", "/api/xhs/creative/runs", "/api/tasks"].includes(path)) return reply({ items: [], total: 0 });
      if (path === "/api/model-configs") return reply({ items: [
        { id: 1, name: "模拟文本模型", model_type: "text", model_name: "mock-text", base_url: "https://example.invalid/v1", has_api_key: true, is_default: true },
        { id: 2, name: "模拟图片模型", model_type: "image", model_name: "mock-image", base_url: "https://example.invalid/v1", has_api_key: true, is_default: true },
      ] });
      if (path === "/api/tasks/scheduler/status") return reply({ enabled: false, running: false, jobs: [], recent_tasks: [] });
      if (path === "/api/xhs/login-sessions/pc/qrcode") {
        qrCalls++;
        assert.equal(route.request().postDataJSON().sync_creator, false);
        return reply({ session_id: 1, status: "expired", qr_url: "" });
      }
      throw new Error(`未模拟接口：${route.request().method()} ${path}`);
    });
    await page.goto(baseUrl);
    await page.getByRole("heading", { name: "热点创作", exact: true }).waitFor();
    assert.equal(new URL(page.url()).pathname, "/platforms/xhs/hot-content");
    for (const name of ["热点创作", "账号绑定", "模型配置", "高级功能"]) {
      assert.ok(await page.getByRole("menuitem", { name: new RegExp(name) }).isVisible());
    }
    const advanced = page.getByRole("menuitem", { name: /高级功能/ });
    const publishing = page.getByRole("menuitem", { name: /发布中心/ });
    assert.equal(await publishing.isVisible(), false);
    await advanced.click();
    await publishing.waitFor({ state: "visible" });
    for (const name of ["图片工坊", "视频工坊", "竞品监控", "自动运营", "任务中心", "平台选择"]) {
      assert.ok(await page.getByRole("menuitem", { name: new RegExp(name) }).isVisible());
    }
    await advanced.click();
    await publishing.waitFor({ state: "hidden" });
    await page.getByRole("button", { name: "收起侧栏", exact: true }).click();
    await page.getByRole("button", { name: "展开侧栏", exact: true }).click();

    await page.getByRole("menuitem", { name: /模型配置/ }).click();
    await page.getByRole("heading", { name: "模型配置", exact: true }).waitFor();
    assert.equal(await page.getByText("模拟图片模型", { exact: true }).isVisible(), false);
    const advancedModels = page.getByRole("button", { name: /高级模型功能/ });
    await advancedModels.click();
    await page.getByText("模拟图片模型", { exact: true }).waitFor();
    await advancedModels.click();
    await page.getByText("模拟图片模型", { exact: true }).waitFor({ state: "hidden" });

    await page.getByRole("menuitem", { name: /账号绑定/ }).click();
    await page.getByRole("heading", { name: "账号绑定", exact: true }).waitFor();
    await page.getByRole("button", { name: /绑定账号$/ }).click();
    const drawer = page.getByRole("dialog");
    await drawer.waitFor();
    assert.equal(await drawer.getByText("Creator", { exact: true }).isVisible(), false);
    assert.equal(await drawer.getByText("登录 PC 后同步 Creator 账号", { exact: true }).isVisible(), false);
    await drawer.getByRole("button", { name: /高级登录选项/ }).click();
    await drawer.getByText("Creator", { exact: true }).waitFor();
    assert.ok(await drawer.getByText("Cookie", { exact: true }).isVisible());
    assert.ok(await drawer.getByText("登录 PC 后同步 Creator 账号", { exact: true }).isVisible());
    assert.ok(qrCalls > 0);

    // Existing deep links are preserved and automatically reveal the advanced menu.
    await page.goto(`${baseUrl}/tasks`);
    await page.getByRole("heading", { name: "任务中心", exact: true }).waitFor();
    await page.getByRole("menuitem", { name: /发布中心/ }).waitFor({ state: "visible" });
    await page.goto(`${baseUrl}/login`);
    await page.getByRole("heading", { name: "热点创作", exact: true }).waitFor();
    assert.equal(new URL(page.url()).pathname, "/platforms/xhs/hot-content");
    assert.equal(await page.getByRole("menuitem", { name: /发布中心/ }).isVisible(), false);
    if (process.env.XHS_TEST_SCREENSHOT) await page.screenshot({ path: process.env.XHS_TEST_SCREENSHOT });
    assert.deepEqual(errors, []);
    console.log("导航验证通过：默认热点创作、三个核心入口、高级菜单展开收起、图片模型/登录选项折叠、侧栏折叠、旧页面直达。全部使用模拟接口。");
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
