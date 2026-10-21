// Real HTML renderer + actual workflow validators + synthetic model responses.
// No frontend/backend server, platform account, database or network service is needed.
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const path = require("node:path");
const { chromium } = require("playwright");

async function main() {
  const root = path.resolve(__dirname, "../..");
  const html = execFileSync(process.env.XHS_TEST_PYTHON || path.join(root, ".venv/bin/python"), [path.join(__dirname, "report_fixture.py")], {
    cwd: root, encoding: "utf8", maxBuffer: 10 * 1024 * 1024,
  });
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [], requests = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("request", request => requests.push(request.url()));
    await page.route("**/*", route => route.abort());
    await page.setContent(html);
    assert.equal(await page.locator("article").count(), 3);
    assert.equal(await page.locator("article").first().getByRole("heading", { name: /^文案 [12] ·/ }).count(), 2);
    assert.equal(await page.getByText("标题评分：8.00 分 · 通过", { exact: true }).count(), 3);
    assert.equal(await page.getByText("正文评分：8.00 分 · 通过", { exact: true }).count(), 6);
    assert.equal(await page.getByText("终审评分：8.00 分 · 通过", { exact: true }).count(), 6);
    assert.equal(await page.locator("details[open]").count(), 0);
    const audit = page.locator("details").filter({ has: page.locator(":scope > summary", { hasText: "评审问题与返工历史（内部记录）" }) });
    await audit.locator(":scope > summary").click();
    const image = audit.locator("details").filter({ has: page.locator(":scope > summary", { hasText: /^配图 · 版本/ }) });
    await image.locator(":scope > summary").click();
    await image.getByText("历史过程（仅供追溯）", { exact: true }).click();
    await image.getByText("本次修改对象：copy_0_0", { exact: true }).waitFor();
    assert.equal(await page.locator("script, img, iframe").count(), 0);
    assert.equal(await page.evaluate(() => window.__reportXss), undefined);
    assert.ok((await page.locator("body").innerText()).includes("模型请求"));
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    assert.deepEqual(requests, []);
    assert.deepEqual(errors, []);
    console.log("报告浏览器验证通过：真实 HTML 渲染、3 标题 × 2 版评分、默认折叠、返工记录、脚本文本转义、窄屏与零网络请求。");
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
