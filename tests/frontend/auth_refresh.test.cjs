const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { createRequire } = require("node:module");

const frontendRequire = createRequire(path.resolve(__dirname, "../../frontend/package.json"));
const axios = frontendRequire("axios");
const ts = frontendRequire("typescript");
const source = ts.transpileModule(fs.readFileSync(path.resolve(__dirname, "../../frontend/src/lib/api.ts"), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function harness(handle) {
  const storage = new Map([["spider_xhs_refresh_token", "test-refresh"]]);
  const calls = [];
  const messages = [];
  const module = { exports: {} };
  const client = {
    ...axios,
    create: options => axios.create({ ...options, adapter: async config => {
      calls.push({ url: config.url, authorization: config.headers.Authorization, timeout: config.timeout });
      const { status = 200, data = {} } = await handle(config);
      const response = { data, status, statusText: String(status), headers: {}, config };
      if (status >= 400) throw new axios.AxiosError("Mock request failed", "ERR_BAD_REQUEST", config, null, response);
      return response;
    } }),
  };
  vm.runInNewContext(source, {
    module, exports: module.exports,
    window: { localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    } },
    require: name => {
      if (name === "axios") return { ...client, default: client };
      if (name === "antd") return { message: { error: value => messages.push(value) } };
      if (name === "./platforms") return { fallbackPlatforms: [] };
      throw new Error(`Unexpected import: ${name}`);
    },
  });
  return { api: module.exports, storage, calls, messages };
}

test("失效刷新凭证只请求一次，清理登录状态并结束等待", async () => {
  const h = harness(async () => ({ status: 401 }));
  let expired = 0;
  h.api.onAuthExpired(() => expired++);
  await assert.rejects(h.api.bootstrapAuth());
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].url, "/auth/refresh");
  assert.equal(h.api.hasRefreshToken(), false);
  assert.equal(h.api.getAccessToken(), null);
  assert.equal(expired, 1);
});

test("并发业务请求共享一次刷新，成功后各自重放", async () => {
  const gate = deferred();
  let initialRequests = 0;
  const h = harness(async config => {
    if (config.url === "/auth/refresh") {
      await gate.promise;
      return { data: { access_token: "fresh-token" } };
    }
    if (!config.headers.Authorization) {
      if (++initialRequests === 2) gate.resolve();
      return { status: 401 };
    }
    assert.equal(config.headers.Authorization, "Bearer fresh-token");
    return { data: { items: [] } };
  });
  await Promise.all([h.api.fetchAccounts(), h.api.fetchCreativeHistory()]);
  assert.equal(h.calls.filter(call => call.url === "/auth/refresh").length, 1);
  assert.equal(h.calls.length, 5);
});

test("并发刷新失败均结束，只通知一次失效", async () => {
  const gate = deferred();
  let initialRequests = 0;
  const h = harness(async config => {
    if (config.url === "/auth/refresh") await gate.promise;
    else if (++initialRequests === 2) gate.resolve();
    return { status: 401 };
  });
  const results = await Promise.allSettled([h.api.fetchAccounts(), h.api.fetchCreativeHistory()]);
  assert.ok(results.every(result => result.status === "rejected"));
  assert.equal(h.calls.filter(call => call.url === "/auth/refresh").length, 1);
  assert.equal(h.messages.length, 1);
});

test("迟到的旧凭证401复用已刷新的令牌", async () => {
  const refreshed = deferred();
  const lateRequest = deferred();
  const h = harness(async config => {
    if (config.url === "/auth/refresh") return { data: { access_token: "fresh-token" } };
    if (config.headers.Authorization) return { data: { items: [] } };
    if (config.url === "/xhs/creative/runs") {
      lateRequest.resolve();
      await refreshed.promise;
    }
    return { status: 401 };
  });
  const late = h.api.fetchCreativeHistory();
  await lateRequest.promise;
  await h.api.fetchAccounts();
  refreshed.resolve();
  await late;
  assert.equal(h.calls.filter(call => call.url === "/auth/refresh").length, 1);
});

test("登录密码错误不会用旧凭证自动刷新", async () => {
  const h = harness(async () => ({ status: 401 }));
  await assert.rejects(h.api.login({ username: "tester", password: "wrong-password" }));
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].url, "/auth/login");
});

test("认证超时在15秒内受限，失败清理状态", async () => {
  const h = harness(async config => {
    assert.equal(config.timeout, 15000);
    throw new axios.AxiosError("Mock timeout", "ECONNABORTED", config);
  });
  await assert.rejects(h.api.bootstrapAuth());
  assert.equal(h.calls.length, 1);
  assert.equal(h.api.hasRefreshToken(), false);
});

test("退出后迟到的刷新结果不会恢复旧会话", async () => {
  const gate = deferred();
  const h = harness(async () => {
    await gate.promise;
    return { data: { access_token: "stale-token" } };
  });
  const pending = h.api.refreshAccessToken();
  h.api.clearAuthTokens();
  gate.resolve();
  await assert.rejects(pending, /session changed/);
  assert.equal(h.api.getAccessToken(), null);
});

test("重新登录后旧刷新失败不会清除新会话", async () => {
  const gate = deferred();
  const h = harness(async config => {
    if (config.url === "/auth/login") return { data: { access_token: "new-access", refresh_token: "new-refresh" } };
    await gate.promise;
    return { status: 401 };
  });
  const pending = h.api.refreshAccessToken();
  await h.api.login({ username: "tester", password: "test-password" });
  gate.resolve();
  await assert.rejects(pending);
  assert.equal(h.api.getAccessToken(), "new-access");
  assert.equal(h.storage.get("spider_xhs_refresh_token"), "new-refresh");
});

test("刷新成功但重放仍401时停止重试并退出登录", async () => {
  const h = harness(async config => config.url === "/auth/refresh"
    ? { data: { access_token: "fresh-token" } }
    : { status: 401 });
  let expired = 0;
  h.api.onAuthExpired(() => expired++);
  await assert.rejects(h.api.fetchAccounts());
  assert.equal(h.calls.length, 3);
  assert.equal(h.api.hasRefreshToken(), false);
  assert.equal(expired, 1);
});

test("有效凭证恢复登录，用户信息请求也有认证超时", async () => {
  const h = harness(async config => {
    assert.equal(config.timeout, 15000);
    if (config.url === "/auth/refresh") return { data: { access_token: "valid-access" } };
    assert.equal(config.url, "/auth/me");
    assert.equal(config.headers.Authorization, "Bearer valid-access");
    return { data: { id: 1, username: "测试用户" } };
  });
  const user = await h.api.bootstrapAuth();
  assert.equal(user.username, "测试用户");
  assert.equal(h.calls.length, 2);
});

test("畸形刷新响应不作为有效登录凭证", async () => {
  for (const access_token of [undefined, null, "", "  ", 123]) {
    const h = harness(async () => ({ data: { access_token } }));
    await assert.rejects(h.api.bootstrapAuth(), /Invalid access token/);
    assert.equal(h.api.hasRefreshToken(), false);
    assert.equal(h.calls.length, 1);
  }
});
