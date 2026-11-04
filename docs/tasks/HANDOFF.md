# 交接状态（换设备继续工作看这里）

> 最后更新：2026-11-04 15:20
> 变更历史见 [project-change-log.md](./project-change-log.md)，完整闭环计划见 [closed-loop-plan.md](./closed-loop-plan.md)。

## 1. 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 创作结果 → 草稿工坊 | ✅ 完成 |
| M2 | 草稿 → 发布（预检、送发布弹窗、发布中心校验/错误翻译/笔记链接） | ✅ 完成 |
| M3 | 可选内置生图（没配图片模型就不显示按钮） | ✅ 完成 |
| M4 | 定时发布状态同步（`POST /publish/jobs/sync`） | ✅ 完成 |
| M3.5 | 浏览器代签：话题 ID、账号导入 | ✅ 打通并验证 |
| 前端入口 | 话题搜索、签名浏览器状态条 | ✅ 完成 |
| M5 | 数据回流（发布后互动数据 → 我的发布） | ⬜ 未开始 |
| 打包 | exe / 一键安装 | ⬜ 暂缓 |

## 2. 未验证 / 已知问题

- **带话题发布未做端到端验证**：话题实体已能正确构造（`hash_tag` + `id/link/name`），但因**发布频率风控**，今日 13:52、14:38 两次发布被拒（`{"code":-1,"success":false}`）。第二天再试一次即可确认话题是否变蓝。
- **登录接口风控**：短时间内大量重试会把「概率 406」升级为「一律 406」，并连带影响发布。已加冷却（10 分钟，`services/login_gate.py`）；**不要连续点登录/发布**。
- **签名浏览器可能已退出**：状态条显示「未运行」时点「打开登录窗口」即可，profile 在 `data/browser/`，登录态持久保留。
- **测试数据**：数据库里可能残留「测试」开头的草稿/发布任务，可手动删除。

## 3. 关键实现位置（换设备先看这几个）

| 能力 | 文件 |
|---|---|
| 浏览器代签（CDP） | `backend/app/services/browser_bridge.py` |
| 话题缓存与实体解析 | `backend/app/services/topic_lookup.py`、`models.TopicCache` |
| 签名相关接口 | `backend/app/api/platforms/xhs/signer.py` |
| 登录风控冷却 | `backend/app/services/login_gate.py` |
| 发布与话题实体注入 | `backend/app/api/publish.py`（`entities_for_topics`） |
| 发布失败翻译 | `backend/app/adapters/xhs/creator_api_adapter.py`（`_publish_failure_message`） |
| 发布弹窗话题搜索 | `frontend/src/pages/platforms/xhs/rewrite-page.tsx` |
| 账号导入与状态条 | `frontend/src/pages/platforms/xhs/accounts-page.tsx` |

## 4. 在新设备上恢复

```powershell
git clone https://github.com/rain-zimujun/XHS_ALL_IN_ONE.git
cd XHS_ALL_IN_ONE

# 后端
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py            # http://127.0.0.1:8000

# 前端（另开终端）
npm --prefix frontend install
npm --prefix frontend run dev                 # http://127.0.0.1:5178
```

注意：

- `data/` 不入库（含数据库、素材、浏览器 profile 与 Cookie）。换设备等于**空数据库**，需要重新登录/导入账号；
- 签名浏览器用的是本机 Edge，路径自动探测；登录态不跨设备，需要在新设备扫码一次；
- 后端健康检查：`GET http://127.0.0.1:8000/api/health`。

## 5. 继续工作的建议顺序

1. **M5 数据回流**（约 1 天）：发布后拉取点赞/收藏/评论，新增 `note_metrics` 快照表，在「数据洞察」增加「我的发布」。
2. **素材/草稿治理**（约半天）：草稿保留上限（`max_kept_drafts`，默认 100，超限提醒不自动删）、测试数据一键清理。
3. **打包**（1~2 天）：PyInstaller + 前端静态托管 + 首次启动向导（数据目录、账号登录）。

## 6. 常用命令

```powershell
# 测试
$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe -m pytest tests/backend -q

# 前端构建
npm --prefix frontend run build

# 话题搜索（需签名浏览器已登录）
curl -X POST http://127.0.0.1:8000/api/xhs/signer/topics `
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" `
  -d '{"keyword":"穿搭","limit":10}'
```
