# 小红书全域运营器 · XHS_ALL_IN_ONE

> 基于 [cv-cat/XHS_ALL_IN_ONE](https://github.com/cv-cat/XHS_ALL_IN_ONE) 二次开发的个人版本。
> 上游提供小红书账号接入、采集 SDK 与平台底座；本仓库的主要工作集中在 **AI 内容创作链路**，以及围绕它的前后端体验、质量评审与任务恢复。

一句话：从关键词搜索一批笔记，选出样本，交给一组 Agent 做研究、选题、写标题正文、出配图方案，再用评审把关、返工收敛，最后导出一份可以直接拿去用的内容方案。

**⚠️ 本项目仅供学习交流使用，禁止任何商业化行为**

---

## 这个版本做了什么

### 1. 用 LangGraph 把创作链路做成可回溯的状态图

13 个节点：样本分析 → 样本证据评审 → 读者需求 → 选题 → 选题评审 → 标题 → 标题评审 → 正文 → 正文评审 → 配图 → 终审 → 返工计划 → 结果组装。

评审结论经条件边决定返工回到哪一阶段，而不是整条重跑。每个节点只允许写自己声明的状态字段，越权写入在运行时直接中断，阶段交接因此可审计。

### 2. 两层校验，挡住模型"编造引用"

- 第一层：每次请求按当次的合法 ID（样本、结论、选题、标题、正文）动态生成严格 JSON Schema，通过 `response_format` 下发，越界引用在 Schema 层就被拒。
- 第二层：程序再核对来源归属、逐字引用、数量与返工范围——Schema 只能保证形状，保证不了关系。

"结论 claim ↔ 来源 note" 的绑定由程序反查生成，不接受模型自报；逐字引用允许空白与字段标注差异，但必须能在单一字段内命中 ≥8 字连续原文。

### 3. 长任务可恢复

按输入指纹做批次缓存与结果复用：样本按 2～3 篇、标题按候选 ≤5 个、正文按锁定标题 2～3 版、配图与终审逐篇。失败接续与主动暂停都只补未完成批次。

配合请求预算（72～512 次）、90 分钟执行窗口、5 次退避重试（15/30/60/90/120 秒）与批内三路并发。

### 4. 返工只改该改的

分阶段评分 rubric 与阻断/建议分级，未通过项定位到具体对象与字段；返工范围由代码强制，其余内容由服务端恢复原稿后再校验；已通过且文本未变的候选"冻结判定"不再重复送审，每阶段最多 3 轮。

### 5. 上下文按角色裁剪

每个角色只收到本阶段需要的字段白名单，未参与判断的上游产物不进请求。实测选题请求从 12,209 字符降到 6,036，标题 15,787→6,065，正文 16,745→8,140。

### 6. 前端重做

React 19 + TypeScript + Ant Design，17 个页面；创作任务有阶段进度、请求计数、评审明细、失败诊断与原句对照，结果支持多版本切换、复制与 HTML/JSON 报告导出。

---

## 平台原有能力（继承自上游）

账号矩阵（PC / Creator 双端、Cookie 加密存储、定时健康巡检）、笔记发现（关键词搜索、URL 直查）、数据抓取（批量 URL / 评论、Excel 导出）、内容库、草稿工坊、图片工坊、视频工坊、发布中心（立即 / 定时）、自动运营、数据洞察、竞品监控、任务中心、通知与模型配置。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 后端 | Python 3.11 / FastAPI / SQLAlchemy / Alembic / APScheduler |
| AI 编排 | LangGraph / LangChain / Pydantic / 严格 JSON Schema |
| 前端 | React 19 / TypeScript / Vite / Ant Design |
| 存储 | SQLite（默认）/ MySQL |
| 部署 | Docker Compose / uvicorn |

---

## 快速开始

环境：Python 3.11+、Node.js 20+。

```bash
git clone https://github.com/rain-zimujun/XHS_ALL_IN_ONE.git
cd XHS_ALL_IN_ONE

python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

npm ci
cd frontend && npm ci && cd ..

python main.py --with-frontend
```

启动后访问：

- 前端 http://127.0.0.1:5178
- 接口文档 http://127.0.0.1:8000/docs

首次使用注册本地账号即可；数据库与运行数据会在项目目录下自动创建（已在 `.gitignore` 中忽略）。

> 国内网络如果卡在包源，可换镜像：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
> `npm ci --registry=https://registry.npmmirror.com`

---

## 配置

- 基础配置：`config/default.yaml`（数据库、端口、CORS、调度开关、密钥）
- 覆盖方式：`CONFIG_FILE` 指定的 YAML < `.env` < 环境变量
- 模型：在页面「模型配置」里填任意 OpenAI 兼容端点（火山引擎、阿里云百炼、中转站均可），文本模型与图片模型分开配置

真实创作需要两样东西：一个已绑定的小红书账号，和一个能跑通的模型端点。两者都可以先用本地模拟数据跑通界面流程，不必一上来就接真账号。

---

## 测试

```bash
pytest tests/backend -q                     # 后端
node tests/browser/hot_content_smoke.cjs    # 浏览器流程（需先起前端，Playwright + 本机 Chrome）
```

后端现有 958 项测试。其中 12 项是改动前就已存在的过期断言（重试次数从 3→5、请求预算调整后仍按旧值断言），不影响运行，修复在进行中；其余 946 项通过。

浏览器流程测试用模拟接口驱动真实 Chrome，覆盖"搜索 → 采集 → 生成 → 多版本切换 → 复制 → 导出报告"整条主流程。它验证的是前端与状态机有没有被改坏，不验证模型输出质量——后者靠真实链路人工验收。

---

## 目录结构（只列改造重点）

```
backend/app/
├── services/creative_graph/        # 创作链路核心
│   ├── workflow.py                 # 13 节点状态图与条件路由
│   ├── state.py                    # 全局状态与各节点可写字段
│   ├── agents.py                   # 各角色 Agent 与结构化输出绑定
│   ├── *_schema.py                 # 按请求动态生成的严格 Schema
│   ├── adaptive_batches.py         # 分批策略
│   ├── batch_runner.py             # 批内并发、缓存与合并
│   ├── recovery.py                 # 失败接续与暂停恢复
│   └── context.py                  # 按角色裁剪上下文
├── services/creative_retries.py    # 请求预算、重试与诊断
└── api/                            # FastAPI 路由（约 139 个）
frontend/src/pages/platforms/xhs/   # 创作、结果、评审等页面
tests/backend/                      # 后端测试
tests/browser/                      # Playwright 流程测试
docs/                               # 开发日志与流程说明
```

---

## 已知限制

- 后台任务只在单进程内运行，没有分布式调度，也没有 LangGraph checkpointer；"接续"是新建子任务从失败节点重入，不是原地续跑。
- 生图模块是独立能力，创作链路目前只产出可直接使用的正反向提示词，尚未自动串联到出图。
- 效果强依赖模型与中转站：弱模型在样本分析阶段就会因为引文不足失败，中转站 504 也会让长任务卡住。
- 真实规模化未验证，定位是本地自用工具而非线上服务。

---

## 与上游的关系

- 小红书账号接入、采集适配、发布与运营等基础能力来自 [cv-cat/XHS_ALL_IN_ONE](https://github.com/cv-cat/XHS_ALL_IN_ONE)，本仓库是它的 fork。
- 界面截图与品牌素材同样来自上游。
- 上游 README 声明 MIT License，但仓库内没有 LICENSE 文件；本仓库沿用同一声明。
- 上游声明：本项目仅供学习交流使用，禁止任何商业化行为。

---

## 致谢

感谢 [cv-cat](https://github.com/cv-cat) 开源的 XHS_ALL_IN_ONE：账号接入与采集适配是整条链路里最枯燥、也最容易翻车的部分，这部分省下来的时间才让创作链路能专心做深。
