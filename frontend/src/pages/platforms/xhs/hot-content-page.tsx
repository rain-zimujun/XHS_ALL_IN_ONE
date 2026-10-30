import { ReloadOutlined, SearchOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Col, Collapse, Empty, Form, Input, Pagination, Progress, Row, Select, Space, Table, Tag, Typography } from "antd";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { analyzeCreativeCollection, cancelCreativeRun, collectCreativeNotes, fetchAccounts, fetchCreativeHistory, fetchCreativeRun, fetchModelConfigs, resumeCreativeRun, searchCreativeNotes } from "../../../lib/api";
import type { ModelConfig, PlatformAccount, XhsSearchNote } from "../../../types";
import type { CreativeAnalysisOptions, CreativeHistoryItem, CreativeRun, CreativeSearch } from "../../../types/creative";
import { CreativeResults } from "./creative-results";
import { CreativePipelineProgress } from "./creative-pipeline";
import { CreativeReviewAudit } from "./creative-reviews";
import { creativeSamples, formatPublicationTime } from "../../../lib/creative-samples";

const { Title, Text, Paragraph } = Typography;
const sortOptions = [
  { value: 0, label: "综合排序" }, { value: 1, label: "最新发布" },
  { value: 2, label: "最多点赞" }, { value: 4, label: "最多收藏" }, { value: 3, label: "最多评论" },
];
const timeOptions = [
  { value: 0, label: "不限时间" }, { value: 1, label: "一天内" },
  { value: 2, label: "一周内" }, { value: 3, label: "半年内" },
];
const statusNames: Record<string, string> = { pending: "等待中", running: "进行中", completed: "已完成", failed: "失败", cancelled: "已停止" };

export function HotContentPage() {
  const [accounts, setAccounts] = useState<PlatformAccount[]>([]);
  const [accountId, setAccountId] = useState<number>();
  const [keyword, setKeyword] = useState("AI 服装设计");
  const [limit, setLimit] = useState<20 | 30 | 50>(20);
  const [sort, setSort] = useState(2);
  const [noteTime, setNoteTime] = useState(2);
  const [noteType, setNoteType] = useState(2);
  const [loadingAccounts, setLoadingAccounts] = useState(true);
  const [searching, setSearching] = useState(false);
  const [run, setRun] = useState<CreativeRun | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [collection, setCollection] = useState<CreativeRun | null>(null);
  const [starting, setStarting] = useState(false);
  const [history, setHistory] = useState<CreativeHistoryItem[]>([]);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotal, setHistoryTotal] = useState(0);
  const historyPageSize = 20;
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [modelId, setModelId] = useState<number>();
  const [preprocessModelId, setPreprocessModelId] = useState<number>();
  const [ideaCount, setIdeaCount] = useState<1 | 3 | 5>(3);
  const [variantCount, setVariantCount] = useState<2 | 3>(3);
  const [titleLength, setTitleLength] = useState(26);
  const [workflow, setWorkflow] = useState<NonNullable<CreativeAnalysisOptions["workflow"]>>("multi_agent");
  const [audience, setAudience] = useState("电商、服装行业、AI、时尚行业、营销行业、美工、后期从业者");
  const [instruction, setInstruction] = useState("符合小红书爆款文案的调性，文案尽可能精简、抓眼球");
  const collecting = collection?.status === "running" || collection?.status === "pending";

  async function loadAccounts() {
    setLoadingAccounts(true);
    try {
      const items = (await fetchAccounts("xhs")).filter(a => a.sub_type === "pc");
      setAccounts(items);
      setAccountId(id => items.some(a => a.id === id) ? id : items[0]?.id);
    } catch { setError("账号列表加载失败，请重试。"); }
    finally { setLoadingAccounts(false); }
  }
  useEffect(() => { void loadAccounts(); }, []);

  async function loadModels() {
    try {
      const items = (await fetchModelConfigs("text")).items;
      setModels(items);
      setModelId(id => items.some(m => m.id === id) ? id : items.find(m => m.is_default)?.id || items[0]?.id);
      setPreprocessModelId(id => items.some(m => m.id === id) ? id : items.find(m => m.is_default)?.id || items[0]?.id);
    } catch { setError("模型配置加载失败，请重试。"); }
  }
  useEffect(() => { void loadModels(); }, []);

  async function loadHistory(page = 1) {
    try {
      const result = await fetchCreativeHistory(page, historyPageSize);
      setHistory(result.items); setHistoryPage(result.page); setHistoryTotal(result.total);
    } catch { /* explicit refresh remains available */ }
  }
  useEffect(() => { void loadHistory(); }, []);

  useEffect(() => {
    if (!collecting || !collection) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const updated = await fetchCreativeRun(collection!.id);
        if (disposed) return;
        setCollection(updated);
        if (updated.status !== "running" && updated.status !== "pending") { void loadHistory(); return; }
      } catch { if (!disposed) setError("进度暂时无法更新，后台任务仍可能在运行。"); }
      if (!disposed) timer = setTimeout(poll, 2000);
    }
    timer = setTimeout(poll, 1000);
    return () => { disposed = true; clearTimeout(timer); };
  }, [collection?.id, collecting]);

  function analysisOptions(): CreativeAnalysisOptions | undefined {
    if (!modelId || (workflow === "langgraph" && !preprocessModelId)) return undefined;
    return {
      model_id: modelId,
      ...(workflow === "langgraph" && preprocessModelId ? { preprocess_model_id: preprocessModelId } : {}),
      idea_count: ideaCount, variant_count: variantCount, title_max_length: titleLength, audience, instruction, workflow,
    };
  }

  async function startCollection(withAnalysis = false) {
    if (!run || selected.length < 3) return;
    if (withAnalysis && !analysisOptions()) return;
    setStarting(true); setError("");
    try { setCollection(await collectCreativeNotes(run.id, selected, withAnalysis ? analysisOptions() : undefined)); void loadHistory(); }
    catch { setError("无法开始采集，请检查选取的笔记与账号状态。"); }
    finally { setStarting(false); }
  }

  async function analyzeAgain() {
    const options = analysisOptions();
    if (!collection || !options) return;
    setStarting(true); setError("");
    try { setCollection(await analyzeCreativeCollection(collection.id, options)); void loadHistory(); }
    catch { setError("无法开始分析，请检查模型配置与有效样本数量。"); }
    finally { setStarting(false); }
  }

  async function stopTask() {
    if (!collection) return;
    try { setCollection(await cancelCreativeRun(collection.id)); void loadHistory(); }
    catch { setError("停止请求未成功，请重试。"); }
  }

  async function resumeFailedTask() {
    if (!collection) return;
    setStarting(true); setError("");
    try { setCollection(await resumeCreativeRun(collection.id)); void loadHistory(); }
    catch { setError("无法接续该任务，请刷新历史任务并检查原模型配置。接续支持保存了完整状态且累计请求次数未用尽的 LangGraph 模型服务失败或执行超时任务。"); }
    finally { setStarting(false); }
  }

  async function openHistory(id: number) {
    try {
      const item = await fetchCreativeRun(id);
      const savedSearch = item.payload.search;
      setAccountId(savedSearch.account_id); setKeyword(savedSearch.keyword); setLimit(savedSearch.limit);
      setSort(savedSearch.sort_type_choice); setNoteTime(savedSearch.note_time); setNoteType(savedSearch.note_type);
      if (item.payload.analysis) {
        setModelId(item.payload.analysis.model_id); setIdeaCount(item.payload.analysis.idea_count);
        setPreprocessModelId(item.payload.analysis.preprocess_model_id || item.payload.analysis.model_id);
        setVariantCount(item.payload.analysis.variant_count || 3);
        setTitleLength(item.payload.analysis.title_max_length || 26);
        setWorkflow(item.payload.analysis.workflow || "multi_agent");
        setAudience(item.payload.analysis.audience); setInstruction(item.payload.analysis.instruction);
      }
      if (item.task_type === "creative_search") { setCollection(null); setRun(item); setSelected((item.payload.items || []).map(n => n.note_id)); }
      else { setRun(null); setSelected([]); setCollection(item); }
    } catch { setError("历史任务加载失败。"); }
  }

  async function search() {
    if (!accountId || !keyword.trim()) return;
    setSearching(true); setError(""); setRun(null); setSelected([]); setCollection(null);
    const payload: CreativeSearch = {
      account_id: accountId, keyword: keyword.trim(), limit,
      sort_type_choice: sort, note_time: noteTime, note_type: noteType,
    };
    try {
      const result = await searchCreativeNotes(payload);
      setRun(result);
      setSelected((result.payload.items || []).map(note => note.note_id));
      void loadHistory();
    } catch { setError("搜索未完成，请检查账号状态或稍后重试。"); }
    finally { setSearching(false); }
  }

  const notes = creativeSamples(run, collection);
  return (
    <Space orientation="vertical" size={20} style={{ width: "100%" }}>
      <div>
        <Title level={3} style={{ marginBottom: 6 }}>热点创作</Title>
        <Text type="secondary">搜索与你有关的热门内容，选取样本，再生成自己的标题、文案与配图方案。</Text>
      </div>
      {error && <Alert type="error" title={error} closable onClose={() => setError("")} />}
      {!loadingAccounts && !accounts.length && <Alert type="info" title="先绑定小红书账号" description={<span>在 <Link to="/platforms/xhs/accounts">账号绑定</Link> 添加 PC 账号并扫码登录，再回来搜索。</span>} />}
      <Card title="1. 搜索内容" extra={<Button icon={<ReloadOutlined />} onClick={() => void loadAccounts()} loading={loadingAccounts}>刷新账号</Button>}>
        <Form layout="vertical" onFinish={() => void search()} disabled={searching || collecting || starting}>
          <Row gutter={16}>
            <Col xs={24} md={8}><Form.Item label="小红书账号"><Select aria-label="小红书账号" placeholder="选择已绑定的 PC 账号" value={accountId} onChange={setAccountId} options={accounts.map(a => ({ value: a.id, label: `${a.nickname || `账号 ${a.id}`}${a.status === "expired" ? "（登录已失效）" : ""}` }))} /></Form.Item></Col>
            <Col xs={24} md={16}><Form.Item label="搜索关键词"><Input aria-label="搜索关键词" value={keyword} onChange={e => setKeyword(e.target.value)} maxLength={120} placeholder="例如：AI 服装设计、AI 模特、虚拟试衣" /></Form.Item></Col>
            <Col xs={12} md={6}><Form.Item label="排序方式"><Select aria-label="排序方式" value={sort} onChange={setSort} options={sortOptions} /></Form.Item></Col>
            <Col xs={12} md={6}><Form.Item label="发布时间"><Select aria-label="发布时间" value={noteTime} onChange={setNoteTime} options={timeOptions} /></Form.Item></Col>
            <Col xs={12} md={6}><Form.Item label="笔记类型"><Select aria-label="笔记类型" value={noteType} onChange={setNoteType} options={[{ value: 2, label: "图文笔记" }, { value: 1, label: "视频笔记" }, { value: 0, label: "不限类型" }]} /></Form.Item></Col>
            <Col xs={12} md={6}><Form.Item label="目标数量"><Select aria-label="目标数量" value={limit} onChange={setLimit} options={[20, 30, 50].map(value => ({ value, label: `${value} 篇` }))} /></Form.Item></Col>
          </Row>
          <Space wrap>
            <Button type="primary" htmlType="submit" icon={<SearchOutlined />} loading={searching} disabled={!accountId || !keyword.trim()}>搜索 {limit} 篇内容</Button>
            <Text type="secondary">自动翻页、去重；保留平台排序，数量不足时据实显示。</Text>
          </Space>
        </Form>
      </Card>
      <Card title={run || !collection ? "2. 选取分析样本" : "2. 已采集样本"} extra={<Tag color="blue">{run || !collection ? `已选 ${selected.length} / ${notes.length} 篇` : `已采集 ${collection.payload.sources?.length || 0} / ${notes.length} 篇`}</Tag>}>
        {run && <Paragraph type="secondary">本批结果：{run.payload.search.keyword} · {sortOptions.find(o => o.value === run.payload.search.sort_type_choice)?.label} · {timeOptions.find(o => o.value === run.payload.search.note_time)?.label}</Paragraph>}
        <Paragraph type="secondary">搜索接口可能不提供发布时间；采集详情后会自动补全。缺失时间不代表采集失败。</Paragraph>
        {run?.payload.warnings?.map(w => <Alert key={w} type="warning" title={w} style={{ marginBottom: 12 }} />)}
        <Table<XhsSearchNote>
          rowKey="note_id" dataSource={notes} loading={searching} scroll={{ x: 760 }}
          pagination={{ pageSize: 10, showSizeChanger: false }}
          locale={{ emptyText: <Empty description={searching ? "正在搜索…" : "输入关键词搜索后，内容会出现在这里"} /> }}
          rowSelection={run ? { selectedRowKeys: selected, onChange: keys => setSelected(keys.map(String)), getCheckboxProps: () => ({ disabled: collecting || starting }) } : undefined}
          columns={[
            { title: "笔记", key: "note", render: (_, note) => <Space align="start">{note.cover_url && <img src={note.cover_url} referrerPolicy="no-referrer" alt="笔记封面" style={{ width: 48, height: 64, objectFit: "cover", borderRadius: 4 }} />}<div><a href={note.note_url} target="_blank" rel="noreferrer">{note.title || "无标题笔记"}</a><br /><Text type="secondary">{note.author_name}</Text></div></Space> },
            { title: "点赞", dataIndex: "likes", width: 90 },
            { title: "收藏", dataIndex: "collects", width: 90 },
            { title: "评论", dataIndex: "comments", width: 90 },
            { title: "发布时间", key: "time", width: 150, render: (_, note) => formatPublicationTime(note) },
          ]}
        />
        <Space wrap>
          {run && <Button onClick={() => void startCollection()} loading={starting} disabled={searching || collecting || selected.length < 3}>仅采集选中的 {selected.length} 篇正文</Button>}
          <Text type="secondary">至少选 3 篇用于综合分析；采集正文和互动数据，图片由你们自行准备。</Text>
        </Space>
      </Card>
      <Card title="3. AI 创作设置" extra={<Button onClick={() => void loadModels()} icon={<ReloadOutlined />}>刷新模型</Button>}>
        {!models.length && <Alert type="info" title="先添加文字模型" description={<span>前往 <Link to="/models">模型配置</Link> 添加文字模型 API；只生成文案和图片提示词，不调用生图接口。</span>} style={{ marginBottom: 16 }} />}
        <Form layout="vertical" disabled={starting || collecting || searching}>
          <Row gutter={16}>
            <Col xs={24} md={12}><Form.Item label="文字模型"><Select aria-label="文字模型" value={modelId} onChange={setModelId} placeholder="选择文字模型" options={models.map(m => ({ value: m.id, label: `${m.name} · ${m.model_name}`, disabled: !m.has_api_key || !m.base_url }))} /></Form.Item></Col>
            {workflow === "langgraph" && <Col xs={24} md={12}><Form.Item label="轻量预处理模型" extra="只处理分批样本摘要与证据；后续创作和评审仍使用上方文字模型。"><Select aria-label="轻量预处理模型" value={preprocessModelId} onChange={setPreprocessModelId} placeholder="选择预处理模型" options={models.map(m => ({ value: m.id, label: `${m.name} · ${m.model_name}`, disabled: !m.has_api_key || !m.base_url }))} /></Form.Item></Col>}
            <Col xs={12} md={6}><Form.Item label="标题数量"><Select aria-label="标题数量" value={ideaCount} onChange={setIdeaCount} options={[1, 3, 5].map(value => ({ value, label: `${value} 个标题` }))} /></Form.Item></Col>
            <Col xs={12} md={6}><Form.Item label="每个标题的文案数"><Select aria-label="每个标题的文案数" value={variantCount} onChange={setVariantCount} options={[2, 3].map(value => ({ value, label: `${value} 个版本` }))} /></Form.Item></Col>
            <Col xs={12} md={6}><Form.Item label="标题长度上限" extra="按字符计（含标点）。过短会迫使模型截断词语。"><Select aria-label="标题长度上限" value={titleLength} onChange={setTitleLength} options={[20, 24, 26, 30, 34].map(value => ({ value, label: `${value} 字` }))} /></Form.Item></Col>
            <Col span={24}><Form.Item label="目标受众"><Input aria-label="目标受众" maxLength={300} value={audience} onChange={e => setAudience(e.target.value)} /></Form.Item></Col>
            <Col span={24}><Form.Item label="创作要求"><Input.TextArea aria-label="创作要求" rows={3} maxLength={2000} value={instruction} onChange={e => setInstruction(e.target.value)} placeholder="默认写可直接发布的作品／灵感分享，参考样本口吻，不夹带制作指导。可补充：语气自然、少用术语；需要教程时请明确写出。" /></Form.Item></Col>
          </Row>
          <Collapse style={{ marginBottom: 16 }} items={[{ key: "workflow", label: "高级生成选项", children: <Form.Item label="生成方式">
            <Select aria-label="生成方式" value={workflow} onChange={setWorkflow} options={[
              { value: "langgraph", label: "LangGraph 分阶段创作与评审（试用）" },
              { value: "multi_agent", label: "多 Agent 分阶段创作与评审（推荐）" }, { value: "legacy", label: "旧版一次综合生成（对照用）" },
            ]} />
          </Form.Item> }]} />
          <Space wrap>
            <Button type="primary" onClick={() => void startCollection(true)} loading={starting} disabled={!analysisOptions() || !run || selected.length < 3 || collecting || searching}>采集并生成 {ideaCount} 个标题 × {variantCount} 版文案</Button>
            {collection && (collection.payload.sources?.length || 0) >= 3 && <Button onClick={() => void analyzeAgain()} disabled={!analysisOptions() || collecting || searching} loading={starting}>使用已采集正文重新生成</Button>}
          </Space>
        </Form>
        <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>正文参考本批有效样本的共同语气、开头、分段与互动方式，保持原创。每个标题对应 {variantCount} 版不同文案，每版都有独立配图建议与提示词；共生成 {ideaCount * variantCount} 版，数量越多通常耗时和费用越高。</Paragraph>
        {workflow !== "legacy" && <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>当前使用{workflow === "langgraph" ? " LangGraph 试用流程" : "多 Agent"}：样本 → 受众 → 选题 → 标题 → 正文 → 配图 → 终审。{workflow === "langgraph" ? `配图与终审逐篇处理，样本分批较多时标题和正文评审也会拆到单项；每批完成后保存进度。当前数量首次全部通过约需 ${7 + 2 * ideaCount + 2 * ideaCount * variantCount} 次模型请求，累计上限会按采集数量和输出批次数自适应（至少 72 次、最多 512 次），每轮执行窗口为 90 分钟。` : "首次全部通过通常为 10 次模型请求，任务总计最多 36 次，每轮执行窗口为 30 分钟。"}每个生产阶段最多质量返工 3 次，执行窗口到期后不再开始新请求。模型评分仅供参考，仍需人工确认。</Paragraph>}
        <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>点击生成会将样本文字发送给所选模型服务，按其规则计费；每次请求读取超时为 3 分钟。多 Agent 与 LangGraph 遇到临时网络故障最多额外重试 5 次，通常分别等待 15、30、60、90、120 秒；格式修复另计，最多一次。超时的原请求仍可能完成并计费。LangGraph 模型服务失败或执行窗口到期，可在累计请求次数未用尽时从失败阶段继续；停止任务无法撤回已发出的请求。</Paragraph>
      </Card>
      {collection && <Card title={`创作任务 #${collection.id} · ${collection.payload.search.keyword}`} extra={collecting && <Button onClick={() => void stopTask()}>停止任务</Button>}>
        <Text>{collection.status === "failed" || collection.status === "cancelled" ? statusNames[collection.status] : collection.payload.stage || statusNames[collection.status] || collection.status}</Text>
        <Progress percent={collection.progress} status={collection.status === "failed" ? "exception" : collecting ? "active" : "normal"} />
        {collection.payload.pipeline && <CreativePipelineProgress pipeline={collection.payload.pipeline} />}
        <Paragraph>已保存 {collection.payload.sources?.length || 0} 篇正文，失败 {collection.payload.failures?.length || 0} 篇。</Paragraph>
        {collection.payload.error && <Alert type="error" title={collection.payload.error} />}
        {collection.resume_available && <Alert type="info" title="可以从失败阶段继续" description={<Space orientation="vertical">
          <Text>复用原任务的素材、创作设置、已完成稿件和有效批次，继续未完成部分。本轮执行窗口为 90 分钟，累计请求次数不清零。会再次调用原模型，可能产生费用；不会重新采集或重跑已完成阶段。若是模型超时或中转站中断，建议先等待几分钟再继续，避免连续失败。</Text>
          <Button type="primary" onClick={() => void resumeFailedTask()} loading={starting} disabled={collecting || searching}>从失败阶段继续</Button>
        </Space>} />}
        {collection.payload.failures?.map(f => <Paragraph key={f.note_id} type="secondary">{f.title || f.note_id}：{f.reason}</Paragraph>)}
      </Card>}
      {collection?.payload.result && <CreativeResults key={`results-${collection.id}`} run={collection} onUpdate={setCollection} />}
      {collection?.payload.pipeline && <CreativeReviewAudit key={`audit-${collection.id}`} pipeline={collection.payload.pipeline} />}
      <Card title={`历史任务${historyTotal ? `（共 ${historyTotal} 条）` : ""}`} extra={<Button onClick={() => void loadHistory(historyPage)} icon={<ReloadOutlined />}>刷新</Button>}>
        <Space wrap>{history.map(item => <Button key={item.id} disabled={collecting || starting || searching} onClick={() => void openHistory(item.id)}>#{item.id} {item.keyword} · {item.task_type === "creative_search" ? "搜索" : item.task_type === "creative_analysis" ? "分析" : "采集/创作"} · {statusNames[item.status] || item.status}</Button>)}</Space>
        {!history.length && <Text type="secondary">完成搜索或采集后，会保留历史记录。</Text>}
        {historyTotal > historyPageSize && <Pagination current={historyPage} pageSize={historyPageSize} total={historyTotal} showSizeChanger={false} showQuickJumper onChange={page => void loadHistory(page)} style={{ marginTop: 16 }} />}
      </Card>
    </Space>
  );
}
