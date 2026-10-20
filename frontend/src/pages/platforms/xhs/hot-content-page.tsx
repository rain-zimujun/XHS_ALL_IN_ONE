import { ReloadOutlined, SearchOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Col, Empty, Form, Input, Progress, Row, Select, Space, Table, Tag, Typography } from "antd";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { cancelCreativeRun, collectCreativeNotes, fetchAccounts, fetchCreativeHistory, fetchCreativeRun, searchCreativeNotes } from "../../../lib/api";
import type { PlatformAccount, XhsSearchNote } from "../../../types";
import type { CreativeHistoryItem, CreativeRun, CreativeSearch } from "../../../types/creative";

const { Title, Text, Paragraph } = Typography;
const sortOptions = [
  { value: 0, label: "综合排序" }, { value: 1, label: "最新发布" },
  { value: 2, label: "最多点赞" }, { value: 4, label: "最多收藏" }, { value: 3, label: "最多评论" },
];
const timeOptions = [
  { value: 0, label: "不限时间" }, { value: 1, label: "一天内" },
  { value: 2, label: "一周内" }, { value: 3, label: "半年内" },
];

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

  async function loadHistory() {
    try { setHistory(await fetchCreativeHistory()); } catch { /* explicit refresh remains available */ }
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

  async function startCollection() {
    if (!run || selected.length < 3) return;
    setStarting(true); setError("");
    try { setCollection(await collectCreativeNotes(run.id, selected)); void loadHistory(); }
    catch { setError("无法开始采集，请检查选取的笔记与账号状态。"); }
    finally { setStarting(false); }
  }

  async function openHistory(id: number) {
    try {
      const item = await fetchCreativeRun(id);
      if (item.task_type === "creative_search") { setRun(item); setSelected((item.payload.items || []).map(n => n.note_id)); }
      else setCollection(item);
    } catch { setError("历史任务加载失败。"); }
  }

  async function search() {
    if (!accountId || !keyword.trim()) return;
    setSearching(true); setError(""); setRun(null); setSelected([]);
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

  const notes = run?.payload.items || [];
  return (
    <Space orientation="vertical" size={20} style={{ width: "100%" }}>
      <div>
        <Title level={3} style={{ marginBottom: 6 }}>热点创作</Title>
        <Text type="secondary">搜索与你有关的热门内容，选取样本，再生成自己的标题、文案与配图方案。</Text>
      </div>
      {error && <Alert type="error" title={error} closable onClose={() => setError("")} />}
      {!loadingAccounts && !accounts.length && <Alert type="info" title="先绑定小红书账号" description={<span>在 <Link to="/platforms/xhs/accounts">账号矩阵</Link> 添加 PC 账号并扫码登录，再回来搜索。</span>} />}
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
      <Card title="2. 选取分析样本" extra={<Tag color="blue">已选 {selected.length} / {notes.length} 篇</Tag>}>
        {run && <Paragraph type="secondary">本批结果：{run.payload.search.keyword} · {sortOptions.find(o => o.value === run.payload.search.sort_type_choice)?.label} · {timeOptions.find(o => o.value === run.payload.search.note_time)?.label}</Paragraph>}
        {run?.payload.warnings?.map(w => <Alert key={w} type="warning" title={w} style={{ marginBottom: 12 }} />)}
        <Table<XhsSearchNote>
          rowKey="note_id" dataSource={notes} loading={searching} scroll={{ x: 760 }}
          pagination={{ pageSize: 10, showSizeChanger: false }}
          locale={{ emptyText: <Empty description={searching ? "正在搜索…" : "输入关键词搜索后，内容会出现在这里"} /> }}
          rowSelection={{ selectedRowKeys: selected, onChange: keys => setSelected(keys.map(String)), getCheckboxProps: () => ({ disabled: collecting || starting }) }}
          columns={[
            { title: "笔记", key: "note", render: (_, note) => <Space align="start">{note.cover_url && <img src={note.cover_url} referrerPolicy="no-referrer" alt="笔记封面" style={{ width: 48, height: 64, objectFit: "cover", borderRadius: 4 }} />}<div><a href={note.note_url} target="_blank" rel="noreferrer">{note.title || "无标题笔记"}</a><br /><Text type="secondary">{note.author_name}</Text></div></Space> },
            { title: "点赞", dataIndex: "likes", width: 90 },
            { title: "收藏", dataIndex: "collects", width: 90 },
            { title: "评论", dataIndex: "comments", width: 90 },
            { title: "发布时间", key: "time", width: 130, render: (_, note) => { const n = Number(note.timestamp); return n > 0 && Number.isFinite(n) ? new Date(n > 1e12 ? n : n * 1000).toLocaleDateString("zh-CN") : "暂未返回"; } },
          ]}
        />
        <Space wrap>
          <Button type="primary" onClick={() => void startCollection()} loading={starting} disabled={searching || collecting || selected.length < 3}>采集选中的 {selected.length} 篇正文</Button>
          <Text type="secondary">至少选 3 篇用于综合分析；采集正文和互动数据，图片由你们自行准备。</Text>
        </Space>
      </Card>
      {collection && <Card title={`采集任务 #${collection.id}`} extra={collecting && <Button onClick={async () => { setCollection(await cancelCreativeRun(collection.id)); }}>停止任务</Button>}>
        <Text>{collection.payload.stage || collection.status}</Text>
        <Progress percent={collection.progress} status={collection.status === "failed" ? "exception" : collecting ? "active" : "normal"} />
        <Paragraph>已保存 {collection.payload.sources?.length || 0} 篇正文，失败 {collection.payload.failures?.length || 0} 篇。</Paragraph>
        {collection.payload.error && <Alert type="error" title={collection.payload.error} />}
        {collection.payload.failures?.map(f => <Paragraph key={f.note_id} type="secondary">{f.title || f.note_id}：{f.reason}</Paragraph>)}
      </Card>}
      <Card title="历史任务" extra={<Button onClick={() => void loadHistory()} icon={<ReloadOutlined />}>刷新</Button>}>
        <Space wrap>{history.map(item => <Button key={item.id} disabled={collecting || starting || searching} onClick={() => void openHistory(item.id)}>#{item.id} {item.keyword} · {item.task_type === "creative_search" ? "搜索" : "采集"} · {item.status}</Button>)}</Space>
        {!history.length && <Text type="secondary">完成搜索或采集后，会保留历史记录。</Text>}
      </Card>
    </Space>
  );
}
