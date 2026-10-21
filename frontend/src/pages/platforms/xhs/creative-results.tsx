import { Alert, Button, Card, Col, Collapse, Row, Select, Space, Tabs, Tag, Typography } from "antd";
import { useState } from "react";

import { downloadExportFile, exportCreativeReport } from "../../../lib/api";
import { currentDecision } from "../../../lib/creative-reviews";
import { ReviewDetails } from "./creative-reviews";

import type { CreativeIdea, CreativePipeline, CreativeQualityReview, CreativeRun, CreativeVariant } from "../../../types/creative";

const { Paragraph, Title, Text } = Typography;

function QualityBadge({ label, review }: { label: string; review?: CreativeQualityReview }) {
  if (!review) return null;
  return <Tag color={review.passed ? "success" : "error"} data-testid={`quality-${label}`}>
    {label} {review.overall_score.toFixed(1)} 分 · {review.passed ? "通过" : "待修改"}
  </Tag>;
}

function VariantContent({ title, variant }: { title: string; variant: CreativeVariant }) {
  const [copied, setCopied] = useState<"body" | "note" | null>(null);
  const [copyError, setCopyError] = useState(false);
  const tags = variant.tags.map(t => `#${t.replace(/^#+/, "")}`).join(" ");
  async function copyNote(kind: "body" | "note") {
    setCopyError(false);
    try {
      await navigator.clipboard.writeText(kind === "body" ? variant.body : [title, variant.body, tags].join("\n\n"));
      setCopied(kind);
    } catch { setCopyError(true); }
  }
  return <Row gutter={[16, 16]} data-testid="creative-variant-content">
    <Col xs={24} xl={12}>
      <Card size="small" title="正文文案">
        {copyError && <Alert type="warning" title="浏览器未允许复制，请选中正文手动复制。" style={{ marginBottom: 12 }} />}
        <Space wrap style={{ marginBottom: 16 }}>
          <Button onClick={() => void copyNote("body")}>{copied === "body" ? "已复制正文" : "复制正文"}</Button>
          <Button onClick={() => void copyNote("note")}>{copied === "note" ? "已复制完整笔记" : "复制完整笔记"}</Button>
        </Space>
        <Paragraph copyable={{ text: variant.body }} style={{ whiteSpace: "pre-wrap" }} data-testid="creative-body">{variant.body}</Paragraph>
        <Title level={5}>话题标签</Title>
        <Paragraph copyable={{ text: tags }}>{variant.tags.map((tag, i) => <Tag key={i}>{tag}</Tag>)}</Paragraph>
      </Card>
    </Col>
    <Col xs={24} xl={12}>
      <Card size="small" title="配图建议与生图提示词" data-testid="creative-images">
        <Paragraph type="secondary">以下图片与当前文案对应，切换文案时一起切换；图片由你自行准备。</Paragraph>
        <Paragraph copyable={{ text: variant.cover_text }}>封面文字：{variant.cover_text}</Paragraph>
        {variant.image_plan.map((plan, i) => <Card key={i} size="small" title={`图 ${i + 1} · ${plan.purpose}`} style={{ marginBottom: 12 }}>
          <Text strong>配图建议</Text>
          <Paragraph style={{ whiteSpace: "pre-wrap" }}>{plan.composition}</Paragraph>
          <Text strong>生图提示词</Text>
          <Paragraph copyable style={{ whiteSpace: "pre-wrap" }}>{plan.prompt}</Paragraph>
          <Text strong>负向提示词</Text>
          <Paragraph copyable>{plan.negative_prompt}</Paragraph>
        </Card>)}
      </Card>
    </Col>
  </Row>;
}

function IdeaContent({ idea, sources, pipeline }: { idea: CreativeIdea; sources: NonNullable<CreativeRun["payload"]["sources"]>; pipeline?: CreativePipeline }) {
  const legacy = !idea.variants;
  const variants: CreativeVariant[] = idea.variants || [{
    label: "原有文案", body: idea.body, tags: idea.tags, cover_text: idea.cover_text, image_plan: idea.image_plan,
  }];
  return <Space orientation="vertical" size={16} style={{ width: "100%" }}>
    <Card title="文案版本">
      <Paragraph type="secondary">当前标题</Paragraph>
      <Title level={4} copyable data-testid="creative-selected-title">{idea.title}</Title>
      <Space wrap style={{ marginBottom: 12 }}><QualityBadge label="标题" review={idea.quality?.title} /></Space>
      {legacy && <Alert type="info" title="这是旧版结果，仅保留原有的 1 版文案和配图。" description="如需每个标题有 2～3 版，请点击上方“使用已采集正文重新生成”（会调用文字模型）。仅切换标题或文案不会调用模型。" style={{ marginBottom: 16 }} />}
      <Tabs defaultActiveKey="0" destroyOnHidden items={variants.map((variant, i) => ({
        key: String(i), label: `文案 ${i + 1} · ${variant.label}`,
        children: <>
          <Space wrap style={{ marginBottom: 12 }}>
            <QualityBadge label="正文" review={variant.quality?.copy} />
            <QualityBadge label="终审" review={variant.quality?.final} />
          </Space>
          <VariantContent title={idea.title} variant={variant} />
          {variant.quality && <Collapse style={{ marginTop: 16 }} items={[{ key: "review", label: "当前文案评审详情（正文与终审）", children: <>
            <Title level={5}>正文评审</Title>
            <ReviewDetails quality={variant.quality.copy} decision={currentDecision(pipeline, "copy", variant.variant_id)} />
            <Title level={5}>最终评审</Title>
            <ReviewDetails quality={variant.quality.final} decision={currentDecision(pipeline, "final_review", variant.variant_id)} />
          </> }]} />}
        </>,
      }))} />
      {variants.some(variant => variant.quality) && <Text type="secondary">正文和终审评分对应当前文案版本。</Text>}
    </Card>
    {idea.quality?.title && <Collapse items={[{ key: "title-review", label: "当前标题评审详情", children: <ReviewDetails quality={idea.quality.title} decision={currentDecision(pipeline, "title", idea.candidate_id)} /> }]} />}
    <Collapse items={[{
      key: "references", label: "推荐理由与参考来源", children: <>
        <Paragraph>{idea.reason}</Paragraph>
        {legacy && idea.alternative_titles && <>
          <Text type="secondary">旧版备选标题（仅供替换措辞，不代表新增文案）</Text>
          {idea.alternative_titles.map((title, i) => <Paragraph key={i} copyable>{title}</Paragraph>)}
        </>}
        <Space wrap>{idea.source_note_ids.map(id => {
          const source = sources.find(n => n.note_id === id);
          return source?.note_url
            ? <a key={id} href={source.note_url} target="_blank" rel="noreferrer">{source.title || id}</a>
            : <Text key={id}>{source?.title || id}</Text>;
        })}</Space>
      </>,
    }]} />
  </Space>;
}

export function CreativeResults({ run, onUpdate }: { run: CreativeRun; onUpdate: (updated: CreativeRun) => void }) {
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState("");
  const [selectedIdeaId, setSelectedIdeaId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"original" | "title" | "copy" | "final">("original");
  const [filterBy, setFilterBy] = useState<"title" | "copy" | "final">("final");
  const [minimumScore, setMinimumScore] = useState(0);
  const { result, analysis_metadata: metadata, sources = [] } = run.payload;
  if (!result) return null;
  const scoreFor = (idea: CreativeIdea, metric: "title" | "copy" | "final") => {
    if (metric === "title") return idea.quality?.title?.overall_score;
    const variantScores = (idea.variants || []).map(variant => metric === "copy" ? variant.quality?.copy?.overall_score : variant.quality?.final?.overall_score).filter((score): score is number => typeof score === "number");
    return variantScores.length ? Math.max(...variantScores) : undefined;
  };
  const sortedIdeas = [...result.ideas].sort((a, b) => {
    if (sortBy === "original") return result.ideas.indexOf(a) - result.ideas.indexOf(b);
    const metric = sortBy;
    return (scoreFor(b, metric) ?? -1) - (scoreFor(a, metric) ?? -1);
  });
  const visibleIdeas = sortedIdeas.filter(idea => {
    if (minimumScore === 0) return true;
    const score = scoreFor(idea, filterBy);
    return score !== undefined && score >= minimumScore;
  });
  const selectedIdea = visibleIdeas.find(idea => {
    const index = result.ideas.indexOf(idea);
    return (idea.idea_id || `legacy-${index}`) === selectedIdeaId;
  }) || visibleIdeas[0];
  const selectedIndex = selectedIdea ? result.ideas.indexOf(selectedIdea) : 0;
  const idea = selectedIdea;
  async function exportAgain() {
    setExporting(true); setError("");
    try { onUpdate(await exportCreativeReport(run.id)); }
    catch { setError("报告导出失败，请检查本地输出目录是否可写。"); }
    finally { setExporting(false); }
  }
  return <Space orientation="vertical" size={16} style={{ width: "100%" }} data-testid="creative-results">
    <Card title="4. 创作结果 · 先选标题，再选文案">
      <Alert type="info" title={metadata?.disclaimer || "内容仅供创作参考，不保证流量；配图建议未分析原图。"} style={{ marginBottom: 16 }} />
      <Title level={5}>标题</Title>
      <Paragraph type="secondary">点击标题，在下方选择它的文案版本，正文与配图建议会同步显示。切换不会再次调用模型。</Paragraph>
      {result.ideas.some(item => item.quality?.title || item.variants?.some(v => v.quality)) && <Space wrap style={{ marginBottom: 12 }}>
        <Select aria-label="评分排序" value={sortBy} onChange={setSortBy} style={{ minWidth: 190 }} options={[
          { value: "original", label: "按原顺序" }, { value: "title", label: "标题评分从高到低" },
          { value: "copy", label: "正文评分从高到低" }, { value: "final", label: "终审评分从高到低" },
        ]} />
        <Select aria-label="评分筛选依据" value={filterBy} onChange={setFilterBy} style={{ minWidth: 150 }} options={[
          { value: "title", label: "按标题评分筛选" }, { value: "copy", label: "按正文评分筛选" }, { value: "final", label: "按终审评分筛选" },
        ]} />
        <Select aria-label="最低评分" value={minimumScore} onChange={setMinimumScore} style={{ minWidth: 150 }} options={[
          { value: 0, label: "显示全部评分" }, { value: 6.5, label: "至少 6.5 分" }, { value: 8, label: "至少 8 分" },
        ]} />
        <Text type="secondary">正文/终审按该标题下最高分版本排序或筛选；开启最低分后，无评分旧稿不会被误认为达标。</Text>
      </Space>}
      <Row gutter={[12, 12]}>
        {visibleIdeas.map(item => {
          const i = result.ideas.indexOf(item);
          return <Col xs={24} md={12} xl={8} key={item.idea_id || i}>
          <Button block type={item === idea ? "primary" : "default"} aria-pressed={item === idea}
            data-testid={`creative-title-${i}`} onClick={() => setSelectedIdeaId(item.idea_id || `legacy-${i}`)}
            style={{ height: "auto", minHeight: 72, whiteSpace: "normal", textAlign: "left", padding: "12px 16px" }}>
            <Space orientation="vertical" align="start"><span>标题 {i + 1} · {item.title}<br /><small>{item.variants?.length || 1} 版文案及对应配图</small></span>
              <QualityBadge label="标题" review={item.quality?.title} /></Space>
          </Button>
          </Col>;
        })}
        {!visibleIdeas.length && <Col span={24}><Text type="secondary">没有达到当前分数条件的标题。</Text></Col>}
      </Row>
    </Card>
    {idea && <IdeaContent key={`${run.id}-${selectedIndex}`} idea={idea} sources={sources} pipeline={run.payload.pipeline} />}
    <Collapse items={[{
      key: "analysis", label: "样本分析与文案风格依据", children: <>
        <Title level={5}>本批样本的文案风格</Title>
        <Paragraph>{result.style_summary || "此旧版结果未单独记录风格分析，重新生成后可查看。"}</Paragraph>
        <Title level={5}>内容特点</Title>
        <Paragraph>{result.summary}</Paragraph>
        {result.patterns.map((pattern, i) => <Paragraph key={i}>· {pattern}</Paragraph>)}
        <Text type="secondary">基于 {metadata?.sample_count || sources.length} 篇有效正文 · 模型 {metadata?.model_name} · {metadata?.excerpt_strategy === "head_and_tail" ? "长正文保留首尾" : "正文从开头截取"}，每篇最多使用 {metadata?.excerpt_chars_per_note} 字符进行分析，完整正文保留在任务中。</Text>
      </>,
    }]} />
    <Card title="报告下载">
      {(error || run.payload.export_error) && <Alert type="warning" title={error || run.payload.export_error} style={{ marginBottom: 12 }} />}
      <Space wrap>
        {run.payload.report_files?.map(file => <Button key={file.file_name} onClick={async () => {
          setError("");
          try { await downloadExportFile(file.download_url, file.file_name); }
          catch { setError("报告下载失败，可尝试重新导出后下载。"); }
        }}>下载 {file.file_name.endsWith(".html") ? "HTML 阅读版" : "JSON 完整数据"}</Button>)}
        <Button onClick={() => void exportAgain()} loading={exporting}>重新导出报告</Button>
      </Space>
      {run.payload.report_directory && <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>服务端固定输出文件夹：<Text code copyable>{run.payload.report_directory}</Text></Paragraph>}
    </Card>
  </Space>;
}
