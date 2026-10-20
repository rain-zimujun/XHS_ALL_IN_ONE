import { Alert, Card, Divider, Space, Tag, Typography } from "antd";

import type { CreativeRun } from "../../../types/creative";

const { Paragraph, Title, Text } = Typography;

export function CreativeResults({ run }: { run: CreativeRun }) {
  const { result, analysis_metadata: metadata, sources = [] } = run.payload;
  if (!result) return null;
  return <Space orientation="vertical" size={16} style={{ width: "100%" }} data-testid="creative-results">
    <Card title="4. 创作方案与配图建议">
      <Alert type="info" title={metadata?.disclaimer || "内容仅供创作参考，不保证流量；配图建议未分析原图。"} style={{ marginBottom: 16 }} />
      <Paragraph>{result.summary}</Paragraph>
      {result.patterns.map((p, i) => <Paragraph key={i}>· {p}</Paragraph>)}
      <Text type="secondary">基于 {metadata?.sample_count || sources.length} 篇有效正文 · 模型 {metadata?.model_name} · 每篇最多使用前 {metadata?.excerpt_chars_per_note} 字符进行分析，完整正文保留在任务中。</Text>
    </Card>
    {result.ideas.map((idea, index) => <Card key={index} title={`方案 ${index + 1}`}>
      <Title level={4} copyable>{idea.title}</Title>
      <Paragraph type="secondary">推荐理由：{idea.reason}</Paragraph>
      <Paragraph copyable style={{ whiteSpace: "pre-wrap" }}>{idea.body}</Paragraph>
      <Paragraph copyable={{ text: idea.tags.map(t => `#${t.replace(/^#/, "")}`).join(" ") }}>{idea.tags.map((t, i) => <Tag key={i}>{t}</Tag>)}</Paragraph>
      <Title level={5}>备选标题</Title>
      {idea.alternative_titles.map((t, i) => <Paragraph key={i} copyable>{t}</Paragraph>)}
      <Divider />
      <Title level={5}>配图脚本</Title>
      <Paragraph copyable>封面文案：{idea.cover_text}</Paragraph>
      {idea.image_plan.map((plan, i) => <Card key={i} size="small" title={`图 ${i + 1} · ${plan.purpose}`} style={{ marginBottom: 12 }}>
        <Paragraph>{plan.composition}</Paragraph>
        <Paragraph><Text strong>生图提示词</Text></Paragraph>
        <Paragraph copyable style={{ whiteSpace: "pre-wrap" }}>{plan.prompt}</Paragraph>
        <Paragraph><Text strong>负向提示词</Text></Paragraph>
        <Paragraph copyable>{plan.negative_prompt}</Paragraph>
      </Card>)}
      <Text type="secondary">参考样本：</Text>
      <Space wrap>{idea.source_note_ids.map(id => {
        const source = sources.find(n => n.note_id === id);
        return <a key={id} href={source?.note_url} target="_blank" rel="noreferrer">{source?.title || id}</a>;
      })}</Space>
    </Card>)}
  </Space>;
}
