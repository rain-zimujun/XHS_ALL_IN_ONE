import { Alert, Card, Collapse, Space, Tag, Typography } from "antd";
import { criterionNames, readReview, reviewFieldNames } from "../../../lib/creative-reviews";
import type { CreativePipeline, CreativeQualityReview, CreativeReviewDecision } from "../../../types/creative";

const { Paragraph, Text } = Typography;
const stageNames: Record<string, string> = { topic: "选题", title: "标题", copy: "正文", image: "配图", final_review: "终审" };

export function ReviewDetails({ decision, quality }: { decision?: CreativeReviewDecision; quality?: CreativeQualityReview }) {
  const review = decision?.quality || quality;
  if (!review) return <Text type="secondary">尚无可展示的评审记录。</Text>;
  return <div data-testid="creative-review-detail" style={{ overflowWrap: "anywhere" }}>
    <Paragraph><Tag color={review.passed ? "success" : "warning"}>{review.passed ? "通过" : "待修改"} · {review.overall_score.toFixed(2)} 分</Tag><Text type="secondary">模型评价仅供辅助，仍需人工确认。</Text></Paragraph>
    {review.scores.map((score, i) => <Paragraph key={i}><Text strong>{criterionNames[score.criterion] || score.criterion} · {score.score} 分：</Text>{score.comment}</Paragraph>)}
    {decision?.issues.map((issue, i) => <Card key={i} size="small" style={{ marginBottom: 12 }} title={<Space wrap>
      <Tag color={issue.severity === "blocking" ? "error" : "gold"}>{issue.severity === "blocking" ? "阻断问题" : "改进建议"}</Tag>
      <span>{reviewFieldNames[issue.field] || issue.field}</span>
      {issue.responsible_stage && <Text type="secondary">负责修改：{stageNames[issue.responsible_stage]}</Text>}
    </Space>}>
      <Paragraph>原文定位：{issue.quote}</Paragraph>
      <Paragraph>问题：{issue.problem}</Paragraph>
      <Paragraph>修改建议：{issue.revision_instruction}</Paragraph>
    </Card>)}
    {review.blocking_issues.map((issue, i) => <Paragraph key={i} type="danger">阻断：{issue}</Paragraph>)}
    {!decision?.issues.length && review.revision_instructions.map((instruction, i) => <Paragraph key={i}>修改建议：{instruction}</Paragraph>)}
    {decision?.preserve.map((value, i) => <Paragraph key={i}>建议保留：{value}</Paragraph>)}
    {decision?.comparison && <Paragraph>版本比较：{decision.comparison}</Paragraph>}
    {review.reviewer_notes && <Paragraph>{review.reviewer_notes}</Paragraph>}
    {!decision && <Text type="secondary">此结果仅保存汇总评分，没有可关联的逐项问题记录。</Text>}
  </div>;
}

function ReviewBatch({ output }: { output: unknown }) {
  const review = readReview(output);
  return <>
    {review.summary && <Paragraph>{review.summary}</Paragraph>}
    {!review.decisions.length && <Paragraph type="secondary">尚无可展示的逐项评审。</Paragraph>}
    <Collapse items={review.decisions.map((decision, i) => ({ key: `${decision.itemId}-${i}`, label: `对象 ${decision.itemId} · ${decision.quality.passed ? "通过" : "待修改"} · ${decision.quality.overall_score.toFixed(2)} 分`, children: <ReviewDetails decision={decision} /> }))} />
    {review.limitations.map((value, i) => <Paragraph key={i} type="secondary">评审局限：{value}</Paragraph>)}
  </>;
}

export function CreativeReviewAudit({ pipeline }: { pipeline: CreativePipeline }) {
  return <Collapse data-testid="creative-review-audit" items={[{ key: "audit", label: "分析与评审详情 · 问题及返工记录", children: <>
    <Alert type="info" title="以下为内部评审记录，不会加入复制正文；历史评分不代表当前成稿已通过。" style={{ marginBottom: 12 }} />
    <Collapse items={Object.entries(stageNames).flatMap(([key, label]) => {
      const stage = pipeline.stages[key];
      if (!stage) return [];
      const history = stage.history || [];
      return [{ key, label: `${label} · 版本 ${stage.revision} · 质量返工 ${stage.quality_revisions} 次`, children: <div data-testid={`review-stage-${key}`}>
        {stage.status === "invalidated" && <Alert type="warning" title="上游已变更，本阶段旧评审失效。" />}
        <Paragraph strong>最近保存的评审</Paragraph>
        <ReviewBatch output={stage.review_result} />
        <Collapse style={{ marginTop: 12 }} items={[{ key: "history", label: `历史过程（${history.length} 条，仅供追溯）`, children: <>
          {history.map((entry, i) => <div key={i} style={{ marginBottom: 16 }}>
            <Paragraph>记录 {i + 1} · {entry.role === "reviewer" ? "评审" : "生成"} · 版本 {entry.revision} · {entry.finished_at}</Paragraph>
            {entry.target_ids?.length > 0 && <Paragraph>本次修改对象：{entry.target_ids.join("、")}</Paragraph>}
            {entry.role === "reviewer" && <ReviewBatch output={entry.output} />}
          </div>)}
        </> }]} />
      </div> }];
    })} />
  </> }]} />;
}
