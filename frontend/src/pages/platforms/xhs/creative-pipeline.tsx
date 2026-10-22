import { Alert, Col, Collapse, Row, Space, Tag, Typography } from "antd";
import type { CreativePipeline, CreativeStageStatus } from "../../../types/creative";
import { readFailedAttempts } from "../../../lib/creative-diagnostics";

export const pipelineStageNames: Record<string, string> = {
  sample_analysis: "样本分析", audience: "读者需求", topic: "选题与评审", title: "标题与评审",
  copy: "正文与评审", image: "配图策划", final_review: "最终评审",
};
const statuses: Record<CreativeStageStatus, { label: string; color: string }> = {
  pending: { label: "等待", color: "default" }, running: { label: "进行中", color: "processing" },
  completed: { label: "已完成", color: "success" }, failed: { label: "失败", color: "error" },
  skipped: { label: "跳过", color: "default" }, revision_required: { label: "待修改", color: "warning" },
  invalidated: { label: "上游变更，待重新校验", color: "warning" }, cancelled: { label: "已停止", color: "default" },
};

export function CreativePipelineProgress({ pipeline }: { pipeline: CreativePipeline }) {
  const failures = Object.entries(pipeline.stages).flatMap(([key, item]) => {
    const attempts = readFailedAttempts(item.metadata);
    if (!attempts.length && !item.error) return [];
    return [{
      key,
      label: `${pipelineStageNames[key] || key} · 失败详情`,
      children: <Space orientation="vertical" style={{ width: "100%" }}>
        {item.error && <Typography.Text type="danger">{item.error}</Typography.Text>}
        {pipeline.error_code && key === pipeline.current_stage && <Typography.Text>错误类型：{pipeline.error_code}</Typography.Text>}
        {attempts.map((attempt, index) => <div key={index}>
          <Typography.Text strong>第 {attempt.attempt} 次 · 耗时 {(attempt.duration_ms / 1000).toFixed(1)} 秒 · {attempt.status}</Typography.Text>
          <Typography.Paragraph copyable style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{attempt.error}</Typography.Paragraph>
        </div>)}
        <Typography.Text type="secondary">详情仅含字段位置与检查原因，不包含完整模型响应。历史任务只能显示当时已保存的诊断信息。</Typography.Text>
      </Space>,
    }];
  });
  return <Space orientation="vertical" size={12} style={{ width: "100%", marginBottom: 16 }} data-testid="creative-pipeline">
    <Typography.Text strong>{pipeline.activity || "多 Agent 创作"}</Typography.Text>
    <Row gutter={[8, 12]} aria-label="生成阶段进度">
      {Object.entries(pipelineStageNames).map(([key, name]) => {
        const item = pipeline.stages[key];
        if (!item) return null;
        const status = statuses[item.status] || statuses.pending;
        return <Col xs={24} sm={12} lg={6} key={key} data-testid={`creative-stage-${key}`}>
          <Space orientation="vertical" size={2}>
            <Typography.Text>{name}</Typography.Text>
            <Tag color={status.color}>{status.label}</Tag>
            <Typography.Text type="secondary">版本 {item.revision} · 质量返工 {item.quality_revisions}/{pipeline.max_quality_revisions}</Typography.Text>
          </Space>
        </Col>;
      })}
    </Row>
    <Typography.Text type="secondary">已发出 {pipeline.request_count} / {pipeline.max_requests} 次模型请求（包括评审、格式修复和质量返工）。阶段进度不是剩余耗时估计。</Typography.Text>
    {pipeline.status === "revision_required" && <Alert type="warning" title="达到返工上限，仍有内容未通过评审" description="稿件、评分和修改意见已保留，不会将其作为已通过成稿。重新生成会创建新任务并再次调用模型，不会自动续跑。" />}
    {pipeline.status === "failed" && <Alert type="info" title="已保存完成阶段的稿件与评审" description="本次没有继续调用模型。若是超时，请先核对中转服务记录，避免重复计费。" />}
    {failures.length > 0 && <Collapse items={failures} />}
    {pipeline.status === "cancelled" && <Typography.Text type="secondary">已停止后续阶段；已发出的模型请求无法撤回。</Typography.Text>}
  </Space>;
}
