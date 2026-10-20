import {
  ArrowRightOutlined,
  DatabaseOutlined,
  LockOutlined,
  FileTextOutlined,
  RobotOutlined,
  UserOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  Row,
  Segmented,
  Space,
  Typography,
} from "antd";
import { type FormEvent, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { z } from "zod";

import { useAuth } from "../../hooks/use-auth";

const { Title, Text, Paragraph } = Typography;

type AuthMode = "login" | "register";

const credentialsSchema = z.object({
  username: z
    .string()
    .trim()
    .min(3, "账号至少 3 个字符")
    .max(80, "账号不能超过 80 个字符"),
  password: z
    .string()
    .min(6, "密码至少 6 个字符")
    .max(128, "密码不能超过 128 个字符"),
});

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export function LoginPage() {
  const auth = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mode, setMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event?: FormEvent<HTMLFormElement>) {
    if (event) event.preventDefault();
    setError(null);

    const parsed = credentialsSchema.safeParse({ username, password });
    if (!parsed.success) {
      setError(parsed.error.issues[0]?.message ?? "请检查账号和密码。");
      return;
    }

    if (mode === "register" && password !== confirmPassword) {
      setError("两次输入的密码不一致。");
      return;
    }

    setIsSubmitting(true);
    try {
      if (mode === "login") {
        await auth.login(parsed.data);
      } else {
        await auth.register(parsed.data);
      }
      const from = (
        location.state as { from?: { pathname?: string } } | null
      )?.from?.pathname;
      navigate(from || "/platforms/xhs/hot-content", { replace: true });
    } catch (caughtError) {
      setError(
        errorMessage(
          caughtError,
          mode === "login"
            ? "账号不存在或密码错误，请检查后重试。"
            : "注册失败，该平台账号可能已存在。"
        )
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0a0a",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <Row
        gutter={48}
        align="middle"
        style={{ maxWidth: 960, width: "100%" }}
      >
        {/* Left side: marketing copy */}
        <Col xs={24} md={12}>
          <Space align="center" size={12} style={{ marginBottom: 32 }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: 10,
                background:
                  "linear-gradient(135deg, #1668dc 0%, #4e8ff7 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontWeight: 800,
                fontSize: 18,
                color: "#fff",
              }}
            >
              X
            </div>
            <div>
              <Text
                type="secondary"
                style={{
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  display: "block",
                }}
              >
                Spider_XHS
              </Text>
              <Text strong style={{ fontSize: 14 }}>
                热点创作工作台
              </Text>
            </div>
          </Space>

          <Title
            level={2}
            style={{
              color: "#fff",
              marginBottom: 12,
              lineHeight: 1.4,
            }}
          >
            从热点选题，到自己的文案与配图方案。
          </Title>
          <Paragraph
            type="secondary"
            style={{ fontSize: 15, marginBottom: 40 }}
          >
            绑定小红书账号，搜索筛选内容，让 AI 综合分析并生成标题、正文和配图提示词。图片由你准备，发布由你决定。
          </Paragraph>

          <Row gutter={24}>
            <Col span={8}>
              <Space orientation="vertical"><DatabaseOutlined style={{ fontSize: 24, color: "#4e8ff7" }} /><Text>搜索筛选</Text><Text type="secondary">20 / 30 / 50 篇</Text></Space>
            </Col>
            <Col span={8}>
              <Space orientation="vertical"><RobotOutlined style={{ fontSize: 24, color: "#4e8ff7" }} /><Text>AI 创作</Text><Text type="secondary">文案与配图建议</Text></Space>
            </Col>
            <Col span={8}>
              <Space orientation="vertical"><FileTextOutlined style={{ fontSize: 24, color: "#4e8ff7" }} /><Text>报告留存</Text><Text type="secondary">下载与历史回看</Text></Space>
            </Col>
          </Row>
        </Col>

        {/* Right side: login form */}
        <Col xs={24} md={12}>
          <Card
            style={{
              background: "#1a1a1a",
              borderColor: "#303030",
              borderRadius: 12,
            }}
            styles={{
              body: { padding: "32px 28px" },
            }}
          >
            <Space
              align="center"
              size={8}
              style={{ marginBottom: 20 }}
            >
              <LockOutlined
                style={{ fontSize: 16, color: "rgba(255,255,255,0.65)" }}
              />
              <Text strong style={{ fontSize: 15 }}>
                {mode === "login" ? "平台登录" : "注册平台账号"}
              </Text>
            </Space>

            <div style={{ marginBottom: 24 }}>
              <Segmented
                value={mode}
                onChange={(val) => {
                  setMode(val as AuthMode);
                  setError(null);
                }}
                options={[
                  { label: "登录", value: "login" },
                  { label: "注册", value: "register" },
                ]}
                block
              />
            </div>

            <form onSubmit={handleSubmit}>
              <Form layout="vertical" component="div">
                <Form.Item label="平台账号" style={{ marginBottom: 16 }}>
                  <Input
                    prefix={<UserOutlined />}
                    placeholder="请输入账号"
                    autoComplete="username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    size="large"
                  />
                </Form.Item>

                <Form.Item label="密码" style={{ marginBottom: 16 }}>
                  <Input.Password
                    prefix={<LockOutlined />}
                    placeholder="请输入密码"
                    autoComplete={
                      mode === "login" ? "current-password" : "new-password"
                    }
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    size="large"
                  />
                </Form.Item>

                {mode === "register" && (
                  <Form.Item label="确认密码" style={{ marginBottom: 16 }}>
                    <Input.Password
                      prefix={<LockOutlined />}
                      placeholder="请再次输入密码"
                      autoComplete="new-password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      size="large"
                    />
                  </Form.Item>
                )}

                {error && (
                  <Alert
                    message={error}
                    type="error"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                )}

                <Button
                  type="primary"
                  htmlType="submit"
                  size="large"
                  block
                  loading={isSubmitting}
                  disabled={auth.isChecking}
                  icon={<ArrowRightOutlined />}
                  iconPosition="end"
                >
                  {mode === "login" ? "进入工作台" : "创建并进入"}
                </Button>
              </Form>
            </form>

            <Text
              type="secondary"
              style={{
                display: "block",
                textAlign: "center",
                marginTop: 16,
                fontSize: 12,
              }}
            >
              {mode === "login"
                ? "登录后选择小红书工作区开始运营。"
                : "注册后会自动进入平台选择页。"}
            </Text>
          </Card>
        </Col>
      </Row>
    </div>
  );
}
