import { Collapse, Drawer, Segmented, Space, Typography, message } from "antd";
import { useState } from "react";

import type { PlatformAccount } from "../../types";
import { CookieImportPanel } from "./cookie-import-panel";
import { PhoneLoginPanel } from "./phone-login-panel";
import { QrLoginPanel } from "./qr-login-panel";

type AddAccountDrawerProps = {
  open: boolean;
  onClose: () => void;
  onBound: () => void;
};

type AccountType = "pc" | "creator";
type LoginMethod = "qr" | "phone" | "cookie";

const accountTypeOptions = [
  { label: "PC", value: "pc" as const },
  { label: "Creator", value: "creator" as const },
];

const loginMethodOptions = [
  { label: "二维码", value: "qr" as const },
  { label: "手机验证码", value: "phone" as const },
  { label: "Cookie", value: "cookie" as const },
];

export function AddAccountDrawer({ open, onClose, onBound }: AddAccountDrawerProps) {
  const [accountType, setAccountType] = useState<AccountType>("pc");
  const [method, setMethod] = useState<LoginMethod>("qr");
  const [advancedOpen, setAdvancedOpen] = useState(false);

  function handleConfirmed(account: PlatformAccount) {
    const actionText = account.action === "updated" ? "绑定信息已更新" : "已完成绑定";
    message.success(`${account.nickname || "账号"} ${actionText}`);
    onBound();
  }

  return (
    <Drawer
      title={
        <div>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.45)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
            XHS Account
          </div>
          <div style={{ fontSize: 18, fontWeight: 600, color: "rgba(255,255,255,0.88)" }}>添加小红书账号</div>
        </div>
      }
      placement="right"
      width={420}
      open={open}
      onClose={onClose}
      destroyOnClose
      styles={{
        header: { background: "#1f1f1f", borderBottom: "1px solid #303030" },
        body: { background: "#141414", padding: 24 },
      }}
    >
      <Typography.Paragraph type="secondary">热点创作默认使用 PC 扫码登录。当前：{accountType === "pc" ? "PC 账号" : "Creator 账号"} · {loginMethodOptions.find(option => option.value === method)?.label}。</Typography.Paragraph>
      <Collapse size="small" style={{ marginBottom: 24 }} activeKey={advancedOpen ? ["advanced"] : []}
        onChange={keys => setAdvancedOpen(keys.includes("advanced"))}
        items={[{ key: "advanced", label: "高级登录选项", children: <Space orientation="vertical" style={{ width: "100%" }} size={16}>
          <Segmented block value={accountType} options={accountTypeOptions} onChange={val => setAccountType(val as AccountType)} />
          <Segmented block value={method} options={loginMethodOptions} onChange={val => setMethod(val as LoginMethod)} />
        </Space> }]} />

      {method === "qr" ? (
        <QrLoginPanel accountType={accountType} onConfirmed={handleConfirmed} showAdvanced={advancedOpen} />
      ) : method === "cookie" ? (
        <CookieImportPanel accountType={accountType} onImported={handleConfirmed} />
      ) : (
        <PhoneLoginPanel accountType={accountType} onConfirmed={handleConfirmed} />
      )}
    </Drawer>
  );
}
