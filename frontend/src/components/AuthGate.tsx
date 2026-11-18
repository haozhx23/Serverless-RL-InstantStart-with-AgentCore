import { Button, Input } from "antd";
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { useAuthStatus } from "../api/hooks";
import { api } from "../api/client";

// Only shown when the backend was started with SRL_AUTH_TOKEN (needed off loopback).
export default function AuthGate({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  const status = useAuthStatus();
  const qc = useQueryClient();
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  if (status.isLoading) return null;
  if (status.error) {
    return <div style={{ padding: 40 }}>{t("shell.auth.backendDown", { error: String(status.error) })}</div>;
  }
  if (!status.data?.required || status.data.authenticated) return <>{children}</>;
  const submit = async () => {
    try {
      await api("/api/auth/login", { body: { token } });
      setError(null);
      await qc.invalidateQueries();
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <div style={{ minHeight: "100%", display: "grid", placeItems: "center" }}>
      <div className="section-card" style={{ width: 380, padding: 24 }}>
        <h1 className="display" style={{ fontSize: 20, marginTop: 0 }}>Serverless RL InstantStart</h1>
        <p className="secondary" style={{ fontSize: 13 }}>{t("shell.auth.prompt")}</p>
        <Input.Password value={token} onChange={(e) => setToken(e.target.value)} onPressEnter={submit} autoFocus
                        placeholder={t("shell.auth.placeholder")} />
        {error && <p style={{ color: "var(--error)", fontSize: 13 }}>{error}</p>}
        <Button type="primary" block style={{ marginTop: 12 }} onClick={submit}>{t("shell.auth.signIn")}</Button>
      </div>
    </div>
  );
}
