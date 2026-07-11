import type { ReactNode } from "react";

interface SectionHeaderProps {
  step: string;
  title: string;
  text: string;
  action?: ReactNode;
}

export function SectionHeader({ step, title, text, action }: SectionHeaderProps) {
  return (
    <div className="section-header">
      <div>
        <span className="eyebrow">{step}</span>
        <h2>{title}</h2>
        <p>{text}</p>
      </div>
      {action && <div className="section-actions">{action}</div>}
    </div>
  );
}

interface MetricCardProps {
  icon: string;
  label: string;
  value: ReactNode;
  detail?: string;
  tone?: "default" | "warning" | "success";
}

export function MetricCard({ icon, label, value, detail, tone = "default" }: MetricCardProps) {
  return (
    <div className={`metric ${tone}`}>
      <span className="metric-icon">{icon}</span>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        {detail && <small>{detail}</small>}
      </div>
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="empty-state">{children}</div>;
}
