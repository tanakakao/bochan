import ExecutionLogs from "../ExecutionLogs";
import { SectionHeader } from "../components/Common";

export default function LogsPage() {
  return (
    <>
      <SectionHeader
        step="6 · LOGS"
        title="実行履歴を確認する"
        text="FastAPIの構造化JSONLログから、直近の最適化処理とエラー詳細を確認します。"
      />
      <ExecutionLogs />
    </>
  );
}
