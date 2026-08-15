# bochan Web Workbench UI

bochan Webは、データ読込からモデル設定、候補提案、実験結果の追加までを一つのReactワークベンチで扱います。

バックエンドAPI、データアップロード形式、モデル学習、候補生成、Plotly可視化の契約とは分離し、フロントエンド側では画面・状態・設定の責務を明示的に分けています。

## 画面構成

詳細モードの主ワークフローは次の5段階です。

| Step | Page | Purpose |
|---|---|---|
| 1 | Data | CSV/Excel読込、保存モデル・プロジェクト読込、データ概要 |
| 2 | Select | 目的変数、説明変数、数値・カテゴリ・組成式の選択 |
| 3 | Model | モデル、前処理、欠損処理、診断などの設定 |
| 4 | Suggest | 目的・探索範囲・提案件数と、必要に応じた高度な候補生成設定 |
| 5 | Results | 候補表、予測、不確実性、診断、Plotly可視化 |

Results取得後は`Experiment`から実験値を追加し、データ更新→再学習→次候補提案のサイクルへ進めます。

補助フローとして`Conversation`があり、質問形式で必要な設定を組み立てて候補提案まで進められます。

簡易モードでは通常操作を`Data → Select → Results`に絞り、内部では同じworkbench設定・実行経路を利用します。

## 共通レイアウト

- 上部: ブランド、現在の工程、FastAPI接続状態、テーマ・チュートリアル操作
- 左側: 対話モード、簡易/詳細モード切替、ワークフローナビゲーション
- 中央: 現在のページとエラー表示
- 右側: データ、探索条件、最新結果のコンテキスト
- 下部: API状態、モード、行数、候補数
- 実行中: 共通の処理中オーバーレイと進捗表示

画面シェルは`web/src/components/workbench/`に集約し、`App.tsx`はそれらを組み立てるだけのcomposition rootとしています。

主な所有先は次の通りです。

```text
App.tsx
└─ components/workbench/
   ├─ WorkbenchHeader.tsx
   ├─ WorkbenchLeftRail.tsx
   ├─ WorkbenchContextRail.tsx
   ├─ WorkbenchStatusBar.tsx
   ├─ WorkbenchErrorAlert.tsx
   ├─ WorkbenchBusyOverlay.tsx
   ├─ useWorkbenchShell.ts
   ├─ workbenchPages.ts
   └─ workbenchPresentation.ts
```

`useWorkbenchShell.ts`は、補助ページのhash routing、簡易/詳細モードに応じた表示工程、進捗表示、右サイドバー開閉など、分析設定とは独立したshell状態だけを扱います。

## 状態管理

ページ側の公開APIは`useWorkbench()`に統一したままですが、`WorkbenchContext.tsx`自身には個別状態を集中させません。

内部状態は責務ごとのhook/moduleに分割しています。

```text
context/
├─ WorkbenchContext.tsx
│  └─ domain stateを合成し、upload/import/executeを調停
├─ useWorkbenchRuntimeState.ts
│  └─ theme / API health / busy / error / current step
├─ useWorkbenchSelectionState.ts
│  └─ dataset / targets / features / search variables
├─ useWorkbenchRunSettings.ts
│  └─ model / acquisition / search / diagnostics
├─ useWorkbenchResultState.ts
│  └─ latest result / fitted-model signature
├─ workbenchValidation.ts
│  └─ pure validation / derived state
├─ workbenchDefaults.ts
│  └─ dataset読込時の初期値生成
└─ workbenchTypes.ts
   └─ context contract / workflow types
```

この構成により、ページを移動しても一貫した状態を共有しつつ、UI shell、選択状態、モデル設定、validation、API実行を独立して変更できます。

## ページ遷移条件

Contextの`canOpenStep()`が主ワークフローの遷移条件を一元管理します。

- Data: 常に開ける
- Select: データ読込後
- Model: 目的変数と説明変数を選択後
- Suggest: モデル設定が妥当になった後
- Results: 候補生成または保存モデル読込で結果が存在するとき
- Experiment: データと結果の両方が存在するとき

簡易モードではModel/Suggestを直接表示せず、Select画面から同じ設定・実行基盤を利用して候補生成します。

## API

フロントエンドからは既存のWeb APIを使用します。代表的なエンドポイントは次の通りです。

```text
GET  /api/v1/health
POST /api/v1/datasets
POST /api/v1/regression/run
```

モデルartifact、実験履歴、可視化sessionなどの追加APIも各機能moduleから呼び出します。`WorkbenchContext.tsx`の分割はAPI request/response契約を変更しません。

## 開発起動

バックエンド:

```bash
pip install -e ".[web]"
uvicorn bochan.serving.webapp.app:app --reload --port 8000
```

フロントエンド:

```bash
cd web
npm install
npm run dev
```

ブラウザ:

```text
http://localhost:5173
```

## 本番ビルド確認

```bash
cd web
npm run build
```

このビルドではTypeScriptのstrict checkとVite buildを実行します。

UIの責務境界については`tests/test_webapp_workbench_architecture.py`でも静的に検証し、`App.tsx`や`WorkbenchContext.tsx`へ機能が再集中しないようにしています。
