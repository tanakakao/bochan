# bochan Web Workbench UI

bochan Webの既存単一目的回帰ワークフローを、malchan Webと共通のデザイン言語を持つページ型ワークベンチへ再構成しています。

バックエンドAPI、データアップロード形式、回帰モデル学習、候補生成、Plotly可視化、JSONLログの契約は変更していません。

## 画面構成

| Step | Page | Purpose |
|---|---|---|
| 1 | Data | CSV/Excel読込、データ概要、プレビュー、列プロファイル |
| 2 | Prepare | 数値目的変数と数値・カテゴリ説明変数の選択 |
| 3 | Optimize | 最適化方向、GPモデル、獲得関数、候補生成設定、探索空間 |
| 4 | Results | 候補表、予測平均・標準偏差・獲得値、Plotly可視化 |
| 5 | Logs | FastAPIの構造化JSONL実行ログ |

## 共通レイアウト

- 上部: ブランド、工程ストリップ、FastAPI接続状態、テーマ切替
- 左側: ページナビゲーション
- 中央: 現在のページ
- 右側: データ、探索条件、最新結果のコンテキスト
- 下部: API状態、行数、候補数

画面幅が狭い場合は右側コンテキストを非表示にし、左ナビゲーションを横スクロール型へ変更します。

## 状態管理

`web/src/context/WorkbenchContext.tsx`で次の状態を共有します。

- 読み込んだデータセット
- 目的変数と説明変数
- 各探索変数の範囲、刻み、固定値
- 最適化方向
- モデルと獲得関数
- 候補生成パラメータ
- 最新の候補生成結果
- API接続状態
- ライト・ダークテーマ

ページを移動してもこれらの状態は維持されます。

## ページ遷移条件

- DataとLogsは常に開けます。
- Prepareはデータ読込後に開けます。
- Optimizeは目的変数と1つ以上の説明変数を設定すると開けます。
- Resultsは候補生成が成功すると開けます。

## API

既存のWeb APIをそのまま使用します。

```text
GET  /api/v1/health
POST /api/v1/datasets
POST /api/v1/regression/run
GET  /api/v1/logs
```

`POST /api/v1/regression/run`のリクエスト構造や候補レスポンスは変更していません。

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

このビルドではTypeScriptの厳格チェックとViteの生成処理が実行されます。
