# LLM documentation

`bochan` のLLM連携に関する補足ドキュメントをこのディレクトリにまとめます。

LLMはベイズ最適化そのものを置き換えるのではなく、設定設計、候補集合の生成、候補説明を補助します。モデル学習、獲得関数評価、制約処理、候補の最終評価は `bochan` 側で実行されます。

## Guides

- [LLM-assisted planning and candidate generation](../../README_LLM.md) — LLM連携全体の入口。設定提案、候補集合生成、provider設定、FastAPI連携を説明します。
- [Candidate explanation](candidate_explanation.md) — 最終候補をモデル根拠、総合説明、物理、化学、製造、開発、リスクの観点から説明する方法です。`overall_interpretation` の説明もこのガイドに統合しています。
- [Hybrid and constrained Bayesian optimization](hybrid_constraints.md) — 回帰・分類・ordinalを組み合わせたHybridモデルと、目的・出力制約・候補制約を扱います。
- [LLM-selected acquisition](../llm_selected_acquisition.md) — `AcquisitionConfig(name="llm_selected")` の選択・検証フローを説明します。

## Documentation ownership

LLM機能の実装は主に `src/bochan/llm/` と `src/bochan/api/llm/` が所有します。このディレクトリは機能別の詳細ガイドを置く場所とし、個別の `README_LLM_*.md` をリポジトリrootへ追加しない方針です。
