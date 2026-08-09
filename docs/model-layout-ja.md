# モデルパッケージ構成

bochan のサロゲートモデルは、タスクを第一階層、実装ファミリを第二階層として配置します。

- 回帰の外部 estimator: `bochan.models.regression.external`
- 二値分類: `bochan.models.classification.binary`
- 多クラス分類: `bochan.models.classification.multiclass`
- 順序モデル: `bochan.models.ordinal`

`bochan.models.classification.common` と `bochan.models.external` は、具体的な公開タスクモデルではなく共有内部実装を保持します。

Random Forest は boosting 手法ではないため `regression.boosting` には置きません。NGBoost と Random Forest の共通点は、Tensor から NumPy へ跨ぐ外部 estimator 境界であるため、両者を `regression.external` に配置します。
