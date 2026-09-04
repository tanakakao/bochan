# 35. Lookahead / Non-myopic Bayesian Optimization

EIやUCBの多くは「次の1回」の価値を基準にするmyopic decisionです。残りbudgetが複数回あるなら、今の観測が将来のdecisionをどう改善するかまで考えるnon-myopic BOが理論的には自然です。

## 1. Myopic decision

```math
x_{t+1}=\arg\max_x\alpha(x;D_t)
```

次の評価直後のutilityを中心に選びます。

## 2. Lookahead

2-stepなら

```math
x_{t+1}^*=\arg\max_x E_{y\mid x,D_t}[V_{t+2}(D_t\cup\{x,y\})]
```

のように、未知の観測 `y` の後に行う次のdecisionまで評価します。

## 3. Fantasy

future observationは未知なのでposteriorからfantasy sampleを生成します。

```math
y^{(s)}\sim p(y\mid x,D)
```

各fantasy datasetでfuture posteriorとfuture actionを構築し、期待utilityを近似します。

## 4. Knowledge Gradient

KGは「1回観測した後に最終decisionのposterior valueがどれだけ改善するか」を評価するvalue-of-information criterionです。

EIが直接的なobjective improvementを見るのに対し、KGは**学習によるdecision quality改善**を評価します。

## 5. Multi-step Lookahead

複数stageをtreeとして考えると

```text
current candidate
 ├─ fantasy outcome 1 -> next candidate -> ...
 ├─ fantasy outcome 2 -> next candidate -> ...
 └─ ...
```

となります。horizonが増えるほどnested optimizationとfantasy branchingで計算量が急増します。

## 6. いつ価値があるか

- 評価budgetが明確
- 1回の観測が後続探索を大きく変える
- explorationの価値が高い
- high-cost experimentで短期greedy decisionを避けたい

場合に有力です。

## 7. いつ避けるか

- surrogate fitting自体が不安定
- candidate optimizationが既に重い
- horizonに対してbudgetが大きい
- simple EI/NEIで十分な問題

では計算costに見合わないことがあります。

## 8. Multi-fidelityとの接続

future actionにfidelityも含めれば

```math
(x_{t+1},s_{t+1})
```

の情報価値を評価できます。30章のMFBOと自然に接続します。

## 9. bochanでの位置付け

bochanではstandard EI/UCB系とlookahead系を同じ「acquisition」という名前だけで扱うより、計算graph、fantasy、inner optimizerなど追加責務を意識する必要があります。

実務では

```text
EI / NEI baseline
 -> KG
 -> 2-step lookahead
 -> deeper horizon only if justified
```

という段階的比較が推奨されます。

Non-myopic BOの利点は未来を完全に予測することではなく、**今の観測が将来の選択肢をどれだけ改善するかをdecision criterionへ含めること**です。
