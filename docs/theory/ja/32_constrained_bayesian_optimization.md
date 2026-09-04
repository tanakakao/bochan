# 32. 制約付きBayesian Optimization

実問題では目的関数だけでなく、安全性、品質規格、製造可能性、予算などの制約を満たす必要があります。制約付きBOは

```math
\max_x f(x)\quad\mathrm{s.t.}\quad g_j(x)\le0
```

を少ない評価回数で解く枠組みです。

## 1. 制約の種類

**Deterministic constraint** は入力だけから判定できる制約です。範囲、整数条件、組成和、禁止組合せなどが該当します。これは候補生成・acquisition optimization時に直接処理するのが基本です。

**Outcome constraint** は実験・simulationをしないと分からない制約です。

```math
g(x)+\epsilon_g\le0
```

のようにsurrogateを構築して確率的に扱います。

## 2. Probability of Feasibility

制約posteriorがGaussianなら

```math
\mathrm{PoF}(x)=P(g(x)\le0\mid D)
```

を計算できます。複数制約を独立と近似すれば

```math
P(\mathrm{feasible})\approx\prod_j P(g_j(x)\le0)
```

です。相関制約ではjoint probabilityを考える必要があります。

## 3. Constrained acquisition

代表的な直感は

```math
\alpha_c(x)=\alpha(x)P(\mathrm{feasible}\mid x)
```

です。改善量が大きくても実現可能性が低いcandidateはdown-weightされます。

bochanではfeasibility処理をbase acquisitionと分離して考えることで、EI/UCBや分類系acquisitionへ再利用しやすくなります。

## 4. Feasibility-first

初期dataにfeasible pointが存在しない場合、improvement基準が安定しません。その場合はまず

```math
x^*=\arg\max_x P(\mathrm{feasible}\mid x)
```

としてfeasible region発見を優先し、その後objective optimizationへ移行する戦略が有効です。

## 5. Chance constraint

process variation `\xi` がある場合は

```math
P(g(x+\xi)\le0)\ge1-\delta
```

のようなchance constraintを使えます。これは31章のRobust BOと接続します。

## 6. Classification constraint

合格/不合格しか観測できない場合はlatent regressionではなくbinary classifierで

```math
p_{feas}(x)=P(y=1\mid x,D)
```

をmodel化できます。bochanのbinary probability-of-feasibility系はこの問題に対応する考え方です。

## 7. Multi-objectiveとの組合せ

多目的問題でもPareto improvementをfeasible region内で評価します。概念的には

```text
posterior samples
 -> objective transform
 -> feasibility evaluation
 -> feasible hypervolume improvement
```

となります。

## 8. 材料・工程での例

- 強度最大化 subject to 導電率 >= threshold
- 性能最大化 subject to 原料cost <= limit
- 組成探索 subject to composition closure
- 温度最適化 subject to equipment upper limit
- MLIP/DFT探索 subject to structure stability

入力から確実に判定できる制約と、実験しないと分からない制約を混在させないことが重要です。

## 9. 実務上の確認事項

1. 制約の向き `<=0` / `>=0` を統一する
2. deterministic constraintはsurrogate化しない
3. feasibility probabilityのcalibrationを確認する
4. feasible initial pointがあるか確認する
5. 複数制約の相関を無視してよいか検討する
6. robust constraintが必要か判断する

制約付きBOでは「良いcandidate」ではなく、**実行可能で、その中で価値の高いcandidate**を選ぶことが中心です。
