# 24. 材料モデルと探索workflowの選び方

この章は、前章までの理論を「どのモデルをいつ使うか」という実装判断へ落とし込みます。

## 1. 最初に確認する4点

モデル名から選ぶのではなく、まず次を確認します。

1. 入力として何を持っているか
2. 何を予測したいか
3. uncertaintyが必要か
4. 候補を予測するだけか、次の実験を選びたいか

## 2. 入力情報から選ぶ

| 入力 | 第一候補 | 主な用途 |
|---|---|---|
| 通常の表形式特徴量 | GP / tree / neural surrogate | 一般BO/回帰 |
| 組成式 | CrabNet / Roost系 | 組成property予測 |
| 組成式 + process | composition encoder + GP/DKL | 材料・工程BO |
| 結晶構造 | ALIGNN等 | structure property予測 |
| 原子構造 + PES関連量 | MACE / CHGNet / M3GNet / ALIGNN-FF | energy/force/stress, relaxation |

## 3. DirectかResidual GPか

pretrained modelの予測だけで十分ならdirect modeが最も単純です。

```math
\hat y=f_0(x)
```

対象domainのDFT/実験データがあり、baselineを補正したいならresidual GPを検討します。

```math
y=f_0(x)+\delta(x)+\epsilon
```

Residual GPが特に有効なのは、baselineに有用な情報が残っている一方でsystematic biasが存在する場合です。

baselineが対象domainで完全に破綻している場合、residualが非常に複雑になり、必ずしも有利ではありません。その場合は別encoder、fine-tuning、通常のsurrogateなども比較すべきです。

## 4. GPかDKLか

低～中次元でデータが少なく、解釈しやすいkernel構造が欲しい場合は通常GPが扱いやすい選択です。

raw representationが高次元・非線形で、encoderによる表現学習が有効ならDKLを検討できます。

```math
k_{\mathrm{DKL}}(x,x')
=k_0(h_\theta(x),h_\theta(x'))
```

ただし小規模データで大きなencoderをend-to-end学習するとoverfitしやすいため、pretrained/frozen encoder + GPも重要なbaselineです。

## 5. relaxationを入れるか

候補が未緩和構造なら、energy landscape上で不自然な初期構造を直接rankするより、まずrelaxationする方が物理的に自然な場合があります。

```text
initial structures
  -> MLIP relaxation
  -> relaxed structures
  -> surrogate/acquisition
```

一方、目的が「未緩和構造の生成器そのものを評価する」場合や、relaxation costが問題になる場合は必ずしも前処理すべきではありません。

## 6. RankかAcquisitionか

posterior meanだけで候補を並べるならrankです。

```math
x^*=\arg\max_x \mu(x)
```

新しい観測を取得しながら探索するならacquisitionを使います。

```math
x_{t+1}=\arg\max_x\alpha(x;\mathcal D_t)
```

この違いは本質的です。rankは現在のbest predictionを選ぶのに対し、BO/ALはuncertaintyを意思決定へ使います。

## 7. acquisitionの簡易選択

| 目的 | 代表的な選択 |
|---|---|
| 良い物性を探索 | EI / LogEI |
| explorationを調整 | UCB |
| noisy observation | NEI / LogNEI |
| uncertaintyを減らす | posterior variance |
| 分類情報を得る | entropy / BALD |
| 空間全体のposterior改善 | NIPV |
| 複数目的 | EHVI / NEHVI系 |

獲得関数名だけでなく、model posteriorがその獲得関数の要求する意味を持っているか確認する必要があります。

## 8. fidelityを区別する

pretrained MLIPとDFT、DFTと実験など計算精度・コストが異なる情報源がある場合、単なるresidual correctionとmulti-fidelity BOを混同しないようにします。

Residual model:

```math
y_H(x)=f_0(x)+\delta(x)
```

Multi-fidelity model:

```math
y=f(x,s),\qquad s\in\mathcal S_{\mathrm{fidelity}}
```

後者では「どの `x` を評価するか」に加えて「どのfidelityで評価するか」も意思決定対象になり得ます。

## 9. 推奨する段階的導入

新しい材料problemでは、最初から最も複雑なworkflowへ進むより次の順序が安全です。

```text
1. simple baseline
2. pretrained direct model
3. calibrated/residual surrogate
4. uncertainty diagnostics
5. BO/AL
6. constraints / multi-objective
7. robust / multi-fidelity / hierarchical search
```

各段階でholdout predictionだけでなく、uncertainty calibrationと逐次探索performanceを評価します。

## 10. bochanにおける設計原則

bochanでは、backend、quantity、model mode、workflow modeを分離して考えます。

```text
backend
  × quantity
  × model mode
  × workflow mode
```

例えば

```text
MACE
  × energy
  × residual GP
  × relax + acquisition
```

という一つのworkflowになります。

この直交的な整理によって、特定backendに探索ロジックを埋め込まず、物理モデルと意思決定ロジックを分離できます。

実際のMLIP APIと対応backendについては [Unified MLIP workflows](../../materials/mlip-workflows.md) を参照してください。
