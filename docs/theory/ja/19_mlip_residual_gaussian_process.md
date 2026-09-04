# 19. MLIP + Residual Gaussian Process

pretrained MLIPは強力なprior knowledgeを持ちますが、対象材料・計算条件・実験条件に対するbiasを持つことがあります。Residual GPは、そのbaselineを捨てずに**観測された誤差だけを確率的に補正する**方法です。

## 1. 基本モデル

真の目的量を

```math
y(x)=f_{MLIP}(x)+\delta(x)+\epsilon
```

と分解します。

- `f_MLIP(x)`: pretrained modelのbaseline
- `delta(x)`: baselineのsystematic residual
- `epsilon`: 観測noise

GPを

```math
\delta(x)\sim\mathcal{GP}(m(x),k(x,x'))
```

と置きます。通常はzero meanから始められます。

## 2. 学習対象

観測値 `y_i` とMLIP予測 `b_i=f_MLIP(x_i)` から

```math
r_i=y_i-b_i
```

を作り、GPは

```math
\mathcal D_r=\{(x_i,r_i)\}_{i=1}^{n}
```

を学習します。

したがって最終予測平均は

```math
\mu_y(x)=f_{MLIP}(x)+\mu_r(x)
```

です。baselineをdeterministicと扱う場合、posterior varianceは主にresidual GPから得られます。

```math
\sigma_y^2(x)\approx\sigma_r^2(x)
```

これは「MLIPそのもののepistemic uncertaintyを完全に表現している」という意味ではありません。**観測されたresidualに基づく不確かさ**です。

## 3. なぜ直接GPより有利になり得るか

直接GPは

```math
y(x)\sim GP
```

として全関数を少量データから学習します。一方Residual GPでは大域的な物理傾向をpretrained MLIPへ任せます。

MLIPが十分に良い場合、残差関数は元の目的関数より

- 振幅が小さい
- 滑らか
- 少量データで学習しやすい

可能性があります。

ただしMLIPのbiasが複雑、または対象domainがpretraining distributionから大きく外れている場合、この利点は弱くなります。

## 4. Bayesian Optimizationとの接続

BOではposterior

```math
f(x)\mid\mathcal D
```

の平均と不確かさを使って候補を選びます。Residual GPでも最終posteriorを通常のsurrogateとして扱えるため、EIやUCBなどを適用できます。

例えばUCBは最大化問題で

```math
\alpha_{UCB}(x)=\mu_y(x)+\sqrt{\beta}\sigma_y(x)
```

となります。

baselineが良い領域は平均予測を活用し、residual GPの不確かな領域では探索価値を残せます。

## 5. Active Learningとの接続

目的が最適値探索ではなくモデル改善なら、posterior varianceなどを直接使えます。

```math
\alpha_{var}(x)=\sigma_y^2(x)
```

分類・情報理論型workflowではpredictive entropy、BALD、NIPVなど別の基準を用います。

「BOで良い材料を探す」と「ALでモデルを良くする」は同じではないため、獲得関数は目的に合わせて選択します。

## 6. Energy residual

scalar energyでは最も直感的です。

```math
r_E=E_{target}-E_{MLIP}
```

対象DFT条件や実験的なenergy proxyに対するsystematic offsetを学習できます。

## 7. Force residual

原子数 `N` が固定なら

```math
r_F\in\mathbb R^{3N}
```

です。各成分を独立出力として扱うだけでなく、multi-output / multitask GPによって成分間相関を表現できます。

異なる原子数を同一dense output tensorへ直接積むことはできないため、bochanの現行force residual contractはfixed topologyを要求します。

## 8. Stress residual

full stress tensorをflattenすると

```math
r_\sigma\in\mathbb R^9
```

です。物理的には対称tensorで独立成分を減らせる場合がありますが、bochanの共通contractはbackend間の一貫性を優先して9成分を保持します。

## 9. Multi-fidelityとの違い

Residual GPとmulti-fidelity GPは関連しますが同義ではありません。

Residual GP:

```math
high(x)=baseline(x)+residual(x)
```

Multi-fidelity model:

```math
f(x,s),\qquad s\in\{low,high,\ldots\}
```

後者はfidelity自体を入力・taskとしてモデル化し、観測costを考慮した意思決定まで拡張できます。pretrained MLIPを固定baselineとして補正するだけならResidual GPの方が単純です。

## 10. bochanでの対応

bochanでは共通factoryにより

```text
backend x quantity x model_mode
```

を指定します。

```python
model = create_material_model(
    "mace",
    "energy",
    "residual_gp",
    structures=structures,
    train_X=train_X,
    train_Y=train_Y,
)
```

実装詳細とbackend別requirementsは [Unified MLIP workflows](../../materials/mlip-workflows.md) を参照してください。

次章では、このposteriorを構造relaxationと組み合わせ、次に評価する構造を選ぶ方法を扱います。
