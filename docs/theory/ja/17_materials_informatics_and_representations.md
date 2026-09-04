# 17. マテリアルズインフォマティクスと材料表現

本章では、材料探索をベイズ最適化へ接続する前に、**何を入力として表現し、何を予測・最適化するのか**を整理します。

## 1. 材料探索の階層

材料開発では、同じ「材料」でも複数の設計空間があります。

```text
元素集合
  -> 組成
  -> 結晶構造
  -> プロセス条件
  -> 状態・組織
  -> 物性
```

組成を `c`、構造を `S`、プロセス条件を `p` とすると、一般には

```math
y=f(c,S,p)+\epsilon
```

と考えられます。どの変数まで観測・制御できるかによって、必要なsurrogate modelと探索方法が変わります。

## 2. 組成表現

組成式だけを使う場合、原子座標は未知です。例えば `LiFePO4` は元素種と比率を持ちますが、結晶中の座標を直接表しません。

組成ベースモデルは概念的に

```math
\phi_{comp}: c \mapsto z_{comp}
```

という埋め込みを学習し、

```math
\hat y=g(z_{comp},p)
```

として物性を予測します。CrabNetやRoostはこの領域に属します。

組成探索は候補生成が容易な一方、polymorphや局所構造の差を組成だけから完全には区別できません。

## 3. 結晶構造表現

周期結晶を

```math
S=(Z,R,L)
```

と書きます。`Z`は元素種、`R`は原子座標、`L`は格子です。

構造モデルには少なくとも次の対称性への配慮が必要です。

- 原子の並べ替えに対する不変性
- 周期境界条件
- 並進に対する不変性
- energyの回転不変性
- forceの回転共変性

グラフニューラルネットワークでは原子をnode、近傍関係をedgeとして表し、message passingによって局所環境から構造表現を構築します。

## 4. プロセス条件との統合

材料特性が構造だけで決まらない場合、構造表現 `z_S` とプロセス変数 `p` を結合できます。

```math
x=[z_S,p]
```

bochanのstructure-index型workflowでは、有限個の構造bankを

```math
x=[i_{structure},p_1,\ldots,p_k]
```

として扱うこともできます。これは「構造候補の選択」と「連続・離散プロセス条件の最適化」を同じ候補空間へ載せるための実装上の表現です。

## 5. 何を最適化するのか

結晶構造BOで常にenergyを最小化するわけではありません。目的は例えば次のように設定できます。

- formation energy / total energy
- band gap
- elastic property
- conductivity
- experimentally measured performance
- 複数物性のPareto最適化

重要なのは、**relaxationの目的とBOの目的を区別すること**です。relaxationでは通常、与えられた構造近傍でenergyを下げます。一方BOでは、relax後の構造に対する任意の目的物性を探索できます。

## 6. bochanでの対応

現在の材料系機能は大きく次の層に分かれます。

```text
composition models       CrabNet / Roost etc.
structure models         ALIGNN etc.
MLIP                     MACE / CHGNet / M3GNet / ALIGNN-FF
probabilistic correction residual GP
sequential decision      BO / Active Learning
runtime workflow         relaxation -> rank/acquisition
```

実装方法は [Unified MLIP workflows](../../materials/mlip-workflows.md) を参照してください。

## 7. モデル選択の目安

| 手元の情報 | 主な選択肢 |
|---|---|
| 組成式のみ | composition encoder / GP / DKL |
| 結晶構造あり | structure GNN / GP / DKL |
| 構造からenergy・forceが必要 | MLIP |
| pretrained MLIPを少量データで補正 | residual GP |
| 次に評価する材料を選びたい | BO / Active Learning |
| 原子位置を安定構造へ近づけたい | structure relaxation |

次章では、MLIPが構造から何を学習しているのかをenergy・force・stressの関係から説明します。
