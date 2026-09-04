# 22. 組成から結晶構造へ：材料表現の階層

材料探索では、最初から完全な結晶構造が分かっているとは限りません。組成、候補構造、緩和構造、DFT/実験値という順に情報が増えるため、すべてを単一モデルで扱うより、情報レベルに応じてsurrogateを選ぶ方が自然です。

## 1. 情報の階層

概念的には

```text
composition
    ↓
candidate structure
    ↓
relaxed structure
    ↓
physics-based prediction
    ↓
DFT / experiment
```

と整理できます。

組成 `c` だけでは構造 `s` は一意に定まりません。

```math
p(s\mid c)
```

は一般に複数のmodeを持ちます。同じ化学式でも結晶多形、格子定数、原子配置、欠陥状態によって物性が異なるためです。

## 2. 組成モデルと構造モデルの違い

組成モデルは

```math
p(y\mid c)
```

を近似します。一方、structure-aware modelは

```math
p(y\mid c,s)
```

または構造自体に組成が含まれるため単に

```math
p(y\mid s)
```

を近似します。

構造情報が得られれば条件付けが増えるため、原理的には組成だけの場合より物性差を説明しやすくなります。ただし構造生成・構造最適化のコストが追加されます。

## 3. ALIGNNの位置づけ

ALIGNNのようなstructure modelは、原子graphに加えてbond-angle情報を表現し、局所幾何を物性予測へ利用します。

```text
atomic structure
  -> atom/bond graph
  -> line graph / angular interactions
  -> learned structure representation
  -> property
```

したがってCrabNet/RoostとALIGNNの違いは単なるモデルarchitectureの違いではなく、入力に利用できる物理情報のレベルの違いでもあります。

## 4. property modelとMLIPの違い

structure-aware property modelが、例えばband gapや形成エネルギーなど目的物性を直接学習するのに対し、MLIPは主としてpotential energy surfaceを近似し、energy/force/stressを提供します。

```text
structure model:
structure -> target property

MLIP:
structure -> energy / forces / stress -> relaxation / dynamics
```

そのためALIGNN property encoderとALIGNN-FFは、名前が近くても役割を分けて考える必要があります。

## 5. 構造生成を含む探索

組成から候補構造を生成できる場合、探索は階層化できます。

```math
c^* = \arg\max_c \; \mathbb E_{s\sim p(s\mid c)}[U(c,s)]
```

あるいは有限個の候補構造を列挙し、

```math
(c^*,s^*)=\arg\max_{c,s} U(c,s)
```

として扱えます。

実務では構造生成自体の不確かさや計算コストが大きいため、まずcomposition modelで候補を絞り、その後structure-aware model/MLIPへ進めるmulti-stage screeningが扱いやすい設計です。

## 6. 緩和前と緩和後

候補構造 `s_0` をMLIPで緩和して

```math
s_{\mathrm{relax}}=R_{\mathrm{MLIP}}(s_0)
```

を得ると、BOが評価すべき入力を緩和後構造へ揃えられます。

これは重要です。未緩和構造の幾何的な歪みをそのままsurrogateへ入力すると、「材料候補の本質的な差」と「初期構造生成方法の差」が混在する可能性があるためです。

## 7. 段階的な探索戦略

典型例は次の通りです。

```text
Stage 1: element/composition search
  CrabNet / Roost + GP/DKL
        ↓
Stage 2: structure screening
  ALIGNN / structure surrogate
        ↓
Stage 3: structural relaxation
  MACE / CHGNet / M3GNet / ALIGNN-FF
        ↓
Stage 4: uncertainty-aware selection
  residual GP + BO/AL
        ↓
Stage 5: high-fidelity evaluation
  DFT / experiment
```

全候補へ高価なDFTを実行するのではなく、安価なモデルを前段filterとして使う考え方です。

## 8. 注意点

この階層をそのまま「精度の序列」と解釈してはいけません。事前学習domain、目的物性、構造品質、化学空間の外挿度によって性能は変わります。重要なのは、利用可能な情報と意思決定に必要な不確かさに合わせてモデルを選ぶことです。
