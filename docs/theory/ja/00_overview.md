# 00. 全体像と読書ガイド

このディレクトリは、`bochan` の理論リファレンスです。個別機能の説明だけでなく、**問題設定 → probabilistic model → decision criterion → implementation → materials discovery workflow** を一つの流れとして理解できる構成を目指します。

## 1. bochanの理論レイヤ

```text
Problem formulation
        ↓
Representation / search space
        ↓
Probabilistic model
        ↓
Posterior / uncertainty
        ↓
Objective / constraints / risk
        ↓
Acquisition function
        ↓
Candidate optimization
        ↓
Experiment / simulation
        ↓
Data update
```

Materials Informaticsではさらにcomposition、crystal structure、MLIP、DFT、experimentをこの共通loopへ接続します。

## 2. 3つの逐次設計問題

Bayesian Optimizationは高いutilityを持つcandidateを少ないevaluationで探索します。

```math
x^*\in\arg\max_{x\in\mathcal X}u(x)
```

Active Learningはmodel uncertaintyやinformation gainを基準にdata acquisitionを行います。

```math
x_{t+1}\in\arg\max_x I(\text{future observation};\text{learning target}\mid\mathcal D_t)
```

Level-set Estimationはoptimumではなくthresholdを満たす領域やboundaryを推定します。

```math
L_h^+=\{x:f(x)\ge h\}
```

同じsurrogateを利用できても、何を価値とするかは異なります。

## 3. 基本的なsequential loop

```text
training data
    -> model fitting
    -> posterior
    -> objective transform
    -> acquisition
    -> acquisition optimization
    -> candidate post-processing
    -> evaluation
    -> updated data
```

`bochan`ではmodel、objective、acquisition、optimizerを分離して考えます。

## 4. 区別すべき不確かさ

| 不確かさ | 例 | 主な扱い |
|---|---|---|
| Epistemic | data不足によるmodel uncertainty | GP posterior、AL |
| Aleatoric | measurement/process noise | likelihood、noise model |
| Input uncertainty | 設定値の実現誤差 | perturbation、Robust BO |
| Fidelity uncertainty | low/high fidelity間の差 | Multi-fidelity model |
| Domain shift | pretrained modelとtarget domainの差 | residual、transfer、retraining |

これらを一つのposterior varianceとして解釈しないことが重要です。

## 5. Theory map

### Part I — 基礎

01 Gaussian Process Models、02 Bayesian Optimization、03 Acquisition Functions、04 Active Learning、05 Level-set Estimation、06 Classification / Ordinal BO、07 Multi-objective / Constraints、08 Input Perturbation / Risk、09 Shape Conventionsを扱います。最初に読む場合は `01 -> 02 -> 03` が基本です。

### Part II — Model families

10 Regression Models and Likelihoods、11 Classification Models、12 Ordinal Models、13 Heteroscedastic / Robust Models、14 Deep / High-dimensional Models、15 Heterogeneous Multi-output、16 Level-set Mathematics and Implementationを扱います。ここでは「どのposteriorを作るか」に重点を置きます。

### Part III — Materials representations and MLIPs

17 Materials Informatics and Representations、18 Machine-learning Interatomic Potentials、19 MLIP + Residual Gaussian Process、20 Structure Relaxation + BO / AL、21 Composition Models、22 Composition to Crystal Structure、23 Composition + Structure + Process Optimization、24 Materials Model / Workflow Selectionを扱います。

### Part IV — 選択ガイド

25 Acquisition Selection、26 Active Learning Selection、27 GP Model Selection、28 Response Distribution / Noise Model Selectionです。「何を使えばよいか」を判断するときはこのPartから逆引きできます。

### Part V — Advanced Bayesian Optimization

29 Multi-fidelity / Multi-task / Transfer / Residual、30 Multi-fidelity BO、31 Robust BO、32 Constrained BO、33 Multi-objective BO、34 Batch / Parallel BO、35 Lookahead / Non-myopic BO、36 Information-theoretic BO、37 Classification / Ordinal BO、38 Level-set / Boundary Search、39 Mixed / Discrete / Combinatorial BO、40 High-dimensional BO、41 Diagnosticsを扱います。

### Part VI — Materials Discovery

42 Composition-space Exploration、43 Crystal-structure Exploration、44 Hierarchical MLIP / DFT / Experiment Workflow、45 Materials Active Learning、46 Closed-loop Materials Discoveryです。46章が理論編全体の統合地点です。

## 6. 推奨学習ルート

通常のBOは `01 -> 02 -> 03 -> 25 -> 41`、model選択は `01 -> 10 -> 27 -> 28 -> 41`、Active Learningは `01 -> 04 -> 26 -> 38 -> 45` を推奨します。

Multi-objective / constraintsは `07 -> 32 -> 33 -> 25`、robustな工程条件探索は `08 -> 31 -> 32 -> 33`、Multi-fidelityは `29 -> 30 -> 35 -> 44` と進みます。特に29章でResidual GPとMulti-fidelityを区別してから30章へ進むことを推奨します。

組成探索は `17 -> 21 -> 39 -> 42 -> 45`、結晶構造・MLIPは `17 -> 18 -> 19 -> 20 -> 43 -> 44`、closed-loop materials discoveryは `42 -> 43 -> 44 -> 45 -> 46` が基本ルートです。

## 7. 問題から章を逆引きする

| やりたいこと | 最初に読む章 |
|---|---:|
| 通常の回帰BO | 01, 02, 03 |
| acquisitionを選びたい | 25 |
| GP modelを選びたい | 27 |
| Binary / multiclass / ordinal | 11, 12, 37 |
| Active Learning | 04, 26 |
| 境界を知りたい | 05, 38 |
| 制約付き探索 | 32 |
| 複数目的 | 33 |
| 並列実験 | 34 |
| Lookahead | 35 |
| MES / PES / JES | 36 |
| categorical / integer / 元素選択 | 39 |
| 高次元 | 40 |
| BOの挙動がおかしい | 41 |
| 工程ばらつきを考慮 | 31 |
| low/high fidelity | 29, 30 |
| 組成式を扱う | 21, 42 |
| 結晶構造を扱う | 18, 20, 43 |
| MLIPをtarget dataで補正 | 19 |
| MLIP→DFT→実験 | 44 |
| 材料AL | 45 |
| 自動材料探索loop | 46 |

## 8. Materials Discoveryの統合像

```text
candidate elements
      ↓
composition search
      ↓
structure candidates
      ↓
MLIP relaxation
      ↓
probabilistic surrogate / residual GP
      ↓
BO / AL / robust / multi-fidelity acquisition
      ↓
DFT / experiment
      ↓
validated observations
      ↓
model update
      └──────────────→ next iteration
```

bochanの中心的な役割はphysics engineそのものではなく、**representation・probabilistic model・uncertainty・decision criterion・workflowを接続すること**です。

## 9. 実装を読むときの視点

```text
1. search space
2. training data / transforms
3. model / likelihood
4. posterior shape
5. objective transform
6. constraints / risk
7. acquisition
8. acquisition optimizer
9. candidate post-processing
10. sequential result
```

特にTensor shape、objective space、constraint space、posterior spaceを混同しないことが重要です。

## 10. 主要記号

| 記号 | 意味 |
|---|---|
| `n` | 観測数 |
| `d` | 入力次元 |
| `q` | batch candidate数 |
| `m` | output / objective数 |
| `K` | class数 |
| `n_w` | input perturbation scenario数 |
| `x` | design variable |
| `s` | fidelity |
| `t` | task index |
| `f` | latent function |
| `y` | observation |
| `u` | utility |
| `alpha(x)` | acquisition function |

## 11. この理論編の使い方

すべてを順番に読む必要はありません。まず **Model Selection → Acquisition Selection → Problem-specific chapter** の3段階で必要な章を選び、実装で問題が出たら41章のdiagnosticsへ戻る使い方を推奨します。

Materials Informatics用途では17〜24章で材料representationを理解した後、42〜46章へ進むと、個別modelからclosed-loop discoveryまで一貫して理解できます。
