# 18. 機械学習原子間ポテンシャル（MLIP）

MLIPは、量子化学計算で得られるpotential energy surfaceを機械学習で近似し、構造に対するenergy・force・stressを高速に評価するモデルです。

## 1. Potential Energy Surface

Born-Oppenheimer近似の下では、原子核配置 `R` に対するenergyを

```math
E=E(R,Z,L)
```

とみなせます。MLIPはこの写像を

```math
E_\theta(S)\approx E_{ref}(S)
```

として学習します。

典型的には全energyを局所的な原子寄与へ分解します。

```math
E_\theta(S)=\sum_{i=1}^{N} e_{\theta,i}(\mathcal N_i)
```

ここで `N_i` はcutoff内の局所原子環境です。この局所性により、異なる原子数の構造にも適用しやすくなります。

## 2. Forceはenergyの勾配

原子 `i` に働くforceは

```math
\mathbf F_i=-\frac{\partial E}{\partial \mathbf r_i}
```

です。したがってenergyを微分可能なモデルとして学習すれば、autogradによりforceを得られます。

これは重要な物理的一貫性を与えます。energyとforceを完全に独立した回帰器として学習すると、一般には保存力場になりません。

学習損失は例えば

```math
\mathcal L=
\lambda_E\lVert E_\theta-E_{ref}\rVert^2+
\lambda_F\sum_i\lVert \mathbf F_{\theta,i}-\mathbf F_{ref,i}\rVert^2+
\lambda_\sigma\lVert \sigma_\theta-\sigma_{ref}\rVert^2
```

のように複数の教師信号を組み合わせます。

## 3. Stress

stressはcell deformationに対するenergy応答と関係します。strain `epsilon` に対して概念的には

```math
\sigma \propto \frac{1}{V}\frac{\partial E}{\partial \epsilon}
```

です。

実装では符号、Voigt表現、単位系に注意が必要です。bochanの共通MLIP層ではbackend固有の暗黙変換を増やさず、各calculatorが返すASE-facing contractを尊重する方針を取っています。

## 4. 不変性と共変性

energyは系全体を回転しても変わらないため

```math
E(QR)=E(R)
```

であるべきです。一方forceは回転とともに回転し

```math
F(QR)=QF(R)
```

となる必要があります。

MACEのようなequivariant modelは、この性質をarchitectureへ強く組み込みます。一般的なmessage-passing modelでも、距離・角度などの幾何情報を用いて対称性を考慮します。

## 5. MACE / CHGNet / M3GNet / ALIGNN-FF

bochanではこれらを共通MLIP backendとして扱いますが、内部表現は同一ではありません。

| backend | 理論上の特徴 | bochanでの役割 |
|---|---|---|
| MACE | equivariant message passing / many-body表現 | energy, force, stress, relaxation |
| CHGNet | crystal graph + pretrained interatomic potential | energy, force, stress, relaxation |
| M3GNet / MatGL | many-body graph potential | energy, force, stress, relaxation |
| ALIGNN-FF | atomistic line-graph系force field | energy, force, stress, relaxation |

bochanはarchitectureの差を消すのではなく、**下流workflowのcontractを揃える**設計です。

## 6. pretrained MLIPの意味

pretrained MLIPを使うと、ユーザー固有の教師データがなくても構造からenergy/force/stressを推論できます。ただしこれは「対象系に対して正確である」ことを保証しません。

学習分布から離れた組成・構造ではsystematic biasが生じ得ます。そこでbochanでは、pretrained predictionをbaselineとして残差だけをGPで学習する構成を提供します。

## 7. Direct predictionとResidual prediction

Direct mode:

```math
\hat y(x)=f_{MLIP}(x)
```

Residual GP mode:

```math
\hat y(x)=f_{MLIP}(x)+\delta_{GP}(x)
```

後者は次章で詳しく扱います。

## 8. bochanのtensor contract

- Energy: scalar
- Force: 固定topologyの構造bankでは `3N` 成分
- Stress: full `3 x 3` をflattenした9成分

force residualでは構造間で原子数が変わると出力次元も変わるため、固定原子数という制約が重要です。

実装APIは [Unified MLIP workflows](../../materials/mlip-workflows.md) を参照してください。
