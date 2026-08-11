# High-dimensional Gaussian regression models

このフォルダには、高次元の連続説明変数を扱うGaussian regression modelを配置します。

現在の主なモデルは次のとおりです。

| 手法 | normal input | mixed input | 特徴 |
|---|---|---|---|
| PCA | `PCAGaussianGPModel` | `PCAGaussianMixedGPModel` | データ分散を保つ線形射影を事前に固定 |
| REMBO | `REMBOGaussianGPModel` | `REMBOGaussianMixedGPModel` | 固定ランダム射影による低次元化 |
| SAAS | `SaasGaussianGPModel` | `SaasGaussianMixedGPModel` | 有効な少数変数をBayesianに選択 |
| VAE | `VAEGaussianGPModel` | `VAEGaussianMixedGPModel` | 入力復元と目的変数回帰を同時に考慮した非線形潜在表現 |

## VAE-GP regression

`VAEGaussianGPModel`と`VAEGaussianMixedGPModel`は、VAEによる非線形な次元削減とGaussian GPを組み合わせたモデルです。

学習時は、次の3項を同時に最小化します。

```text
gp_weight * negative_gp_marginal_log_likelihood
+ reconstruction_weight * reconstruction_mse
+ kl_weight * kl_divergence
```

GPと獲得関数には、確率的なサンプルではなくencoderの潜在平均を使用します。これにより、同じ候補点に対する獲得関数値を決定論的に評価できます。decoderの復元損失にはreparameterized sampleを使用します。

### normal input

```text
raw X
  ↓ input_transform
preprojection X
  ↓ VAE encoder
latent mean Z
  ↓ SingleTaskGP
posterior of Y
```

### mixed input

mixed版では、カテゴリ列をVAEへ含めません。連続列だけをVAEで射影し、カテゴリ列は整数コードのまま保持します。

```text
raw mixed X
  ↓ input_transform（連続列のみ）
  ├─ continuous X → VAE encoder → latent Z
  └─ categorical X → unchanged

[latent Z, categorical X]
  ↓ MixedSingleTaskGP
posterior of Y
```

reconstruction MSEも連続列だけで計算します。カテゴリ列はdecoderで生成せず、入力値をそのまま保持します。

## normal inputを直接使用する例

```python
import torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from bochan.fit import fit_vae_gp
from bochan.models.regression.gaussian.high_dim import VAEGaussianGPModel


torch.manual_seed(0)
dtype = torch.double

train_X = torch.rand(40, 10, dtype=dtype)
train_Y = (
    torch.sin(2.0 * torch.pi * train_X[:, 0])
    + 0.5 * train_X[:, 1]
    - 0.2 * train_X[:, 2]
).unsqueeze(-1)

bounds = torch.stack(
    [
        torch.zeros(train_X.shape[-1], dtype=dtype),
        torch.ones(train_X.shape[-1], dtype=dtype),
    ]
)

model = VAEGaussianGPModel(
    train_X=train_X,
    train_Y=train_Y,
    latent_dim=3,
    hidden_dims=[32, 16],
    reconstruction_weight=1.0,
    kl_weight=1e-3,
    gp_weight=1.0,
    input_transform=Normalize(
        d=train_X.shape[-1],
        bounds=bounds,
    ),
    outcome_transform=Standardize(m=1),
)

fit_result = fit_vae_gp(
    model,
    num_epochs=300,
    lr=1e-2,
    clip_grad_norm=10.0,
    verbose=True,
)

X_test = torch.rand(5, train_X.shape[-1], dtype=dtype)
posterior = model.posterior(X_test)

print(posterior.mean.shape)      # [5, 1]
print(posterior.variance.shape)  # [5, 1]
print(fit_result.final_loss)
```

`fit_gpytorch_mll`ではなく、専用の`fit_vae_gp`を使用してください。VAEのencoder・decoderと内部GPをAdamで共同最適化するためです。

## mixed inputを直接使用する例

次の例では第1列をカテゴリ列とします。

```python
import torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from bochan.fit import fit_vae_gp
from bochan.models.regression.gaussian.high_dim import VAEGaussianMixedGPModel


dtype = torch.double
cat_dims = [1]
cont_dims = [0, 2, 3, 4]

continuous_X = torch.rand(40, 4, dtype=dtype)
category = torch.randint(0, 3, (40, 1)).to(dtype=dtype)
train_X = torch.cat(
    [continuous_X[:, :1], category, continuous_X[:, 1:]],
    dim=-1,
)
train_Y = (
    torch.sin(2.0 * torch.pi * continuous_X[:, :1])
    + 0.3 * continuous_X[:, 1:2]
    + 0.2 * category
)

bounds = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, 1.0, 1.0],
    ],
    dtype=dtype,
)

model = VAEGaussianMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=cat_dims,
    category_counts={1: 3},
    latent_dim=2,
    hidden_dims=[16, 8],
    input_transform=Normalize(
        d=train_X.shape[-1],
        bounds=bounds,
        indices=cont_dims,
    ),
    outcome_transform=Standardize(m=1),
)

fit_result = fit_vae_gp(
    model,
    num_epochs=300,
    lr=1e-2,
    clip_grad_norm=10.0,
)

posterior = model.posterior(train_X[:5])
```

内部GPが見る入力は次の形になります。

```text
[latent_0, latent_1, category]
```

カテゴリ列の内部位置は`model.latent_cat_dims`で確認できます。

```python
print(model.latent_cat_dims)  # [2]
```

## high-level API

normal inputでは`model_type="vae"`を指定します。

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig


bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="regression",
        model_type="vae",
        model_kwargs={
            "latent_dim": 3,
            "hidden_dims": [32, 16],
            "reconstruction_weight": 1.0,
            "kl_weight": 1e-3,
            "gp_weight": 1.0,
        },
    ),
    fit_config=FitConfig(
        num_epochs=300,
        lr=1e-2,
        clip_grad_norm=10.0,
    ),
    bounds=bounds,
)

bo.fit(train_X, train_Y)
```

mixed inputでは`cat_dims`を指定すると、registryが`VAEGaussianMixedGPModel`を選択します。

```python
bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="regression",
        model_type="vae",
        cat_dims=[1],
        input_transform=Normalize(
            d=train_X.shape[-1],
            bounds=bounds,
            indices=[0, 2, 3, 4],
        ),
        model_kwargs={
            "category_counts": {1: 3},
            "latent_dim": 2,
            "hidden_dims": [16, 8],
        },
    ),
    fit_config=FitConfig(num_epochs=300, lr=1e-2),
    bounds=bounds,
)

bo.fit(train_X, train_Y)
```

mixed inputの`input_transform`はカテゴリ列を変更してはいけません。`Normalize`を直接渡す場合は、`indices`へ連続列だけを指定してください。

## 潜在変数の取得

`encode()`はraw-space入力を受け取ります。

```python
Z = model.encode(train_X)
print(Z.shape)  # [n, latent_dim]
```

mixed版でも`encode()`が返すのは連続列の潜在表現だけです。カテゴリを含む内部GP入力は`transform_inputs()`で確認します。

```python
projected_X = model.transform_inputs(train_X)
print(projected_X.shape)  # [n, latent_dim + num_categorical_columns]
```

学習された潜在分布からサンプリングする場合は、`sample=True`を指定します。

```python
Z_sample = model.encode(train_X, sample=True)
```

平均と対数分散も確認できます。

```python
Z, mu, logvar = model.encode(
    train_X,
    sample=False,
    return_stats=True,
)
```

## decoderによる復元

normal版では、潜在変数からそのまま入力全体を復元できます。

```python
X_reconstructed = model.reconstruct(train_X, raw_space=True)
X_from_Z = model.decode(Z, raw_space=True)
```

mixed版では`reconstruct()`が連続列だけを復元し、カテゴリ列を元入力から保持します。

```python
X_reconstructed = mixed_model.reconstruct(train_X, raw_space=True)
```

潜在変数からmixed入力を復元する場合は、カテゴリ値を明示的に渡します。カテゴリはVAEへ符号化されていないため、`Z`だけから推定はしません。

```python
X_from_Z = mixed_model.decode(
    Z,
    categorical_X=train_X[:, mixed_model.cat_dims],
    raw_space=True,
)
```

`input_transform`が`untransform()`を持たない場合、raw-spaceへの復元はできません。`raw_space=False`を指定するとpreprojection-spaceで返します。

## BoTorch獲得関数との接続

normal版は`optimize_acqf`へ接続できます。

```python
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf


acqf = UpperConfidenceBound(model=model, beta=0.1)

candidates, acq_value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=256,
)
```

mixed版ではカテゴリ候補を`fixed_features_list`で列挙し、`optimize_acqf_mixed`を使用します。

```python
from botorch.optim import optimize_acqf_mixed


acqf = UpperConfidenceBound(model=mixed_model, beta=0.1)

candidates, acq_value = optimize_acqf_mixed(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=256,
    fixed_features_list=[
        {1: 0.0},
        {1: 1.0},
        {1: 2.0},
    ],
)
```

どちらも探索対象はraw-spaceです。現時点では潜在空間`Z`を直接最適化してdecoderで`X`へ戻す方式ではありません。

## 主な引数

| 引数 | 意味 | 初期値 |
|---|---|---|
| `latent_dim` | GPへ渡す連続潜在次元数 | `2` |
| `hidden_dims` | encoderの隠れ層。decoderは逆順で構築 | 自動設定 |
| `activation` | 隠れ層の活性化関数 | `"silu"` |
| `decoder_output_activation` | decoder最終層の活性化関数 | `"identity"` |
| `reconstruction_weight` | 連続入力の復元誤差の重み | `1.0` |
| `kl_weight` | KL divergenceの重み | `1e-3` |
| `gp_weight` | GP負の対数周辺尤度の重み | `1.0` |
| `logvar_min` | encoder log varianceの下限 | `-10.0` |
| `logvar_max` | encoder log varianceの上限 | `10.0` |
| `category_counts` | original-spaceのカテゴリ列ごとのカテゴリ数 | 自動推定 |
| `cont_kernel_factory` | mixed GPの連続カーネル生成関数 | `None` |

## 重みの調整

最初は次の設定を基準にしてください。

```python
reconstruction_weight = 1.0
kl_weight = 1e-3
gp_weight = 1.0
```

- 復元精度が低い場合：`reconstruction_weight`を上げる
- 潜在空間が学習点周辺に強く集中する場合：`kl_weight`を上げる
- 予測性能が低く復元だけが改善する場合：`gp_weight`を上げる、または`reconstruction_weight`を下げる
- 損失が非有限になる場合：`lr`を下げ、`clip_grad_norm`を指定する

KL項を強くしすぎるとposterior collapseが起こり、入力によらず潜在変数が似た値になることがあります。`mu.std(dim=0)`やGPの予測精度と合わせて確認してください。

## PCA・REMBOとの使い分け

| 観点 | PCA | REMBO | VAE-GP |
|---|---|---|---|
| 射影 | 線形・データ依存 | 線形・ランダム固定 | 非線形・学習可能 |
| 目的変数を考慮 | しない | しない | GP損失を通じて考慮 |
| decoder | 線形逆変換 | 基本的になし | neural decoder |
| 学習安定性 | 高い | 高い | 初期値・重みに依存 |
| 少数データ | 比較的扱いやすい | 比較的扱いやすい | 過学習に注意 |
| 表現力 | 低～中 | 低 | 高い |

VAE-GPを評価するときは、`SingleTaskGP`または`MixedSingleTaskGP`、PCA、REMBO、必要に応じてSAASと比較してください。VAE-GPは必ずPCAより高精度になるモデルではありません。データ数が少ない場合や入力構造がほぼ線形の場合は、PCAのほうが安定することがあります。

## 現在の対応範囲

現在のVAE-GP実装は次の範囲を対象とします。

- Gaussian single-output regression
- normal continuous input
- mixed continuous / categorical input
- mixedでは連続列だけをVAE射影し、カテゴリ列を保持
- raw-spaceでのBayesian optimization
- full-batch exact GP
- VAEとGPの共同学習

現在は次に対応していません。

- classification / ordinal regression
- 潜在空間を直接探索するoptimizer
- `condition_on_observations()`による高速更新
- 大規模データ向けmini-batch variational GP

新しい観測を追加する場合は、raw-spaceの学習データを連結してモデル全体を再構築・再学習してください。

## 関連実装

```text
src/bochan/models/components/vae.py
src/bochan/models/regression/gaussian/high_dim/vae.py
src/bochan/models/regression/gaussian/high_dim/vae_mixed.py
src/bochan/fit/vae.py
src/bochan/api/model_registry.py
tests/test_regression_vae_single_output.py
tests/test_regression_vae_mixed_single_output.py
```
