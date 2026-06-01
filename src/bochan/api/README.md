# bochan API

`bochan.api` は、モデル生成・学習・獲得関数生成・候補点最適化を分離したまま、
外側からは `BayesianOptimizer` クラスとして扱うための高レベル API です。

## 基本方針

内部処理は以下の4段階に分かれています。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)

acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer` は、この流れを薄く包むクラスです。

```python
bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
)

bo.fit(train_X, train_Y)

posterior = bo.predict(test_X)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
)
```

## モデルクラスを直接渡す例

```python
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    ModelConfig,
    OptimizeConfig,
)

model_config = ModelConfig(
    model_cls=SingleTaskGP,
    task_type="regression",
    model_type="base",
)

fit_config = FitConfig(
    mll_cls=ExactMarginalLogLikelihood,
    fit_func=fit_gpytorch_mll,
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="qEI",
    acqf_cls=qExpectedImprovement,
    acqf_kwargs={
        "best_f": train_Y.max(),
    },
)

opt_config = OptimizeConfig(
    q=3,
    num_restarts=10,
    raw_samples=256,
)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

## registry を使う例

ネストした registry にも対応しています。

```python
model_registry = {
    "normal": {
        "regression": {
            "base": SingleTaskGP,
        },
    },
    "mixed": {
        "regression": {
            "base": MixedSingleTaskGP,
        },
    },
}

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    cat_dims=[],
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
    model_registry=model_registry,
)
```

`cat_dims` が空なら `"normal"`、非空なら `"mixed"` としてモデルを解決します。

## ask/tell 形式

```python
candidates, acq_value = bo.ask(acq_config, opt_config)

new_Y = evaluate(candidates)

bo.tell(candidates, new_Y, refit=True)
```

## 複数獲得関数の比較

```python
results = bo.compare_acquisitions(
    acq_configs=[ei_config, ucb_config, straddle_config],
    opt_config=opt_config,
)

for name, result in results.items():
    print(name, result.candidates, result.acq_value)
```
