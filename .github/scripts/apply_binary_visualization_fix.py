from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "src/bochan/visualization/utils.py"
TEST = ROOT / "tests/test_binary_visualization_probability.py"


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one {name} block, found {text.count(old)}."
        )
    return text.replace(old, new, 1)


def update_utils() -> None:
    text = UTILS.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''def get_model(obj: Any) -> Any:
    """BayesianOptimizer / ModelBundle / model から model 本体を取り出す。"""

    if hasattr(obj, "model") and getattr(obj, "model") is not None:
        return getattr(obj, "model")
    if hasattr(obj, "bundle") and getattr(obj, "bundle") is not None:
        return getattr(obj.bundle, "model")
    return obj
''',
        '''def get_model(obj: Any) -> Any:
    """BayesianOptimizer / ModelBundle / model から model 本体を取り出す。

    ``posterior`` を持つ model wrapper 自身を先に返す。binary classification
    wrapper の ``model`` 属性は latent GP なので、先に unwrap すると
    probability posterior ではなく latent model を参照してしまう。
    """

    if callable(getattr(obj, "probability_posterior", None)) or callable(
        getattr(obj, "posterior", None)
    ):
        return obj
    if hasattr(obj, "bundle") and getattr(obj, "bundle") is not None:
        bundle_model = getattr(obj.bundle, "model", None)
        if bundle_model is not None:
            return bundle_model
    if hasattr(obj, "model") and getattr(obj, "model") is not None:
        return getattr(obj, "model")
    return obj
''',
        name="get_model",
    )
    text = replace_once(
        text,
        '''def prediction_mean_std(obj: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    """予測平均と標準偏差を配列で返す内部 helper。"""

    X_t = to_tensor_like(X, obj)
    if hasattr(obj, "predict"):
        try:
            mean, var = obj.predict(X_t, return_type="mean_variance")
        except TypeError:
            posterior = get_model(obj).posterior(X_t)
            mean, var = posterior.mean, posterior.variance
    else:
        posterior = get_model(obj).posterior(X_t)
        mean, var = posterior.mean, posterior.variance
    mean_arr = ensure_2d(mean)
    std_arr = np.sqrt(np.clip(ensure_2d(var), 0.0, None))
    return mean_arr, std_arr
''',
        '''def prediction_mean_std(obj: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    """予測平均と標準偏差を配列で返す内部 helper。

    binary classification model が ``probability_posterior`` を提供する場合は
    それを最優先する。これにより visualization は latent ``f(x)`` ではなく、
    model likelihood と整合した ``p(y=1 | x)`` を表示する。
    """

    X_t = to_tensor_like(X, obj)
    model = get_model(obj)
    probability_posterior = getattr(model, "probability_posterior", None)

    if callable(probability_posterior):
        posterior = probability_posterior(X_t)
        mean, var = posterior.mean, posterior.variance
    elif hasattr(obj, "predict"):
        try:
            mean, var = obj.predict(X_t, return_type="mean_variance")
        except TypeError:
            posterior = model.posterior(X_t)
            mean, var = posterior.mean, posterior.variance
    else:
        posterior = model.posterior(X_t)
        mean, var = posterior.mean, posterior.variance

    mean_arr = ensure_2d(mean)
    std_arr = np.sqrt(np.clip(ensure_2d(var), 0.0, None))
    return mean_arr, std_arr
''',
        name="prediction_mean_std",
    )
    UTILS.write_text(text, encoding="utf-8")


def add_tests() -> None:
    TEST.write_text(
        '''from __future__ import annotations

import numpy as np
import torch

from bochan.visualization.utils import get_model, prediction_mean_std


class _Posterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _LatentModel:
    def posterior(self, X: torch.Tensor) -> _Posterior:
        mean = torch.full(X.shape[:-1] + (1,), 4.0, dtype=X.dtype)
        variance = torch.ones_like(mean)
        return _Posterior(mean, variance)


class _BinaryWrapper:
    def __init__(self) -> None:
        self.model = _LatentModel()
        self.train_X = torch.zeros(2, 1, dtype=torch.double)
        self.probability_calls = 0
        self.posterior_calls = 0

    def probability_posterior(self, X: torch.Tensor) -> _Posterior:
        self.probability_calls += 1
        mean = torch.tensor([[0.2], [0.8]], dtype=X.dtype, device=X.device)
        variance = mean * (1.0 - mean)
        return _Posterior(mean, variance)

    def posterior(self, X: torch.Tensor) -> _Posterior:
        self.posterior_calls += 1
        raise AssertionError("probability_posterior must be preferred")


class _Optimizer:
    def __init__(self, model: _BinaryWrapper) -> None:
        self.model = model
        self.train_X = model.train_X
        self.predict_calls = 0

    def predict(self, X: torch.Tensor, return_type: str):
        self.predict_calls += 1
        raise AssertionError("binary probability_posterior must be preferred")


def test_get_model_keeps_binary_wrapper_instead_of_unwrapping_latent_gp() -> None:
    model = _BinaryWrapper()

    assert get_model(model) is model
    assert get_model(model) is not model.model


def test_prediction_uses_probability_posterior_for_direct_binary_model() -> None:
    model = _BinaryWrapper()
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    mean, std = prediction_mean_std(model, X)

    assert model.probability_calls == 1
    assert model.posterior_calls == 0
    np.testing.assert_allclose(mean[:, 0], [0.2, 0.8])
    np.testing.assert_allclose(std[:, 0], np.sqrt([0.16, 0.16]))
    assert np.all((0.0 <= mean) & (mean <= 1.0))


def test_prediction_uses_nested_binary_probability_posterior_before_predict() -> None:
    model = _BinaryWrapper()
    optimizer = _Optimizer(model)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    mean, _ = prediction_mean_std(optimizer, X)

    assert optimizer.predict_calls == 0
    assert model.probability_calls == 1
    np.testing.assert_allclose(mean[:, 0], [0.2, 0.8])
''',
        encoding="utf-8",
    )


def main() -> None:
    update_utils()
    add_tests()


if __name__ == "__main__":
    main()
