from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/visualization/utils.py"
text = path.read_text(encoding="utf-8")

anchor = "import pandas as pd\n"
imports = '''
from bochan.acquisition.binary.epistemic import binary_probability_moments
'''
if imports.strip() not in text:
    text = text.replace(anchor, anchor + imports, 1)

old = '''def prediction_mean_std(obj: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
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
'''
new = '''def _is_binary_prediction_object(obj: Any, model: Any) -> bool:
    """Return whether visualization should use binary uncertainty semantics."""
    bundle = getattr(obj, "bundle", None)
    task_type = getattr(bundle, "task_type", None)
    if task_type is None:
        config = getattr(obj, "model_config", None)
        task_type = getattr(config, "task_type", None)
    if str(task_type).lower() == "binary":
        return True
    module_name = type(model).__module__.lower()
    class_name = type(model).__name__.lower()
    return "classification.binary" in module_name or "binary" in class_name


def prediction_mean_std(
    obj: Any,
    X: Any,
    *,
    uncertainty_kind: str = "epistemic",
    num_uncertainty_samples: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictive mean and standard deviation for visualization.

    For binary classification the default band is probability epistemic
    uncertainty ``sqrt(Var_f[p(y=1|f)])``.  ``uncertainty_kind`` may also be
    ``aleatoric`` or ``observation`` / ``bernoulli``.  The latter reproduces
    the old ``sqrt(p(1-p))`` label-variance band.
    """
    X_t = to_tensor_like(X, obj)
    model = get_model(obj)

    if _is_binary_prediction_object(obj, model):
        mean, epistemic, aleatoric, observation = binary_probability_moments(
            model,
            X_t,
            num_samples=num_uncertainty_samples,
        )
        key = str(uncertainty_kind).lower()
        if key == "epistemic":
            var = epistemic
        elif key == "aleatoric":
            var = aleatoric
        elif key in {"observation", "bernoulli", "total_label"}:
            var = observation
        else:
            raise ValueError(
                "binary uncertainty_kind must be 'epistemic', 'aleatoric', "
                "'observation', or 'bernoulli'."
            )
    else:
        if hasattr(obj, "predict"):
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
'''
if text.count(old) != 1:
    raise RuntimeError("prediction_mean_std block was not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

path = root / "src/bochan/visualization/data.py"
text = path.read_text(encoding="utf-8")
old = '''def prediction_dataframe(obj: Any, X: Any, *, target_cols: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bochan モデルの予測平均・標準偏差を DataFrame で返す。"""

    mean, std = prediction_mean_std(obj, X)
'''
new = '''def prediction_dataframe(
    obj: Any,
    X: Any,
    *,
    target_cols: Sequence[str] | None = None,
    uncertainty_kind: str = "epistemic",
    num_uncertainty_samples: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return predictive mean and uncertainty band as DataFrames.

    Binary classification defaults to probability epistemic uncertainty.
    """
    mean, std = prediction_mean_std(
        obj,
        X,
        uncertainty_kind=uncertainty_kind,
        num_uncertainty_samples=num_uncertainty_samples,
    )
'''
if text.count(old) != 1:
    raise RuntimeError("prediction_dataframe block was not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
