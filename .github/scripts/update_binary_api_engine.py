from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str, name: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one {name} block in {path}.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


configs = ROOT / "src/bochan/api/configs.py"
replace_once(
    configs,
    '''@dataclass
class PredictionResult:
    """予測結果。"""

    posterior: Any
    mean: Any | None = None
    variance: Any | None = None
''',
    '''@dataclass
class PredictionResult:
    """予測結果。

    binary の ``mean`` はクラス1確率です。``variance_kind`` が
    ``bernoulli_observation`` の場合、variance は ``p * (1 - p)`` であり、
    確率推定値そのものの epistemic variance ではありません。
    """

    posterior: Any
    mean: Any | None = None
    variance: Any | None = None
    task_type: str | None = None
    prediction_space: str | None = None
    variance_kind: str | None = None
''',
    "PredictionResult",
)

engine = ROOT / "src/bochan/api/engine.py"
text = engine.read_text(encoding="utf-8")
pattern = re.compile(
    r'''    def predict\(\n        self,\n        X: Any,\n        \*,\n        return_type: str = "posterior",\n        return_result: bool = False,\n        posterior_kwargs: dict\[str, Any\] \| None = None,\n    \) -> Any:\n        """予測を行う。"""\n.*?        raise ValueError\("Unknown return_type\. Expected 'posterior', 'mean', 'variance', or 'mean_variance'\."\)\n''',
    re.DOTALL,
)
replacement = '''    def predict(
        self,
        X: Any,
        *,
        return_type: str = "posterior",
        return_result: bool = False,
        posterior_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """予測を行う。

        binary classification では、利用可能なら ``probability_posterior``
        を優先する。mean はクラス1確率、variance は通常 Bernoulli 観測分散。
        """
        self._check_fitted()
        posterior_kwargs = posterior_kwargs or {}
        task_type = str(self.bundle.task_type)

        probability_posterior = getattr(self.model, "probability_posterior", None)
        if task_type == "binary" and callable(probability_posterior):
            posterior = probability_posterior(X, **posterior_kwargs)
        else:
            posterior = self.model.posterior(X, **posterior_kwargs)

        mean = getattr(posterior, "mean", None)
        variance = getattr(posterior, "variance", None)

        if task_type == "binary":
            prediction_space = "probability"
            observation_noise = posterior_kwargs.get("observation_noise", False)
            has_observation_noise = observation_noise is not False and observation_noise is not None
            variance_kind = (
                "bernoulli_observation_plus_noise"
                if has_observation_noise
                else "bernoulli_observation"
            )
        else:
            prediction_space = "outcome"
            variance_kind = "posterior"

        if return_result:
            return PredictionResult(
                posterior=posterior,
                mean=mean,
                variance=variance,
                task_type=task_type,
                prediction_space=prediction_space,
                variance_kind=variance_kind,
            )
        if return_type == "posterior":
            return posterior
        if return_type == "mean":
            return mean
        if return_type == "variance":
            return variance
        if return_type == "mean_variance":
            return mean, variance
        raise ValueError(
            "Unknown return_type. Expected 'posterior', 'mean', 'variance', or 'mean_variance'."
        )
'''
updated, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError("BayesianOptimizer.predict block was not found.")
engine.write_text(updated, encoding="utf-8")
