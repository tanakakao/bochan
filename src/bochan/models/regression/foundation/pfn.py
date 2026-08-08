"""Prior-data Fitted Network surrogate model for single-output regression."""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import torch
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor, nn

_PUBLIC_MODEL_FILES = {
    "hebo_plus": "model_hebo_morebudget_9_unused_features_3.pt",
    "hebo_plus_userprior": "hebo_morebudget_9_unused_features_3_userpriorperdim2_8.pt",
    "bnn": "model_sampled_warp_simple_mlp_for_hpob_46.pt",
}
_PUBLIC_MODEL_BASE_URL = (
    "https://github.com/automl/PFNs4BO/raw/main/pfns4bo/final_models"
)


def _require_regression_data(train_X: Tensor, train_Y: Tensor) -> Tensor:
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    if not train_X.is_floating_point():
        raise TypeError("train_X must be a floating-point tensor.")
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError("PFN currently supports single-output regression only.")
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of observations.")
    if train_X.shape[0] == 0:
        raise ValueError("PFN requires at least one training observation.")
    if not train_Y.is_floating_point():
        raise TypeError("train_Y must be a floating-point tensor.")
    if not torch.isfinite(train_X).all() or not torch.isfinite(train_Y).all():
        raise ValueError("PFN training data must be finite.")
    return train_Y


def _resolve_bounds(train_X: Tensor, bounds: Tensor | Sequence[Sequence[float]] | None) -> Tensor:
    if bounds is None:
        lower = train_X.min(dim=0).values
        upper = train_X.max(dim=0).values
        bounds_tensor = torch.stack((lower, upper))
    else:
        bounds_tensor = torch.as_tensor(bounds, dtype=train_X.dtype, device=train_X.device)
    if bounds_tensor.shape != (2, train_X.shape[-1]):
        raise ValueError(
            f"bounds must have shape [2, {train_X.shape[-1]}], got {tuple(bounds_tensor.shape)}."
        )
    if not torch.isfinite(bounds_tensor).all():
        raise ValueError("bounds must be finite.")
    if torch.any(bounds_tensor[1] <= bounds_tensor[0]):
        if bounds is None:
            raise ValueError(
                "PFN could not infer non-degenerate bounds from train_X. "
                "Pass the full search-space bounds explicitly."
            )
        raise ValueError("Every upper bound must be greater than its lower bound.")
    return bounds_tensor


def _default_cache_dir() -> Path:
    base = os.environ.get("BOCHAN_CACHE_DIR")
    if base:
        return Path(base).expanduser() / "pfns4bo"
    return Path.home() / ".cache" / "bochan" / "pfns4bo"


def _import_pfns4bo() -> Any:
    try:
        import pfns4bo
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "PFN public checkpoints require the legacy 'pfns4bo' Python package "
            "for checkpoint class definitions. Install the code package with "
            "`pip install pfns4bo==0.1.5 --no-deps` so bochan can keep its newer "
            "PyTorch / BoTorch / scikit-learn dependency stack."
        ) from exc
    return pfns4bo


def _download_public_checkpoint(model_name: str, destination: Path) -> None:
    filename = _PUBLIC_MODEL_FILES[model_name]
    url = f"{_PUBLIC_MODEL_BASE_URL}/{filename}.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as compressed:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(compressed, output)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pfns4bo_pretrained(
    model_name: str = "hebo_plus",
    *,
    model_path: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    device: torch.device | str = "cpu",
    download_if_missing: bool = True,
) -> nn.Module:
    """Load an official PFNs4BO checkpoint without changing bochan dependencies.

    The upstream checkpoints are full PyTorch pickles. The ``pfns4bo`` package is
    therefore imported only to provide the original class definitions during
    deserialization. Known public checkpoints are downloaded from the official
    automl/PFNs4BO repository into bochan's user cache on first use.
    """
    normalized = str(model_name).lower().replace("-", "_")
    if normalized not in _PUBLIC_MODEL_FILES:
        raise ValueError(
            f"Unknown PFNs4BO model_name={model_name!r}. "
            f"Expected one of {sorted(_PUBLIC_MODEL_FILES)}."
        )

    _import_pfns4bo()
    if model_path is None:
        root = Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
        path = root / _PUBLIC_MODEL_FILES[normalized]
        if not path.exists():
            if not download_if_missing:
                raise FileNotFoundError(f"PFNs4BO checkpoint was not found at {path}.")
            _download_public_checkpoint(normalized, path)
    else:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"PFNs4BO checkpoint was not found at {path}.")

    load_kwargs: dict[str, Any] = {"map_location": device}
    try:
        model = torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:  # pragma: no cover - compatibility with older torch
        model = torch.load(path, **load_kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("The PFNs4BO checkpoint did not contain a torch.nn.Module.")
    return model


class PFNPosterior(Posterior):
    """Marginal PFN bar-distribution posterior used for prediction and inspection.

    Version 1 intentionally exposes marginal moments and bucket probabilities but
    does not claim a joint reparameterizable posterior across q points. Native PFN
    EI / PI / UCB should therefore be used for optimization.
    """

    def __init__(self, mean: Tensor, variance: Tensor, logits: Tensor) -> None:
        self._mean = mean
        self._variance = variance.clamp_min(0.0)
        self.logits = logits

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    @property
    def probabilities(self) -> Tensor:
        return self.logits.softmax(dim=-1)

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    def _extended_shape(self, sample_shape: torch.Size = torch.Size()) -> torch.Size:  # noqa: B008
        return sample_shape + self._mean.shape

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        raise NotImplementedError(
            "PFNPosterior v1 does not expose reparameterized joint samples. "
            "Use PFNExpectedImprovement / PFNProbabilityOfImprovement / "
            "PFNUpperConfidenceBound for candidate optimization."
        )


class PFNRegressorModel(Model):
    """In-context PFNs4BO surrogate for continuous single-output regression."""

    posterior_family = "pfn_bar"

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        bounds: Tensor | Sequence[Sequence[float]] | None = None,
        pretrained_model: nn.Module | None = None,
        model_name: str = "hebo_plus",
        model_path: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        download_if_missing: bool = True,
        target_standardize: bool = True,
        inference_dtype: torch.dtype = torch.float32,
        max_eval_points: int = 4096,
        strict_bounds: bool = True,
        bounds_atol: float = 1e-7,
        style: Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        train_Y = _require_regression_data(train_X, train_Y)
        resolved_bounds = _resolve_bounds(train_X, bounds)
        if max_eval_points <= 0:
            raise ValueError("max_eval_points must be positive.")
        if bounds_atol < 0:
            raise ValueError("bounds_atol must be non-negative.")

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        self.register_buffer("bounds", resolved_bounds.detach().clone())
        target_mean = train_Y.mean()
        target_scale = train_Y.std(unbiased=False)
        if not torch.isfinite(target_scale) or target_scale <= torch.finfo(train_Y.dtype).eps:
            target_scale = torch.ones((), dtype=train_Y.dtype, device=train_Y.device)
        self.register_buffer("target_mean", target_mean.detach().clone())
        self.register_buffer("target_scale", target_scale.detach().clone())

        self.model_name = str(model_name).lower().replace("-", "_")
        self.model_path = None if model_path is None else str(model_path)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.download_if_missing = bool(download_if_missing)
        self.target_standardize = bool(target_standardize)
        self.inference_dtype = inference_dtype
        self.max_eval_points = int(max_eval_points)
        self.strict_bounds = bool(strict_bounds)
        self.bounds_atol = float(bounds_atol)
        self.style = None if style is None else torch.as_tensor(style)
        self._pfn_model = pretrained_model
        self._is_fitted = False

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self.train_X,)

    @property
    def train_targets(self) -> Tensor:
        return self.train_Y.squeeze(-1)

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def pfn_model(self) -> nn.Module:
        if self._pfn_model is None:
            raise RuntimeError("PFN checkpoint is not loaded. Call fit() first.")
        return self._pfn_model

    @property
    def criterion(self) -> Any:
        criterion = getattr(self.pfn_model, "criterion", None)
        if criterion is None:
            raise AttributeError("The PFN model does not expose its bar-distribution criterion.")
        return criterion

    def make_mll(self, **_: Any) -> None:
        return None

    def fit(self, _fit_target: Any | None = None, **_: Any) -> Self:
        """Load/freeze the pretrained PFN; no task-specific parameter fitting is done."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this PFN model instance.")
        if self._pfn_model is None:
            self._pfn_model = load_pfns4bo_pretrained(
                self.model_name,
                model_path=self.model_path,
                cache_dir=self.cache_dir,
                device=self.train_X.device,
                download_if_missing=self.download_if_missing,
            )
        self._pfn_model.to(device=self.train_X.device)
        self._pfn_model.requires_grad_(False)
        self._pfn_model.eval()
        self._is_fitted = True
        self.eval()
        return self

    def _normalize_X(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self.train_X.shape[-1]:
            raise ValueError(f"Expected {self.train_X.shape[-1]} input features, got {X.shape[-1]}.")
        bounds = self.bounds.to(dtype=X.dtype, device=X.device)
        lower, upper = bounds[0], bounds[1]
        if self.strict_bounds:
            detached = X.detach()
            if torch.any(detached < lower - self.bounds_atol) or torch.any(
                detached > upper + self.bounds_atol
            ):
                raise ValueError("PFN inputs must lie inside the configured search-space bounds.")
        return ((X - lower) / (upper - lower)).clamp(0.0, 1.0)

    def _standardized_targets(self) -> Tensor:
        y = self.train_Y.squeeze(-1)
        if not self.target_standardize:
            return y
        return (y - self.target_mean) / self.target_scale

    def _objective_targets(self, maximize: bool) -> Tensor:
        sign = 1.0 if maximize else -1.0
        return sign * self._standardized_targets()

    def _to_internal_incumbent(self, incumbent: Tensor | float, maximize: bool) -> Tensor:
        value = torch.as_tensor(incumbent, dtype=self.train_Y.dtype, device=self.train_Y.device)
        if self.target_standardize:
            value = (value - self.target_mean) / self.target_scale
        return value if maximize else -value

    def _predict_logits(self, X: Tensor, *, context_y: Tensor | None = None) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("PFN model is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")
        normalized_train_X = self._normalize_X(self.train_X).to(dtype=self.inference_dtype)
        normalized_X = self._normalize_X(X).to(dtype=self.inference_dtype)
        original_shape = normalized_X.shape[:-1]
        flat_X = normalized_X.reshape(-1, normalized_X.shape[-1])
        targets = self._standardized_targets() if context_y is None else context_y
        targets = targets.to(device=flat_X.device, dtype=self.inference_dtype).reshape(-1)
        normalized_train_X = normalized_train_X.to(device=flat_X.device)

        style = self.style
        if style is not None:
            style = style.to(device=flat_X.device, dtype=self.inference_dtype)

        chunks: list[Tensor] = []
        for start in range(0, flat_X.shape[0], self.max_eval_points):
            x_eval = flat_X[start : start + self.max_eval_points]
            x_full = torch.cat((normalized_train_X, x_eval), dim=0).unsqueeze(1)
            y_full = targets.unsqueeze(1)
            output = self.pfn_model(
                (style, x_full, y_full),
                single_eval_pos=normalized_train_X.shape[0],
            )
            if isinstance(output, dict):
                if "standard" not in output:
                    raise RuntimeError("PFN output dictionary does not contain 'standard' logits.")
                output = output["standard"]
            if not isinstance(output, Tensor):
                raise TypeError("PFN forward output must be a Tensor or a dict containing a Tensor.")
            expected_total = normalized_train_X.shape[0] + x_eval.shape[0]
            if output.shape[0] == expected_total:
                output = output[normalized_train_X.shape[0] :]
            elif output.shape[0] != x_eval.shape[0]:
                raise RuntimeError(
                    "Unexpected PFN output sequence length: "
                    f"got {output.shape[0]}, expected {x_eval.shape[0]} or {expected_total}."
                )
            if output.ndim == 3:
                probabilities = output.softmax(dim=-1).mean(dim=1)
                output = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
            elif output.ndim != 2:
                raise RuntimeError(
                    f"Expected PFN logits with 2 or 3 dimensions, got shape {tuple(output.shape)}."
                )
            chunks.append(output)
        logits = torch.cat(chunks, dim=0)
        return logits.reshape(*original_shape, logits.shape[-1])

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any | None = None,
    ) -> PFNPosterior:
        if output_indices not in (None, [0]):
            raise ValueError("PFNRegressorModel has only output index 0.")
        if observation_noise is not False:
            raise NotImplementedError("PFN v1 does not expose a separate observation-noise posterior.")
        if posterior_transform is not None:
            raise NotImplementedError("PFN v1 does not support BoTorch posterior_transform.")

        logits = self._predict_logits(X)
        internal_mean = self.criterion.mean(logits)
        internal_variance = self.criterion.variance(logits)
        if self.target_standardize:
            mean = self.target_mean.to(internal_mean) + self.target_scale.to(internal_mean) * internal_mean
            variance = self.target_scale.to(internal_variance).square() * internal_variance
        else:
            mean = internal_mean
            variance = internal_variance
        return PFNPosterior(mean=mean.unsqueeze(-1), variance=variance.unsqueeze(-1), logits=logits)

    def native_acquisition(
        self,
        X: Tensor,
        *,
        kind: str,
        incumbent: Tensor | float | None = None,
        maximize: bool = True,
        rest_prob: float = (1.0 - 0.6826894921370859) / 2.0,
    ) -> Tensor:
        """Evaluate PFNs4BO's bar-distribution EI, PI, or quantile-UCB for q=1."""
        if X.shape[-2] != 1:
            raise ValueError("PFN native acquisitions currently support q=1 only.")
        context_y = self._objective_targets(maximize=maximize)
        logits = self._predict_logits(X, context_y=context_y)
        normalized_kind = str(kind).lower().replace("-", "_")

        if normalized_kind in {"ei", "expected_improvement"}:
            if incumbent is None:
                incumbent = self.train_Y.max() if maximize else self.train_Y.min()
            best_internal = self._to_internal_incumbent(incumbent, maximize=maximize)
            values = self.criterion.ei(logits, best_internal, maximize=True)
            if self.target_standardize:
                values = values * self.target_scale.to(values)
        elif normalized_kind in {"pi", "probability_of_improvement"}:
            if incumbent is None:
                incumbent = self.train_Y.max() if maximize else self.train_Y.min()
            best_internal = self._to_internal_incumbent(incumbent, maximize=maximize)
            values = self.criterion.pi(logits, best_internal, maximize=True)
        elif normalized_kind in {"ucb", "upper_confidence_bound"}:
            if not 0.0 < float(rest_prob) < 0.5:
                raise ValueError("rest_prob must be in (0, 0.5).")
            values = self.criterion.ucb(logits, None, rest_prob=float(rest_prob), maximize=True)
            if self.target_standardize:
                sign = 1.0 if maximize else -1.0
                values = sign * self.target_mean.to(values) + self.target_scale.to(values) * values
        else:
            raise ValueError("kind must be one of: ei, pi, ucb.")
        return values.squeeze(-1)

    def forward(self, X: Tensor) -> Tensor:
        return self.posterior(X).mean


__all__ = [
    "PFNPosterior",
    "PFNRegressorModel",
    "load_pfns4bo_pretrained",
]
