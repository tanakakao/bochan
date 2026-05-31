from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor
from torch.nn import Module, ModuleList

from .posterior import HybridPosterior
from .specs import OutputSpec, PosteriorMode

OutputIndex = Union[int, str]


class HybridMultiOutputModel(Model):
    """異種タスクの single-output model 群を multi-output model として扱う wrapper。

    既存の homogeneous multi-output 実装は変更せず、回帰・二値分類・順序回帰・
    多クラス分類を `objective` scale に変換して `[..., q, m]` として束ねる。
    """

    def __init__(self, specs: Sequence[OutputSpec]) -> None:
        super().__init__()
        if len(specs) == 0:
            raise ValueError("At least one OutputSpec must be provided.")

        self.specs = list(specs)
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError(f"OutputSpec.name must be unique. Got {names}.")

        models = []
        for i, spec in enumerate(self.specs):
            if not isinstance(spec.model, Module):
                raise TypeError(
                    f"specs[{i}].model must be torch.nn.Module / BoTorch Model. "
                    f"Got {type(spec.model).__name__}."
                )
            models.append(spec.model)
        self.models = ModuleList(models)

        first_cat_dims = list(getattr(self.models[0], "cat_dims", []))
        self.cat_dims = (
            first_cat_dims
            if all(list(getattr(m, "cat_dims", [])) == first_cat_dims for m in self.models)
            else []
        )

        first_tf = getattr(self.models[0], "input_transform", None)
        self.input_transform = (
            first_tf
            if all(getattr(m, "input_transform", None) is first_tf for m in self.models)
            else None
        )

    @property
    def num_outputs(self) -> int:
        return len(self.specs)

    @property
    def output_names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def task_types(self) -> list[str]:
        return [s.task_type for s in self.specs]

    @property
    def batch_shape(self) -> torch.Size:
        bs = getattr(self.models[0], "batch_shape", torch.Size())
        for m in self.models[1:]:
            if getattr(m, "batch_shape", torch.Size()) != bs:
                raise NotImplementedError("All submodels must have the same batch_shape.")
        return bs

    @property
    def raw_train_X(self) -> Tensor:
        return self._get_raw_train_x(self.models[0])

    @property
    def train_X(self) -> Tensor:
        return self.raw_train_X

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self.raw_train_X,)

    @property
    def train_Y(self) -> Tensor:
        ys = []
        for model in self.models:
            y = self._get_train_y(model)
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            ys.append(y)
        return torch.cat(ys, dim=-1)

    @property
    def train_targets(self) -> Tensor:
        return self.train_Y

    @staticmethod
    def _unwrap_X(X: Union[Tensor, tuple[Tensor, ...]]) -> Tensor:
        return X[0] if isinstance(X, tuple) else X

    @staticmethod
    def _get_raw_train_x(model: Model) -> Tensor:
        for name in ("raw_train_X", "train_inputs_raw", "train_inputs"):
            if not hasattr(model, name):
                continue
            value = getattr(model, name)
            if isinstance(value, tuple):
                return value[0]
            return value
        raise AttributeError(
            f"{model.__class__.__name__} has no raw_train_X / train_inputs_raw / train_inputs."
        )

    @staticmethod
    def _get_train_y(model: Model) -> Tensor:
        for name in ("train_Y", "train_targets"):
            if hasattr(model, name):
                return getattr(model, name)
        raise AttributeError(f"{model.__class__.__name__} has no train_Y / train_targets.")

    def _normalize_output_indices(
        self,
        output_indices: Optional[Union[OutputIndex, Sequence[OutputIndex], Tensor]],
    ) -> list[int]:
        if output_indices is None:
            return list(range(self.num_outputs))
        if torch.is_tensor(output_indices):
            output_indices = output_indices.detach().cpu().tolist()
        if isinstance(output_indices, (int, str)):
            output_indices = [output_indices]

        name_to_idx = {name: i for i, name in enumerate(self.output_names)}
        idcs = []
        for item in output_indices:
            if isinstance(item, str):
                if item not in name_to_idx:
                    raise KeyError(f"Unknown output name {item!r}. Available={self.output_names}.")
                i = name_to_idx[item]
            else:
                i = int(item)
            if i < 0 or i >= self.num_outputs:
                raise IndexError(f"output index {i} is out of range for num_outputs={self.num_outputs}.")
            idcs.append(i)
        return idcs

    @staticmethod
    def _call_accessor(model: Any, names: Sequence[str], X: Tensor, **kwargs: Any):
        last_error: Optional[Exception] = None
        for name in names:
            fn = getattr(model, name, None)
            if not callable(fn):
                continue
            try:
                return fn(X=X, **kwargs)
            except TypeError as e1:
                last_error = e1
                try:
                    return fn(X, **kwargs)
                except TypeError as e2:
                    last_error = e2
                    try:
                        return fn(X)
                    except TypeError as e3:
                        last_error = e3
        if last_error is not None:
            raise last_error
        raise AttributeError(f"{model.__class__.__name__} has none of {tuple(names)}.")

    @staticmethod
    def _call_class_probs(fn, X: Tensor, **kwargs: Any):
        """class_probs 系 accessor を安全に呼ぶ。

        ordinal / multiclass wrapper の `class_probs` は、BoTorch posterior と違って
        `observation_noise` や `posterior_transform` を受け取らない実装がある。
        そのため、まず kwargs 付きで試し、失敗した場合は X のみで再試行する。
        """

        try:
            return fn(X=X, **kwargs)
        except TypeError as e1:
            try:
                return fn(X, **kwargs)
            except TypeError:
                try:
                    return fn(X=X)
                except TypeError:
                    try:
                        return fn(X)
                    except TypeError:
                        raise e1

    @staticmethod
    def _posterior_mean_variance(post: Posterior, name: str) -> tuple[Tensor, Tensor]:
        mean = getattr(post, "mean", None)
        if mean is None:
            raise AttributeError(f"{name} posterior has no mean.")
        var = getattr(post, "variance", None)
        if var is None:
            var = torch.zeros_like(mean)
        return mean, var

    @staticmethod
    def _reduce_extra_sample_dims(t: Tensor, X: Tensor) -> Tensor:
        expected = max(1, X.ndim - 1)
        while t.ndim > expected:
            t = t.mean(dim=0)
        return t

    def _select_scalar(self, t: Tensor, X: Tensor, *, output_index: int, name: str) -> Tensor:
        if not torch.is_tensor(t):
            raise TypeError(f"{name} must be a Tensor. Got {type(t)}.")

        if t.ndim >= X.ndim and t.shape[-1] == 1:
            t = t.squeeze(-1)

        # multi-output posterior: [..., q, m] -> [..., q]
        if t.ndim >= X.ndim and t.shape[-2] == X.shape[-2]:
            if output_index >= t.shape[-1]:
                raise IndexError(
                    f"output_index={output_index} is out of bounds for {name}.shape={tuple(t.shape)}."
                )
            t = t[..., output_index]

        return self._reduce_extra_sample_dims(t, X)

    @staticmethod
    def _stack(values: Sequence[Tensor], name: str) -> Tensor:
        ref = values[0].shape
        out = []
        for i, v in enumerate(values):
            if v.shape != ref:
                try:
                    v = v.expand(ref)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"All {name} tensors must have same shape. "
                        f"0={tuple(ref)}, {i}={tuple(v.shape)}."
                    ) from e
            out.append(v.unsqueeze(-1))
        return torch.cat(out, dim=-1)

    @staticmethod
    def _as_1d(values: Optional[Sequence[float] | Tensor], default: Tensor, name: str) -> Tensor:
        if values is None:
            return default
        out = torch.as_tensor(values, device=default.device, dtype=default.dtype)
        if out.ndim != 1:
            raise ValueError(f"{name} must be 1D. Got shape={tuple(out.shape)}.")
        return out

    @staticmethod
    def _class_utility_stats(probs: Tensor, utilities: Tensor) -> tuple[Tensor, Tensor]:
        if probs.shape[-1] != utilities.numel():
            raise RuntimeError(
                "Number of classes does not match utilities. "
                f"probs={tuple(probs.shape)}, utilities={utilities.numel()}."
            )
        utilities = utilities.reshape(*([1] * (probs.ndim - 1)), utilities.numel())
        mean = (probs * utilities).sum(dim=-1)
        var = (probs * (utilities - mean.unsqueeze(-1)).pow(2)).sum(dim=-1)
        return mean, var

    @staticmethod
    def _ordinal_likelihood(model: Any):
        for name in ("ordinal_likelihood", "likelihood"):
            value = getattr(model, name, None)
            if value is not None:
                return value
        raise AttributeError(f"{model.__class__.__name__} has no ordinal_likelihood / likelihood.")

    @staticmethod
    def _ordinal_cutpoints(likelihood: Any) -> Tensor:
        for name in (
            "get_cutpoints",
            "transformed_cutpoints",
            "cutpoints",
            "thresholds",
            "cuts",
            "boundaries",
            "raw_cutpoints",
            "_ordered_cutpoints",
            "_cutpoints",
        ):
            if not hasattr(likelihood, name):
                continue
            value = getattr(likelihood, name)
            if callable(value):
                value = value()
            if torch.is_tensor(value):
                return value.reshape(-1)
        raise AttributeError("Could not find ordinal cutpoints.")

    @staticmethod
    def _ordinal_probs_from_latent(latent_f: Tensor, cutpoints: Tensor, eps: float = 1e-12) -> Tensor:
        if latent_f.ndim >= 1 and latent_f.shape[-1] == 1:
            latent_f = latent_f.squeeze(-1)
        cutpoints = cutpoints.to(device=latent_f.device, dtype=latent_f.dtype).reshape(-1)
        cdf = torch.sigmoid(cutpoints.view(*([1] * latent_f.ndim), -1) - latent_f.unsqueeze(-1))
        probs = torch.cat([cdf[..., :1], cdf[..., 1:] - cdf[..., :-1], 1.0 - cdf[..., -1:]], dim=-1)
        probs = probs.clamp_min(eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _objective_transform(self, mean: Tensor, var: Tensor, spec: OutputSpec) -> tuple[Tensor, Tensor]:
        if spec.eq_target is not None:
            mean = -torch.abs(mean - float(spec.eq_target)) * spec.weight
            var = var * (spec.weight ** 2)
        else:
            mean = mean * spec.sign * spec.weight
            var = var * (spec.weight ** 2)
        if spec.transform is not None:
            mean = spec.transform(mean)
        return mean, var

    def _regression_stats(self, spec: OutputSpec, X: Tensor, output_mode: PosteriorMode, **kwargs: Any):
        names = ("latent_posterior", "posterior") if output_mode == "latent" else ("posterior",)
        post = self._call_accessor(spec.model, names, X, **kwargs)
        mean, var = self._posterior_mean_variance(post, spec.name)
        mean = self._select_scalar(mean, X, output_index=spec.output_index, name=f"{spec.name}.mean")
        var = self._select_scalar(var, X, output_index=spec.output_index, name=f"{spec.name}.variance")
        if output_mode in ("objective", "expected_utility"):
            mean, var = self._objective_transform(mean, var, spec)
        return mean, var

    def _binary_probability_stats(self, spec: OutputSpec, X: Tensor, **kwargs: Any):
        post = self._call_accessor(spec.model, ("probability_posterior", "posterior"), X, **kwargs)
        p1, var = self._posterior_mean_variance(post, spec.name)
        p1 = self._select_scalar(p1, X, output_index=spec.output_index, name=f"{spec.name}.p1").clamp(0.0, 1.0)
        var = self._select_scalar(var, X, output_index=spec.output_index, name=f"{spec.name}.var").clamp_min(0.0)
        p = 1.0 - p1 if spec.positive_class == 0 else p1
        return p, var, p1

    def _binary_stats(self, spec: OutputSpec, X: Tensor, output_mode: PosteriorMode, **kwargs: Any):
        if output_mode == "latent":
            post = self._call_accessor(spec.model, ("latent_posterior", "posterior_latent", "posterior_f"), X, **kwargs)
            mean, var = self._posterior_mean_variance(post, spec.name)
            return (
                self._select_scalar(mean, X, output_index=spec.output_index, name=f"{spec.name}.latent_mean"),
                self._select_scalar(var, X, output_index=spec.output_index, name=f"{spec.name}.latent_var"),
            )

        p, var, p1 = self._binary_probability_stats(spec, X, **kwargs)
        if output_mode == "probability" or spec.utility_values is None:
            mean = p
        else:
            utilities = self._as_1d(
                spec.utility_values,
                torch.tensor([0.0, 1.0], device=p1.device, dtype=p1.dtype),
                f"{spec.name}.utility_values",
            )
            if utilities.numel() != 2:
                raise ValueError("Binary utility_values must have length 2.")
            mean, var = self._class_utility_stats(torch.stack([1.0 - p1, p1], dim=-1), utilities)

        if output_mode in ("objective", "expected_utility"):
            mean, var = self._objective_transform(mean, var, spec)
        return mean, var

    def _ordinal_class_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                if probs.ndim >= X.ndim + 1:
                    probs = probs.squeeze(-2) if probs.shape[-2] == 1 else probs[..., spec.output_index, :]
                return probs.clamp_min(0.0)

        post = self._call_accessor(spec.model, ("latent_posterior", "posterior_latent", "posterior_f"), X, **kwargs)
        latent, _ = self._posterior_mean_variance(post, spec.name)
        latent = self._select_scalar(latent, X, output_index=spec.output_index, name=f"{spec.name}.latent")
        cutpoints = self._ordinal_cutpoints(self._ordinal_likelihood(spec.model))
        return self._ordinal_probs_from_latent(latent, cutpoints)

    def _ordinal_stats(self, spec: OutputSpec, X: Tensor, output_mode: PosteriorMode, **kwargs: Any):
        if output_mode == "latent":
            post = self._call_accessor(spec.model, ("latent_posterior", "posterior_latent", "posterior_f"), X, **kwargs)
            mean, var = self._posterior_mean_variance(post, spec.name)
            return (
                self._select_scalar(mean, X, output_index=spec.output_index, name=f"{spec.name}.latent_mean"),
                self._select_scalar(var, X, output_index=spec.output_index, name=f"{spec.name}.latent_var"),
            )

        probs = self._ordinal_class_probs(spec, X, **kwargs)
        utilities = self._as_1d(
            spec.utility_values,
            torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype),
            f"{spec.name}.utility_values",
        )
        mean, var = self._class_utility_stats(probs, utilities)
        if output_mode in ("objective", "expected_utility"):
            mean, var = self._objective_transform(mean, var, spec)
        return mean, var

    def _multiclass_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                if probs.ndim >= X.ndim + 1:
                    probs = probs.squeeze(-2) if probs.shape[-2] == 1 else probs[..., spec.output_index, :]
                return probs.clamp_min(0.0)

        post = self._call_accessor(spec.model, ("probability_posterior", "posterior"), X, **kwargs)
        probs, _ = self._posterior_mean_variance(post, spec.name)
        if probs.ndim >= X.ndim + 1:
            probs = probs.squeeze(-2) if probs.shape[-2] == 1 else probs[..., spec.output_index, :]
        return probs.clamp_min(0.0)

    def _multiclass_stats(self, spec: OutputSpec, X: Tensor, output_mode: PosteriorMode, **kwargs: Any):
        if output_mode == "latent":
            post = self._call_accessor(spec.model, ("latent_posterior", "posterior_latent", "posterior_f", "posterior"), X, **kwargs)
            mean, var = self._posterior_mean_variance(post, spec.name)
            return (
                self._select_scalar(mean, X, output_index=spec.output_index, name=f"{spec.name}.latent_mean"),
                self._select_scalar(var, X, output_index=spec.output_index, name=f"{spec.name}.latent_var"),
            )

        probs = self._multiclass_probs(spec, X, **kwargs)
        if output_mode == "probability":
            cls = probs.shape[-1] - 1 if spec.positive_class is None else int(spec.positive_class)
            if cls < 0 or cls >= probs.shape[-1]:
                raise IndexError(f"positive_class={cls} is out of range.")
            mean = probs[..., cls]
            var = mean * (1.0 - mean)
        else:
            utilities = self._as_1d(
                spec.utility_values,
                torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype),
                f"{spec.name}.utility_values",
            )
            mean, var = self._class_utility_stats(probs, utilities)

        if output_mode in ("objective", "expected_utility"):
            mean, var = self._objective_transform(mean, var, spec)
        return mean, var

    def _stats(self, spec: OutputSpec, X: Tensor, output_mode: PosteriorMode, **kwargs: Any):
        if spec.task_type == "regression":
            return self._regression_stats(spec, X, output_mode, **kwargs)
        if spec.task_type == "binary":
            return self._binary_stats(spec, X, output_mode, **kwargs)
        if spec.task_type == "ordinal":
            return self._ordinal_stats(spec, X, output_mode, **kwargs)
        if spec.task_type == "multiclass":
            return self._multiclass_stats(spec, X, output_mode, **kwargs)
        raise RuntimeError(f"Unsupported task_type={spec.task_type!r}.")

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[Union[OutputIndex, Sequence[OutputIndex], Tensor]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        *,
        output_mode: PosteriorMode = "objective",
        **kwargs: Any,
    ) -> HybridPosterior:
        if output_mode not in {"objective", "mean", "latent", "probability", "expected_utility"}:
            raise ValueError(f"Unknown output_mode={output_mode!r}.")

        X = self._unwrap_X(X)
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("observation_noise", observation_noise)
        call_kwargs.setdefault("posterior_transform", None)

        means, variances = [], []
        for i in self._normalize_output_indices(output_indices):
            mean_i, var_i = self._stats(self.specs[i], X, output_mode, **call_kwargs)
            means.append(mean_i)
            variances.append(var_i)

        post = HybridPosterior(mean=self._stack(means, "mean"), variance=self._stack(variances, "variance"))
        return posterior_transform(post) if posterior_transform is not None else post

    def objective_posterior(self, X: Tensor, output_indices=None, **kwargs: Any) -> HybridPosterior:
        return self.posterior(X=X, output_indices=output_indices, output_mode="objective", **kwargs)

    def mean_posterior(self, X: Tensor, output_indices=None, **kwargs: Any) -> HybridPosterior:
        return self.posterior(X=X, output_indices=output_indices, output_mode="mean", **kwargs)

    def latent_posterior(self, X: Tensor, output_indices=None, **kwargs: Any) -> HybridPosterior:
        return self.posterior(X=X, output_indices=output_indices, output_mode="latent", **kwargs)

    def probability_posterior(self, X: Tensor, output_indices=None, **kwargs: Any) -> HybridPosterior:
        return self.posterior(X=X, output_indices=output_indices, output_mode="probability", **kwargs)

    def expected_utility_posterior(self, X: Tensor, output_indices=None, **kwargs: Any) -> HybridPosterior:
        return self.posterior(X=X, output_indices=output_indices, output_mode="expected_utility", **kwargs)

    def objective_mean(self, X: Tensor, output_indices=None, **kwargs: Any) -> Tensor:
        return self.objective_posterior(X=X, output_indices=output_indices, **kwargs).mean

    def expected_utility(self, X: Tensor, output_indices=None, **kwargs: Any) -> Tensor:
        return self.expected_utility_posterior(X=X, output_indices=output_indices, **kwargs).mean

    def class_probs_list(self, X: Tensor, output_indices=None, **kwargs: Any) -> list[Tensor]:
        X = self._unwrap_X(X)
        out = []
        for i in self._normalize_output_indices(output_indices):
            spec = self.specs[i]
            if spec.task_type == "binary":
                _, _, p1 = self._binary_probability_stats(spec, X, **kwargs)
                out.append(torch.stack([1.0 - p1, p1], dim=-1))
            elif spec.task_type == "ordinal":
                out.append(self._ordinal_class_probs(spec, X, **kwargs))
            elif spec.task_type == "multiclass":
                out.append(self._multiclass_probs(spec, X, **kwargs))
            else:
                raise TypeError(f"Output {spec.name!r} is regression and has no class probabilities.")
        return out

    def subset_output(self, idcs: Union[OutputIndex, Sequence[OutputIndex], Tensor]) -> Model:
        indices = self._normalize_output_indices(idcs)
        if len(indices) == 1:
            return self.models[indices[0]]
        return self.__class__([self.specs[i] for i in indices])

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "HybridMultiOutputModel":
        X_tensor = self._unwrap_X(X)
        expected = X_tensor.shape[:-1]
        if Y.shape == expected and self.num_outputs == 1:
            Y = Y.unsqueeze(-1)
        if not (Y.shape[:-1] == expected and Y.shape[-1] == self.num_outputs):
            raise ValueError(
                f"Expected Y.shape == X.shape[:-1] + ({self.num_outputs},), "
                f"got X.shape={tuple(X_tensor.shape)}, Y.shape={tuple(Y.shape)}."
            )

        if noise is not None:
            if noise.shape == expected and self.num_outputs == 1:
                noise = noise.unsqueeze(-1)
            if not (noise.shape[:-1] == expected and noise.shape[-1] == self.num_outputs):
                raise ValueError(
                    f"Expected noise.shape == X.shape[:-1] + ({self.num_outputs},), "
                    f"got noise.shape={tuple(noise.shape)}."
                )

        new_specs = []
        for i, spec in enumerate(self.specs):
            fn = getattr(spec.model, "condition_on_observations", None)
            if not callable(fn):
                raise NotImplementedError(f"Submodel {i} ({spec.name!r}) has no condition_on_observations.")
            y_i = Y[..., i : i + 1]
            noise_i = None if noise is None else noise[..., i : i + 1]
            model_i = fn(X=X, Y=y_i, **kwargs) if noise_i is None else fn(X=X, Y=y_i, noise=noise_i, **kwargs)
            new_specs.append(replace(spec, model=model_i))
        return self.__class__(new_specs)


__all__ = ["HybridMultiOutputModel"]
