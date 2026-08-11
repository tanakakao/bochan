from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor
from torch.nn import Module, ModuleList

from .posterior import HybridPosterior
from .specs import OutputIndex, OutputSpec, PosteriorMode


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

        first_cat_dims = self._get_cat_dims(self.models[0])
        self.cat_dims = (
            first_cat_dims
            if all(self._get_cat_dims(model) == first_cat_dims for model in self.models)
            else []
        )

        first_tf = getattr(self.models[0], "input_transform", None)
        self.input_transform = (
            first_tf
            if all(getattr(m, "input_transform", None) is first_tf for m in self.models)
            else None
        )

    @staticmethod
    def _get_cat_dims(model: Module) -> list[int]:
        """モデルのカテゴリ次元を比較可能なリストへ正規化する。"""

        cat_dims = getattr(model, "cat_dims", None)
        return [] if cat_dims is None else list(cat_dims)

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
        batch_shapes = [getattr(m, "batch_shape", torch.Size()) for m in self.models]
        if not batch_shapes:
            return torch.Size()
        first = batch_shapes[0]
        if all(shape == first for shape in batch_shapes):
            return first
        return torch.Size()

    @property
    def train_inputs(self) -> tuple[Tensor, ...]:
        return self.models[0].train_inputs

    @train_inputs.setter
    def train_inputs(self, value: tuple[Tensor, ...]) -> None:
        self._train_inputs = value

    @property
    def train_targets(self) -> Tensor:
        targets = []
        for model in self.models:
            target = getattr(model, "train_targets", None)
            if target is None:
                raise AttributeError(
                    f"{type(model).__name__} does not expose train_targets."
                )
            target = torch.as_tensor(target)
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            targets.append(target)
        return torch.cat(targets, dim=-1)

    @staticmethod
    def _unwrap_X(X: Tensor) -> Tensor:
        return X[0] if isinstance(X, tuple) else X

    def _normalize_output_indices(
        self,
        output_indices: OutputIndex | Sequence[OutputIndex] | Tensor | None,
    ) -> list[int]:
        if output_indices is None:
            return list(range(self.num_outputs))
        if torch.is_tensor(output_indices):
            output_indices = output_indices.detach().cpu().flatten().tolist()
        elif isinstance(output_indices, (int, str)):
            output_indices = [output_indices]
        else:
            output_indices = list(output_indices)

        name_to_index = {spec.name: i for i, spec in enumerate(self.specs)}
        normalized: list[int] = []
        for index in output_indices:
            if isinstance(index, str):
                if index not in name_to_index:
                    raise KeyError(
                        f"Unknown output name {index!r}. "
                        f"Available outputs: {list(name_to_index)}."
                    )
                normalized.append(name_to_index[index])
            else:
                resolved = int(index)
                if resolved < 0:
                    resolved += self.num_outputs
                if resolved < 0 or resolved >= self.num_outputs:
                    raise IndexError(
                        f"output index {index} is out of bounds for {self.num_outputs} outputs."
                    )
                normalized.append(resolved)
        return normalized

    def subset_output(self, idcs: list[int]) -> "HybridMultiOutputModel":
        selected = [self.specs[i] for i in idcs]
        return self.__class__(selected)

    @staticmethod
    def _call_accessor(
        model: Module,
        names: Sequence[str],
        X: Tensor,
        **kwargs: Any,
    ) -> Any:
        for name in names:
            accessor = getattr(model, name, None)
            if callable(accessor):
                try:
                    return accessor(X, **kwargs)
                except TypeError:
                    return accessor(X)
        raise AttributeError(
            f"{type(model).__name__} does not expose any of {tuple(names)}."
        )

    @staticmethod
    def _posterior_mean_variance(
        posterior: Any,
        name: str,
    ) -> tuple[Tensor, Tensor]:
        mean = getattr(posterior, "mean", None)
        variance = getattr(posterior, "variance", None)
        if mean is None or variance is None:
            raise AttributeError(
                f"{name} posterior must expose mean and variance."
            )
        return torch.as_tensor(mean), torch.as_tensor(variance)

    @staticmethod
    def _select_output_tensor(
        tensor: Tensor,
        output_index: int,
        *,
        name: str,
    ) -> Tensor:
        if tensor.ndim == 0:
            return tensor
        if tensor.shape[-1] == 1:
            return tensor.squeeze(-1)
        if output_index >= tensor.shape[-1]:
            raise IndexError(
                f"output_index={output_index} is out of bounds for {name}.shape={tuple(tensor.shape)}."
            )
        return tensor[..., output_index]

    @staticmethod
    def _call_class_probs(fn: Any, X: Tensor, **kwargs: Any) -> Tensor:
        try:
            return fn(X, **kwargs)
        except TypeError:
            return fn(X)

    def _regression_stats(
        self,
        spec: OutputSpec,
        X: Tensor,
        *,
        output_mode: PosteriorMode,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        if output_mode == "latent":
            posterior = self._call_accessor(
                spec.model,
                ("latent_posterior", "posterior"),
                X,
                **kwargs,
            )
        else:
            posterior = self._call_accessor(spec.model, ("posterior",), X, **kwargs)
        mean, variance = self._posterior_mean_variance(posterior, spec.name)
        mean = self._select_output_tensor(mean, spec.output_index, name=f"{spec.name}.mean")
        variance = self._select_output_tensor(
            variance,
            spec.output_index,
            name=f"{spec.name}.variance",
        )
        return mean, variance

    def _binary_probability_stats(
        self,
        spec: OutputSpec,
        X: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        posterior = self._call_accessor(
            spec.model,
            ("probability_posterior", "posterior"),
            X,
            **kwargs,
        )
        p1, variance = self._posterior_mean_variance(posterior, spec.name)
        p1 = self._select_output_tensor(
            p1,
            spec.output_index,
            name=f"{spec.name}.probability",
        ).clamp(0.0, 1.0)
        variance = self._select_output_tensor(
            variance,
            spec.output_index,
            name=f"{spec.name}.variance",
        ).clamp_min(0.0)
        p0 = 1.0 - p1
        return p0, variance, p1

    def _ordinal_class_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return probs
        posterior = self._call_accessor(
            spec.model,
            ("probability_posterior",),
            X,
            **kwargs,
        )
        if torch.is_tensor(posterior):
            return posterior
        mean = getattr(posterior, "mean", None)
        if mean is None:
            raise AttributeError(
                f"{spec.name} ordinal probability posterior must expose probabilities or mean."
            )
        return torch.as_tensor(mean)

    def _multiclass_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return probs
        posterior = self._call_accessor(
            spec.model,
            ("probability_posterior", "posterior"),
            X,
            **kwargs,
        )
        probs, _ = self._posterior_mean_variance(posterior, spec.name)
        return probs

    @staticmethod
    def _utility_tensor(
        spec: OutputSpec,
        *,
        num_classes: int,
        ref: Tensor,
    ) -> Tensor:
        if spec.utility_values is None:
            return torch.arange(num_classes, device=ref.device, dtype=ref.dtype)
        utility = torch.as_tensor(
            spec.utility_values,
            device=ref.device,
            dtype=ref.dtype,
        )
        if utility.numel() != num_classes:
            raise ValueError(
                f"{spec.name}.utility_values must have length {num_classes}, got {utility.numel()}."
            )
        return utility

    def _probability_stats(
        self,
        spec: OutputSpec,
        X: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        if spec.task_type == "binary":
            p0, _, p1 = self._binary_probability_stats(spec, X, **kwargs)
            positive = 1 if spec.positive_class is None else int(spec.positive_class)
            probability = p1 if positive == 1 else p0
            return probability, (probability * (1.0 - probability)).clamp_min(0.0)

        if spec.task_type == "ordinal":
            probs = self._ordinal_class_probs(spec, X, **kwargs).clamp_min(0.0)
        elif spec.task_type == "multiclass":
            probs = self._multiclass_probs(spec, X, **kwargs).clamp_min(0.0)
        else:
            raise ValueError(
                f"output_mode='probability' is unsupported for task_type={spec.task_type!r}."
            )

        positive = (
            int(spec.positive_class)
            if spec.positive_class is not None
            else int(probs.shape[-1] - 1)
        )
        if positive < 0 or positive >= probs.shape[-1]:
            raise IndexError(
                f"positive_class={positive} is out of bounds for {spec.name} with "
                f"{probs.shape[-1]} classes."
            )
        probability = probs[..., positive]
        return probability, (probability * (1.0 - probability)).clamp_min(0.0)

    def _expected_utility_stats(
        self,
        spec: OutputSpec,
        X: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        if spec.task_type == "binary":
            p0, _, p1 = self._binary_probability_stats(spec, X, **kwargs)
            probs = torch.stack([p0, p1], dim=-1)
        elif spec.task_type == "ordinal":
            probs = self._ordinal_class_probs(spec, X, **kwargs).clamp_min(0.0)
        elif spec.task_type == "multiclass":
            probs = self._multiclass_probs(spec, X, **kwargs).clamp_min(0.0)
        else:
            raise ValueError(
                f"output_mode='expected_utility' is unsupported for task_type={spec.task_type!r}."
            )

        utility = self._utility_tensor(
            spec,
            num_classes=int(probs.shape[-1]),
            ref=probs,
        )
        mean = (probs * utility).sum(dim=-1)
        second = (probs * utility.square()).sum(dim=-1)
        variance = (second - mean.square()).clamp_min(0.0)
        return mean, variance

    @staticmethod
    def _objective_transform_stats(
        spec: OutputSpec,
        mean: Tensor,
        variance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        scale = float(spec.sign) * float(spec.weight)
        if spec.eq_target is not None:
            mean = -torch.abs(mean - float(spec.eq_target))
            variance = variance * (float(spec.weight) ** 2)
        else:
            mean = mean * scale
            variance = variance * (scale**2)
        if spec.transform is not None:
            mean = spec.transform(mean)
        return mean, variance

    def _output_stats(
        self,
        spec: OutputSpec,
        X: Tensor,
        *,
        output_mode: PosteriorMode,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        if output_mode == "probability":
            return self._probability_stats(spec, X, **kwargs)
        if output_mode == "expected_utility":
            return self._expected_utility_stats(spec, X, **kwargs)

        if spec.task_type == "regression":
            mean, variance = self._regression_stats(
                spec,
                X,
                output_mode=output_mode,
                **kwargs,
            )
        elif spec.task_type == "binary":
            p0, variance, p1 = self._binary_probability_stats(spec, X, **kwargs)
            if output_mode == "latent":
                posterior = self._call_accessor(
                    spec.model,
                    ("latent_posterior", "posterior"),
                    X,
                    **kwargs,
                )
                mean, variance = self._posterior_mean_variance(posterior, spec.name)
                mean = self._select_output_tensor(
                    mean,
                    spec.output_index,
                    name=f"{spec.name}.latent_mean",
                )
                variance = self._select_output_tensor(
                    variance,
                    spec.output_index,
                    name=f"{spec.name}.latent_variance",
                )
            else:
                positive = 1 if spec.positive_class is None else int(spec.positive_class)
                mean = p1 if positive == 1 else p0
        elif spec.task_type == "ordinal":
            probs = self._ordinal_class_probs(spec, X, **kwargs).clamp_min(0.0)
            utility = self._utility_tensor(
                spec,
                num_classes=int(probs.shape[-1]),
                ref=probs,
            )
            mean = (probs * utility).sum(dim=-1)
            second = (probs * utility.square()).sum(dim=-1)
            variance = (second - mean.square()).clamp_min(0.0)
        elif spec.task_type == "multiclass":
            probs = self._multiclass_probs(spec, X, **kwargs).clamp_min(0.0)
            utility = self._utility_tensor(
                spec,
                num_classes=int(probs.shape[-1]),
                ref=probs,
            )
            mean = (probs * utility).sum(dim=-1)
            second = (probs * utility.square()).sum(dim=-1)
            variance = (second - mean.square()).clamp_min(0.0)
        else:
            raise RuntimeError(f"Unsupported task_type={spec.task_type!r}.")

        if output_mode == "objective":
            return self._objective_transform_stats(spec, mean, variance)
        return mean, variance

    def posterior(
        self,
        X: Tensor,
        output_indices: OutputIndex | Sequence[OutputIndex] | Tensor | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        *,
        output_mode: PosteriorMode = "objective",
        **kwargs: Any,
    ) -> Posterior:
        if observation_noise is not False:
            raise NotImplementedError(
                "HybridMultiOutputModel does not support observation_noise."
            )
        X = self._unwrap_X(X)
        indices = self._normalize_output_indices(output_indices)

        means = []
        variances = []
        for i in indices:
            mean, variance = self._output_stats(
                self.specs[i],
                X,
                output_mode=output_mode,
                **kwargs,
            )
            means.append(mean.unsqueeze(-1))
            variances.append(variance.unsqueeze(-1))

        posterior = HybridPosterior(
            mean=torch.cat(means, dim=-1),
            variance=torch.cat(variances, dim=-1),
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        **kwargs: Any,
    ) -> "HybridMultiOutputModel":
        if Y.shape[-1] != self.num_outputs:
            raise ValueError(
                f"Y.shape[-1] must equal num_outputs={self.num_outputs}. "
                f"Got Y.shape={tuple(Y.shape)}."
            )
        new_specs = []
        for i, spec in enumerate(self.specs):
            model = spec.model
            condition = getattr(model, "condition_on_observations", None)
            if not callable(condition):
                raise NotImplementedError(
                    f"{type(model).__name__} does not support condition_on_observations."
                )
            y_i = Y[..., i : i + 1]
            try:
                new_model = condition(X=X, Y=y_i, **kwargs)
            except TypeError:
                new_model = condition(X, y_i, **kwargs)
            new_specs.append(replace(spec, model=new_model))
        return self.__class__(new_specs)

    def fantasize(
        self,
        X: Tensor,
        sampler: Any,
        observation_noise: Tensor | None = None,
        **kwargs: Any,
    ) -> "HybridMultiOutputModel":
        new_specs = []
        for spec in self.specs:
            fantasize = getattr(spec.model, "fantasize", None)
            if not callable(fantasize):
                raise NotImplementedError(
                    f"{type(spec.model).__name__} does not support fantasize."
                )
            new_model = fantasize(
                X,
                sampler=sampler,
                observation_noise=observation_noise,
                **kwargs,
            )
            new_specs.append(replace(spec, model=new_model))
        return self.__class__(new_specs)


__all__ = ["HybridMultiOutputModel"]
