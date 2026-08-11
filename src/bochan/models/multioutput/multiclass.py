from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor
from torch.nn import ModuleList


class MultiOutputMulticlassProbsPosterior(Posterior):
    """
    multi-output multiclass classification 用の probability posterior。

    Shape:
        mean:     ``batch_shape x q x m x C``
        variance: ``batch_shape x q x m x C``

    Notes:
        各出力でクラス数が同じ場合に、single-output multiclass posterior を
        ``m`` 次元に stack して扱うための簡易 posterior です。
        出力ごとにクラス数が異なる場合は、この posterior では stack できないため、
        wrapper 側の ``class_probs_list`` または ``padded_class_probs`` を使います。
    """

    def __init__(self, posteriors: Sequence[Posterior], *, eps: float = 1e-9) -> None:
        super().__init__()
        if len(posteriors) == 0:
            raise ValueError("At least one posterior is required.")
        self.posteriors = list(posteriors)
        self.eps = float(eps)

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mean.dtype

    @property
    def mean(self) -> Tensor:
        means = [p.mean for p in self.posteriors]
        num_classes = [int(m.shape[-1]) for m in means]
        if len(set(num_classes)) != 1:
            raise ValueError(
                "Outputs have different numbers of classes, so posterior.mean "
                "cannot be stacked into [..., q, m, C]. "
                "Use class_probs_list() or padded_class_probs() instead. "
                f"Got num_classes={num_classes}."
            )
        return torch.stack(means, dim=-2)

    @property
    def variance(self) -> Tensor:
        variances = []
        for p in self.posteriors:
            if hasattr(p, "variance"):
                v = p.variance
            else:
                m = p.mean.clamp(self.eps, 1.0 - self.eps)
                v = m * (1.0 - m)
            variances.append(v)
        num_classes = [int(v.shape[-1]) for v in variances]
        if len(set(num_classes)) != 1:
            raise ValueError(
                "Outputs have different numbers of classes, so posterior.variance "
                "cannot be stacked into [..., q, m, C]. "
                "Use class_probs_list() or padded_class_probs() instead. "
                f"Got num_classes={num_classes}."
            )
        return torch.stack(variances, dim=-2)

    @property
    def event_shape(self) -> torch.Size:
        # [..., q, m, C] のうち q, m, C を event とみなす。
        mean = self.mean
        return torch.Size(mean.shape[-3:])

    @property
    def base_sample_shape(self) -> torch.Size:
        # BoTorch sampler 互換のため、外から見える probability shape を返す。
        # 内部の subposterior は latent GP の base_sample_shape を持つ場合があり、
        # その形は probability shape と一致しないことがある。そのため
        # rsample_from_base_samples では base_samples を best-effort で使い、
        # 合わない場合は通常の rsample にフォールバックする。
        return torch.Size(self.mean.shape)

    @property
    def batch_range(self) -> tuple[int, int]:
        # 最後の3次元を q, m, C とみなし、それ以前を t-batch とする。
        return (0, max(0, self.mean.ndim - 3))

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        sample_shape = torch.Size(sample_shape)

        if base_samples is not None:
            base_samples = base_samples.to(device=self.device, dtype=self.dtype)

        samples = []
        for i, posterior in enumerate(self.posteriors):
            base_i = None
            if base_samples is not None:
                # 想定 shape: sample_shape x batch_shape x q x m x C
                # subposterior には m 次元の i 番目だけ渡す。
                if base_samples.ndim >= 3 and base_samples.shape[-2] == len(self.posteriors):
                    base_i = base_samples[..., i, :]
                else:
                    base_i = None
            try:
                s_i = posterior.rsample(sample_shape=sample_shape, base_samples=base_i)
            except Exception:
                s_i = posterior.rsample(sample_shape=sample_shape)
            samples.append(s_i)

        num_classes = [int(s.shape[-1]) for s in samples]
        if len(set(num_classes)) != 1:
            raise ValueError(
                "Outputs have different numbers of classes, so posterior samples "
                "cannot be stacked into sample_shape x [..., q, m, C]. "
                f"Got num_classes={num_classes}."
            )
        return torch.stack(samples, dim=-2)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """BoTorch NormalMCSampler / SobolQMCNormalSampler 互換の sampling method。

        ``MultiOutputMulticlassProbsPosterior`` は probability posterior の wrapper だが、
        内部の single-output posterior は latent GP の base sample shape を持つことがある。
        BoTorch sampler から渡される ``base_samples`` は wrapper の probability shape に
        基づくため、subposterior の latent shape と一致しない場合がある。

        そのため、まず best-effort で base_samples を各 subposterior に分配し、失敗したら
        通常の ``rsample`` にフォールバックする。これにより qEHVI / qNEHVI の sampler 呼び出しで
        ``NotImplementedError`` にならないようにする。
        """
        try:
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)
        except Exception:
            return self.rsample(sample_shape=sample_shape)

    def class_probs(self) -> Tensor:
        return self.mean

    def predict_class(self) -> Tensor:
        return self.mean.argmax(dim=-1)

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Size:
        return torch.Size(sample_shape) + torch.Size(self.mean.shape[:-1])

    @property
    def batch_shape(self) -> torch.Size:
        mean = self.mean
        if mean.ndim <= 3:
            return torch.Size()
        return torch.Size(mean.shape[:-3])


class MultiOutputMulticlassClassificationModel(Model):
    """
    independent な single-output multiclass classification model 群を
    1 つの multi-output multiclass model として扱う wrapper。

    Notes:
        - この wrapper の ``num_outputs`` は multiclass 出力の個数 ``m`` を表す。
        - 各 submodel の ``num_outputs`` は通常 ``num_classes`` なので、binary / ordinal
          wrapper のように ``num_outputs == 1`` では検証しない。
        - ``posterior(X)`` は probability scale の posterior を返す。
        - 出力ごとのクラス数が異なる場合、``posterior().mean`` / ``class_probs()`` は
          stack できないため、``class_probs_list()`` または ``padded_class_probs()`` を使う。
    """

    def __init__(
        self,
        *models: Model,
        validate_same_train_inputs: bool = True,
    ) -> None:
        super().__init__()
        if len(models) == 0:
            raise ValueError("At least one submodel must be provided.")

        self.models = ModuleList(models)

        for i, model in enumerate(self.models):
            if not hasattr(model, "num_classes"):
                raise TypeError(
                    f"Submodel {i} must represent one multiclass output and have `num_classes`. "
                    f"Got {model.__class__.__name__}."
                )
            if not (hasattr(model, "class_probs") or hasattr(model, "posterior")):
                raise TypeError(
                    f"Submodel {i} must implement class_probs(X) or posterior(X). "
                    f"Got {model.__class__.__name__}."
                )

        first_cat_dims = list(getattr(self.models[0], "cat_dims", []) or [])
        same_cat_dims = all(list(getattr(m, "cat_dims", []) or []) == first_cat_dims for m in self.models)
        self.cat_dims = first_cat_dims if same_cat_dims else []

        if validate_same_train_inputs:
            self._validate_same_train_inputs()

    @property
    def num_outputs(self) -> int:
        """multiclass 出力の個数。"""
        return len(self.models)

    @property
    def batch_shape(self) -> torch.Size:
        batch_shape = getattr(self.models[0], "batch_shape", torch.Size())
        for model in self.models[1:]:
            model_batch_shape = getattr(model, "batch_shape", torch.Size())
            if model_batch_shape != batch_shape:
                raise NotImplementedError(
                    "All submodels must have the same batch_shape. "
                    f"Got {batch_shape} and {model_batch_shape}."
                )
        return batch_shape

    @property
    def num_classes_list(self) -> list[int]:
        return [int(getattr(model, "num_classes")) for model in self.models]

    @property
    def num_classes(self) -> int:
        """全出力で同じクラス数の場合のみ、そのクラス数を返す。"""
        classes = self.num_classes_list
        if len(set(classes)) != 1:
            raise ValueError(
                "Outputs have different numbers of classes. "
                f"Use num_classes_list instead. Got {classes}."
            )
        return classes[0]

    @staticmethod
    def _get_submodel_train_input_raw(model: Model) -> Tensor:
        if hasattr(model, "train_input_raw"):
            return getattr(model, "train_input_raw")
        if hasattr(model, "train_inputs_raw"):
            train_inputs_raw = getattr(model, "train_inputs_raw")
            if isinstance(train_inputs_raw, tuple):
                return train_inputs_raw[0]
            return train_inputs_raw
        if hasattr(model, "raw_train_X"):
            return getattr(model, "raw_train_X")
        if hasattr(model, "train_X"):
            return getattr(model, "train_X")
        if hasattr(model, "train_inputs"):
            train_inputs = getattr(model, "train_inputs")
            if isinstance(train_inputs, tuple):
                return train_inputs[0]
            return train_inputs
        raise AttributeError(
            "Submodel does not have train_input_raw, train_inputs_raw, raw_train_X, "
            "train_X, or train_inputs."
        )

    @staticmethod
    def _get_submodel_train_input(model: Model) -> Tensor:
        if hasattr(model, "train_input"):
            return getattr(model, "train_input")
        if hasattr(model, "transformed_train_inputs"):
            transformed = getattr(model, "transformed_train_inputs")
            if isinstance(transformed, tuple):
                return transformed[0]
            return transformed
        if hasattr(model, "train_inputs"):
            train_inputs = getattr(model, "train_inputs")
            if isinstance(train_inputs, tuple):
                return train_inputs[0]
            return train_inputs
        return MultiOutputMulticlassClassificationModel._get_submodel_train_input_raw(model)

    @staticmethod
    def _get_submodel_train_targets(model: Model) -> Tensor:
        if hasattr(model, "train_targets"):
            return getattr(model, "train_targets")
        if hasattr(model, "train_Y"):
            return getattr(model, "train_Y")
        raise AttributeError("Submodel does not have train_targets or train_Y.")

    def _validate_same_train_inputs(self) -> None:
        ref_X = self._get_submodel_train_input_raw(self.models[0])
        for i, model in enumerate(self.models[1:], start=1):
            X_i = self._get_submodel_train_input_raw(model)
            if X_i.shape != ref_X.shape:
                raise ValueError(
                    "All submodels must have the same raw train input shape. "
                    f"Submodel 0 has {tuple(ref_X.shape)}, but submodel {i} has {tuple(X_i.shape)}."
                )
            if not torch.allclose(X_i, ref_X):
                raise ValueError(
                    "All submodels must have the same raw train inputs. "
                    f"Submodel {i} has different train inputs from submodel 0."
                )

    @property
    def train_input_raw(self) -> Tensor:
        return self._get_submodel_train_input_raw(self.models[0])

    @property
    def train_inputs_raw(self) -> tuple[Tensor]:
        return (self.train_input_raw,)

    @property
    def train_input(self) -> Tensor:
        # wrapper 自体は raw-space X を受け取るため、train_input も raw-space とする。
        return self.train_input_raw

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self.train_input,)

    @property
    def transformed_train_inputs_list(self) -> list[Tensor]:
        return [self._get_submodel_train_input(model) for model in self.models]

    @property
    def train_targets_list(self) -> list[Tensor]:
        return [self._get_submodel_train_targets(model) for model in self.models]

    @property
    def train_targets(self) -> Tensor:
        ys = []
        for model in self.models:
            y = self._get_submodel_train_targets(model)
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            elif y.ndim == 2 and y.shape[-1] == 1:
                pass
            else:
                raise ValueError(
                    "Each submodel target must be [n] or [n, 1]. "
                    f"Got shape={tuple(y.shape)}."
                )
            ys.append(y)
        return torch.cat(ys, dim=-1)

    @property
    def train_X(self) -> Tensor:
        return self.train_input_raw

    @property
    def raw_train_X(self) -> Tensor:
        return self.train_input_raw

    @property
    def train_Y(self) -> Tensor:
        return self.train_targets

    def _normalize_output_indices(self, output_indices: Optional[Sequence[int]]) -> list[int]:
        if output_indices is None:
            return list(range(self.num_outputs))
        idcs = [int(i) for i in output_indices]
        for i in idcs:
            if i < 0 or i >= self.num_outputs:
                raise IndexError(f"output index {i} is out of range for num_outputs={self.num_outputs}.")
        return idcs

    def _select_observation_noise_for_output(
        self,
        observation_noise: bool | Tensor,
        *,
        output_index: int,
        X: Tensor,
    ) -> bool | Tensor:
        if not torch.is_tensor(observation_noise):
            return observation_noise

        point_shape = X.shape[:-1]

        # single-output multiclass posterior 用: [..., q] or [..., q, C]
        if observation_noise.shape == point_shape:
            return observation_noise
        if observation_noise.ndim >= 1 and observation_noise.shape[:-1] == point_shape:
            last = int(observation_noise.shape[-1])
            if last in {1, self.num_classes_list[output_index]}:
                return observation_noise
            if last == self.num_outputs:
                return observation_noise[..., output_index]

        # full multi-output class-wise noise: [..., q, m, C]
        if observation_noise.ndim >= 2 and observation_noise.shape[:-2] == point_shape:
            if observation_noise.shape[-2] == self.num_outputs:
                return observation_noise[..., output_index, :]

        raise ValueError(
            "observation_noise must be bool, Tensor with shape X.shape[:-1], "
            "X.shape[:-1] + (C,), X.shape[:-1] + (num_outputs,), or "
            "X.shape[:-1] + (num_outputs, C). "
            f"Got observation_noise.shape={tuple(observation_noise.shape)}, X.shape={tuple(X.shape)}."
        )

    def _probability_posterior_one(
        self,
        model: Model,
        X: Tensor,
        observation_noise: bool | Tensor = False,
        **kwargs: Any,
    ) -> Posterior:
        post = model.posterior(
            X=X,
            output_indices=None,
            observation_noise=observation_noise,
            posterior_transform=None,
            **kwargs,
        )
        if not hasattr(post, "mean"):
            raise TypeError(f"Expected posterior with mean, got {type(post).__name__}.")
        return post

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> MultiOutputMulticlassProbsPosterior:
        return self.probability_posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def probability_posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> MultiOutputMulticlassProbsPosterior:
        if isinstance(X, tuple):
            X = X[0]
        idcs = self._normalize_output_indices(output_indices)
        posteriors = []
        for i in idcs:
            obs_i = self._select_observation_noise_for_output(
                observation_noise,
                output_index=i,
                X=X,
            )
            posteriors.append(
                self._probability_posterior_one(
                    self.models[i],
                    X=X,
                    observation_noise=obs_i,
                    **kwargs,
                )
            )
        posterior = MultiOutputMulticlassProbsPosterior(posteriors)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def class_probs_list(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        **kwargs: Any,
    ) -> list[Tensor]:
        if isinstance(X, tuple):
            X = X[0]
        idcs = self._normalize_output_indices(output_indices)
        probs_list = []
        for i in idcs:
            obs_i = self._select_observation_noise_for_output(
                observation_noise,
                output_index=i,
                X=X,
            )
            model_i = self.models[i]
            class_probs_fn = getattr(model_i, "class_probs", None)
            if callable(class_probs_fn) and not torch.is_tensor(observation_noise):
                try:
                    probs_i = class_probs_fn(X, **kwargs)
                except TypeError:
                    probs_i = class_probs_fn(X)
            else:
                probs_i = self._probability_posterior_one(
                    model_i,
                    X=X,
                    observation_noise=obs_i,
                    **kwargs,
                ).mean
            probs_list.append(probs_i)
        return probs_list

    def class_probs(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        **kwargs: Any,
    ) -> Tensor:
        probs_list = self.class_probs_list(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs,
        )
        num_classes = [int(p.shape[-1]) for p in probs_list]
        if len(set(num_classes)) != 1:
            raise ValueError(
                "Outputs have different numbers of classes, so class_probs() cannot stack them. "
                "Use class_probs_list() or padded_class_probs() instead. "
                f"Got num_classes={num_classes}."
            )
        return torch.stack(probs_list, dim=-2)

    def padded_class_probs(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        pad_value: float = 0.0,
        **kwargs: Any,
    ) -> Tensor:
        probs_list = self.class_probs_list(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs,
        )
        max_c = max(int(p.shape[-1]) for p in probs_list)
        padded = []
        for p in probs_list:
            if p.shape[-1] == max_c:
                padded.append(p)
                continue
            pad_shape = list(p.shape[:-1]) + [max_c - int(p.shape[-1])]
            pad_tensor = torch.full(
                pad_shape,
                fill_value=float(pad_value),
                device=p.device,
                dtype=p.dtype,
            )
            padded.append(torch.cat([p, pad_tensor], dim=-1))
        return torch.stack(padded, dim=-2)

    def probability_variance(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        **kwargs: Any,
    ) -> Tensor:
        return self.probability_posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs,
        ).variance

    def latent_posterior_list(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> list[Posterior]:
        if isinstance(X, tuple):
            X = X[0]
        idcs = self._normalize_output_indices(output_indices)
        outs = []
        for i in idcs:
            fn = getattr(self.models[i], "latent_posterior", None)
            if not callable(fn):
                raise TypeError(f"Submodel {i} does not implement latent_posterior(X).")
            outs.append(fn(X, **kwargs))
        return outs

    def latent_mean_list(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        return [p.mean for p in self.latent_posterior_list(X, output_indices=output_indices, **kwargs)]

    def predict_class(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        preds = []
        for probs in self.class_probs_list(X=X, output_indices=output_indices, observation_noise=False, **kwargs):
            pred_i = probs.argmax(dim=-1)
            if pred_i.ndim == X.ndim - 1:
                pred_i = pred_i.unsqueeze(-1)
            preds.append(pred_i)
        return torch.cat(preds, dim=-1)

    def _normalize_utility_values(
        self,
        utility_values: Optional[Sequence[Sequence[float] | Tensor] | Tensor],
        *,
        idcs: Sequence[int],
        probs_list: Sequence[Tensor],
    ) -> list[Tensor]:
        if utility_values is None:
            return [
                torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
                for p in probs_list
            ]

        if torch.is_tensor(utility_values):
            u = utility_values
            if u.ndim == 1:
                return [u.to(device=p.device, dtype=p.dtype) for p in probs_list]
            if u.ndim == 2:
                if u.shape[0] == self.num_outputs:
                    return [u[i].to(device=p.device, dtype=p.dtype) for i, p in zip(idcs, probs_list)]
                if u.shape[0] == len(idcs):
                    return [u[j].to(device=p.device, dtype=p.dtype) for j, p in enumerate(probs_list)]
            raise ValueError(
                "Tensor utility_values must be [C], [num_outputs, C], or [selected_outputs, C]."
            )

        values = list(utility_values)
        if len(values) == 0:
            raise ValueError("utility_values must not be empty.")

        # common utility: [u0, u1, ..., uC]
        if all(isinstance(v, (int, float)) for v in values):
            return [torch.as_tensor(values, device=p.device, dtype=p.dtype) for p in probs_list]

        # per-output utility list: length num_outputs or selected outputs
        if len(values) == self.num_outputs:
            return [torch.as_tensor(values[i], device=p.device, dtype=p.dtype) for i, p in zip(idcs, probs_list)]
        if len(values) == len(idcs):
            return [torch.as_tensor(values[j], device=p.device, dtype=p.dtype) for j, p in enumerate(probs_list)]

        raise ValueError(
            "utility_values must be None, common [C], per-output list length num_outputs, "
            "or selected-output list length len(output_indices)."
        )

    def expected_utility(
        self,
        X: Tensor,
        utility_values: Optional[Sequence[Sequence[float] | Tensor] | Tensor] = None,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        idcs = self._normalize_output_indices(output_indices)
        probs_list = self.class_probs_list(X=X, output_indices=idcs, observation_noise=False, **kwargs)
        utilities_list = self._normalize_utility_values(
            utility_values,
            idcs=idcs,
            probs_list=probs_list,
        )
        outs = []
        for p, u in zip(probs_list, utilities_list):
            if u.ndim != 1 or u.numel() != p.shape[-1]:
                raise ValueError(
                    "Each utility vector must have length equal to the number of classes. "
                    f"Got utility shape={tuple(u.shape)}, probs.shape={tuple(p.shape)}."
                )
            u = u.reshape(*([1] * (p.ndim - 1)), p.shape[-1])
            out = (p * u).sum(dim=-1)
            if out.ndim == X.ndim - 1:
                out = out.unsqueeze(-1)
            outs.append(out)
        return torch.cat(outs, dim=-1)

    def subset_output(self, idcs: list[int]) -> Model:
        idcs = self._normalize_output_indices(idcs)
        submodels = [self.models[i] for i in idcs]
        if len(submodels) == 1:
            return submodels[0]
        return self.__class__(*submodels)

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "MultiOutputMulticlassClassificationModel":
        if isinstance(X, tuple):
            X_tensor = X[0]
        else:
            X_tensor = X
        expected_y_prefix = X_tensor.shape[:-1]

        if Y.shape == expected_y_prefix and self.num_outputs == 1:
            Y = Y.unsqueeze(-1)
        if not (Y.shape[:-1] == expected_y_prefix and Y.shape[-1] == self.num_outputs):
            raise ValueError(
                f"Expected Y.shape == X.shape[:-1] + ({self.num_outputs},), "
                f"got X.shape={tuple(X_tensor.shape)}, Y.shape={tuple(Y.shape)}."
            )

        if noise is not None:
            if noise.shape == expected_y_prefix and self.num_outputs == 1:
                noise = noise.unsqueeze(-1)
            if not (
                noise.shape[:-1] == expected_y_prefix
                and noise.shape[-1] == self.num_outputs
            ):
                raise ValueError(
                    f"Expected noise.shape == X.shape[:-1] + ({self.num_outputs},), "
                    f"got X.shape={tuple(X_tensor.shape)}, noise.shape={tuple(noise.shape)}."
                )

        fantasy_models = []
        for i, model in enumerate(self.models):
            Y_i = Y[..., i : i + 1]
            noise_i = None if noise is None else noise[..., i : i + 1]
            if noise_i is None:
                fantasy_i = model.condition_on_observations(X=X, Y=Y_i, **kwargs)
            else:
                fantasy_i = model.condition_on_observations(X=X, Y=Y_i, noise=noise_i, **kwargs)
            fantasy_models.append(fantasy_i)
        return self.__class__(*fantasy_models)


# Short aliases mirroring the naming style of ordinal / binary wrappers.
MultiOutputMulticlassModel = MultiOutputMulticlassClassificationModel
MultiOutputMulticlassClassificationGPModel = MultiOutputMulticlassClassificationModel


__all__ = [
    "MultiOutputMulticlassProbsPosterior",
    "MultiOutputMulticlassClassificationModel",
    "MultiOutputMulticlassModel",
    "MultiOutputMulticlassClassificationGPModel",
]
