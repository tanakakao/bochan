from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch import Tensor
from torch.nn import ModuleList

from bochan.posteriors.bernoulli import SimpleBernoulliPosterior


class MultiOutputBernoulliPosterior(Posterior):
    """
    multi-output binary classification 用の簡易 posterior。

    mean / variance は probability scale の Tensor として扱う。

    Shape:
        mean:     [..., q, m]
        variance: [..., q, m]

    Notes:
        - SimpleBernoulliPosterior は single-output 前提で last dim == 1 を要求する。
        - MultiOutputClassificationModel.probability_posterior では last dim が
          num_outputs になるため、この multi-output 版 posterior を使う。
    """

    def __init__(
        self,
        mean: Optional[Tensor] = None,
        variance: Optional[Tensor] = None,
        probs: Optional[Tensor] = None,
        eps: float = 1e-9,
    ) -> None:
        if probs is not None:
            if mean is not None:
                raise ValueError("Specify either mean or probs, not both.")
            mean = probs
        if mean is None:
            raise ValueError("Either mean or probs must be provided.")
        if mean.ndim < 2:
            raise ValueError(
                f"Expected mean to have at least 2 dims (..., q, m), got {tuple(mean.shape)}."
            )

        mean = mean.clamp(eps, 1.0 - eps)
        if variance is None:
            variance = mean * (1.0 - mean)
        else:
            variance = variance.to(device=mean.device, dtype=mean.dtype).clamp_min(eps)
            if variance.shape != mean.shape:
                try:
                    variance = variance.expand_as(mean)
                except RuntimeError as e:
                    raise ValueError(
                        f"variance must be broadcastable to mean.shape={tuple(mean.shape)}, "
                        f"got {tuple(variance.shape)}."
                    ) from e

        self._mean = mean
        self._variance = variance
        self._eps = float(eps)

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    @property
    def event_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def batch_range(self) -> tuple[int, int]:
        # BoTorch の Posterior API 互換用。
        # 最後の 2 次元を q, m とみなし、それ以前を t-batch とする。
        return (0, max(0, self._mean.ndim - 2))

    def _extended_shape(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> torch.Size:
        """
        BoTorch sampler 用の extended sample shape。
        """
        if sample_shape is None:
            sample_shape = torch.Size()
        return torch.Size(sample_shape) + self._mean.shape

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        """
        probability scale の reparameterized proxy sample を返す。

        Notes:
            離散 Bernoulli sample は微分不能なので、qEHVI / qNEHVI のような
            gradient-based MC acquisition には不向きです。

            ここでは probability posterior を連続値の近似分布として扱い、

                p_sample = clamp(mean + sqrt(variance) * base_sample, 0, 1)

            を返します。
        """
        if sample_shape is None:
            sample_shape = torch.Size()
        sample_shape = torch.Size(sample_shape)

        if base_samples is None:
            base_samples = torch.randn(
                sample_shape + self._mean.shape,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            base_samples = base_samples.to(device=self.device, dtype=self.dtype)

        return self.rsample_from_base_samples(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """
        SobolQMCNormalSampler / NormalMCSampler 用。

        base_samples は標準正規サンプルを想定する。
        戻り値は probability scale の連続 proxy sample。
        """
        sample_shape = torch.Size(sample_shape)
        target_shape = sample_shape + self._mean.shape

        base_samples = base_samples.to(device=self.device, dtype=self.dtype)

        if base_samples.shape != target_shape:
            try:
                base_samples = base_samples.expand(target_shape)
            except RuntimeError as e:
                raise RuntimeError(
                    f"base_samples must be broadcastable to {tuple(target_shape)}, "
                    f"got {tuple(base_samples.shape)}."
                ) from e

        mean = self._mean.expand(target_shape)
        std = self._variance.clamp_min(self._eps).sqrt().expand(target_shape)

        return (mean + std * base_samples).clamp(self._eps, 1.0 - self._eps)

    def sample_bernoulli(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        """
        非微分の Bernoulli 離散サンプルを返す。
        診断・可視化用途向け。
        """
        if sample_shape is None:
            sample_shape = torch.Size()
        sample_shape = torch.Size(sample_shape)
        probs = self._mean.expand(sample_shape + self._mean.shape)
        return torch.bernoulli(probs)

    def sample(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        with torch.no_grad():
            return self.sample_bernoulli(sample_shape=sample_shape)


class MultiOutputBinaryClassificationModel(Model):
    """
    independent な single-output binary classification GP 群を
    1 つの multi-output model として扱うラッパ。

    設計方針:
        - posterior(X): Bernoulli probability scale の multi-output posterior を返す
            - single-output classification model の posterior(X) とスケールを揃える
            - mean は [0, 1] の probability
        - latent_posterior(X): latent f の joint posterior を返す
            - 各 submodel の latent_posterior(X) を使う
            - latent f は負値や 1 超えを取り得る
        - probability_posterior(X): posterior(X) と同じく probability scale の posterior を返す
        - class_probs(X): [..., q, m, 2] のクラス確率を返す
            - 最後の次元は [P(y=0), P(y=1)]
        - predict_class(X): [..., q, m] の 0/1 予測を返す
        - expected_utility(X): utility=[u0, u1] に対する期待効用を返す

    注意:
        - 各 submodel は single-output binary classifier を想定する。
        - submodel は latent_posterior(X) を持つ必要がある。
        - condition_on_observations は、各 submodel 側の実装に委譲する。
    """

    def __init__(self, *models: Model) -> None:
        super().__init__()
        if len(models) == 0:
            raise ValueError("At least one submodel must be provided.")

        self.models = ModuleList(models)

        for i, model in enumerate(self.models):
            if getattr(model, "num_outputs", None) != 1:
                raise ValueError(
                    f"Submodel {i} must be single-output, "
                    f"got num_outputs={getattr(model, 'num_outputs', None)}."
                )
            if not hasattr(model, "latent_posterior"):
                raise TypeError(
                    f"Submodel {i} must implement latent_posterior(X). "
                    "Classification submodel.posterior(X) usually returns "
                    "a Bernoulli probability posterior, which cannot be combined "
                    "as a latent MultitaskMultivariateNormal."
                )

        first_cat_dims = list(getattr(self.models[0], "cat_dims", []))
        same_cat_dims = all(
            list(getattr(m, "cat_dims", [])) == first_cat_dims for m in self.models
        )
        self.cat_dims = first_cat_dims if same_cat_dims else []

    # ---------------------------------------------------------------------
    # Basic BoTorch-style properties
    # ---------------------------------------------------------------------
    @property
    def num_outputs(self) -> int:
        return len(self.models)

    @property
    def batch_shape(self) -> torch.Size:
        batch_shape = getattr(self.models[0], "batch_shape", torch.Size())
        for model in self.models[1:]:
            if getattr(model, "batch_shape", torch.Size()) != batch_shape:
                raise NotImplementedError(
                    "All submodels must have the same batch_shape."
                )
        return batch_shape

    @property
    def num_classes_list(self) -> list[int]:
        # binary classification 固定
        return [2 for _ in range(self.num_outputs)]

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
        ys = []
        for model in self.models:
            y = getattr(model, "train_targets")
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            ys.append(y)
        return torch.cat(ys, dim=-1)

    @staticmethod
    def _get_raw_train_x(model: Model) -> Tensor:
        if hasattr(model, "raw_train_X"):
            return getattr(model, "raw_train_X")
        if hasattr(model, "train_inputs_raw"):
            train_inputs_raw = getattr(model, "train_inputs_raw")
            if isinstance(train_inputs_raw, tuple):
                return train_inputs_raw[0]
            return train_inputs_raw
        if hasattr(model, "train_inputs"):
            train_inputs = getattr(model, "train_inputs")
            if isinstance(train_inputs, tuple):
                return train_inputs[0]
            return train_inputs
        raise AttributeError(
            f"{model.__class__.__name__} has no raw_train_X / "
            "train_inputs_raw / train_inputs."
        )

    @staticmethod
    def _get_train_y(model: Model) -> Tensor:
        if hasattr(model, "train_Y"):
            return getattr(model, "train_Y")
        if hasattr(model, "train_targets"):
            return getattr(model, "train_targets")
        raise AttributeError(
            f"{model.__class__.__name__} has no train_Y / train_targets."
        )

    def _normalize_output_indices(
        self,
        output_indices: Optional[Sequence[int]],
    ) -> list[int]:
        if output_indices is None:
            return list(range(self.num_outputs))

        idcs = [int(i) for i in output_indices]
        for i in idcs:
            if i < 0 or i >= self.num_outputs:
                raise IndexError(
                    f"output index {i} is out of range for "
                    f"num_outputs={self.num_outputs}."
                )
        return idcs

    # ---------------------------------------------------------------------
    # Probability-scale posterior / latent joint posterior
    # ---------------------------------------------------------------------
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> MultiOutputBernoulliPosterior:
        """
        Bernoulli probability scale の multi-output posterior を返す。

        Returns:
            MultiOutputBernoulliPosterior:
                mean / variance の shape は [..., q, m_selected]

        Notes:
            single-output classification model の posterior(X) と揃えるため、
            MultiOutputClassificationModel.posterior(X) も probability scale を返す。

            latent f の posterior が必要な場合は latent_posterior(X) を使う。
        """
        return self.probability_posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """
        latent f の multi-output joint posterior を返す。

        latent f は probability ではないため、mean は負値や 1 超えを取り得る。
        BoTorch の latent-space acquisition / diagnostic で使う。
        """
        if observation_noise is not False:
            raise NotImplementedError(
                "MultiOutputClassificationModel.latent_posterior does not support "
                "observation_noise on the latent posterior. "
                "Use posterior(..., observation_noise=...) / probability_posterior(...) "
                "for probability-scale posterior."
            )

        idcs = self._normalize_output_indices(output_indices)

        mvns: list[MultivariateNormal] = []
        for i in idcs:
            model_i = self.models[i]
            post_i = model_i.latent_posterior(X)
            dist_i = post_i.distribution
            if not isinstance(dist_i, MultivariateNormal):
                raise TypeError(
                    f"Expected MultivariateNormal from submodel.latent_posterior, "
                    f"got {type(dist_i).__name__}."
                )

            # DeepGP の latent posterior は mean.shape = [S, ..., q_like] のように
            # 先頭に prediction sample 次元を持つことがある。
            # MultitaskMultivariateNormal.from_independent_mvns に渡す前に、
            # single-output model と同様にその次元を平均する。
            dist_i = self._average_extra_sample_dims_in_mvn(dist_i, X)
            mvns.append(dist_i)

        joint = MultitaskMultivariateNormal.from_independent_mvns(mvns)
        posterior = GPyTorchPosterior(joint)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def latent_mean(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        latent f の mean を返す。

        Returns:
            Tensor:
                shape = [..., q, m_selected]
        """
        return self.latent_posterior(
            X=X,
            output_indices=output_indices,
            **kwargs,
        ).mean

    # ---------------------------------------------------------------------
    # Probability helpers / probabilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _squeeze_last_singleton(t: Tensor) -> Tensor:
        if t.ndim >= 1 and t.shape[-1] == 1:
            return t.squeeze(-1)
        return t

    @staticmethod
    def _reduce_extra_sample_dims_for_points(t: Tensor, X: Tensor) -> Tensor:
        """
        DeepGP posterior に含まれる先頭の prediction sample 次元を平均する。

        Args:
            t:
                single-output の pointwise tensor。
                通常は [..., q_like]、DeepGP では [S, ..., q_like] になることがある。
            X:
                raw X。X.ndim - 1 が通常の pointwise tensor の ndim。

        Returns:
            Tensor:
                余分な先頭 sample 次元を平均した tensor。

        Notes:
            InputPerturbation により q_like = q * n_w になっている場合、
            最後の point dimension は保持し、先頭の DeepGP sample 次元だけを平均する。
        """
        expected_ndim = X.ndim - 1
        while t.ndim > expected_ndim:
            t = t.mean(dim=0)
        return t

    @staticmethod
    def _average_extra_sample_dims_in_mvn(dist: MultivariateNormal, X: Tensor) -> MultivariateNormal:
        """
        DeepGP latent posterior の extra sample / batch 次元を平均し、
        通常の MultivariateNormal に戻す。

        single-output DeepGP の実装で使っている
        _average_deepgp_latent_distribution と同じ考え方を multi-output wrapper 側にも入れる。
        """
        mean = dist.mean
        covar = dist.covariance_matrix

        expected_mean_ndim = X.ndim - 1
        expected_covar_ndim = X.ndim

        while mean.ndim > expected_mean_ndim:
            mean = mean.mean(dim=0)

        while covar.ndim > expected_covar_ndim:
            covar = covar.mean(dim=0)

        # すでに通常 shape の場合も、MultivariateNormal として返す。
        return MultivariateNormal(mean, covar)

    def _select_observation_noise_for_output(
        self,
        observation_noise: Union[bool, Tensor],
        *,
        output_index: int,
        X: Tensor,
    ) -> Union[bool, Tensor]:
        """
        multi-output noise [..., q, m] / [..., q, m_selected] が来た場合に
        single-output 用 [..., q, 1] に切り出す。
        """
        if not torch.is_tensor(observation_noise):
            return observation_noise

        # single-output 向け shape の場合はそのまま渡す
        if observation_noise.shape == X.shape[:-1]:
            return observation_noise
        if observation_noise.shape[:-1] == X.shape[:-1] and observation_noise.shape[-1] == 1:
            return observation_noise

        # full multi-output noise [..., q, m]
        if observation_noise.shape[:-1] == X.shape[:-1] and observation_noise.shape[-1] == self.num_outputs:
            return observation_noise[..., output_index : output_index + 1]

        raise ValueError(
            "observation_noise must be bool, Tensor with shape X.shape[:-1], "
            "X.shape[:-1] + (1,), or X.shape[:-1] + (num_outputs,). "
            f"Got observation_noise.shape={tuple(observation_noise.shape)}, "
            f"X.shape={tuple(X.shape)}."
        )

    def _probability_posterior_one(
        self,
        model: Model,
        X: Tensor,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> SimpleBernoulliPosterior:
        post = model.posterior(
            X=X,
            output_indices=None,
            observation_noise=observation_noise,
            posterior_transform=None,
            **kwargs,
        )
        if not hasattr(post, "mean") or not hasattr(post, "variance"):
            raise TypeError(
                f"Expected posterior with mean and variance, got {type(post).__name__}."
            )
        return post

    def probability_posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> MultiOutputBernoulliPosterior:
        """
        Bernoulli probability scale の multi-output posterior を返す。

        Returns:
            MultiOutputBernoulliPosterior:
                mean / variance の shape は [..., q, m_selected]
        """
        idcs = self._normalize_output_indices(output_indices)

        means = []
        variances = []
        for i in idcs:
            obs_i = self._select_observation_noise_for_output(
                observation_noise,
                output_index=i,
                X=X,
            )
            post_i = self._probability_posterior_one(
                self.models[i],
                X=X,
                observation_noise=obs_i,
                **kwargs,
            )
            mean_i = self._squeeze_last_singleton(post_i.mean)
            var_i = self._squeeze_last_singleton(post_i.variance)

            # DeepGP の probability posterior は本来 submodel 側で reduction 済みだが、
            # wrapper 側でも保険として [S, ..., q_like] -> [..., q_like] に平均する。
            mean_i = self._reduce_extra_sample_dims_for_points(mean_i, X).clamp(0.0, 1.0)
            var_i = self._reduce_extra_sample_dims_for_points(var_i, X).clamp_min(0.0)

            means.append(mean_i.unsqueeze(-1))
            variances.append(var_i.unsqueeze(-1))

        mean = torch.cat(means, dim=-1)
        variance = torch.cat(variances, dim=-1)
        posterior = MultiOutputBernoulliPosterior(mean=mean, variance=variance)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def mean_probability(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> Tensor:
        """
        P(y=1 | X) を返す。

        Returns:
            Tensor: [..., q, m_selected]
        """
        return self.probability_posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs,
        ).mean

    def probability_variance(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> Tensor:
        """
        probability scale の variance を返す。

        Returns:
            Tensor: [..., q, m_selected]
        """
        return self.probability_posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs,
        ).variance

    def class_probs_list(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> list[Tensor]:
        """
        各出力のクラス確率を list で返す。

        Returns:
            list[Tensor]:
                各要素の shape は [..., q, 2]
                最後の次元は [P(y=0), P(y=1)]
        """
        idcs = self._normalize_output_indices(output_indices)
        probs_list = []
        for local_j, i in enumerate(idcs):
            obs_i = self._select_observation_noise_for_output(
                observation_noise,
                output_index=i,
                X=X,
            )
            post_i = self._probability_posterior_one(
                self.models[i],
                X=X,
                observation_noise=obs_i,
                **kwargs,
            )
            p1 = self._squeeze_last_singleton(post_i.mean)
            p1 = self._reduce_extra_sample_dims_for_points(p1, X).clamp(0.0, 1.0)
            probs_i = torch.stack([1.0 - p1, p1], dim=-1)
            probs_list.append(probs_i)
        return probs_list

    def class_probs(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> Tensor:
        """
        multi-output binary class probabilities を返す。

        Returns:
            Tensor:
                shape は [..., q, m_selected, 2]
                最後の次元は [P(y=0), P(y=1)]
        """
        probs_list = self.class_probs_list(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs,
        )
        return torch.stack(probs_list, dim=-2)

    def padded_class_probs(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        pad_value: float = 0.0,
        **kwargs: Any,
    ) -> Tensor:
        """
        ordinal 版との互換用。
        binary classification では全出力が2クラスなので class_probs と同じ。
        """
        return self.class_probs(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs,
        )

    def predict_class(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        threshold: Union[float, Tensor, Sequence[float]] = 0.5,
        **kwargs: Any,
    ) -> Tensor:
        """
        各出力の 0/1 予測を返す。

        Args:
            threshold:
                scalar または m_selected 個の threshold。

        Returns:
            Tensor: [..., q, m_selected]
        """
        p1 = self.mean_probability(
            X=X,
            output_indices=output_indices,
            observation_noise=False,
            **kwargs,
        )

        threshold_t = torch.as_tensor(threshold, device=p1.device, dtype=p1.dtype)
        if threshold_t.ndim == 0:
            return (p1 >= threshold_t).to(torch.long)

        if threshold_t.numel() != p1.shape[-1]:
            raise ValueError(
                f"threshold must be scalar or have length {p1.shape[-1]}, "
                f"got shape={tuple(threshold_t.shape)}."
            )
        threshold_t = threshold_t.reshape(*([1] * (p1.ndim - 1)), p1.shape[-1])
        return (p1 >= threshold_t).to(torch.long)

    def _normalize_utility_values(
        self,
        utility_values: Optional[Union[Sequence[float], Sequence[Sequence[float]], Tensor]],
        *,
        idcs: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        utility を [m_selected, 2] に正規化する。
        """
        m_selected = len(idcs)

        if utility_values is None:
            return torch.tensor(
                [[0.0, 1.0]] * m_selected,
                device=device,
                dtype=dtype,
            )

        u = torch.as_tensor(utility_values, device=device, dtype=dtype)

        # common utility: [u0, u1]
        if u.ndim == 1:
            if u.numel() != 2:
                raise ValueError(
                    "1D utility_values must have length 2: [u0, u1]."
                )
            return u.reshape(1, 2).expand(m_selected, 2)

        # per-output utility: [m, 2] or [m_selected, 2]
        if u.ndim == 2:
            if u.shape[-1] != 2:
                raise ValueError(
                    "2D utility_values must have shape [m, 2] or [m_selected, 2]."
                )
            if u.shape[0] == self.num_outputs:
                return u[list(idcs)]
            if u.shape[0] == m_selected:
                return u
            raise ValueError(
                f"utility_values first dim must be num_outputs={self.num_outputs} "
                f"or selected outputs={m_selected}, got {u.shape[0]}."
            )

        raise ValueError(
            "utility_values must be None, [2], [m, 2], or [m_selected, 2]."
        )

    def expected_utility(
        self,
        X: Tensor,
        utility_values: Optional[Union[Sequence[float], Sequence[Sequence[float]], Tensor]] = None,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        各出力の expected utility を返す。

        Args:
            utility_values:
                - None: [0, 1] を全出力に使うため、P(y=1) と同じ
                - [u0, u1]: 全出力共通の utility
                - [m, 2]: 出力ごとの utility
                - [m_selected, 2]: output_indices 指定時の選択出力ごとの utility

        Returns:
            Tensor: [..., q, m_selected]
        """
        idcs = self._normalize_output_indices(output_indices)
        probs = self.class_probs(
            X=X,
            output_indices=idcs,
            observation_noise=False,
            **kwargs,
        )
        utilities = self._normalize_utility_values(
            utility_values,
            idcs=idcs,
            device=probs.device,
            dtype=probs.dtype,
        )
        utilities = utilities.reshape(*([1] * (probs.ndim - 2)), len(idcs), 2)
        return (probs * utilities).sum(dim=-1)

    # ---------------------------------------------------------------------
    # BoTorch model helpers
    # ---------------------------------------------------------------------
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
    ) -> "MultiOutputClassificationModel":
        """
        観測を追加した multi-output classification model を返す。

        Args:
            X:
                shape は [..., q, d]
            Y:
                shape は [..., q, m]
            noise:
                optional。shape は [..., q, m] または submodel 側で扱える形。
        """
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
            fantasy_i = model.condition_on_observations(
                X=X,
                Y=Y_i,
                noise=noise_i,
                **kwargs,
            )
            fantasy_models.append(fantasy_i)

        return self.__class__(*fantasy_models)
