from __future__ import annotations

import copy
from typing import Any, Literal, Optional, Sequence

import torch
from torch import Tensor
from torch.distributions import Gamma as TorchGamma

from botorch.models.transforms.input import ChainedInputTransform, InputTransform, Normalize
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import _OneDimensionalLikelihood


GammaLink = Literal["softplus", "exp"]


def clone_input_transform(input_transform: Optional[InputTransform]) -> Optional[InputTransform]:
    """input_transform を安全に複製する。"""
    return None if input_transform is None else copy.deepcopy(input_transform)


def to_device_dtype_transform(
    input_transform: Optional[InputTransform],
    ref: Tensor,
) -> Optional[InputTransform]:
    """InputTransform を ref と同じ device / dtype に移す。"""
    if input_transform is None:
        return None
    if hasattr(input_transform, "to"):
        input_transform = input_transform.to(device=ref.device, dtype=ref.dtype)
    return input_transform


def positive_mean_from_latent(
    f: Tensor,
    *,
    link: GammaLink = "softplus",
    exp_clip: float = 20.0,
    min_mean: float = 1e-8,
) -> Tensor:
    """latent f から Gamma mean μ > 0 を作る。"""
    if link == "softplus":
        return torch.nn.functional.softplus(f).clamp_min(min_mean)
    if link == "exp":
        return f.clamp(max=float(exp_clip)).exp().clamp_min(min_mean)
    raise ValueError(f"Unknown Gamma link: {link}")


def positive_concentration_from_raw(
    raw_concentration: Tensor,
    *,
    min_concentration: float = 1e-6,
) -> Tensor:
    """raw concentration parameter から正の concentration κ を作る。"""
    return torch.nn.functional.softplus(raw_concentration).clamp_min(min_concentration)


def prepare_positive_targets(train_Y: Tensor, ref: Tensor, *, min_value: float = 1e-12) -> Tensor:
    """Gamma 回帰用の正値 target を [n] に整形する。"""
    y = torch.as_tensor(train_Y, device=ref.device, dtype=ref.dtype)
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if (y <= 0).any():
        raise ValueError("Gamma targets must be strictly positive.")
    return y.clamp_min(float(min_value)).contiguous()


def ensure_2d_col(y: Tensor) -> Tensor:
    """1D tensor を [n, 1] にする。"""
    return y.unsqueeze(-1) if y.ndim == 1 else y


def normalize_dims(dims: Sequence[int], d: int) -> list[int]:
    """負の index を許容して feature dimension index を正規化する。"""
    out: list[int] = []
    for idx in dims:
        j = int(idx)
        if j < 0:
            j = d + j
        if j < 0 or j >= d:
            raise ValueError(f"dim index {idx} is out of range for d={d}.")
        if j not in out:
            out.append(j)
    return out


def get_cont_dims(d: int, cat_dims: Sequence[int]) -> list[int]:
    """カテゴリ列以外の連続列 index を返す。"""
    cat = set(normalize_dims(cat_dims, d))
    return [j for j in range(d) if j not in cat]


def expand_raw_X_to_match_transformed_q(X: Tensor, X_tf: Tensor) -> Tensor:
    """InputPerturbation 後の X_tf と比較できるように raw X の q 次元を展開する。"""
    if X.shape == X_tf.shape:
        return X
    if X.ndim < 2 or X_tf.ndim < 2 or X.shape[-1] != X_tf.shape[-1]:
        return X
    if X.shape[:-2] == X_tf.shape[:-2]:
        q = X.shape[-2]
        q_like = X_tf.shape[-2]
        if q_like == q:
            return X
        if q > 0 and q_like % q == 0:
            return X.repeat_interleave(q_like // q, dim=-2)
    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)
    return X


def check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
) -> None:
    """mixed model で input_transform がカテゴリ列を変えていないか確認する。"""
    if cat_dims is None or len(cat_dims) == 0:
        return
    cat_idx = [int(i) for i in cat_dims]
    if X.shape[-1] != X_tf.shape[-1]:
        raise ValueError(
            "For mixed Gamma models, input_transform must preserve feature dimension. "
            f"raw dim={X.shape[-1]}, transformed dim={X_tf.shape[-1]}."
        )
    X_cmp = expand_raw_X_to_match_transformed_q(X, X_tf)
    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            "Could not align raw X with transformed X for categorical column check. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, X_cmp.shape={tuple(X_cmp.shape)}."
        )
    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(f"input_transform must not modify categorical columns: cat_dims={cat_idx}.")


def apply_input_transform_for_training(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
    name: str = "input_transform",
) -> Tensor:
    """学習用 X に input_transform を適用する。InputPerturbation による点数増加は許さない。"""
    if input_transform is None:
        return X
    was_training = getattr(input_transform, "training", False)
    if hasattr(input_transform, "train"):
        input_transform.train()
    X_tf = input_transform(X)
    if not was_training and hasattr(input_transform, "eval"):
        input_transform.eval()
    if X_tf.shape[-2] != X.shape[-2]:
        raise RuntimeError(
            f"{name} expanded training inputs. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}. "
            "For InputPerturbation, use transform_on_train=False during fitting."
        )
    check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def apply_input_transform_for_eval(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """posterior / acquisition 評価用 transform。InputPerturbation による q 展開を許す。"""
    if input_transform is None:
        return X
    X_tf = input_transform(X)
    check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def extract_normalize_only_transform(
    input_transform: Optional[InputTransform],
) -> Optional[InputTransform]:
    """ChainedInputTransform から Normalize のみを抽出する。"""
    if input_transform is None:
        return None
    if isinstance(input_transform, Normalize):
        return copy.deepcopy(input_transform)
    if isinstance(input_transform, ChainedInputTransform):
        for key in input_transform.keys():
            tf = input_transform[key]
            if isinstance(tf, Normalize):
                return copy.deepcopy(tf)
    return None


def align_like(t: Tensor, ref: Tensor) -> Tensor:
    """t を ref と同じ shape にできる範囲で揃える。"""
    if t.shape == ref.shape:
        return t
    if ref.ndim >= 1 and ref.shape[-1] == 1:
        ref_no_last = ref.squeeze(-1)
        if t.shape == ref_no_last.shape:
            return t.unsqueeze(-1)
        if t.ndim >= 1 and t.shape[-1] == 1 and t.squeeze(-1).shape == ref_no_last.shape:
            return t
        if (
            t.ndim == ref.ndim
            and t.shape[-1] == 1
            and t.shape[:-2] == ref.shape[:-2]
            and t.shape[-2] > 0
            and ref.shape[-2] % t.shape[-2] == 0
        ):
            return t.repeat_interleave(ref.shape[-2] // t.shape[-2], dim=-2)
    while t.ndim < ref.ndim:
        t = t.unsqueeze(0)
    if t.shape == ref.shape:
        return t
    if t.ndim >= 2 and t.transpose(-1, -2).shape == ref.shape:
        return t.transpose(-1, -2)
    if t.numel() == ref.numel():
        return t.reshape_as(ref)
    return t.expand_as(ref)


def select_inducing_points(
    X: Tensor,
    num_inducing_points: int,
    inducing_points: Optional[Tensor] = None,
) -> Tensor:
    """inducing points を選ぶ。指定があればそれを使い、なければ training X から選ぶ。"""
    if inducing_points is not None:
        return torch.as_tensor(inducing_points, device=X.device, dtype=X.dtype).contiguous()
    n = X.shape[-2]
    m = min(int(num_inducing_points), int(n))
    perm = torch.randperm(n, device=X.device)[:m]
    return X[perm].clone().contiguous()


def build_default_gamma_covar_module(
    train_X: Tensor,
    *,
    ard_num_dims: Optional[int] = None,
    nu: float = 2.5,
) -> ScaleKernel:
    """Gamma latent GP 用のデフォルト Matern kernel を作る。"""
    if ard_num_dims is None:
        ard_num_dims = train_X.shape[-1]
    return ScaleKernel(
        MaternKernel(
            nu=float(nu),
            ard_num_dims=int(ard_num_dims),
        )
    ).to(train_X)


class GammaLogLikelihood(_OneDimensionalLikelihood):
    """
    latent GP f から Gamma mean μ を作る likelihood。

    Gamma 分布は mean / concentration parameterization で扱います。

        concentration = κ
        rate = κ / μ
        mean = μ
        variance = μ^2 / κ

    Args:
        link:
            `"softplus"` または `"exp"`。
        init_concentration:
            concentration κ の初期値。
        learn_concentration:
            concentration を学習するかどうか。
        exp_clip:
            `link="exp"` のときの clipping 上限。
        min_mean:
            mean μ の下限。
        min_concentration:
            concentration κ の下限。
    """

    def __init__(
        self,
        link: GammaLink = "softplus",
        init_concentration: float = 10.0,
        learn_concentration: bool = True,
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
        min_concentration: float = 1e-6,
    ) -> None:
        super().__init__()
        self.link = str(link)
        self.exp_clip = float(exp_clip)
        self.min_mean = float(min_mean)
        self.min_concentration = float(min_concentration)

        init = torch.as_tensor(float(init_concentration)).clamp_min(self.min_concentration)
        # softplus(raw) ~= init となるように逆変換する。
        raw = torch.log(torch.expm1(init))
        if learn_concentration:
            self.register_parameter("raw_concentration", torch.nn.Parameter(raw.clone()))
        else:
            self.register_buffer("raw_concentration", raw.clone())

    @property
    def concentration(self) -> Tensor:
        """Gamma concentration κ を返す。"""
        return positive_concentration_from_raw(
            self.raw_concentration,
            min_concentration=self.min_concentration,
        )

    def mean_from_f(self, f: Tensor) -> Tensor:
        """latent f から Gamma mean μ を返す。"""
        return positive_mean_from_latent(
            f,
            link=self.link,  # type: ignore[arg-type]
            exp_clip=self.exp_clip,
            min_mean=self.min_mean,
        )

    def gamma_params_from_f(self, f: Tensor) -> tuple[Tensor, Tensor]:
        """latent f から Gamma(concentration, rate) の parameter を返す。"""
        mean = self.mean_from_f(f)
        concentration = self.concentration.to(device=f.device, dtype=f.dtype)
        rate = concentration / mean.clamp_min(self.min_mean)
        return concentration.expand_as(mean), rate

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> TorchGamma:
        concentration, rate = self.gamma_params_from_f(function_samples)
        return TorchGamma(concentration=concentration, rate=rate)

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any,
    ) -> Tensor:
        y = observations
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        y = y.to(device=function_dist.mean.device, dtype=function_dist.mean.dtype).clamp_min(self.min_mean)

        def log_prob_lambda(f: Tensor) -> Tensor:
            concentration, rate = self.gamma_params_from_f(f)
            return TorchGamma(concentration=concentration, rate=rate).log_prob(y)

        return self.quadrature(log_prob_lambda, function_dist)

    def log_marginal(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any,
    ) -> Tensor:
        return self.expected_log_prob(observations, function_dist, *params, **kwargs)


class GammaPosterior(Posterior):
    """
    Gamma mean / observation distribution の簡易 posterior。

    Notes:
        `mean` は Gamma mean μ の平均近似です。
        `variance` は観測 Gamma 分散 μ^2/κ と latent uncertainty の近似和です。
        `rsample()` は differentiable な mean sample を返します。
        観測値 sample が必要な場合は `sample_observations()` を使います。
    """

    def __init__(
        self,
        latent_posterior: GPyTorchPosterior,
        likelihood: GammaLogLikelihood,
        *,
        add_observation_noise: bool = True,
    ) -> None:
        super().__init__()
        self.latent_posterior = latent_posterior
        self.likelihood = likelihood
        self.add_observation_noise = bool(add_observation_noise)

    @property
    def device(self) -> torch.device:
        return self.latent_posterior.mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.latent_posterior.mean.dtype

    @property
    def event_shape(self) -> torch.Size:
        return self.mean.shape[-2:]

    @property
    def base_sample_shape(self) -> torch.Size:
        return self.latent_posterior.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return self.latent_posterior.batch_range

    @property
    def mean(self) -> Tensor:
        f_mean = self.latent_posterior.mean
        mean = self.likelihood.mean_from_f(f_mean)
        if mean.ndim == f_mean.ndim - 1:
            mean = mean.unsqueeze(-1)
        return mean

    @property
    def variance(self) -> Tensor:
        mean = self.mean
        latent_var = align_like(self.latent_posterior.variance, mean).clamp_min(0.0)
        if self.add_observation_noise:
            concentration = self.likelihood.concentration.to(device=mean.device, dtype=mean.dtype)
            obs_var = mean.pow(2) / concentration.clamp_min(self.likelihood.min_concentration)
            return obs_var + latent_var
        return latent_var

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        f_samples = self.latent_posterior.rsample(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )
        return self.likelihood.mean_from_f(f_samples)

    def sample_observations(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        """Gamma observation sample を返す。非 reparameterized sample。"""
        if sample_shape is None:
            sample_shape = torch.Size()
        mean_samples = self.rsample(sample_shape=sample_shape)
        concentration = self.likelihood.concentration.to(device=mean_samples.device, dtype=mean_samples.dtype)
        rate = concentration / mean_samples.clamp_min(self.likelihood.min_mean)
        return TorchGamma(concentration=concentration.expand_as(mean_samples), rate=rate).sample()


__all__ = [
    "GammaLink",
    "GammaLogLikelihood",
    "GammaPosterior",
    "align_like",
    "apply_input_transform_for_eval",
    "apply_input_transform_for_training",
    "build_default_gamma_covar_module",
    "check_categorical_columns_unchanged",
    "clone_input_transform",
    "ensure_2d_col",
    "expand_raw_X_to_match_transformed_q",
    "extract_normalize_only_transform",
    "get_cont_dims",
    "normalize_dims",
    "positive_concentration_from_raw",
    "positive_mean_from_latent",
    "prepare_positive_targets",
    "select_inducing_points",
    "to_device_dtype_transform",
]
