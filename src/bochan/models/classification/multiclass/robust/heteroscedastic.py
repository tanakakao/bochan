from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.posteriors import Posterior
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.models.components.multiclass import (
    MulticlassProbsPosterior,
    extract_normalize_only_transform,
    prepare_class_targets,
)
from bochan.models.classification.multiclass import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
)


def _align_like(t: Tensor, ref: Tensor) -> Tensor:
    """t を ref と同じ shape にできる範囲で揃える。"""
    if t.shape == ref.shape:
        return t
    while t.ndim < ref.ndim:
        t = t.unsqueeze(0)
    if t.shape == ref.shape:
        return t
    if t.numel() == ref.numel():
        return t.reshape_as(ref)
    return t.expand_as(ref)


class HeteroscedasticMulticlassPosterior(Posterior):
    """多クラス probability posterior に class-wise extra variance を加える wrapper。"""

    def __init__(
        self,
        base_posterior: MulticlassProbsPosterior,
        extra_noise_var: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.base_posterior = base_posterior
        self.extra_noise_var = extra_noise_var

    @property
    def device(self) -> torch.device:
        return self.base_posterior.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base_posterior.dtype

    @property
    def event_shape(self) -> torch.Size:
        return self.base_posterior.event_shape

    @property
    def base_sample_shape(self) -> torch.Size:
        return self.base_posterior.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return self.base_posterior.batch_range

    @property
    def mean(self) -> Tensor:
        return self.base_posterior.mean

    @property
    def variance(self) -> Tensor:
        var = self.base_posterior.variance
        if self.extra_noise_var is None:
            return var
        return var + _align_like(self.extra_noise_var, var)

    def rsample(self, sample_shape: Optional[torch.Size] = None, base_samples: Optional[Tensor] = None) -> Tensor:
        return self.base_posterior.rsample(sample_shape=sample_shape, base_samples=base_samples)

    def class_probs(self) -> Tensor:
        return self.mean

    def predict_class(self) -> Tensor:
        return self.mean.argmax(dim=-1)


def _fit_variational_multiclass_mll(
    model: MulticlassClassificationGPModel | MulticlassClassificationMixedGPModel,
    *,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> None:
    """補助多クラス分類モデルを簡易 training loop で fit する。"""
    mll = model.make_mll()
    mll.train()
    x_tensor = model.model.train_inputs[0]
    y_tensor = model.train_targets
    if batch_size is None:
        batch_size = x_tensor.shape[-2]
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)
    optimizer = torch.optim.Adam(mll.parameters(), lr=float(lr))
    for _ in range(int(num_epochs)):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            output = mll.model(xb)
            loss = -mll(output, yb).mean()
            loss.backward()
            optimizer.step()
    mll.eval()
    model.eval()
    model.likelihood.eval()


def _one_hot_targets(y: Tensor, num_classes: int, ref: Tensor) -> Tensor:
    """target を one-hot [n, C] に変換する。"""
    y = y.long()
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    y = y.reshape(-1)
    return torch.nn.functional.one_hot(y, num_classes=int(num_classes)).to(device=ref.device, dtype=ref.dtype)


def _prepare_noise_targets_for_gp(noise_targets: Tensor, train_X: Tensor, *, num_classes: Optional[int] = None) -> Tensor:
    """
    ノイズ target を SingleTaskGP / MixedSingleTaskGP 用の ``[n, m]`` に整形する。

    multiclass では class-wise noise target が ``[n, C]`` であるべきだが、
    posterior の batch shape により ``[1, n, C]`` や ``[n, C, 1]`` が混ざることがある。
    ここで補助 GP に渡す直前の形に正規化する。
    """
    n = int(train_X.shape[-2])
    y = torch.as_tensor(noise_targets, device=train_X.device, dtype=train_X.dtype)

    # 余分な singleton dimension を削除する。ただし [n, 1] は単出力として維持する。
    while y.ndim > 2 and y.shape[0] == 1:
        y = y.squeeze(0)
    while y.ndim > 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)

    if y.ndim == 1:
        if y.shape[0] != n:
            raise ValueError(
                "noise_targets length must match train_X n. "
                f"Got noise_targets.shape={tuple(y.shape)}, train_X.shape={tuple(train_X.shape)}."
            )
        y = y.unsqueeze(-1)
    elif y.ndim == 2:
        if y.shape[0] == n:
            pass
        elif y.shape[-1] == n:
            y = y.transpose(-1, -2)
        else:
            raise ValueError(
                "noise_targets must have shape [n], [n, m], or [m, n]. "
                f"Got noise_targets.shape={tuple(y.shape)}, train_X.shape={tuple(train_X.shape)}."
            )
    else:
        raise ValueError(
            "noise_targets could not be converted to [n, m]. "
            f"Got noise_targets.shape={tuple(y.shape)}, train_X.shape={tuple(train_X.shape)}."
        )

    if num_classes is not None and y.shape[-1] not in {1, int(num_classes)}:
        raise ValueError(
            "class-wise noise target output dimension mismatch. "
            f"Expected 1 or num_classes={int(num_classes)}, got {y.shape[-1]}."
        )

    return y.contiguous().clamp_min(1e-12)


def _estimate_multiclass_noise_targets(
    model: MulticlassClassificationGPModel | MulticlassClassificationMixedGPModel,
    train_X: Tensor,
    train_Y: Tensor,
    *,
    num_classes: int,
    min_noise: float = 1e-6,
) -> Tensor:
    """one-hot target と class probability の残差二乗から class-wise noise target を作る。"""
    with torch.no_grad():
        probs = model.class_probs(train_X)
        probs = _prepare_noise_targets_for_gp(probs, train_X, num_classes=num_classes)
        y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        y_oh = _one_hot_targets(y, num_classes=num_classes, ref=probs)
        return (y_oh - probs).pow(2).clamp_min(float(min_noise))


def _fit_noise_model_single(
    train_X: Tensor,
    noise_targets: Tensor,
    input_transform: Optional[InputTransform],
    *,
    num_classes: Optional[int] = None,
) -> SingleTaskGP:
    noise_targets = _prepare_noise_targets_for_gp(noise_targets, train_X, num_classes=num_classes)
    model = SingleTaskGP(train_X=train_X, train_Y=noise_targets.log(), input_transform=input_transform)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


def _fit_noise_model_mixed(
    train_X: Tensor,
    noise_targets: Tensor,
    cat_dims: Sequence[int],
    input_transform: Optional[InputTransform],
    *,
    num_classes: Optional[int] = None,
) -> MixedSingleTaskGP:
    noise_targets = _prepare_noise_targets_for_gp(noise_targets, train_X, num_classes=num_classes)
    model = MixedSingleTaskGP(
        train_X=train_X,
        train_Y=noise_targets.log(),
        cat_dims=list(cat_dims),
        input_transform=input_transform,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


class _HeteroscedasticMulticlassMixin:
    """多クラス heteroscedastic model 用 mixin。"""

    def predict_noise_logvar(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        logvar = self.noise_model.posterior(X).mean
        if ref_like is not None:
            logvar = _align_like(logvar, ref_like)
        return logvar

    def predict_noise_var(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        return self.predict_noise_logvar(X, ref_like=ref_like).exp().clamp_min(1e-12)

    def predict_noise_std(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        return self.predict_noise_var(X, ref_like=ref_like).sqrt()

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = True,
        posterior_transform=None,
        **kwargs: Any,
    ) -> HeteroscedasticMulticlassPosterior:
        if torch.is_tensor(observation_noise):
            extra_noise = observation_noise
            base_post = super().posterior(X, output_indices=output_indices, posterior_transform=None, **kwargs)
        else:
            base_post = super().posterior(X, output_indices=output_indices, posterior_transform=None, **kwargs)
            extra_noise = self.predict_noise_var(X, ref_like=base_post.mean) if observation_noise else None
        posterior = HeteroscedasticMulticlassPosterior(base_posterior=base_post, extra_noise_var=extra_noise)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


class HeteroscedasticMulticlassClassificationGPModel(_HeteroscedasticMulticlassMixin, MulticlassClassificationGPModel):
    """class-wise extra variance GP を持つ多クラス分類モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        temperature: float = 1.0,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 300,
        aux_batch_size: Optional[int] = None,
        aux_shuffle: bool = True,
        min_noise: float = 1e-6,
        train_Yvar: Optional[Tensor] = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        if num_classes is None:
            y_tmp = train_Y.squeeze(-1) if train_Y.ndim > 1 and train_Y.shape[-1] == 1 else train_Y
            num_classes = int(torch.as_tensor(y_tmp).max().item()) + 1
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        noise_tf = extract_normalize_only_transform(input_transform)
        if train_Yvar is None:
            aux_model = MulticlassClassificationGPModel(
                train_X=train_X,
                train_Y=train_Y,
                num_classes=num_classes,
                input_transform=noise_tf,
                num_inducing_points=num_inducing_points,
                temperature=temperature,
            )
            _fit_variational_multiclass_mll(
                aux_model,
                lr=aux_lr,
                num_epochs=aux_num_epochs,
                batch_size=aux_batch_size,
                shuffle=aux_shuffle,
            )
            noise_targets = _estimate_multiclass_noise_targets(
                aux_model,
                train_X,
                train_Y,
                num_classes=num_classes,
                min_noise=min_noise,
            )
        else:
            noise_targets = torch.as_tensor(train_Yvar, device=train_X.device, dtype=train_X.dtype).clamp_min(float(min_noise))
        self.noise_model = _fit_noise_model_single(train_X, noise_targets, noise_tf, num_classes=num_classes)
        self.noise_input_transform = noise_tf
        self.min_noise = float(min_noise)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            input_transform=input_transform,
            num_inducing_points=num_inducing_points,
            temperature=temperature,
        )


class HeteroscedasticMulticlassClassificationMixedGPModel(_HeteroscedasticMulticlassMixin, MulticlassClassificationMixedGPModel):
    """mixed 入力版の class-wise heteroscedastic 多クラス分類モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        temperature: float = 1.0,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 300,
        aux_batch_size: Optional[int] = None,
        aux_shuffle: bool = True,
        min_noise: float = 1e-6,
        train_Yvar: Optional[Tensor] = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        if num_classes is None:
            y_tmp = train_Y.squeeze(-1) if train_Y.ndim > 1 and train_Y.shape[-1] == 1 else train_Y
            num_classes = int(torch.as_tensor(y_tmp).max().item()) + 1
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        noise_tf = extract_normalize_only_transform(input_transform)
        if train_Yvar is None:
            aux_model = MulticlassClassificationMixedGPModel(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=cat_dims,
                num_classes=num_classes,
                input_transform=noise_tf,
                num_inducing_points=num_inducing_points,
                temperature=temperature,
            )
            _fit_variational_multiclass_mll(
                aux_model,
                lr=aux_lr,
                num_epochs=aux_num_epochs,
                batch_size=aux_batch_size,
                shuffle=aux_shuffle,
            )
            noise_targets = _estimate_multiclass_noise_targets(
                aux_model,
                train_X,
                train_Y,
                num_classes=num_classes,
                min_noise=min_noise,
            )
        else:
            noise_targets = torch.as_tensor(train_Yvar, device=train_X.device, dtype=train_X.dtype).clamp_min(float(min_noise))
        self.noise_model = _fit_noise_model_mixed(train_X, noise_targets, cat_dims, noise_tf, num_classes=num_classes)
        self.noise_input_transform = noise_tf
        self.min_noise = float(min_noise)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            num_classes=num_classes,
            input_transform=input_transform,
            num_inducing_points=num_inducing_points,
            temperature=temperature,
        )


__all__ = [
    "HeteroscedasticMulticlassPosterior",
    "HeteroscedasticMulticlassClassificationGPModel",
    "HeteroscedasticMulticlassClassificationMixedGPModel",
]
