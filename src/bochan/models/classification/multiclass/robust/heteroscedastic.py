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

from bochan.models.classification.multiclass._components import (
    MulticlassProbsPosterior,
    extract_normalize_only_transform,
    prepare_class_targets,
)
from bochan.models.classification.multiclass import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
)


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out

def _expand_q_like_to_ref(t: Tensor, ref: Tensor) -> Tensor | None:
    """Expand raw q-like noise to an InputPerturbation-expanded reference.

    Examples:
        ``[B, q, C]``     -> ``[B, q * n_w, 1, C]``
        ``[B, q, 1, C]``  -> ``[B, q * n_w, 1, C]``
        ``[1, B, q, C]``  -> ``[B, q * n_w, 1, C]``
    """
    if ref.ndim < 4 or t.shape[-1] != ref.shape[-1]:
        return None

    b = int(ref.shape[0])
    q_ref = int(ref.shape[1])
    c = int(ref.shape[-1])

    work = t
    while work.ndim > 0 and work.shape[0] == 1 and work.ndim > 3:
        work = work.squeeze(0)

    # [B, q, C]
    if work.ndim == 3 and work.shape[0] == b and work.shape[-1] == c:
        q_like = int(work.shape[1])
        if q_like == q_ref:
            return work.unsqueeze(-2).expand_as(ref)
        if q_like > 0 and q_ref % q_like == 0:
            return work.repeat_interleave(q_ref // q_like, dim=1).unsqueeze(-2).expand_as(ref)
        if q_like > q_ref and q_like % q_ref == 0:
            reduced = work.reshape(b, q_ref, q_like // q_ref, c).mean(dim=2)
            return reduced.unsqueeze(-2).expand_as(ref)

    # [B, q, 1, C]
    if work.ndim == 4 and work.shape[0] == b and work.shape[-1] == c and work.shape[-2] == 1:
        q_like = int(work.shape[1])
        if q_like == q_ref:
            return work.expand_as(ref)
        if q_like > 0 and q_ref % q_like == 0:
            return work.repeat_interleave(q_ref // q_like, dim=1).expand_as(ref)
        if q_like > q_ref and q_like % q_ref == 0:
            reduced = work.reshape(b, q_ref, q_like // q_ref, 1, c).mean(dim=2)
            return reduced.expand_as(ref)

    return None

def _align_like(t: Tensor, ref: Tensor) -> Tensor:
    """Align a noise tensor to a reference probability tensor.

    Heteroscedastic noise models are often evaluated on the raw candidate batch,
    while the base multiclass posterior may be evaluated after one-to-many input
    transforms such as InputPerturbation. Common shape pairs are:

    - noise logvar: ``[1, B, 1, C]``
      ref_like:     ``[B, n_w, 1, C]``
    - noise logvar: ``[B, q, C]`` or ``[B, q, 1, C]``
      ref_like:     ``[B, q * n_w, 1, C]``
    """
    t = torch.as_tensor(t, device=ref.device, dtype=ref.dtype)

    if t.shape == ref.shape:
        return t

    # Exact element count: reshape is unambiguous.
    if t.numel() == ref.numel():
        return t.reshape_as(ref)

    # Handle q-like raw candidate axis before generic broadcasting, because
    # [B, q, C] must become [B, q * n_w, 1, C], not [1, B, q, C].
    expanded_q = _expand_q_like_to_ref(t, ref)
    if expanded_q is not None:
        return expanded_q

    # Standard broadcasting may already work.
    try:
        return t.expand_as(ref)
    except RuntimeError:
        pass

    # Remove leading singleton axes, then try suffix broadcast.
    t_work = t
    while t_work.ndim > 0 and t_work.shape[0] == 1 and t_work.ndim >= ref.ndim:
        t_work = t_work.squeeze(0)
    if t_work.ndim <= ref.ndim:
        # Try q-like expansion again after dropping leading singleton axes.
        expanded_q = _expand_q_like_to_ref(t_work, ref)
        if expanded_q is not None:
            return expanded_q
        view_shape = (1,) * (ref.ndim - t_work.ndim) + tuple(t_work.shape)
        try:
            return t_work.reshape(view_shape).expand_as(ref)
        except RuntimeError:
            pass

    # InputPerturbation pattern:
    #   t   = [1, B, 1, C] or [B, 1, C]
    #   ref = [B, W, 1, C]
    if ref.ndim >= 4 and t.shape[-1] == ref.shape[-1]:
        b = int(ref.shape[0])
        c = int(ref.shape[-1])

        # [1, B, 1, C] -> [B, 1, 1, C] -> [B, W, 1, C]
        if t.ndim == ref.ndim and t.shape[0] == 1 and t.shape[1] == b and t.shape[-1] == c:
            moved = t.squeeze(0)
            while moved.ndim < ref.ndim - 1:
                moved = moved.unsqueeze(-2)
            moved = moved.reshape(b, *([1] * (ref.ndim - 2)), c)
            return moved.expand_as(ref)

        # [B, 1, C] -> [B, 1, 1, C] -> [B, W, 1, C].
        # If the middle dimension is not 1, it is q-like and should have been
        # handled by _expand_q_like_to_ref above.
        if t.ndim == ref.ndim - 1 and t.shape[0] == b and t.shape[-1] == c and t.shape[1] == 1:
            moved = t
            while moved.ndim < ref.ndim:
                moved = moved.unsqueeze(-2)
            return moved.expand_as(ref)

        # [B, C] -> [B, 1, 1, C] -> [B, W, 1, C]
        if t.ndim == 2 and t.shape[0] == b and t.shape[-1] == c:
            moved = t.reshape(b, *([1] * (ref.ndim - 2)), c)
            return moved.expand_as(ref)

    # If t has extra axes, average them until a broadcastable representation is found.
    t_work = t
    while t_work.ndim > 0:
        if t_work.ndim <= ref.ndim:
            expanded_q = _expand_q_like_to_ref(t_work, ref)
            if expanded_q is not None:
                return expanded_q
            view_shape = (1,) * (ref.ndim - t_work.ndim) + tuple(t_work.shape)
            try:
                return t_work.reshape(view_shape).expand_as(ref)
            except RuntimeError:
                pass
        # Prefer reducing singleton / sample-like leading axes first.
        if t_work.shape[0] == 1:
            t_work = t_work.squeeze(0)
        else:
            t_work = t_work.mean(dim=0)

    if t.numel() == 1:
        return t.reshape(()).expand_as(ref)

    raise RuntimeError(
        "Could not align heteroscedastic noise tensor to reference. "
        f"t.shape={tuple(t.shape)}, ref.shape={tuple(ref.shape)}."
    )


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


def _squeeze_extra_singletons(y: Tensor, *, n: int) -> Tensor:
    """
    補助 GP の target に不要な singleton batch 次元を削除する。

    ``[n, 1]`` は単出力 target として残し、3D 以上に現れる singleton のみ削除する。
    例: ``[1, n, C] -> [n, C]``, ``[n, 1, C] -> [n, C]``, ``[n, C, 1] -> [n, C]``。
    """
    _ = n
    while y.ndim > 2:
        squeezed = False
        for dim, size in enumerate(y.shape):
            if size == 1:
                y = y.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            break
    return y


def _prepare_noise_targets_for_gp(noise_targets: Tensor, train_X: Tensor, *, num_classes: Optional[int] = None) -> Tensor:
    """
    ノイズ target を SingleTaskGP / MixedSingleTaskGP 用の ``[n, m]`` に整形する。

    multiclass では class-wise noise target が ``[n, C]`` であるべきだが、
    posterior の batch shape により ``[1, n, C]``, ``[n, 1, C]``, ``[n, C, 1]``
    が混ざることがある。ここで補助 GP に渡す直前の形に正規化する。
    """
    n = int(train_X.shape[-2])
    y = torch.as_tensor(noise_targets, device=train_X.device, dtype=train_X.dtype)
    y = _squeeze_extra_singletons(y, n=n)

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
        # 最後の保険: 要素数が n で割り切れるなら [n, m] に落とす。
        if y.numel() % n == 0:
            y = y.reshape(n, y.numel() // n)
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
        num_inducing: int = 128,
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
                num_inducing=num_inducing,
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
        noise_model = _fit_noise_model_single(train_X, noise_targets, noise_tf, num_classes=num_classes)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            input_transform=input_transform,
            num_inducing=num_inducing,
            temperature=temperature,
        )
        self.noise_model = noise_model
        self.noise_input_transform = noise_tf
        self.min_noise = float(min_noise)


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
        num_inducing: int = 128,
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
                num_inducing=num_inducing,
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
        noise_model = _fit_noise_model_mixed(train_X, noise_targets, cat_dims, noise_tf, num_classes=num_classes)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            num_classes=num_classes,
            input_transform=input_transform,
            num_inducing=num_inducing,
            temperature=temperature,
        )
        self.noise_model = noise_model
        self.noise_input_transform = noise_tf
        self.min_noise = float(min_noise)


__all__ = [
    "HeteroscedasticMulticlassPosterior",
    "HeteroscedasticMulticlassClassificationGPModel",
    "HeteroscedasticMulticlassClassificationMixedGPModel",
]
