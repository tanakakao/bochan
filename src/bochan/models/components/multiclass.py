from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
from torch import Tensor

from botorch.models.transforms.input import ChainedInputTransform, InputTransform, Normalize
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.kernels import MaternKernel, ScaleKernel


def clone_input_transform(input_transform: Optional[InputTransform]) -> Optional[InputTransform]:
    """input_transform を安全に複製する。"""
    return None if input_transform is None else copy.deepcopy(input_transform)


def to_device_dtype_transform(input_transform: Optional[InputTransform], ref: Tensor) -> Optional[InputTransform]:
    """InputTransform を ref と同じ device / dtype に移す。"""
    if input_transform is None:
        return None
    if hasattr(input_transform, 'to'):
        input_transform = input_transform.to(device=ref.device, dtype=ref.dtype)
    return input_transform


def prepare_class_targets(train_Y: Tensor, ref: Tensor, *, num_classes: Optional[int] = None) -> Tensor:
    """多クラス分類用 target を LongTensor [n] に整形する。"""
    y = torch.as_tensor(train_Y, device=ref.device)
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    y = y.long().contiguous()
    if (y < 0).any():
        raise ValueError('Multiclass targets must be non-negative integer labels.')
    if num_classes is not None and (y >= int(num_classes)).any():
        raise ValueError(f'Targets must be smaller than num_classes={num_classes}.')
    return y


def infer_num_classes(train_Y: Tensor, num_classes: Optional[int] = None) -> int:
    """target からクラス数を推定する。"""
    if num_classes is not None:
        return int(num_classes)
    y = train_Y.squeeze(-1) if train_Y.ndim > 1 and train_Y.shape[-1] == 1 else train_Y
    return int(y.max().item()) + 1


def normalize_dims(dims: Sequence[int], d: int) -> list[int]:
    """負の index を許容して feature dimension index を正規化する。"""
    out = []
    for idx in dims:
        j = int(idx)
        if j < 0:
            j = d + j
        if j < 0 or j >= d:
            raise ValueError(f'dim index {idx} is out of range for d={d}.')
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
        q, q_like = X.shape[-2], X_tf.shape[-2]
        if q_like == q:
            return X
        if q > 0 and q_like % q == 0:
            return X.repeat_interleave(q_like // q, dim=-2)
    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)
    return X


def check_categorical_columns_unchanged(X: Tensor, X_tf: Tensor, cat_dims: Optional[Sequence[int]]) -> None:
    """mixed model で input_transform がカテゴリ列を変えていないか確認する。"""
    if cat_dims is None or len(cat_dims) == 0:
        return
    cat_idx = [int(i) for i in cat_dims]
    if X.shape[-1] != X_tf.shape[-1]:
        raise ValueError('For mixed multiclass models, input_transform must preserve feature dimension.')
    X_cmp = expand_raw_X_to_match_transformed_q(X, X_tf)
    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(f'Could not align raw X with transformed X: {X.shape} vs {X_tf.shape}.')
    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(f'input_transform must not modify categorical columns: cat_dims={cat_idx}.')


def apply_input_transform_for_training(X: Tensor, input_transform: Optional[InputTransform], *, cat_dims: Optional[Sequence[int]] = None, name: str = 'input_transform') -> Tensor:
    """学習用 X に input_transform を適用する。InputPerturbation による点数増加は許さない。"""
    if input_transform is None:
        return X
    was_training = getattr(input_transform, 'training', False)
    if hasattr(input_transform, 'train'):
        input_transform.train()
    X_tf = input_transform(X)
    if not was_training and hasattr(input_transform, 'eval'):
        input_transform.eval()
    if X_tf.shape[-2] != X.shape[-2]:
        raise RuntimeError(f'{name} expanded training inputs. Use transform_on_train=False during fitting.')
    check_categorical_columns_unchanged(X, X_tf, cat_dims)
    return X_tf


def apply_input_transform_for_eval(X: Tensor, input_transform: Optional[InputTransform], *, cat_dims: Optional[Sequence[int]] = None) -> Tensor:
    """posterior / acquisition 評価用 transform。InputPerturbation による q 展開を許す。"""
    if input_transform is None:
        return X
    X_tf = input_transform(X)
    check_categorical_columns_unchanged(X, X_tf, cat_dims)
    return X_tf


def extract_normalize_only_transform(input_transform: Optional[InputTransform]) -> Optional[InputTransform]:
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


def select_inducing_points(X: Tensor, num_inducing_points: int, inducing_points: Optional[Tensor] = None, *, num_classes: Optional[int] = None) -> Tensor:
    """inducing points を選ぶ。num_classes 指定時は [C, m, d] に展開する。"""
    if inducing_points is not None:
        Z = torch.as_tensor(inducing_points, device=X.device, dtype=X.dtype).contiguous()
    else:
        n = X.shape[-2]
        m = min(int(num_inducing_points), int(n))
        perm = torch.randperm(n, device=X.device)[:m]
        Z = X[perm].clone().contiguous()
    if num_classes is not None and Z.ndim == 2:
        Z = Z.unsqueeze(0).expand(int(num_classes), *Z.shape).contiguous()
    return Z


def build_default_multiclass_covar_module(train_X: Tensor, *, num_classes: int, ard_num_dims: Optional[int] = None, nu: float = 2.5) -> ScaleKernel:
    """多クラス latent GP 用の batch Matern kernel を作る。"""
    if ard_num_dims is None:
        ard_num_dims = train_X.shape[-1]
    batch_shape = torch.Size([int(num_classes)])
    return ScaleKernel(
        MaternKernel(nu=float(nu), ard_num_dims=int(ard_num_dims), batch_shape=batch_shape),
        batch_shape=batch_shape,
    ).to(train_X)


def move_class_dim_to_last(t: Tensor, *, num_classes: int) -> Tensor:
    """class batch dimension を最後に移動する。"""
    c = int(num_classes)
    if t.shape[-1] == c:
        return t
    if t.shape[0] == c:
        return t.movedim(0, -1)
    if t.ndim >= 2 and t.shape[-2] == c:
        return t.movedim(-2, -1)
    for dim, size in enumerate(t.shape):
        if size == c:
            return t.movedim(dim, -1)
    raise RuntimeError(f'Could not find class dimension of size {c} in tensor shape {tuple(t.shape)}.')


class MulticlassProbsPosterior(Posterior):
    """多クラス分類の probability posterior。`mean` は class probability。"""

    def __init__(self, latent_posterior: GPyTorchPosterior, *, num_classes: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.latent_posterior = latent_posterior
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)

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
    def logits(self) -> Tensor:
        logits = move_class_dim_to_last(self.latent_posterior.mean, num_classes=self.num_classes)
        if logits.shape[-1] == 1 and logits.ndim >= 3 and logits.shape[-2] == self.num_classes:
            logits = logits.squeeze(-1).movedim(-1, -2)
        return logits / self.temperature

    @property
    def mean(self) -> Tensor:
        return torch.softmax(self.logits, dim=-1)

    @property
    def variance(self) -> Tensor:
        p = self.mean
        return p * (1.0 - p)

    def rsample(self, sample_shape: Optional[torch.Size] = None, base_samples: Optional[Tensor] = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        latent_samples = self.latent_posterior.rsample(sample_shape=sample_shape, base_samples=base_samples)
        logits = move_class_dim_to_last(latent_samples, num_classes=self.num_classes)
        return torch.softmax(logits / self.temperature, dim=-1)

    def class_probs(self) -> Tensor:
        return self.mean

    def predict_class(self) -> Tensor:
        return self.mean.argmax(dim=-1)


__all__ = [
    'MulticlassProbsPosterior',
    'apply_input_transform_for_eval',
    'apply_input_transform_for_training',
    'build_default_multiclass_covar_module',
    'check_categorical_columns_unchanged',
    'clone_input_transform',
    'extract_normalize_only_transform',
    'get_cont_dims',
    'infer_num_classes',
    'move_class_dim_to_last',
    'normalize_dims',
    'prepare_class_targets',
    'select_inducing_points',
    'to_device_dtype_transform',
]
