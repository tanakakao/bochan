
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any

import torch
from torch import Tensor

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.models.transforms.input import ChainedInputTransform, InputTransform, Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from linear_operator.operators import DiagLinearOperator


__all__ = [
    "HeteroscedasticAuxResult",
    "IdentityInputTransform",
    "HeteroscedasticNoiseModelMixin",
    "HeteroscedasticLatentPosteriorMixin",
    "clone_input_transform",
    "align_like",
    "ensure_2d_col",
    "prepare_train_yvar",
    "extract_normalize_only_transform",
    "make_normalize_only_transform",
    "check_categorical_columns_unchanged",
    "expand_observation_noise_tensor",
    "fit_exact_mll",
    "fit_variational_classifier_mll",
    "fit_noise_model_single",
    "fit_noise_model_mixed",
    "predict_noise_var_from_log_noise_model",
    "compute_regression_log_var_from_residuals",
    "prepare_conditioning_data",
    "concat_optional_train_yvar",
]


def clone_input_transform(
    input_transform: Optional[InputTransform],
) -> Optional[InputTransform]:
    """input_transform を安全に複製する。"""
    return None if input_transform is None else copy.deepcopy(input_transform)


def ensure_2d_col(y: Tensor) -> Tensor:
    """Tensor を [N, 1] 形状にそろえる。"""
    if y.ndim == 1:
        return y.unsqueeze(-1)
    return y


def align_like(t: Tensor, ref: Tensor) -> Tensor:
    """
    t を ref と同じ shape にできる範囲でそろえる。

    InputPerturbation により q -> q * n_w へ展開された場合は、
    q 次元を repeat_interleave して ref に合わせる。
    """
    if t.shape == ref.shape:
        return t

    if ref.ndim >= 1 and ref.shape[-1] == 1:
        ref_no_last = ref.squeeze(-1)

        if t.shape == ref_no_last.shape:
            return t.unsqueeze(-1)

        if t.ndim >= 1 and t.shape[-1] == 1 and t.squeeze(-1).shape == ref_no_last.shape:
            return t

        if (
            t.ndim == ref_no_last.ndim
            and t.shape[:-1] == ref_no_last.shape[:-1]
            and t.shape[-1] > 0
            and ref_no_last.shape[-1] % t.shape[-1] == 0
        ):
            n_w = ref_no_last.shape[-1] // t.shape[-1]
            return t.repeat_interleave(n_w, dim=-1).unsqueeze(-1)

        if (
            t.ndim == ref.ndim
            and t.shape[-1] == 1
            and t.shape[:-2] == ref.shape[:-2]
            and t.shape[-2] > 0
            and ref.shape[-2] % t.shape[-2] == 0
        ):
            n_w = ref.shape[-2] // t.shape[-2]
            return t.repeat_interleave(n_w, dim=-2)

    while t.dim() < ref.dim():
        t = t.unsqueeze(0)

    if t.shape == ref.shape:
        return t

    if t.ndim >= 2 and t.transpose(-1, -2).shape == ref.shape:
        return t.transpose(-1, -2)

    if t.numel() == ref.numel():
        return t.reshape_as(ref)

    try:
        return t.expand_as(ref)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot align tensor shape {tuple(t.shape)} to ref shape {tuple(ref.shape)}."
        ) from exc


def expand_raw_X_to_match_transformed_q(X: Tensor, X_tf: Tensor) -> Tensor:
    """InputPerturbation 後の q 展開に合わせて raw X を展開する。"""
    if X.shape == X_tf.shape:
        return X

    if X.ndim < 2 or X_tf.ndim < 2:
        return X

    if X.shape[-1] != X_tf.shape[-1]:
        return X

    if X.shape[:-2] == X_tf.shape[:-2]:
        q = X.shape[-2]
        q_like = X_tf.shape[-2]
        if q_like == q:
            return X
        if q > 0 and q_like % q == 0:
            n_w = q_like // q
            return X.repeat_interleave(n_w, dim=-2)

    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)

    return X


def check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
    *,
    name: str = "input_transform",
) -> None:
    """
    mixed model 用に input_transform がカテゴリ列を変更していないか確認する。

    InputPerturbation により q -> q*n_w へ展開される場合も考慮する。
    """
    if cat_dims is None or len(cat_dims) == 0:
        return

    if X.shape[-1] != X_tf.shape[-1]:
        raise ValueError(
            f"{name} must preserve feature dimension for mixed models. "
            f"raw dim={X.shape[-1]}, transformed dim={X_tf.shape[-1]}."
        )

    cat_idx = [int(i) for i in cat_dims]
    X_cmp = expand_raw_X_to_match_transformed_q(X, X_tf)

    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            f"Could not align raw X with transformed X in {name}. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}."
        )

    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(
            f"{name} must not modify categorical columns. "
            f"X_cat.shape={tuple(X_cmp[..., cat_idx].shape)}, "
            f"X_tf_cat.shape={tuple(X_tf[..., cat_idx].shape)}."
        )


def _normalize_indices(indices: Sequence[int], d: int) -> list[int]:
    """負の index を正規化し、重複を除いて昇順にする。"""
    out: list[int] = []
    for idx in indices:
        j = int(idx)
        if j < 0:
            j = d + j
        if not 0 <= j < d:
            raise IndexError(f"Index {idx} is out of bounds for dimension {d}.")
        out.append(j)
    return sorted(set(out))


def extract_normalize_only_transform(
    input_transform: Optional[InputTransform],
) -> Optional[InputTransform]:
    """
    ChainedInputTransform から Normalize のみを抽出する。

    Heteroscedastic noise model 学習では InputPerturbation などの
    q 展開 transform を使わず、Normalize だけを補助モデルに渡す。
    """
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


def make_normalize_only_transform(
    input_transform: Optional[InputTransform],
    train_X: Tensor,
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Optional[InputTransform]:
    """
    noise model 用の Normalize transform を作る。

    mixed model ではカテゴリ列を変換しないように、Normalize(indices=cont_dims)
    に変換できる場合は変換する。変換できない場合はカテゴリ列チェックで検出する。
    """
    tf = extract_normalize_only_transform(input_transform)
    if tf is None:
        return None

    tf = tf.to(train_X) if hasattr(tf, "to") else tf

    if cat_dims is None or len(cat_dims) == 0:
        return tf

    d = train_X.shape[-1]
    cat_dims = _normalize_indices(cat_dims, d=d)
    cont_dims = [i for i in range(d) if i not in set(cat_dims)]

    if not isinstance(tf, Normalize):
        return tf

    indices = getattr(tf, "indices", None)
    if indices is not None:
        # 既に indices がある場合はカテゴリ列を含んでいないかだけ確認する。
        idx_list = [int(i) for i in indices.view(-1).tolist()] if torch.is_tensor(indices) else [int(i) for i in indices]
        bad = sorted(set(idx_list).intersection(cat_dims))
        if bad:
            raise ValueError(
                f"Normalize indices must exclude categorical dims. Got {bad}."
            )
        return tf

    bounds = getattr(tf, "bounds", None)
    if isinstance(bounds, Tensor):
        if bounds.shape[-1] == d:
            return Normalize(
                d=d,
                bounds=bounds.to(train_X),
                indices=cont_dims,
            )
        if bounds.shape[-1] == len(cont_dims):
            return Normalize(
                d=d,
                bounds=bounds.to(train_X),
                indices=cont_dims,
            )

    # bounds が復元できない場合は、あとでカテゴリ列チェックに任せる。
    return tf


def prepare_train_yvar(
    train_Yvar: Optional[Tensor],
    ref_X: Tensor,
    min_noise: float,
) -> Optional[Tensor]:
    """train_Yvar を [N, 1] の正値 tensor に整形する。"""
    if train_Yvar is None:
        return None

    out = torch.as_tensor(train_Yvar, device=ref_X.device, dtype=ref_X.dtype)
    out = ensure_2d_col(out)

    if out.shape[-2] != ref_X.shape[-2]:
        raise ValueError(
            f"train_Yvar must have N={ref_X.shape[-2]}, got shape={tuple(out.shape)}."
        )

    return out.clamp_min(float(min_noise))


def fit_exact_mll(model) -> Any:
    """ExactMarginalLogLikelihood で GP を学習する。"""
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


def fit_variational_classifier_mll(
    model,
    *,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> None:
    """
    VariationalELBO で分類用 approximate GP を学習する簡易 helper。

    model は ApproximateGPyTorchModel 互換で、
    model.model / model.likelihood / model.train_targets を持つ想定。
    """
    train_inputs = model.model.train_inputs
    x_tensor = train_inputs[0] if isinstance(train_inputs, (tuple, list)) else train_inputs
    y_tensor = model.model.train_targets

    if y_tensor.ndim > 1 and y_tensor.shape[-1] == 1:
        y_tensor = y_tensor.squeeze(-1)

    y_tensor = y_tensor.to(dtype=x_tensor.dtype, device=x_tensor.device)

    if batch_size is None:
        batch_size = x_tensor.shape[-2]

    mll = VariationalELBO(
        likelihood=model.likelihood,
        model=model.model,
        num_data=y_tensor.numel(),
    )

    optimizer = torch.optim.Adam(mll.parameters(), lr=float(lr))
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle)

    model.train()
    model.likelihood.train()
    for _ in range(int(num_epochs)):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model.model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    model.likelihood.eval()


def fit_noise_model_single(
    train_X: Tensor,
    train_Y_log_var: Tensor,
    input_transform: Optional[InputTransform] = None,
) -> SingleTaskGP:
    """log variance を回帰する SingleTaskGP noise model を学習する。"""
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=ensure_2d_col(train_Y_log_var),
        input_transform=clone_input_transform(input_transform),
    )
    return fit_exact_mll(model)


def fit_noise_model_mixed(
    train_X: Tensor,
    train_Y_log_var: Tensor,
    cat_dims: Sequence[int],
    input_transform: Optional[InputTransform] = None,
) -> MixedSingleTaskGP:
    """log variance を回帰する MixedSingleTaskGP noise model を学習する。"""
    model = MixedSingleTaskGP(
        train_X=train_X,
        train_Y=ensure_2d_col(train_Y_log_var),
        cat_dims=list(cat_dims),
        input_transform=clone_input_transform(input_transform),
    )
    return fit_exact_mll(model)


def predict_noise_var_from_log_noise_model(
    noise_model,
    X: Tensor,
    *,
    ref_like: Optional[Tensor] = None,
    min_noise: float = 1e-12,
) -> Tensor:
    """noise_model.posterior(X).mean を log variance とみなして variance を返す。"""
    with torch.no_grad():
        logvar = noise_model.posterior(X).mean
        if ref_like is not None:
            logvar = align_like(logvar, ref_like)
        return logvar.exp().clamp_min(float(min_noise))


def compute_regression_log_var_from_residuals(
    base_model,
    train_X: Tensor,
    train_Y: Tensor,
    *,
    min_noise: float = 1e-6,
) -> Tensor:
    """回帰 base model の残差二乗から log variance target を作る。"""
    with torch.no_grad():
        mean = base_model.posterior(train_X, observation_noise=False).mean
        mean = align_like(mean, ensure_2d_col(train_Y.to(mean)))
        residual_sq = (mean - ensure_2d_col(train_Y.to(mean))).pow(2)
        return residual_sq.clamp_min(float(min_noise)).log()


def expand_observation_noise_tensor(
    observation_noise: Tensor,
    X: Tensor,
) -> Tensor:
    """user-provided observation_noise を X の point shape に合わせる。"""
    noise = torch.as_tensor(observation_noise, device=X.device, dtype=X.dtype)
    target_shape = X.shape[:-1] + (1,)

    if noise.ndim == 0:
        return noise.expand(target_shape)

    if noise.shape == X.shape[:-1]:
        return noise.unsqueeze(-1)

    if noise.shape == target_shape:
        return noise

    if noise.ndim >= 1 and noise.shape[-1] == 1:
        return noise

    raise ValueError(
        f"observation_noise must broadcast to {target_shape}, got {tuple(noise.shape)}."
    )


def prepare_conditioning_data(
    X: Tensor,
    Y: Tensor,
    noise: Optional[Tensor],
    *,
    expected_input_dim: int,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    condition_on_observations 用に raw-space X / Y / noise を flatten する。

    fantasy batch ではなく、通常の新規観測追加を想定する。
    """
    if isinstance(X, tuple):
        X = X[0]

    if X.ndim < 2:
        raise ValueError("X must have shape [q, d] or [batch, q, d].")

    if X.shape[-1] != expected_input_dim:
        raise ValueError(
            f"Expected raw input dim {expected_input_dim}, got {X.shape[-1]}."
        )

    expected_y_shape = X.shape[:-1]

    if Y.ndim == X.ndim and Y.shape[-1] == 1:
        Y = Y.squeeze(-1)

    if Y.shape != expected_y_shape:
        raise NotImplementedError(
            "Only non-fantasy observations with Y.shape == X.shape[:-1] "
            "or trailing singleton output dim are supported. "
            f"Got X.shape={tuple(X.shape)}, Y.shape={tuple(Y.shape)}."
        )

    if noise is not None:
        if noise.ndim == X.ndim and noise.shape[-1] == 1:
            noise = noise.squeeze(-1)
        if noise.shape != expected_y_shape:
            raise ValueError(
                "noise must match X.shape[:-1] or have trailing singleton output dim. "
                f"Got X.shape={tuple(X.shape)}, noise.shape={tuple(noise.shape)}."
            )

    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1).to(dtype=X.dtype, device=X.device)
    noise_flat = None if noise is None else noise.reshape(-1).to(dtype=X.dtype, device=X.device)

    return X_flat, Y_flat, noise_flat


def concat_optional_train_yvar(
    old_Y: Tensor,
    old_Yvar: Optional[Tensor],
    new_Y: Tensor,
    new_Yvar: Optional[Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Tensor]:
    """old/new の noise tensor を結合する。片方がない場合は 0 埋めする。"""
    if old_Yvar is None and new_Yvar is None:
        return None

    old = torch.zeros_like(old_Y, dtype=dtype, device=device) if old_Yvar is None else ensure_2d_col(old_Yvar).reshape(-1).to(dtype=dtype, device=device)
    new = torch.zeros_like(new_Y, dtype=dtype, device=device) if new_Yvar is None else ensure_2d_col(new_Yvar).reshape(-1).to(dtype=dtype, device=device)
    return torch.cat([old.reshape(-1), new.reshape(-1)], dim=0)


@dataclass
class HeteroscedasticAuxResult:
    """heteroscedastic 補助モデルの学習結果。"""

    noise_model: object
    predicted_noise_var: Tensor
    noise_input_transform: Optional[InputTransform]


class IdentityInputTransform(InputTransform):
    """何もしない input transform。"""

    def transform(self, X: Tensor) -> Tensor:
        return X

    def untransform(self, X: Tensor) -> Tensor:
        return X

    def preprocess_transform(self, X: Tensor) -> Tensor:
        return X


class HeteroscedasticNoiseModelMixin:
    """
    log variance noise_model を持つモデル用 mixin。

    noise_model は raw X を受け取り、内部で Normalize のみを適用する想定。
    """

    def _set_transformed_inputs(self) -> None:
        """BoTorch eval 時の自動 transformed train_inputs 更新を無効化する。"""
        return None

    def predict_noise_logvar(
        self,
        X: Tensor,
        ref_like: Optional[Tensor] = None,
    ) -> Tensor:
        logvar = self.noise_model.posterior(X).mean
        if ref_like is not None:
            logvar = align_like(logvar, ref_like)
        return logvar

    def predict_noise_var(
        self,
        X: Tensor,
        ref_like: Optional[Tensor] = None,
    ) -> Tensor:
        return self.predict_noise_logvar(X, ref_like=ref_like).exp().clamp_min(1e-12)

    def predict_noise_std(
        self,
        X: Tensor,
        ref_like: Optional[Tensor] = None,
    ) -> Tensor:
        return self.predict_noise_var(X, ref_like=ref_like).sqrt()


class HeteroscedasticLatentPosteriorMixin(HeteroscedasticNoiseModelMixin):
    """
    GPyTorchPosterior 系の latent posterior に heteroscedastic noise を足す mixin。

    observation_noise=True のときだけ対角 noise を covariance に加える。
    """

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs,
    ):
        base_post = super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=False,
            posterior_transform=None,
            **kwargs,
        )

        base_mean = base_post.mean
        obs_noise = None

        if torch.is_tensor(observation_noise):
            obs_noise = expand_observation_noise_tensor(observation_noise, X)
            obs_noise = align_like(obs_noise, base_mean)
        elif observation_noise:
            obs_noise = self.predict_noise_var(X, ref_like=base_mean)

        if obs_noise is None:
            posterior = base_post
        else:
            obs_noise = align_like(obs_noise, base_mean)
            base_dist = base_post.distribution

            if isinstance(base_dist, MultitaskMultivariateNormal):
                noise_flat = obs_noise.reshape(*obs_noise.shape[:-len(base_dist.event_shape)], -1)
                noise_diag = torch.diag_embed(noise_flat)
                dist = MultitaskMultivariateNormal(
                    base_dist.mean,
                    base_dist.covariance_matrix + noise_diag,
                )
            else:
                diag = DiagLinearOperator(obs_noise.squeeze(-1))
                dist = MultivariateNormal(
                    base_dist.mean,
                    base_dist.lazy_covariance_matrix + diag,
                )

            posterior = GPyTorchPosterior(dist)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior
