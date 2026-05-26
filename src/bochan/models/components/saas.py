from __future__ import annotations

"""SAAS 系モデルで共有する helper 群。

このモジュールは regression / classification / ordinal の各 SAAS 実装から
共通利用することを想定する。主な責務は次の 3 つ。

1. MAP-SAAS 用の covar_module 構築
2. mixed 入力の one-hot encode / decode
3. raw-space / encoded-space の bounds, input_transform, conditioning data 整形

Notes:
    - raw-space はユーザーが渡す元の特徴量空間。
    - encoded-space はカテゴリ列を one-hot 展開した内部特徴量空間。
    - mixed SAAS では内部 GP は encoded-space を見る。
    - public な ``train_inputs_raw`` は raw-space を保持することを推奨する。
"""

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

import torch
from torch import Tensor

from botorch.models.map_saas import add_saas_prior
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel


@dataclass(frozen=True)
class CategoricalSpec:
    """raw-space の 1 つのカテゴリ列に関する metadata。

    Args:
        source_dim: raw-space でのカテゴリ列 index。
        categories: 学習データで確認されたカテゴリ値。
        encoded_indices: encoded-space における one-hot block の列 index。
    """

    source_dim: int
    categories: Tensor
    encoded_indices: Tensor


def to_device_dtype_transform(input_transform: Any | None, ref: Tensor) -> Any | None:
    """input_transform を ref と同じ device / dtype に移す。"""
    if input_transform is None:
        return None
    if hasattr(input_transform, "to"):
        try:
            return input_transform.to(device=ref.device, dtype=ref.dtype)
        except TypeError:
            return input_transform.to(ref)
    return input_transform


def infer_ard_num_dims(train_X: Tensor, input_transform: Any | None = None) -> int:
    """input_transform 適用後の入力次元数を推定する。

    SAAS prior を貼る ARD kernel の ``ard_num_dims`` 決定に使う。
    学習用 transform として扱うため、可能なら一時的に train mode にする。
    """
    input_transform = to_device_dtype_transform(input_transform, train_X)
    if input_transform is None:
        return int(train_X.shape[-1])

    was_training = bool(getattr(input_transform, "training", False))
    if hasattr(input_transform, "train"):
        input_transform.train()
    with torch.no_grad():
        transformed = input_transform(train_X)
    if hasattr(input_transform, "eval") and not was_training:
        input_transform.eval()
    return int(transformed.shape[-1])


def build_map_saas_covar_module(
    train_X: Tensor,
    input_transform: Any | None = None,
    tau: float | Tensor | None = None,
    log_scale: bool = True,
    nu: float = 2.5,
    outputscale: bool = True,
) -> ScaleKernel | MaternKernel:
    """MAP-SAAS style の Matern kernel を構築する。

    Args:
        train_X: 学習入力。input_transform 適用前の空間でよい。
        input_transform: train_X に適用される入力変換。
        tau: SAAS の global shrinkage parameter。None の場合は推定対象。
        log_scale: BoTorch の ``add_saas_prior`` に渡す log-scale flag。
        nu: Matern kernel の smoothness parameter。
        outputscale: True の場合は ``ScaleKernel`` で包む。

    Returns:
        SAAS prior 付き kernel。
    """
    ard_num_dims = infer_ard_num_dims(train_X=train_X, input_transform=input_transform)
    base_kernel = MaternKernel(
        nu=float(nu),
        ard_num_dims=ard_num_dims,
        batch_shape=torch.Size([]),
    ).to(train_X)
    add_saas_prior(base_kernel=base_kernel, tau=tau, log_scale=bool(log_scale))
    if not outputscale:
        return base_kernel
    return ScaleKernel(base_kernel=base_kernel, batch_shape=torch.Size([])).to(train_X)


def flatten_targets(y: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    """target を [n] にそろえる。"""
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    y = y.reshape(-1)
    return y if dtype is None else y.to(dtype=dtype)


def flatten_optional_noise(noise: Optional[Tensor]) -> Optional[Tensor]:
    """optional noise を [n] にそろえる。"""
    if noise is None:
        return None
    if noise.ndim > 1 and noise.shape[-1] == 1:
        noise = noise.squeeze(-1)
    return noise.reshape(-1)


def concat_optional_noise(
    old_Y: Tensor,
    old_Yvar: Optional[Tensor],
    new_Y: Tensor,
    new_Yvar: Optional[Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Tensor]:
    """old/new の Yvar を連結する。片方だけ None の場合は 0 で補う。"""
    if old_Yvar is None and new_Yvar is None:
        return None
    if old_Yvar is None:
        old_Yvar = torch.zeros_like(old_Y, dtype=dtype, device=device)
    else:
        old_Yvar = flatten_optional_noise(old_Yvar).to(dtype=dtype, device=device)
    if new_Yvar is None:
        new_Yvar = torch.zeros_like(new_Y, dtype=dtype, device=device)
    else:
        new_Yvar = flatten_optional_noise(new_Yvar).to(dtype=dtype, device=device)
    return torch.cat([old_Yvar, new_Yvar], dim=0)


def prepare_mixed_conditioning_data(
    X: Tensor,
    Y: Tensor,
    noise: Optional[Tensor],
    *,
    raw_dim: int,
    encoded_dim: int,
    decode_fn: Callable[[Tensor], Tensor],
    target_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """mixed one-hot wrapper 用の condition_on_observations 入力を整形する。

    Args:
        X: raw-space または encoded-space の観測点。
        Y: X.shape[:-1] と一致する target。
        noise: optional noise。shape は Y と同じ。
        raw_dim: raw-space の特徴量次元。
        encoded_dim: encoded-space の特徴量次元。
        decode_fn: encoded-space X を raw-space X に戻す関数。
        target_dtype: Y の dtype。classification なら X.dtype、ordinal なら long など。

    Returns:
        ``(X_raw_flat, Y_flat, noise_flat)``。
    """
    if isinstance(X, tuple):
        X = X[0]
    if X.dim() < 2:
        raise ValueError("X must have shape [q, d] or [batch, q, d].")

    expected_y_shape = X.shape[:-1]
    if Y.dim() == X.dim() and Y.shape[-1] == 1:
        Y = Y.squeeze(-1)
    if Y.shape != expected_y_shape:
        raise NotImplementedError(
            "This wrapper supports non-fantasy observations with "
            "Y.shape == X.shape[:-1] or trailing singleton output dim. "
            f"Got X.shape={tuple(X.shape)}, Y.shape={tuple(Y.shape)}."
        )

    if noise is not None:
        if noise.dim() == X.dim() and noise.shape[-1] == 1:
            noise = noise.squeeze(-1)
        if noise.shape != expected_y_shape:
            raise ValueError(
                "noise must match X.shape[:-1] or have trailing singleton dim. "
                f"Got X.shape={tuple(X.shape)}, noise.shape={tuple(noise.shape)}."
            )

    X_flat = X.reshape(-1, X.shape[-1])
    if X.shape[-1] == raw_dim:
        X_raw_flat = X_flat
    elif X.shape[-1] == encoded_dim:
        X_raw_flat = decode_fn(X_flat)
    else:
        raise ValueError(
            f"Expected X last dim to be raw_dim={raw_dim} or encoded_dim={encoded_dim}, "
            f"got {X.shape[-1]}."
        )

    Y_flat = Y.reshape(-1)
    if target_dtype is not None:
        Y_flat = Y_flat.to(dtype=target_dtype)
    noise_flat = None if noise is None else noise.reshape(-1).to(dtype=X.dtype)
    return X_raw_flat, Y_flat, noise_flat


class OneHotEncodingMixin:
    """mixed SAAS 用 one-hot encoding mixin。

    サブクラスでは ``_init_one_hot_encoding(train_X, cat_dims)`` を先に呼び、
    その後 ``_encode_X`` / ``decode_inputs`` / ``transform_bounds`` などを利用する。
    """

    _raw_dim: int
    _encoded_dim: int
    _cat_dims: list[int]
    _cat_specs: Mapping[int, CategoricalSpec]

    def _init_one_hot_encoding(
        self,
        train_X: Tensor,
        cat_dims: Optional[Sequence[int]],
    ) -> Tensor:
        """カテゴリ仕様を推定し、train_X を encoded-space に変換する。"""
        self._raw_dim = int(train_X.shape[-1])
        self._cat_dims = sorted(set(int(i) for i in (cat_dims or [])))
        self._validate_cat_dims(self._cat_dims, raw_dim=self._raw_dim)
        self._cat_specs = self._infer_cat_specs(train_X=train_X, cat_dims=self._cat_dims)
        encoded = self._encode_X(train_X)
        self._encoded_dim = int(encoded.shape[-1])
        return encoded

    @property
    def raw_dim(self) -> int:
        return self._raw_dim

    @property
    def encoded_dim(self) -> int:
        return self._encoded_dim

    @property
    def cat_dims(self) -> list[int]:
        return list(self._cat_dims)

    @property
    def encoded_cat_dims(self) -> Dict[int, List[int]]:
        return {
            d: [int(i) for i in spec.encoded_indices.tolist()]
            for d, spec in self._cat_specs.items()
        }

    @staticmethod
    def _validate_cat_dims(cat_dims: Sequence[int], *, raw_dim: int) -> None:
        for d in cat_dims:
            if int(d) < 0 or int(d) >= int(raw_dim):
                raise ValueError(f"cat_dim {d} is out of range for raw_dim={raw_dim}.")

    @staticmethod
    def _infer_cat_specs(train_X: Tensor, cat_dims: Sequence[int]) -> Mapping[int, CategoricalSpec]:
        specs: MutableMapping[int, CategoricalSpec] = {}
        encoded_cursor = 0
        cat_dim_set = set(int(d) for d in cat_dims)
        for d in range(train_X.shape[-1]):
            if d not in cat_dim_set:
                encoded_cursor += 1
                continue
            col = train_X[..., d]
            if col.dtype.is_floating_point:
                col = col.round()
            categories = torch.unique(col).sort().values
            if categories.numel() == 0:
                raise ValueError(f"Categorical column {d} is empty.")
            encoded_indices = torch.arange(
                encoded_cursor,
                encoded_cursor + len(categories),
                dtype=torch.long,
                device=train_X.device,
            )
            specs[d] = CategoricalSpec(
                source_dim=d,
                categories=categories,
                encoded_indices=encoded_indices,
            )
            encoded_cursor += len(categories)
        return dict(specs)

    def _encode_X(self, X: Tensor) -> Tensor:
        """raw-space X を encoded-space X に変換する。"""
        if X.shape[-1] != self._raw_dim:
            raise ValueError(f"Expected raw input dim {self._raw_dim}, got {X.shape[-1]}.")
        if not self._cat_specs:
            return X

        pieces: list[Tensor] = []
        for d in range(self._raw_dim):
            if d not in self._cat_specs:
                pieces.append(X[..., d : d + 1])
                continue
            spec = self._cat_specs[d]
            x_d = X[..., d : d + 1]
            if x_d.dtype.is_floating_point:
                x_d = x_d.round()
            oh = (x_d == spec.categories).to(dtype=X.dtype)
            if torch.any(oh.sum(dim=-1) == 0):
                raise ValueError(
                    f"Input includes unseen category in raw dim {d}. "
                    f"Known categories: {spec.categories.tolist()}"
                )
            pieces.append(oh)
        return torch.cat(pieces, dim=-1)

    def decode_inputs(self, X_encoded: Tensor) -> Tensor:
        """encoded-space X を raw-space X に戻す。カテゴリ列は argmax で復元する。"""
        if X_encoded.shape[-1] != self.encoded_dim:
            raise ValueError(
                f"Expected encoded input dim {self.encoded_dim}, got {X_encoded.shape[-1]}."
            )
        raw_pieces: list[Tensor] = []
        for d in range(self._raw_dim):
            if d not in self._cat_specs:
                raw_idx = self._raw_to_single_encoded_index(d)
                raw_pieces.append(X_encoded[..., raw_idx : raw_idx + 1])
                continue
            spec = self._cat_specs[d]
            block = X_encoded[..., spec.encoded_indices]
            idx = block.argmax(dim=-1)
            vals = spec.categories[idx].unsqueeze(-1).to(dtype=X_encoded.dtype)
            raw_pieces.append(vals)
        return torch.cat(raw_pieces, dim=-1)

    def transform_bounds(self, bounds: Tensor) -> Tensor:
        """raw-space bounds [2, raw_dim] を encoded-space bounds [2, encoded_dim] へ変換する。"""
        if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] != self._raw_dim:
            raise ValueError(f"Expected bounds shape [2, {self._raw_dim}], got {tuple(bounds.shape)}.")
        pieces: list[Tensor] = []
        for d in range(self._raw_dim):
            if d not in self._cat_specs:
                pieces.append(bounds[:, d : d + 1])
                continue
            spec = self._cat_specs[d]
            cat_bounds = torch.zeros(2, len(spec.categories), dtype=bounds.dtype, device=bounds.device)
            cat_bounds[1, :] = 1.0
            pieces.append(cat_bounds)
        return torch.cat(pieces, dim=-1)

    def get_optimize_acqf_mixed_fixed_features_list(self) -> list[dict[int, float]]:
        """encoded-space 用の ``fixed_features_list`` を返す。"""
        if not self._cat_specs:
            return []
        per_dim: list[list[dict[int, float]]] = []
        for d in self._cat_dims:
            spec = self._cat_specs[d]
            assignments: list[dict[int, float]] = []
            for active in range(len(spec.categories)):
                assignments.append({
                    int(idx): float(i == active)
                    for i, idx in enumerate(spec.encoded_indices.tolist())
                })
            per_dim.append(assignments)
        out: list[dict[int, float]] = []
        for combo in product(*per_dim):
            merged: dict[int, float] = {}
            for part in combo:
                merged.update(part)
            out.append(merged)
        return out

    def _raw_to_single_encoded_index(self, raw_dim: int) -> int:
        encoded_idx = 0
        for d in range(self._raw_dim):
            if d == raw_dim:
                return encoded_idx
            encoded_idx += len(self._cat_specs[d].categories) if d in self._cat_specs else 1
        raise RuntimeError(f"Invalid raw dimension {raw_dim}.")

    def _encoded_indices_for_raw_dim(self, raw_dim: int) -> list[int]:
        raw_dim = int(raw_dim)
        if raw_dim in self._cat_specs:
            return [int(i) for i in self._cat_specs[raw_dim].encoded_indices.tolist()]
        return [int(self._raw_to_single_encoded_index(raw_dim))]

    def _raw_indices_to_encoded_indices(self, raw_indices: Optional[Sequence[int]]) -> Optional[list[int]]:
        if raw_indices is None:
            return None
        encoded: list[int] = []
        for raw_idx in raw_indices:
            encoded.extend(self._encoded_indices_for_raw_dim(int(raw_idx)))
        out: list[int] = []
        seen: set[int] = set()
        for idx in encoded:
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        return out

    def _canonicalize_inducing_points_for_encoded_space(self, inducing_points: Optional[Tensor]) -> Optional[Tensor]:
        """inducing_points を encoded-space にそろえる。raw/encoded の両方を許容する。"""
        if inducing_points is None:
            return None
        ref = getattr(self, "train_X_raw", None)
        if ref is None:
            ref = getattr(self, "train_inputs_raw", (None,))[0]
        inducing_points = torch.as_tensor(inducing_points, device=ref.device, dtype=ref.dtype)
        if inducing_points.ndim != 2:
            raise ValueError("inducing_points must be [m, d_raw] or [m, d_encoded].")
        if inducing_points.shape[-1] == self.raw_dim:
            return self._encode_X(inducing_points)
        if inducing_points.shape[-1] == self.encoded_dim:
            return inducing_points.contiguous()
        raise ValueError(
            f"inducing_points feature dim must be raw_dim={self.raw_dim} "
            f"or encoded_dim={self.encoded_dim}, got {inducing_points.shape[-1]}."
        )

    @staticmethod
    def _expand_X_to_match_q(X: Tensor, X_tf: Tensor) -> Tensor:
        """InputPerturbation 後の q 展開に合わせて X を repeat する。"""
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

    def _check_encoded_categorical_blocks_unchanged(
        self,
        X_encoded: Tensor,
        X_tf: Tensor,
        *,
        name: str = "input_transform",
    ) -> None:
        """one-hot block が input_transform で変更されていないか確認する。"""
        if not self._cat_specs:
            return
        if X_encoded.shape[-1] != self.encoded_dim or X_tf.shape[-1] != self.encoded_dim:
            return
        X_cmp = self._expand_X_to_match_q(X_encoded, X_tf)
        if X_cmp.shape[:-1] != X_tf.shape[:-1]:
            raise RuntimeError(
                f"Could not align encoded X with transformed X in {name}. "
                f"X_encoded.shape={tuple(X_encoded.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
                f"X_cmp.shape={tuple(X_cmp.shape)}."
            )
        for raw_dim, spec in self._cat_specs.items():
            before = X_cmp[..., spec.encoded_indices]
            after = X_tf[..., spec.encoded_indices]
            if not torch.allclose(after, before):
                raise ValueError(
                    f"{name} must not modify one-hot categorical block for raw dim {raw_dim}. "
                    f"before.shape={tuple(before.shape)}, after.shape={tuple(after.shape)}."
                )

    def _to_encoded_feature_space(self, X: Tensor) -> Tensor:
        """raw/encoded のどちらの X でも encoded-space にそろえる。"""
        if isinstance(X, tuple):
            X = X[0]
        if X.shape[-1] == self.raw_dim:
            return self._encode_X(X)
        if X.shape[-1] == self.encoded_dim:
            return X
        raise ValueError(
            f"Expected raw dim {self.raw_dim} or encoded dim {self.encoded_dim}, got {X.shape[-1]}."
        )

    def _maybe_expand_input_transform(self, input_transform: Any | None) -> Any | None:
        """raw-space 用 input_transform を encoded-space 用に拡張する。"""
        if input_transform is None or not self._cat_specs:
            return input_transform
        try:
            input_transform = deepcopy(input_transform)
        except Exception:
            pass
        return self._expand_transform_to_encoded_space(input_transform)

    def _expand_transform_to_encoded_space(self, transform: Any) -> Any:
        if transform is None:
            return None

        child_items = self._get_transform_child_items(transform)
        if child_items:
            for key, child in child_items:
                new_child = self._expand_transform_to_encoded_space(child)
                self._set_child_transform(transform, key, new_child)
            return transform

        if isinstance(transform, Normalize):
            return self._expand_normalize_to_encoded_space(transform)

        self._expand_perturbation_tensors_to_encoded_space(transform)

        bounds = getattr(transform, "bounds", None)
        if isinstance(bounds, Tensor):
            new_bounds = self._expand_bounds_tensor_to_encoded_space(bounds)
            if new_bounds is not None:
                self._set_attr_if_present(transform, "bounds", new_bounds)

        for attr in ("categorical_idx", "categorical_indices", "categorical_dims", "cat_dims", "indices"):
            value = getattr(transform, attr, None)
            encoded_value = self._maybe_map_index_value_to_encoded(value)
            if encoded_value is not None:
                self._set_attr_if_present(transform, attr, encoded_value)
        return transform

    @staticmethod
    def _get_transform_child_items(transform: Any) -> list[tuple[Any, Any]]:
        """ChainedInputTransform / ModuleDict 風 transform の子 transform を取得する。"""
        try:
            items = list(transform.items())
        except Exception:
            items = []
        if items:
            return items

        transforms = getattr(transform, "transforms", None)
        if transforms is not None:
            try:
                items = list(transforms.items())
            except Exception:
                items = []
        return items

    @staticmethod
    def _set_child_transform(transform: Any, key: Any, child: Any) -> None:
        """ChainedInputTransform / ModuleDict 風 transform の子 transform を差し替える。"""
        try:
            transform[key] = child
            return
        except Exception:
            pass

        transforms = getattr(transform, "transforms", None)
        if transforms is not None:
            try:
                transforms[key] = child
                return
            except Exception:
                pass
            try:
                setattr(transforms, str(key), child)
                return
            except Exception:
                pass

        try:
            setattr(transform, str(key), child)
        except Exception:
            pass

    def _expand_perturbation_tensors_to_encoded_space(self, transform: Any) -> None:
        """InputPerturbation などの raw-space perturbation tensor を encoded-space に拡張する。

        additive perturbation ではカテゴリ one-hot block に 0 を入れ、
        multiplicative perturbation ではカテゴリ one-hot block に 1 を入れる。
        これにより raw-space の ``[..., raw_dim]`` perturbation を
        encoded-space の ``[..., encoded_dim]`` 入力へ安全に適用できる。
        """
        fill_value = 1.0 if bool(getattr(transform, "multiplicative", False)) else 0.0
        for attr in (
            "perturbation_set",
            "_perturbation_set",
            "perturbations",
            "_perturbations",
        ):
            value = getattr(transform, attr, None)
            if not isinstance(value, Tensor):
                continue
            encoded_value = self._expand_raw_feature_tensor_to_encoded_space(
                value,
                fill_value=fill_value,
            )
            if encoded_value is not None:
                self._set_attr_if_present(transform, attr, encoded_value)

    def _expand_raw_feature_tensor_to_encoded_space(
        self,
        value: Tensor,
        *,
        fill_value: float = 0.0,
    ) -> Optional[Tensor]:
        """raw-space feature tensor ``[..., raw_dim]`` を ``[..., encoded_dim]`` へ拡張する。"""
        if value.shape[-1] == self.encoded_dim:
            return value
        if value.shape[-1] != self.raw_dim:
            return None

        pieces: list[Tensor] = []
        for d in range(self.raw_dim):
            if d not in self._cat_specs:
                pieces.append(value[..., d : d + 1])
                continue
            spec = self._cat_specs[d]
            cat_piece = value.new_full(
                (*value.shape[:-1], len(spec.categories)),
                fill_value,
            )
            pieces.append(cat_piece)
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _set_attr_if_present(obj: Any, name: str, value: Any) -> None:
        if hasattr(obj, name):
            try:
                setattr(obj, name, value)
            except Exception:
                pass

    def _maybe_map_index_value_to_encoded(self, value: Any) -> Any | None:
        if value is None:
            return None
        is_tensor = isinstance(value, Tensor)
        if is_tensor:
            flat = value.detach().cpu().view(-1).tolist()
        elif isinstance(value, (list, tuple, set)):
            flat = list(value)
        else:
            return None
        if len(flat) == 0:
            return value
        try:
            raw_indices = [int(i) for i in flat]
        except Exception:
            return None
        if max(raw_indices) >= self.raw_dim and max(raw_indices) < self.encoded_dim:
            return value
        if min(raw_indices) < 0 or max(raw_indices) >= self.raw_dim:
            return None
        encoded = self._raw_indices_to_encoded_indices(raw_indices)
        if is_tensor:
            return torch.tensor(encoded, dtype=value.dtype, device=value.device)
        if isinstance(value, tuple):
            return tuple(encoded)
        if isinstance(value, set):
            return set(encoded)
        return encoded

    def _expand_bounds_tensor_to_encoded_space(self, bounds: Tensor) -> Optional[Tensor]:
        if bounds.shape[-1] == self.encoded_dim:
            return bounds
        if bounds.shape[-1] != self.raw_dim:
            return None
        if bounds.ndim == 2 and bounds.shape[0] == 2:
            return self.transform_bounds(bounds).to(dtype=bounds.dtype, device=bounds.device)
        if bounds.ndim >= 3 and bounds.shape[-2] == 2:
            flat = bounds.reshape(-1, 2, bounds.shape[-1])
            enc = torch.stack([self.transform_bounds(b).to(bounds) for b in flat], dim=0)
            return enc.reshape(*bounds.shape[:-2], 2, self.encoded_dim)
        return None

    def _expand_normalize_to_encoded_space(self, transform: Normalize) -> Normalize:
        bounds = getattr(transform, "bounds", None)
        indices = getattr(transform, "indices", None)

        raw_indices = None
        if indices is not None:
            if isinstance(indices, Tensor):
                raw_indices = [int(i) for i in indices.view(-1).tolist()]
            else:
                raw_indices = [int(i) for i in indices]

        if isinstance(bounds, Tensor) and bounds.ndim >= 3 and bounds.shape[-2] == 2:
            # BoTorch Normalize の bounds は batch 付きになる場合があるため、代表 slice を使う。
            bounds_2d = bounds.reshape(-1, 2, bounds.shape[-1])[0]
        else:
            bounds_2d = bounds

        if raw_indices is None:
            if isinstance(bounds_2d, Tensor):
                if bounds_2d.shape[-1] == self.encoded_dim:
                    return Normalize(d=self.encoded_dim, bounds=bounds_2d.to(dtype=bounds.dtype, device=bounds.device))
                if bounds_2d.shape[-1] == self.raw_dim:
                    return Normalize(
                        d=self.encoded_dim,
                        bounds=self.transform_bounds(bounds_2d).to(dtype=bounds_2d.dtype, device=bounds_2d.device),
                    )
            # bounds が復元できない場合は encoded 全次元 Normalize にする。
            return Normalize(d=self.encoded_dim)

        encoded_indices = self._raw_indices_to_encoded_indices(raw_indices)
        if not isinstance(bounds_2d, Tensor):
            raise ValueError("Normalize(indices=...) requires explicit bounds for SAAS mixed wrappers.")
        if bounds_2d.shape[-1] == self.raw_dim:
            selected_bounds = bounds_2d[:, raw_indices]
        elif bounds_2d.shape[-1] == len(raw_indices):
            selected_bounds = bounds_2d
        elif bounds_2d.shape[-1] == self.encoded_dim:
            return Normalize(d=self.encoded_dim, bounds=bounds_2d, indices=encoded_indices)
        else:
            raise ValueError(
                f"Normalize bounds shape {tuple(bounds_2d.shape)} is incompatible with "
                f"raw_dim={self.raw_dim}, encoded_dim={self.encoded_dim}."
            )

        # raw indices の bounds を encoded indices に展開する。
        # カテゴリ列は [0, 1] に固定する。
        piece_map: dict[int, Tensor] = {}
        for raw_idx, col_pos in zip(raw_indices, range(len(raw_indices))):
            enc_indices = self._encoded_indices_for_raw_dim(raw_idx)
            if raw_idx in self._cat_specs:
                b = torch.zeros(2, len(enc_indices), dtype=bounds_2d.dtype, device=bounds_2d.device)
                b[1, :] = 1.0
            else:
                b = selected_bounds[:, col_pos : col_pos + 1]
            for j, enc_idx in enumerate(enc_indices):
                piece_map[enc_idx] = b[:, j : j + 1]
        ordered_indices = sorted(piece_map)
        expanded_bounds = torch.cat([piece_map[i] for i in ordered_indices], dim=-1)
        return Normalize(d=self.encoded_dim, bounds=expanded_bounds, indices=ordered_indices)


__all__ = [
    "CategoricalSpec",
    "OneHotEncodingMixin",
    "to_device_dtype_transform",
    "infer_ard_num_dims",
    "build_map_saas_covar_module",
    "flatten_targets",
    "flatten_optional_noise",
    "concat_optional_noise",
    "prepare_mixed_conditioning_data",
]
