from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor
from botorch.utils.sampling import draw_sobol_samples


@torch.no_grad()
def optimize_qordinal_nipv_by_sampling(
    acq_function,
    bounds: Tensor,
    q: int,
    n_candidates: int = 256,
    batch_limit: int = 16,
):
    X_cand = draw_sobol_samples(
        bounds=bounds,
        n=n_candidates,
        q=q,
    ).to(bounds)  # [n_candidates, q, d]

    vals = []
    for xb in X_cand.split(batch_limit, dim=0):
        vals.append(acq_function(xb))
    vals = torch.cat(vals, dim=0)  # [n_candidates]

    best = vals.argmax()
    return X_cand[best], vals[best]


def _set_acqf_pending(acq_function, X_pending: Optional[Tensor]) -> None:
    """
    acq_function に X_pending を設定する。
    set_X_pending があればそれを使い、なければ属性へ直接代入する。
    """
    if hasattr(acq_function, "set_X_pending"):
        acq_function.set_X_pending(X_pending)
    else:
        acq_function.X_pending = X_pending


def _get_acqf_pending(acq_function) -> Optional[Tensor]:
    """acq_function から現在の X_pending を取得する。"""
    return getattr(acq_function, "X_pending", None)


def _apply_fixed_features(X: Tensor, fixed_features: dict[int, float]) -> Tensor:
    """
    X[..., d] に対して fixed_features を上書きする。

    Args:
        X: [..., d]
        fixed_features: {dim_index: fixed_value}
    """
    if not fixed_features:
        return X

    X = X.clone()
    for idx, val in fixed_features.items():
        X[..., idx] = torch.as_tensor(val, dtype=X.dtype, device=X.device)
    return X


def _evaluate_acqf_in_batches(
    acq_function,
    X: Tensor,
    batch_limit: int,
) -> Tensor:
    """
    候補集合 X を分割して acq_function に通す。

    Args:
        X: [n, q, d]
        batch_limit: 1回に評価する候補数

    Returns:
        vals: [n]
    """
    vals = []
    for xb in X.split(batch_limit, dim=0):
        vals.append(acq_function(xb))
    return torch.cat(vals, dim=0)


def _filter_exact_duplicates_against_pending(
    X: Tensor,
    X_pending: Optional[Tensor],
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> Tensor:
    """
    X のうち、X_pending と完全一致に近い候補を除外する。

    Args:
        X: [n, 1, d]
        X_pending: [m, d] or None

    Returns:
        filtered X: [n_filtered, 1, d]
    """
    if X_pending is None or X_pending.numel() == 0:
        return X

    Xp = X_pending.to(dtype=X.dtype, device=X.device)
    X_flat = X[:, 0, :]  # [n, d]

    # [n, m, d] -> [n, m]
    same = torch.isclose(
        X_flat[:, None, :],
        Xp[None, :, :],
        atol=atol,
        rtol=rtol,
    ).all(dim=-1)

    keep = ~same.any(dim=-1)
    if keep.any():
        return X[keep]
    return X


@torch.no_grad()
def optimize_qordinal_nipv_mixed_by_sampling(
    acq_function,
    bounds: Tensor,
    q: int,
    fixed_features_list: Optional[Sequence[dict[int, float]]] = None,
    n_candidates: int = 256,
    batch_limit: int = 16,
    hard_avoid_duplicates: bool = True,
    duplicate_atol: float = 1e-8,
    duplicate_rtol: float = 1e-5,
):
    """
    qOrdinalNegIntegratedPosteriorVariance 向けの mixed 版サンプリング最適化。

    方針:
        - q > 1 は逐次 greedy に 1 点ずつ選ぶ
        - 各 step で Sobol により連続候補を生成
        - fixed_features_list を全列挙して mixed 候補を作る
        - すでに選んだ点を X_pending に追加して次点選択へ進む

    Args:
        acq_function:
            forward(X:[batch, q, d]) -> [batch] を返す獲得関数。
            X_pending を参照できる実装を想定。
        bounds:
            [2, d]
        q:
            提案点数
        fixed_features_list:
            例: [{3: 0.0}, {3: 0.5}, {3: 1.0}]
            None または空なら通常の連続サンプリングと同じ。
        n_candidates:
            各 step の連続候補数
        batch_limit:
            acq_function 評価時の分割サイズ
        hard_avoid_duplicates:
            True のとき、X_pending と完全一致の候補を事前除外する
        duplicate_atol, duplicate_rtol:
            完全一致判定の許容誤差

    Returns:
        best_X: [q, d]
        best_val: scalar Tensor
    """
    device = bounds.device
    dtype = bounds.dtype

    if fixed_features_list is None or len(fixed_features_list) == 0:
        fixed_features_list = [{}]

    # 元の pending を保存
    base_pending = _get_acqf_pending(acq_function)
    selected = []

    try:
        for _ in range(q):
            # 現在までに選んだ点を pending に追加
            if len(selected) == 0:
                current_pending = base_pending
            else:
                selected_tensor = torch.stack(selected, dim=0)  # [k, d]
                if base_pending is None or base_pending.numel() == 0:
                    current_pending = selected_tensor
                else:
                    current_pending = torch.cat(
                        [base_pending.to(selected_tensor), selected_tensor], dim=0
                    )

            _set_acqf_pending(acq_function, current_pending)

            # 連続候補を生成: [n_candidates, 1, d]
            X_base = draw_sobol_samples(
                bounds=bounds,
                n=n_candidates,
                q=1,
            ).to(device=device, dtype=dtype)

            # fixed_features を全列挙して mixed 候補を作る
            X_mixed_list = []
            for ff in fixed_features_list:
                X_ff = _apply_fixed_features(X_base, ff)
                X_mixed_list.append(X_ff)

            X_cand = torch.cat(X_mixed_list, dim=0)  # [n_candidates * n_ff, 1, d]

            # 完全重複を事前除外
            if hard_avoid_duplicates:
                X_cand_filtered = _filter_exact_duplicates_against_pending(
                    X=X_cand,
                    X_pending=current_pending,
                    atol=duplicate_atol,
                    rtol=duplicate_rtol,
                )
                if X_cand_filtered.shape[0] > 0:
                    X_cand = X_cand_filtered

            vals = _evaluate_acqf_in_batches(
                acq_function=acq_function,
                X=X_cand,
                batch_limit=batch_limit,
            )  # [n_total]

            best_idx = vals.argmax()
            best_x = X_cand[best_idx, 0, :].clone()  # [d]
            selected.append(best_x)

        best_X = torch.stack(selected, dim=0)  # [q, d]

        # 最終 batch 値は「元の pending」に戻したうえで評価
        _set_acqf_pending(acq_function, base_pending)
        best_val = acq_function(best_X.unsqueeze(0)).squeeze(0)

        return best_X, best_val

    finally:
        # 例外時も pending を戻す
        _set_acqf_pending(acq_function, base_pending)