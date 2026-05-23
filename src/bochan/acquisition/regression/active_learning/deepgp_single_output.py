from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform


QReduceType = Literal["mean", "sum", "max", "min"]
OutputReduceType = Literal["mean", "sum", "max", "min"]


def _reduce(t: Tensor, dim: int, mode: str) -> Tensor:
    if mode == "mean":
        return t.mean(dim=dim)
    if mode == "sum":
        return t.sum(dim=dim)
    if mode == "max":
        return t.max(dim=dim).values
    if mode == "min":
        return t.min(dim=dim).values
    raise ValueError(f"Unknown reduction mode: {mode}")


class _DeepPosteriorAcquisitionBase(AcquisitionFunction):
    """
    DeepGP / DeepMixedGP のように posterior() は持つが
    fantasize() を持たないモデル向けの共通基底クラス。

    対応内容:
        - posterior.mean / posterior.variance のみで評価できる獲得関数
        - q-batch 内の重複ペナルティ
        - X_pending / X_observed への近接ペナルティ
        - InputPerturbation 後の q * n_w 出力に対する risk objective 集約

    注意:
        qNegIntegratedPosteriorVariance のように fantasy model を作る
        厳密な獲得関数ではなく、DeepGP 用の実用 proxy です。
    """

    def __init__(
        self,
        model,
        q_reduction: QReduceType = "mean",
        output_reduction: OutputReduceType = "mean",
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_penalty: float = 0.0,
        hard_duplicate_tol: float = 1e-8,
        eps: float = 1e-12,
        objective: Optional[Any] = None,
        n_w: Optional[int] = None,
    ) -> None:
        super().__init__(model=model)

        self.q_reduction = q_reduction
        self.output_reduction = output_reduction
        self.eps = eps

        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)
        self.hard_duplicate_tol = float(hard_duplicate_tol)

        self.objective = objective

        if n_w is None and objective is not None:
            n_w = getattr(objective, "n_w", None)

        self.n_w = int(n_w) if n_w is not None else None

        self.X_pending: Optional[Tensor] = None
        self.X_observed: Optional[Tensor] = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = X_pending

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = X_observed

    def _reduce_outputs_if_needed(self, t: Tensor) -> Tensor:
        """
        posterior mean / variance の出力次元 m を集約する。

        想定入力:
            - [..., q]
            - [..., q, m]
        """
        if t.ndim >= 3:
            return _reduce(t, dim=-1, mode=self.output_reduction)
        return t

    def _reduce_q(self, t: Tensor) -> Tensor:
        """q 個の候補のスコアを 1 つにまとめる。"""
        return _reduce(t, dim=-1, mode=self.q_reduction)

    def _risk_reduce_n_w(
        self,
        score_per_point: Tensor,
        q: int,
        *,
        context: str = "score",
    ) -> Tensor:
        """
        InputPerturbation によって [..., q * n_w] になったスコアを
        [..., q] に戻す。

        objective がある場合:
            VaR / CVaR / Expectation などで n_w 方向を集約する。

        objective がないが n_w がある場合:
            n_w 方向を単純平均する。

        objective も n_w もない場合:
            何もしない。
        """
        if self.n_w is None:
            if self.objective is not None:
                raise ValueError(
                    "objective を使う場合は n_w が必要です。"
                    "VaR(alpha=..., n_w=...) のように n_w を持つ objective を渡すか、"
                    "獲得関数側で n_w=... を指定してください。"
                )
            return score_per_point

        expected_qnw = q * self.n_w

        if score_per_point.shape[-1] != expected_qnw:
            if self.objective is None:
                return score_per_point

            raise RuntimeError(
                f"{context}: risk objective を使う場合、最後の次元は "
                f"q * n_w = {q} * {self.n_w} = {expected_qnw} である必要があります。"
                f"しかし score_per_point.shape[-1] = {score_per_point.shape[-1]} でした。\n"
                "InputPerturbation が model.posterior(X) 内で有効になっているか、"
                "n_w が InputPerturbation の摂動数と一致しているか確認してください。"
            )

        if self.objective is None:
            return score_per_point.reshape(
                *score_per_point.shape[:-1],
                q,
                self.n_w,
            ).mean(dim=-1)

        # BoTorch の RiskMeasureMCObjective は通常、
        # samples: [sample_shape, batch_shape..., q * n_w, m]
        # を想定する。
        #
        # ここでは posterior mean/std から作ったスコアを
        # deterministic sample として扱う。
        pseudo_samples = score_per_point.unsqueeze(0).unsqueeze(-1)

        risk_score = self.objective(pseudo_samples)

        # 多くの場合: [1, batch_shape..., q] -> [batch_shape..., q]
        if risk_score.ndim >= 1 and risk_score.shape[0] == 1:
            risk_score = risk_score.squeeze(0)

        if risk_score.shape[-1] != q:
            raise RuntimeError(
                f"{context}: objective 適用後の最後の次元は q={q} を想定していますが、"
                f"risk_score.shape={tuple(risk_score.shape)} でした。"
            )

        return risk_score

    def _aggregate_point_scores(
        self,
        score_per_point: Tensor,
        X: Tensor,
        *,
        context: str = "score",
    ) -> Tensor:
        """
        [..., q] または [..., q * n_w] のスコアを
        risk 集約 + q 集約して [...] にする。
        """
        q = X.shape[-2]
        score_per_q = self._risk_reduce_n_w(
            score_per_point=score_per_point,
            q=q,
            context=context,
        )
        return self._reduce_q(score_per_q)

    @staticmethod
    def _pairwise_sq_dists(X: Tensor, Y: Tensor) -> Tensor:
        """
        二乗距離行列を返す。

        Args:
            X: [..., q, d]
            Y: [n, d]

        Returns:
            [..., q, n]
        """
        if Y.ndim != 2:
            raise ValueError("Y must have shape [n, d].")

        expand_shape = [1] * (X.ndim - 2) + list(Y.shape)
        Y_expanded = Y.view(*expand_shape)

        return (X.unsqueeze(-2) - Y_expanded).pow(2).sum(dim=-1)

    def _same_batch_penalty(self, X: Tensor) -> Tensor:
        if self.same_batch_penalty_weight <= 0 or X.shape[-2] <= 1:
            return torch.zeros(X.shape[:-2], dtype=X.dtype, device=X.device)

        d2 = (X.unsqueeze(-2) - X.unsqueeze(-3)).pow(2).sum(dim=-1)

        q = X.shape[-2]
        eye = torch.eye(q, dtype=torch.bool, device=X.device)
        while eye.ndim < d2.ndim:
            eye = eye.unsqueeze(0)

        valid_mask = ~eye

        soft_pen = torch.exp(-self.same_batch_penalty_beta * d2)
        soft_pen = torch.where(valid_mask, soft_pen, torch.zeros_like(soft_pen))
        soft_pen = soft_pen.sum(dim=(-2, -1))

        if self.hard_duplicate_penalty > 0:
            dup = (d2 <= self.hard_duplicate_tol).to(X.dtype)
            dup = torch.where(valid_mask, dup, torch.zeros_like(dup))
            hard_pen = self.hard_duplicate_penalty * dup.sum(dim=(-2, -1))
        else:
            hard_pen = torch.zeros_like(soft_pen)

        return self.same_batch_penalty_weight * soft_pen + hard_pen

    def _set_penalty_against_reference(
        self,
        X: Tensor,
        ref: Optional[Tensor],
        weight: float,
        beta: float,
    ) -> Tensor:
        if weight <= 0 or ref is None:
            return torch.zeros(X.shape[:-2], dtype=X.dtype, device=X.device)

        if ref.ndim == 3:
            ref = ref.reshape(-1, ref.shape[-1])

        if ref.ndim != 2:
            raise ValueError("Reference points must have shape [n, d] or [*, n, d].")

        d2 = self._pairwise_sq_dists(X, ref)
        pen = torch.exp(-beta * d2).sum(dim=-1)
        pen = self._reduce_q(pen)

        return weight * pen

    def _total_penalty(self, X: Tensor) -> Tensor:
        pen = self._same_batch_penalty(X)

        pen = pen + self._set_penalty_against_reference(
            X=X,
            ref=self.X_pending,
            weight=self.pending_penalty_weight,
            beta=self.pending_penalty_beta,
        )

        pen = pen + self._set_penalty_against_reference(
            X=X,
            ref=self.X_observed,
            weight=self.observed_penalty_weight,
            beta=self.observed_penalty_beta,
        )

        return pen

    def _posterior_mean_std(self, X: Tensor) -> tuple[Tensor, Tensor]:
        post = self.model.posterior(X, observation_noise=False)

        mean = post.mean
        var = post.variance.clamp_min(self.eps)
        std = var.sqrt()

        mean = self._reduce_outputs_if_needed(mean)
        std = self._reduce_outputs_if_needed(std)

        # [..., q, 1] -> [..., q]
        if mean.ndim == X.ndim:
            mean = mean.squeeze(-1)
        if std.ndim == X.ndim:
            std = std.squeeze(-1)

        return mean, std

    def _posterior_variance_score(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X, observation_noise=False)

        var = post.variance.clamp_min(self.eps)
        var = self._reduce_outputs_if_needed(var)

        if var.ndim == X.ndim:
            var = var.squeeze(-1)

        return var


class qDeepPosteriorVariance(_DeepPosteriorAcquisitionBase):
    """
    DeepGP 用の q-batch posterior-variance 獲得関数。

    入力摂動なし:
        score(X) = reduce_q(var(X)) - penalty(X)

    入力摂動あり + objective あり:
        score(X) = reduce_q(risk[var(X + ΔX)]) - penalty(X)

    入力摂動あり + objective なし + n_w あり:
        score(X) = reduce_q(mean_w[var(X + ΔX)]) - penalty(X)
    """

    def __init__(
        self,
        model,
        q_reduction: QReduceType = "mean",
        output_reduction: OutputReduceType = "mean",
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_penalty: float = 0.0,
        hard_duplicate_tol: float = 1e-8,
        eps: float = 1e-12,
        objective: Optional[Any] = None,
        n_w: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            q_reduction=q_reduction,
            output_reduction=output_reduction,
            X_pending=X_pending,
            X_observed=X_observed,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_penalty=hard_duplicate_penalty,
            hard_duplicate_tol=hard_duplicate_tol,
            eps=eps,
            objective=objective,
            n_w=n_w,
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        var_score = self._posterior_variance_score(X)

        score = self._aggregate_point_scores(
            score_per_point=var_score,
            X=X,
            context="qDeepPosteriorVariance",
        )

        return score - self._total_penalty(X)


class qDeepStraddle(_DeepPosteriorAcquisitionBase):
    """
    DeepGP 用の q-batch straddle 獲得関数。

    入力摂動なし:
        score(X) = reduce_q(beta * std(X) - |mean(X) - target|) - penalty(X)

    入力摂動あり + objective あり:
        score(X) = reduce_q(risk[beta * std(X + ΔX) - |mean(X + ΔX) - target|]) - penalty(X)

    入力摂動あり + objective なし + n_w あり:
        score(X) = reduce_q(mean_w[beta * std(X + ΔX) - |mean(X + ΔX) - target|]) - penalty(X)
    """

    def __init__(
        self,
        model,
        target: float = 0.0,
        beta: float = 1.96,
        q_reduction: QReduceType = "mean",
        output_reduction: OutputReduceType = "mean",
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_penalty: float = 0.0,
        hard_duplicate_tol: float = 1e-8,
        eps: float = 1e-12,
        objective: Optional[Any] = None,
        n_w: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            q_reduction=q_reduction,
            output_reduction=output_reduction,
            X_pending=X_pending,
            X_observed=X_observed,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_penalty=hard_duplicate_penalty,
            hard_duplicate_tol=hard_duplicate_tol,
            eps=eps,
            objective=objective,
            n_w=n_w,
        )

        self.target = float(target)
        self.beta = float(beta)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, std = self._posterior_mean_std(X)

        score_per_point = self.beta * std - (mean - self.target).abs()

        score = self._aggregate_point_scores(
            score_per_point=score_per_point,
            X=X,
            context="qDeepStraddle",
        )

        return score - self._total_penalty(X)


class qDeepIntegratedPosteriorVarianceProxy(_DeepPosteriorAcquisitionBase):
    """
    qNegIntegratedPosteriorVariance の DeepGP 向け proxy。

    厳密な qNIPV ではありません。
    fantasy model を作らず、参照点群 X_ref 上の posterior variance を見て、
    候補 X が高不確実領域をどれだけカバーしていそうかを評価します。

    入力摂動ありの場合:
        - X_ref の posterior variance が [n_ref * n_w] に展開される場合、
          objective または mean により [n_ref] に戻します。
        - 候補 X と X_ref の距離重みは nominal X に対して計算します。
          つまり「参照点側の不確実性」はリスク集約されますが、
          距離計算そのものは nominal 候補ベースです。
    """

    def __init__(
        self,
        model,
        X_ref: Tensor,
        kernel_lengthscale: float = 0.2,
        normalize_weights: bool = True,
        q_reduction: QReduceType = "mean",
        output_reduction: OutputReduceType = "mean",
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_penalty: float = 0.0,
        hard_duplicate_tol: float = 1e-8,
        eps: float = 1e-12,
        objective: Optional[Any] = None,
        n_w: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            q_reduction=q_reduction,
            output_reduction=output_reduction,
            X_pending=X_pending,
            X_observed=X_observed,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_penalty=hard_duplicate_penalty,
            hard_duplicate_tol=hard_duplicate_tol,
            eps=eps,
            objective=objective,
            n_w=n_w,
        )

        if X_ref.ndim != 2:
            raise ValueError("X_ref must have shape [n_ref, d].")

        self.register_buffer("X_ref", X_ref.detach().clone())
        self.kernel_lengthscale = float(kernel_lengthscale)
        self.normalize_weights = bool(normalize_weights)

    def _reference_variance(self) -> Tensor:
        """
        X_ref 上の posterior variance を返す。

        入力摂動がある場合:
            ref_var: [n_ref * n_w]
        となりうるため、n_w 方向に risk 集約して
            [n_ref]
        に戻す。
        """
        post = self.model.posterior(self.X_ref, observation_noise=False)

        ref_var = post.variance.clamp_min(self.eps)
        ref_var = self._reduce_outputs_if_needed(ref_var)

        if ref_var.ndim == 2:
            ref_var = ref_var.squeeze(-1)

        n_ref = self.X_ref.shape[-2]

        ref_var = self._risk_reduce_n_w(
            score_per_point=ref_var,
            q=n_ref,
            context="qDeepIntegratedPosteriorVarianceProxy._reference_variance",
        )

        if ref_var.shape[-1] != n_ref:
            raise RuntimeError(
                "X_ref 上の variance は [n_ref] になる必要があります。"
                f"しかし ref_var.shape={tuple(ref_var.shape)} でした。"
            )

        return ref_var

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        ref_var = self._reference_variance()

        d2 = self._pairwise_sq_dists(X, self.X_ref)
        ls2 = max(self.kernel_lengthscale ** 2, self.eps)

        weights = torch.exp(-0.5 * d2 / ls2)

        if self.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        view_shape = [1] * (weights.ndim - 1) + [-1]
        local_scores = (weights * ref_var.view(*view_shape)).sum(dim=-1)

        score = self._reduce_q(local_scores)

        return score - self._total_penalty(X)


__all__ = [
    "QReduceType",
    "OutputReduceType",
    "qDeepPosteriorVariance",
    "qDeepStraddle",
    "qDeepIntegratedPosteriorVarianceProxy",
]