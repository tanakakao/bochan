from __future__ import annotations

import inspect
import math
from typing import Callable, Literal, Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models import ModelListGP
from botorch.models.gpytorch import ModelListGPyTorchModel
from torch import Tensor


ReductionType = Literal["mean", "sum", "max"]
LargeQStrategy = Literal["per_point", "raise"]
UncertaintyScoreType = Literal["variance", "entropy", "least_confidence"]

ROIWeightMode = Literal[
    "none",
    "prob_above",
    "prob_below",
    "target_prob",
    "interval",
    "custom",
]
ROICombineType = Literal["multiply", "add"]

NoiseWeightMode = Literal[
    "none",
    "inverse_linear",
    "exp",
    "custom",
]
NoiseCombineType = Literal["multiply", "subtract"]


class _BinaryClassificationAcqBase(AcquisitionFunction):
    """
    2値分類用獲得関数の共通ベース。

    前提:
      - self.model.model に latent GP があるか、
        もしくは self.model 自体が latent GP として呼べる
      - input_transform があれば self.model.input_transform / transform_inputs を使う
      - heteroscedastic モデルでは必要に応じて
        self.model.noise_model を参照して noise penalty を計算できる

    主な責務:
      - ModelList の unwrap
      - 入力の feature space 変換
      - latent GP の取得
      - pointwise / joint 用 latent_dist の取得
      - pending penalty
      - q 集約
      - BALD / qBALD 用の binary helper
      - ROI weighting
      - hetero noise weighting
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        # ROI
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_aggregate_reduction: ReductionType = "mean",
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # hetero noise
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):
            model = model.models[0]

        super().__init__(model)
        self.reduction = reduction
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.eps = float(eps)

        # ROI
        self.roi_mode = roi_mode
        self.roi_combine = roi_combine
        self.roi_threshold = float(roi_threshold)
        self.roi_target_prob = float(roi_target_prob)
        self.roi_interval = roi_interval
        self.roi_beta = float(roi_beta)
        self.roi_bandwidth = float(roi_bandwidth)
        self.roi_min_weight = float(roi_min_weight)
        self.roi_weight_scale = float(roi_weight_scale)
        self.roi_aggregate_reduction = roi_aggregate_reduction
        self.roi_weight_fn = roi_weight_fn

        # hetero noise
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_model_outputs_log_var = bool(noise_model_outputs_log_var)
        self.noise_weight_fn = noise_weight_fn

        self.objective = objective

        self.set_X_pending(None)

    # =========================================================
    # 基本ユーティリティ
    # =========================================================
    def _prepare_eval(self) -> None:
        """model / likelihood を eval モードにする。"""
        self.model.eval()
        like = getattr(self.model, "likelihood", None)
        if like is not None:
            like.eval()

    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        """
        2D 入力 (n, d) が来た場合は q=1 扱いのため (n, 1, d) にする。
        """
        if X.ndim == 2:
            X = X.unsqueeze(-2)
        return X

    def _map_to_training_feature_space(self, X: Tensor) -> Tensor:
        """
        モデル内部の train_inputs と同じ feature space へ写す。

        優先順:
          1. _to_internal
          2. _to_latent
          3. _to_training_feature_space
          4. そのまま
        """
        X = self._ensure_q_batch(X)

        if hasattr(self.model, "_to_internal"):
            return self.model._to_internal(X)
        if hasattr(self.model, "_to_latent"):
            return self.model._to_latent(X)
        if hasattr(self.model, "_to_training_feature_space"):
            return self.model._to_training_feature_space(X)
        return X

    def _apply_input_transform_single_model(self, model, X: Tensor) -> Tensor:
        """
        1つの submodel に対して、評価に使う特徴空間へ X を写す。
        decomposition wrapper を優先し、なければ通常の input_transform を使う。
        """
        Xt = X[0] if isinstance(X, tuple) else X
    
        if hasattr(model, "_to_internal"):
            return model._to_internal(Xt)
    
        if hasattr(model, "_to_latent"):
            return model._to_latent(Xt)
    
        cls_transform_inputs = getattr(type(model), "transform_inputs", None)
        has_custom_transform_inputs = (
            cls_transform_inputs is not None
            and cls_transform_inputs is not Model.transform_inputs
        )
        if has_custom_transform_inputs:
            return model.transform_inputs(Xt)
    
        it = getattr(model, "input_transform", None)
        if callable(it):
            return it(Xt)
    
        return Xt
    
    
    def _apply_input_transform(self, X):
        """
        単一モデルなら Tensor を返す。
        ModelList なら各 submodel 用に変換した Tensor の list を返す。
        """
        Xt = X[0] if isinstance(X, tuple) else X
    
        submodels = getattr(self.model, "models", None)
        if submodels is not None:
            return [
                self._apply_input_transform_single_model(submodel, Xt)
                for submodel in submodels
            ]
    
        return self._apply_input_transform_single_model(self.model, Xt)

    def _get_latent_gp(self):
        latent_gp = getattr(self.model, "model", None)
        if latent_gp is None:
            latent_gp = self.model
        return latent_gp

    # =========================================================
    # latent posterior helpers
    # =========================================================
    def _as_distribution(self, posterior_or_dist):
        """
        GPyTorchPosterior のような posterior なら .distribution を取り出し、
        gpytorch distribution ならそのまま返す。
        """
        return getattr(posterior_or_dist, "distribution", posterior_or_dist)

    def _numel_from_dist_mean(self, posterior_or_dist) -> int:
        """
        latent distribution / posterior の mean の要素数を返す。
        single-output の最後の dim=1 は除去して数える。
        """
        dist = self._as_distribution(posterior_or_dist)
        mean = dist.mean
        if mean.ndim >= 1 and mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        return int(mean.numel())

    def _maybe_average_deepgp_latent_distribution(self, latent_dist, X_ref: Tensor):
        """
        DeepGP の extra sample / batch 次元を平均化する。

        self.model._average_deepgp_latent_distribution がある場合だけ使う。
        X_ref は latent output の期待 shape を決める参照入力で、
        InputPerturbation 使用時は transform 後の Xt を渡す。
        """
        avg_fn = getattr(self.model, "_average_deepgp_latent_distribution", None)
        if callable(avg_fn):
            return avg_fn(latent_dist, X_ref)
        return latent_dist

    def _call_model_forward_latent(
        self,
        X: Tensor,
        Xt: Tensor,
        expected_numel: int,
    ):
        """
        model.forward(X, apply_input_transform=True) を優先的に使う。

        ClassificationDeepGPModel / ClassificationMixedDeepGPModel のように
        forward が apply_input_transform を受け取れる場合、これが最も安全。
        古い latent_posterior 実装が二重 transform を持っていても回避できる。
        """
        forward = getattr(self.model, "forward", None)
        if not callable(forward):
            return None

        try:
            sig = inspect.signature(forward)
            if "apply_input_transform" not in sig.parameters:
                return None
        except (TypeError, ValueError):
            return None

        try:
            latent_dist = forward(X, apply_input_transform=True)
            latent_dist = self._maybe_average_deepgp_latent_distribution(latent_dist, Xt)
            if self._numel_from_dist_mean(latent_dist) == expected_numel:
                return self._as_distribution(latent_dist)
        except Exception:
            return None

        return None

    def _call_model_latent_posterior_raw(
        self,
        X: Tensor,
        Xt: Tensor,
        expected_numel: int,
    ):
        """
        wrapper model の latent posterior accessor を raw X で呼ぶ。

        注意:
            ここで input_transform 済みの Xt を渡してはいけない。
            Xt を渡すと InputPerturbation が二重に適用される可能性がある。
        """
        for name in ("latent_posterior", "posterior_latent", "posterior_f"):
            fn = getattr(self.model, name, None)
            if not callable(fn):
                continue
            try:
                latent_post = fn(X)
                latent_dist = self._as_distribution(latent_post)
                latent_dist = self._maybe_average_deepgp_latent_distribution(latent_dist, Xt)
                if self._numel_from_dist_mean(latent_dist) == expected_numel:
                    return self._as_distribution(latent_dist)
            except Exception:
                continue
        return None

    def _call_inner_latent_model(
        self,
        X: Tensor,
        Xt: Tensor,
        expected_numel: int,
    ):
        """
        fallback として inner latent model を直接呼ぶ。

        - inner model が input_transform を持つ場合: raw X を渡す
        - inner model が input_transform を持たない場合: transform 済み Xt を渡す

        q-batch の共分散を保つため、まずは flatten せずに呼ぶ。
        それで合わない場合のみ flatten を試す。
        """
        latent_gp = self._get_latent_gp()
        has_inner_transform = getattr(latent_gp, "input_transform", None) is not None

        candidates = []
        if has_inner_transform:
            candidates.append(X)
            candidates.append(X.reshape(-1, X.shape[-1]))
        else:
            candidates.append(Xt)
            candidates.append(Xt.reshape(-1, Xt.shape[-1]))

        for X_eval in candidates:
            try:
                latent_dist = latent_gp(X_eval)
                latent_dist = self._maybe_average_deepgp_latent_distribution(latent_dist, Xt)
                if self._numel_from_dist_mean(latent_dist) == expected_numel:
                    return self._as_distribution(latent_dist)
            except Exception:
                continue

        return None

    def _get_latent_dist_and_orig(self, X: Tensor):
        """
        pointwise / joint 共通の latent distribution 取得。

        Returns:
            latent_dist:
                gpytorch distribution
            orig:
                latent output を reshape する shape。
                InputPerturbation ありなら (*batch, q*n_w)
            Xt:
                input_transform 適用後の X。
                pending penalty / ROI / 距離計算用。

        方針:
            - Xt = input_transform(X) は shape 推定・penalty 用に使う
            - model-level latent posterior accessor には raw X を渡す
            - forward(X, apply_input_transform=True) を持つモデルでは forward を優先する
            - Xt を model.latent_posterior(Xt) に渡さない
        """
        X = self._ensure_q_batch(X)

        Xt = self._apply_input_transform(X)
        if isinstance(Xt, list):
            Xt = Xt[0]

        orig = Xt.shape[:-1]
        expected_numel = int(math.prod(orig))

        # 1. DeepGP wrapper など、forward(..., apply_input_transform=True) があるモデルを優先
        latent_dist = self._call_model_forward_latent(
            X=X,
            Xt=Xt,
            expected_numel=expected_numel,
        )
        if latent_dist is not None:
            return latent_dist, orig, Xt

        # 2. 修正済み latent_posterior / posterior_latent / posterior_f を raw X で呼ぶ
        latent_dist = self._call_model_latent_posterior_raw(
            X=X,
            Xt=Xt,
            expected_numel=expected_numel,
        )
        if latent_dist is not None:
            return latent_dist, orig, Xt

        # 3. fallback: inner latent GP を直接呼ぶ
        latent_dist = self._call_inner_latent_model(
            X=X,
            Xt=Xt,
            expected_numel=expected_numel,
        )
        if latent_dist is not None:
            return latent_dist, orig, Xt

        raise RuntimeError(
            "Unexpected latent output shape in _get_latent_dist_and_orig. "
            f"X.shape={tuple(X.shape)}, "
            f"Xt.shape={tuple(Xt.shape)}, "
            f"expected latent mean numel={expected_numel}. "
            "InputTransform may still be applied twice. "
            "Do not pass transformed Xt to a model that already owns input_transform."
        )

    def _reshape_pointwise_tensor(self, t: Tensor, orig: torch.Size) -> Tensor:
        """
        t を (*orig) に整形する。
        例:
          (B*q,) / (B*q,1) / (B*q,1,1) などを許容。
        """
        expected = math.prod(orig)
        if t.numel() != expected:
            raise RuntimeError(
                f"Unexpected tensor shape: got {tuple(t.shape)}, "
                f"numel={t.numel()}, expected={expected} for orig={tuple(orig)}"
            )
        return t.reshape(*orig)

    def _reduce_q(
        self,
        score_per_point: Tensor,
        reduction: Optional[ReductionType] = None,
    ) -> Tensor:
        """
        score_per_point: (*batch, q)
        return: (*batch,)
        """
        reduction = self.reduction if reduction is None else reduction

        if reduction == "mean":
            return score_per_point.mean(dim=-1)
        if reduction == "sum":
            return score_per_point.sum(dim=-1)
        if reduction == "max":
            return score_per_point.max(dim=-1).values
        raise ValueError(f"Unknown reduction: {reduction}")

    def _check_output_shape(self, out: Tensor, expected: torch.Size, name: str) -> None:
        """獲得関数の出力 shape を検証する。"""
        if out.shape != expected:
            raise RuntimeError(
                f"{name} output shape mismatch: expected {tuple(expected)}, got {tuple(out.shape)}"
            )

    def _squeeze_last_output_dim(self, x: Tensor) -> Tensor:
        if x.ndim >= 1 and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x

    def _transform_inputs_for_latent(self, X: Tensor) -> Tensor:
        """
        latent GP に入れるための入力へ変換する。
        """
        return self._apply_input_transform(X)

    # def _latent_stats_and_mean_prob(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    #     """
    #     latent mean / latent variance / predictive mean probability / Xt を返す。

    #     Returns:
    #         mu_f:      (*batch, q)
    #         var_f:     (*batch, q)
    #         mean_prob: (*batch, q)
    #         Xt:        (*batch, q, d)
    #     """
    #     self._prepare_eval()

    #     Xt = self._transform_inputs_for_latent(X)
    #     latent_dist = self.model.model(Xt)
    #     pred_dist = self.model.likelihood(latent_dist)

    #     mu_f = self._squeeze_last_output_dim(latent_dist.mean)
    #     var_f = self._squeeze_last_output_dim(latent_dist.variance).clamp_min(self.eps)
    #     mean_prob = self._squeeze_last_output_dim(pred_dist.mean).clamp(self.eps, 1.0 - self.eps)

    #     return mu_f, var_f, mean_prob, Xt
    def _latent_mean_and_cov(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """
        latent posterior の mean / covariance を返す。

        InputPerturbation ありの場合は、_get_latent_dist_and_orig が返す orig に合わせて
        q_like = q*n_w の形を期待する。joint 系で q に戻したい場合は、各 acquisition 側で
        q*n_w -> q の block reduction を行う。
        """
        latent_dist, orig, _ = self._get_latent_dist_and_orig(X)

        mu_raw = latent_dist.mean
        cov_raw = latent_dist.covariance_matrix

        if mu_raw.ndim >= 1 and mu_raw.shape[-1] == 1:
            mu_raw = mu_raw.squeeze(-1)

        q_like = int(orig[-1])
        batch_shape = torch.Size(orig[:-1])
        expected_mean = torch.Size(orig)
        expected_cov = batch_shape + torch.Size([q_like, q_like])

        # DeepGP sample dimension 付き: (S, *batch, q_like), (S, *batch, q_like, q_like)
        if (
            mu_raw.ndim >= len(expected_mean) + 1
            and tuple(mu_raw.shape[-len(expected_mean):]) == tuple(expected_mean)
            and cov_raw.ndim >= len(expected_cov) + 1
            and tuple(cov_raw.shape[-len(expected_cov):]) == tuple(expected_cov)
            and tuple(mu_raw.shape) != tuple(expected_mean)
        ):
            mu = mu_raw.mean(dim=0)
            second_moment = cov_raw + mu_raw.unsqueeze(-1) * mu_raw.unsqueeze(-2)
            cov = second_moment.mean(dim=0) - mu.unsqueeze(-1) * mu.unsqueeze(-2)
            return mu, cov

        if tuple(mu_raw.shape) == tuple(expected_mean) and tuple(cov_raw.shape) == tuple(expected_cov):
            return mu_raw, cov_raw

        if mu_raw.numel() == math.prod(expected_mean) and cov_raw.numel() == math.prod(expected_cov):
            return mu_raw.reshape(*expected_mean), cov_raw.reshape(*expected_cov)

        raise RuntimeError(
            f"Unexpected latent posterior shapes: "
            f"X.shape={tuple(X.shape)}, "
            f"orig={tuple(orig)}, "
            f"mean.shape={tuple(mu_raw.shape)}, "
            f"cov.shape={tuple(cov_raw.shape)}"
        )

    # =========================================================
    # pending penalty
    # =========================================================
    def _get_pending_in_feature_space(self) -> Optional[Tensor]:
        """
        X_pending を現在の X と同じ feature space に写す。
        """
        Xp = getattr(self, "X_pending", None)
        if Xp is None or Xp.numel() == 0:
            return None
        return self._apply_input_transform(Xp)

    def _pending_penalty_per_point(self, X: Tensor) -> Tensor:
        """
        X: (*batch, q, d)  ※ feature space 済みを想定
        return: (*batch, q)
        """
        if self.pending_penalty_weight <= 0:
            return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        Xp = self._get_pending_in_feature_space()
        if Xp is None or Xp.numel() == 0:
            return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        d = X.shape[-1]
        X2d = X.reshape(-1, d)           # (B_total*q, d)
        Xp2d = Xp.reshape(-1, d)         # (N_pending, d)

        dists = torch.cdist(X2d, Xp2d)   # (B_total*q, N_pending)
        min_dist = dists.min(dim=-1).values.reshape(*X.shape[:-1])  # (*batch, q)

        penalty = self.pending_penalty_weight * torch.exp(
            -self.pending_penalty_beta * min_dist
        )
        return penalty

    def _pending_penalty_aggregated(
        self,
        X: Tensor,
        reduction: Optional[ReductionType] = None,
    ) -> Tensor:
        """
        pending penalty を q 方向に集約して返す。

        Args:
            X: (*batch, q, d)  ※ feature space 済みを想定
            reduction: "mean" / "sum" / "max"
                None の場合は self.reduction を使う

        Returns:
            (*batch,)
        """
        penalty_per_point = self._pending_penalty_per_point(X)
        return self._reduce_q(penalty_per_point, reduction=reduction)

    # =========================================================
    # ROI helpers
    # =========================================================
    def _roi_weight_from_mean_prob(
        self,
        mean_prob: Tensor,
        X: Optional[Tensor] = None,
    ) -> Tensor:
        """
        mean_prob: (*batch, q)
        return:    (*batch, q)
        """
        if self.roi_mode == "none":
            return torch.ones_like(mean_prob)

        if self.roi_mode == "prob_above":
            raw = torch.sigmoid(self.roi_beta * (mean_prob - self.roi_threshold))

        elif self.roi_mode == "prob_below":
            raw = torch.sigmoid(self.roi_beta * (self.roi_threshold - mean_prob))

        elif self.roi_mode == "target_prob":
            bw = max(self.roi_bandwidth, 1e-12)
            raw = torch.exp(-0.5 * ((mean_prob - self.roi_target_prob) / bw) ** 2)

        elif self.roi_mode == "interval":
            if self.roi_interval is None:
                raise ValueError("roi_interval must be set when roi_mode='interval'.")
            low, high = self.roi_interval
            if low > high:
                raise ValueError(f"roi_interval must satisfy low <= high, got {(low, high)}.")
            raw = (
                torch.sigmoid(self.roi_beta * (mean_prob - low))
                * torch.sigmoid(self.roi_beta * (high - mean_prob))
            )

        elif self.roi_mode == "custom":
            if self.roi_weight_fn is None:
                raise ValueError("roi_weight_fn must be provided when roi_mode='custom'.")
            raw = self.roi_weight_fn(mean_prob, X)

        else:
            raise ValueError(f"Unknown roi_mode: {self.roi_mode}")

        raw = raw.clamp_min(0.0)
        weight = self.roi_min_weight + (1.0 - self.roi_min_weight) * raw
        return weight

    def _apply_roi_weight_per_point(
        self,
        score: Tensor,
        mean_prob: Tensor,
        X: Optional[Tensor] = None,
    ) -> Tensor:
        """
        pointwise score (*batch, q) に ROI weight を適用する。
        """
        weight = self._roi_weight_from_mean_prob(mean_prob, X=X)

        if self.roi_combine == "multiply":
            return score * weight

        if self.roi_combine == "add":
            return score + self.roi_weight_scale * weight

        raise ValueError(f"Unknown roi_combine: {self.roi_combine}")

    def _aggregate_roi_weight(
        self,
        mean_prob: Tensor,
        X: Optional[Tensor] = None,
    ) -> Tensor:
        """
        pointwise ROI weight (*batch, q) を joint score 用に集約する。
        return: (*batch,)
        """
        weight = self._roi_weight_from_mean_prob(mean_prob, X=X)
        return self._reduce_q(weight, reduction=self.roi_aggregate_reduction)

    def _apply_roi_weight_aggregated(
        self,
        score: Tensor,
        mean_prob: Tensor,
        X: Optional[Tensor] = None,
    ) -> Tensor:
        """
        joint / aggregated score (*batch,) に ROI weight を適用する。
        """
        weight = self._aggregate_roi_weight(mean_prob, X=X)

        if self.roi_combine == "multiply":
            return score * weight

        if self.roi_combine == "add":
            return score + self.roi_weight_scale * weight

        raise ValueError(f"Unknown roi_combine: {self.roi_combine}")

    # =========================================================
    # hetero noise helpers
    # =========================================================
    def _get_noise_eval_inputs(self, X: Tensor) -> Tensor:
        """
        noise_model に入れる入力を作る。
        """
        if isinstance(X, tuple):
            X = X[0]

        uses_transformed = getattr(self.model, "noise_model_uses_transformed_inputs", False)
        if uses_transformed:
            return self._transform_inputs_for_latent(X)

        noise_tf = getattr(self.model, "noise_input_transform", None)
        if noise_tf is not None:
            noise_tf.eval()
            return noise_tf(self._ensure_q_batch(X))

        return self._ensure_q_batch(X)

    def _pointwise_noise_var(
        self,
        X: Tensor,
    ) -> Tensor:
        """
        noise_model から予測ノイズ分散を返す。

        Returns:
            noise_var: (*batch, q)
        """
        noise_model = getattr(self.model, "noise_model", None)
        if noise_model is None:
            return torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)

        Xn = self._get_noise_eval_inputs(X)
        noise_post = noise_model.posterior(Xn)
        noise_pred = self._squeeze_last_output_dim(noise_post.mean)

        if self.noise_model_outputs_log_var:
            noise_var = noise_pred.exp()
        else:
            noise_var = noise_pred

        return noise_var.clamp_min(self.eps)

    def _noise_weight_from_var(
        self,
        noise_var: Tensor,
        X: Optional[Tensor] = None,
    ) -> Tensor:
        """
        noise_var: (*batch, q)
        return:    (*batch, q)
        """
        if self.noise_mode == "none":
            return torch.ones_like(noise_var)

        lam = self.noise_penalty_lambda

        if self.noise_mode == "inverse_linear":
            raw = 1.0 / (1.0 + lam * noise_var)

        elif self.noise_mode == "exp":
            raw = torch.exp(-lam * noise_var)

        elif self.noise_mode == "custom":
            if self.noise_weight_fn is None:
                raise ValueError("noise_weight_fn must be provided when noise_mode='custom'.")
            raw = self.noise_weight_fn(noise_var, X)

        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

        raw = raw.clamp_min(0.0)
        weight = self.noise_min_weight + (1.0 - self.noise_min_weight) * raw
        return weight

    def _apply_noise_weight_per_point(
        self,
        score: Tensor,
        X: Tensor,
    ) -> Tensor:
        """
        pointwise score (*batch, q) に noise penalty を適用する。
        """
        noise_var = self._pointwise_noise_var(X)
        weight = self._noise_weight_from_var(noise_var, X=X)

        if self.noise_combine == "multiply":
            return score * weight

        if self.noise_combine == "subtract":
            return score - self.noise_weight_scale * noise_var

        raise ValueError(f"Unknown noise_combine: {self.noise_combine}")

    # =========================================================
    # BALD / qBALD 用 helper
    # =========================================================
    @staticmethod
    def _binary_entropy(p: Tensor, eps: float = 1e-6) -> Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return -(p * p.log() + (1.0 - p) * (1.0 - p).log())

    @staticmethod
    def _make_binary_patterns(q: int, device, dtype) -> Tensor:
        """
        shape: (2^q, q)
        各行が 0/1 のラベルパターン。
        """
        ids = torch.arange(2**q, device=device, dtype=torch.long)
        bitpos = torch.arange(q, device=device, dtype=torch.long)
        patterns = ((ids[:, None] >> bitpos[None, :]) & 1).to(dtype)
        return patterns

    def _pointwise_latent_probs(
        self,
        X: Tensor,
        num_samples: int,
    ) -> tuple[Tensor, torch.Size, Tensor]:
        """
        pointwise latent sample -> sigmoid(prob) へ変換する。

        Returns:
            probs: (S, *batch, q)
            orig:  (*batch, q)
            Xt:    transformed input
        """
        latent_dist, orig, Xt = self._get_latent_dist_and_orig(X)
        f_samples = latent_dist.rsample(torch.Size([num_samples]))

        expected = num_samples * math.prod(orig)
        if f_samples.numel() != expected:
            raise RuntimeError(
                f"Unexpected tensor shape: got {tuple(f_samples.shape)}, "
                f"numel={f_samples.numel()}, expected={expected} "
                f"for shape={(num_samples, *orig)}"
            )

        f_samples = f_samples.reshape(num_samples, *orig)
        probs = torch.sigmoid(f_samples).clamp(self.eps, 1.0 - self.eps)
        return probs, orig, Xt

    def _get_joint_latent_dist(self, X: Tensor):
        """
        joint 用 latent_dist を取得する。

        _get_latent_dist_and_orig と同じ経路を使うことで、InputPerturbation の
        二重適用を避ける。可能な場合は q-batch の共分散を保持する。

        Returns:
            latent_dist
            batch_shape: Xt.shape[:-2]
            q_like: Xt.shape[-2]  # q または q*n_w
            Xt
        """
        latent_dist, orig, Xt = self._get_latent_dist_and_orig(X)
        batch_shape = torch.Size(orig[:-1])
        q_like = int(orig[-1])
        return latent_dist, batch_shape, q_like, Xt

    def _reshape_joint_tensor(
        self,
        t: Tensor,
        batch_shape: torch.Size,
        q: int,
        num_samples: int,
    ) -> Tensor:
        """
        t を (S, *batch_shape, q) に整形する。

        例:
          (S, *batch, q)
          (S, *batch, q, 1)
          (S, *batch, q, 1, 1)
        などを許容。
        """
        expected = num_samples * math.prod(batch_shape) * q
        if t.numel() != expected:
            raise RuntimeError(
                f"Unexpected joint tensor shape: got {tuple(t.shape)}, "
                f"numel={t.numel()}, expected={expected}, "
                f"batch_shape={tuple(batch_shape)}, q={q}, num_samples={num_samples}"
            )
        return t.reshape(num_samples, *batch_shape, q)

    def _joint_latent_probs(
        self,
        X: Tensor,
        num_samples: int,
    ) -> tuple[Tensor, torch.Size, int, Tensor]:
        """
        joint latent sample -> sigmoid(prob) へ変換する。

        Returns:
            probs: (S, *batch, q)
            batch_shape: (*batch,)
            q: joint 点数
            Xt: transformed input
        """
        latent_dist, batch_shape, q, Xt = self._get_joint_latent_dist(X)
        f_samples = latent_dist.rsample(torch.Size([num_samples]))
        f_samples = self._reshape_joint_tensor(
            f_samples,
            batch_shape=batch_shape,
            q=q,
            num_samples=num_samples,
        )
        probs = torch.sigmoid(f_samples).clamp(self.eps, 1.0 - self.eps)
        return probs, batch_shape, q, Xt

    def _joint_predictive_entropy_binary(
        self,
        probs: Tensor,
        *,
        max_joint_q: int,
        large_q_strategy: LargeQStrategy = "raise",
    ) -> Tensor:
        """
        probs: (S, *batch, q)
        return: (*batch,)

        H[p(y_1:q | X, D)] を MC mixture で近似。
        """
        q = probs.shape[-1]
        if q > max_joint_q:
            if large_q_strategy == "per_point":
                mean_prob = probs.mean(dim=0)  # (*batch, q)
                return self._binary_entropy(mean_prob, self.eps).sum(dim=-1)

            raise ValueError(
                f"q={q} is too large for exact joint enumeration "
                f"(max_joint_q={max_joint_q})."
            )

        num_samples = probs.shape[0]
        patterns = self._make_binary_patterns(q, probs.device, probs.dtype)  # (M, q)
        num_patterns = patterns.shape[0]

        log_p = probs.clamp(self.eps, 1.0 - self.eps).log()
        log_1mp = (1.0 - probs.clamp(self.eps, 1.0 - self.eps)).log()

        pattern_view = patterns.view(*([1] * (probs.ndim - 1)), num_patterns, q)

        log_comp = (
            log_p.unsqueeze(-2) * pattern_view
            + log_1mp.unsqueeze(-2) * (1.0 - pattern_view)
        ).sum(dim=-1)  # (S, *batch, M)

        log_mix = torch.logsumexp(log_comp, dim=0) - math.log(num_samples)  # (*batch, M)
        mix = log_mix.exp().clamp_min(self.eps)

        joint_entropy = -(mix * log_mix).sum(dim=-1)  # (*batch,)
        return joint_entropy

    def _conditional_entropy_given_w(self, probs: Tensor) -> Tensor:
        """
        probs: (S, *batch, q)
        return: (*batch,)

        E_w[ H[p(y_1:q | X, w)] ] を計算。
        binary 条件付きでは各点は独立 Bernoulli とみなし和を取る。
        """
        ent_each = self._binary_entropy(probs, self.eps)  # (S, *batch, q)
        return ent_each.sum(dim=-1).mean(dim=0)

    def _apply_objective_to_score(
        self,
        score: Tensor,
        X: Optional[Tensor] = None,
        name: str = "ClassificationAcquisition",
    ) -> Tensor:
        """
        classification acquisition の score に objective を適用する。

        回帰系 objective と異なり、posterior samples ではなく
        entropy / uncertainty / BALD / straddle / PoF などから計算済みの
        score に作用する。
        """
        objective = getattr(self, "objective", None)
        if objective is None:
            return score

        try:
            out = objective(score, X=X)
        except TypeError:
            out = objective(score)

        if not torch.is_tensor(out):
            raise RuntimeError(
                f"{name}: objective must return a Tensor. Got {type(out)}."
            )

        return out