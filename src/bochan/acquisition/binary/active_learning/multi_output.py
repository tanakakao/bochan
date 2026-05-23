from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.utils.transforms import t_batch_mode_transform


ReductionType = Literal["mean", "sum"]
UncertaintyScoreType = Literal["variance", "entropy", "least_confidence"]
MultiOutputMode = Literal["mean", "sum", "max", "min", "weighted_mean", "all_positive"]

NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "exp"]
NoiseCombineType = Literal["multiply", "add"]
EventNoiseAggregateType = Literal["product", "mean", "sum", "max", "min", "weighted_mean"]


class _StackedPosterior:
    """
    ModelList / model.models から得た複数 posterior を multi-output 風に束ねる簡易 posterior。

    mean / variance:
        (*batch, q_like, m)

    rsample(sample_shape):
        sample_shape x *batch x q_like x m
    """

    def __init__(self, posteriors: list) -> None:
        if len(posteriors) == 0:
            raise ValueError("At least one posterior is required.")

        self.posteriors = posteriors
        self._mean = torch.cat([self._ensure_last_output_dim(p.mean) for p in posteriors], dim=-1)

        vars_ = []
        for p in posteriors:
            if hasattr(p, "variance"):
                v = p.variance
            else:
                dist = getattr(p, "distribution", None)
                if dist is None or not hasattr(dist, "variance"):
                    raise AttributeError("Could not extract variance from posterior.")
                v = dist.variance
            vars_.append(self._ensure_last_output_dim(v))
        self._variance = torch.cat(vars_, dim=-1)

    @staticmethod
    def _ensure_last_output_dim(x: Tensor) -> Tensor:
        if x.ndim == 0:
            return x.view(1, 1)
        if x.ndim >= 1 and x.shape[-1] == 1:
            return x
        # single-output posterior が (..., q_like) で返る場合
        return x.unsqueeze(-1)

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        samples = []
        for p in self.posteriors:
            s = p.rsample(sample_shape)
            samples.append(self._ensure_last_output_dim(s))
        return torch.cat(samples, dim=-1)


class _MultiOutputBinaryClassificationAcqBase(AcquisitionFunction):
    """
    多出力 binary classification 用獲得関数の共通ベース。

    想定:
      - probability_posterior(X).mean が (*batch, q_like, m) を返す
      - InputPerturbation 使用時は q_like = q * n_w になる
      - objective は acquisition が計算した pointwise score に作用する
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(model)
        self.reduction = reduction
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.eps = float(eps)
        self.objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None
        self.set_X_pending(None)

    # =========================================================
    # pending utilities
    # =========================================================

    def _coerce_pending_to_tensor(
        self,
        X_pending,
        *,
        ref: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """X_pending を Tensor または None に正規化する。"""
        if X_pending is None:
            return None
        if torch.is_tensor(X_pending):
            out = X_pending
        elif isinstance(X_pending, (list, tuple)):
            tensors = []
            for item in X_pending:
                if item is None:
                    continue
                t = self._coerce_pending_to_tensor(item, ref=ref)
                if t is not None and t.numel() > 0:
                    tensors.append(t)
            if len(tensors) == 0:
                return None
            if len(tensors) == 1:
                out = tensors[0]
            else:
                try:
                    out = torch.cat(tensors, dim=-2)
                except RuntimeError:
                    out = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=-2)
        else:
            raise TypeError(
                "X_pending must be None, Tensor, list, or tuple. "
                f"Got {type(X_pending)}."
            )
        if ref is not None:
            out = out.to(device=ref.device, dtype=ref.dtype)

        # X_pending は acquisition optimization 中の定数として扱う。
        return out.detach()

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        """pending points を raw input space の値として保持する。"""
        self.X_pending = self._coerce_pending_to_tensor(X_pending)

    def _transform_pending_like_candidate(
        self,
        X_pending,
        *,
        ref: Tensor,
    ) -> Optional[Tensor]:
        """X_pending を candidate と同じ距離計算空間へ写す。"""
        Xp = self._coerce_pending_to_tensor(X_pending, ref=ref)
        if Xp is None or Xp.numel() == 0:
            return None
        Xp_t = self._apply_input_transform(Xp)
        Xp_t = self._ensure_q_batch(Xp_t)
        return Xp_t.to(device=ref.device, dtype=ref.dtype)


    # =========================================================
    # objective
    # =========================================================
    def _set_multioutput_classification_objective(
        self,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        self.objective = objective

    @staticmethod
    def _is_classification_score_objective(objective) -> bool:
        """
        自作 ClassificationScoreObjective / MultiOutputClassificationScoreObjective を
        ゆるく判定する。

        注意:
            MultiOutputClassificationInputPerturbationObjective は
            MCMultiOutputObjective を継承する qEHVI / qNEHVI 用 objective。
            class 名や module 名に classification が含まれていても、
            score objective として扱ってはいけない。
        """
        if isinstance(objective, MCMultiOutputObjective):
            return False

        cls_name = objective.__class__.__name__
        module_name = objective.__class__.__module__
        return (
            cls_name in {
                "ClassificationScoreObjective",
                "MultiOutputClassificationScoreObjective",
            }
            or (
                "classification" in module_name
                and hasattr(objective, "n_w")
                and hasattr(objective, "risk_type")
            )
        )

    def _apply_objective_to_pointwise_score(
        self,
        score: Tensor,
        *,
        raw_X: Tensor,
        expanded_X: Tensor,
        name: str,
    ) -> Tensor:
        """
        multi-output classification の pointwise score に objective を適用する。

        Args:
            score:
                output 集約後の pointwise score。
                典型 shape:
                    (*batch, q_like)
                ここで q_like は q または q * n_w。

            raw_X:
                optimize_acqf から渡された元の X。
                shape:
                    (*batch, q, d)

            expanded_X:
                input_transform 後の X。
                shape:
                    (*batch, q_like, d)

        Returns:
            objective=None:
                score をそのまま返す。
            objectiveあり:
                InputPerturbation 集約後の shape (*batch, q) などを返す。
        """
        objective = getattr(self, "objective", None)
        if objective is None:
            return score

        # ------------------------------------------------------------
        # 1. classification score objective:
        #    score は (*batch, q_like) または (*batch, q_like, m) のまま渡す。
        # ------------------------------------------------------------
        if self._is_classification_score_objective(objective):
            try:
                out = objective(score, X=raw_X)
            except TypeError:
                out = objective(score)

            if not torch.is_tensor(out):
                raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

            if out.ndim == raw_X.ndim and out.shape[-1] == 1:
                out = out.squeeze(-1)

            return out

        # ------------------------------------------------------------
        # 2. BoTorch MCMultiOutputObjective:
        #    samples 風に (*batch, q_like, m) として渡す。
        #
        #    例:
        #        score.shape = (*batch, q * n_w)
        #        -> score_in.shape = (*batch, q * n_w, 1)
        #
        #    MultiOutputClassificationInputPerturbationObjective はここで
        #    q * n_w -> q に戻す。
        # ------------------------------------------------------------
        score_in = score
        if isinstance(objective, MCMultiOutputObjective) and score_in.ndim == expanded_X.ndim - 1:
            score_in = score_in.unsqueeze(-1)

        try:
            out = objective(score_in, X=raw_X)
        except TypeError:
            out = objective(score_in)

        if not torch.is_tensor(out):
            raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

        if out.ndim == raw_X.ndim and out.shape[-1] == 1:
            out = out.squeeze(-1)

        return out

    # =========================================================
    # basic utilities
    # =========================================================
    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        if X.ndim == 2:
            X = X.unsqueeze(-2)
        return X

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        """
        acquisition 側で shape 整合・pending penalty に使う X を返す。

        MultiOutputClassificationModel 自体には input_transform が無いことがある。
        その場合でも各 submodel が InputPerturbation を持っていれば、
        posterior.mean は q * n_w に展開される。

        したがって、model.input_transform が無い場合は submodel 側の
        input_transform を fallback として使い、Xt.shape[-2] を posterior.mean の
        q_like と一致させる。
        """
        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return Xt

        # MultiOutputClassificationModel / ModelList など
        # model.models に各 single-output classifier が入っている場合。
        models = getattr(self.model, "models", None)
        if models is not None and len(models) > 0:
            first_model = models[0]
            it = getattr(first_model, "input_transform", None)
            if it is not None:
                Xt = it(X)
                if isinstance(Xt, tuple):
                    Xt = Xt[0]
                return Xt

        return X

    def _set_eval_mode(self) -> None:
        self.model.eval()
        like = getattr(self.model, "likelihood", None)
        if like is not None:
            like.eval()

    @staticmethod
    def _binary_entropy(p: Tensor, eps: float = 1e-6) -> Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return -(p * p.log() + (1.0 - p) * (1.0 - p).log())

    def _check_output_shape(self, out: Tensor, expected: torch.Size, name: str) -> None:
        if out.shape != expected:
            raise RuntimeError(
                f"{name} output shape mismatch: expected {tuple(expected)}, got {tuple(out.shape)}"
            )

    # =========================================================
    # posterior helpers
    # =========================================================
    def _get_latent_posterior(self, X: Tensor):
        """
        latent f scale の posterior を取得する。

        優先順:
          1. model.latent_posterior(X)
          2. model.models の各 submodel.latent_posterior(X) を stack
          3. fallback: model.posterior(X)

        BALD では probability posterior の rsample より、latent posterior から
        sample して sigmoid する方が binary classification model の設計と合う。
        """
        fn = getattr(self.model, "latent_posterior", None)
        if callable(fn):
            return fn(X)

        if hasattr(self.model, "models"):
            posts = []
            for submodel in self.model.models:
                for name in ("latent_posterior", "posterior_latent", "posterior_f"):
                    sub_fn = getattr(submodel, name, None)
                    if callable(sub_fn):
                        posts.append(sub_fn(X))
                        break
                else:
                    # fallback: posterior が latent を返す古い wrapper 用
                    posts.append(submodel.posterior(X))
            return _StackedPosterior(posts)

        return self.model.posterior(X)

    def _get_probability_posterior(self, X: Tensor):
        """
        probability scale の posterior を取得する。

        優先順:
          1. model.probability_posterior(X)
          2. model.models の各 submodel.posterior(X) を stack
          3. model.posterior(X)
        """
        fn = getattr(self.model, "probability_posterior", None)
        if callable(fn):
            return fn(X)

        if hasattr(self.model, "models"):
            posts = []
            for submodel in self.model.models:
                sub_fn = getattr(submodel, "probability_posterior", None)
                if callable(sub_fn):
                    posts.append(sub_fn(X))
                else:
                    posts.append(submodel.posterior(X))
            return _StackedPosterior(posts)

        return self.model.posterior(X)

    def _normalize_mean_shape(self, mean: Tensor, X: Tensor) -> Tensor:
        """
        posterior.mean を (*batch, q_like, m) に正規化する。
        single-output 相当なら末尾に m=1 を追加する。

        注意:
            InputPerturbation 使用時は X に expanded_X を渡す。
            raw_X を渡すと q*n_w が output dim と誤認される。
        """
        X = self._ensure_q_batch(X)
        expected_prefix = X.shape[:-1]  # (*batch, q_like)
        n_points = math.prod(expected_prefix)

        # (*batch, q_like) -> (*batch, q_like, 1)
        if mean.shape == expected_prefix:
            return mean.unsqueeze(-1)

        # (*batch, q_like, m)
        if mean.ndim == X.ndim and mean.shape[:-1] == expected_prefix:
            return mean

        # flatten されていても総要素数から復元
        if mean.numel() % n_points == 0:
            m = mean.numel() // n_points
            if m <= 0:
                raise RuntimeError(
                    f"Invalid inferred output dimension m={m} from mean shape {tuple(mean.shape)}."
                )
            return mean.reshape(*expected_prefix, m)

        raise RuntimeError(
            "Unsupported posterior.mean shape for multi-output binary classification: "
            f"X.shape={tuple(X.shape)}, posterior.mean.shape={tuple(mean.shape)}"
        )

    def _to_probability(
        self,
        x: Tensor,
        *,
        apply_sigmoid_if_needed: bool,
        name: str,
    ) -> Tensor:
        xmin = x.min().item()
        xmax = x.max().item()

        if 0.0 <= xmin and xmax <= 1.0:
            return x.clamp(self.eps, 1.0 - self.eps)

        if apply_sigmoid_if_needed:
            return torch.sigmoid(x).clamp(self.eps, 1.0 - self.eps)

        raise RuntimeError(
            f"{name} is not in [0,1] (min={xmin:.4g}, max={xmax:.4g}). "
            "This acquisition assumes probability output. "
            "Either fix the classifier wrapper or enable sigmoid conversion."
        )

    def _reshape_samples(
        self,
        samples: Tensor,
        X: Tensor,
        num_samples: int,
    ) -> Tensor:
        """
        posterior.rsample(...) を (S, *batch, q_like, m) に整形する。

        注意:
            InputPerturbation 使用時は X に expanded_X を渡す。
        """
        X = self._ensure_q_batch(X)
        expected_prefix = X.shape[:-1]  # (*batch, q_like)
        n_points = math.prod(expected_prefix)
        expected_base = num_samples * n_points

        if samples.numel() % expected_base != 0:
            raise RuntimeError(
                f"Unexpected sample shape: got {tuple(samples.shape)}, "
                f"numel={samples.numel()}, expected a multiple of {expected_base} "
                f"(num_samples={num_samples}, X_prefix={tuple(expected_prefix)})."
            )

        m = samples.numel() // expected_base
        if m <= 0:
            raise RuntimeError(f"Invalid inferred output dimension m={m} from samples.")

        return samples.reshape(num_samples, *expected_prefix, m)

    # =========================================================
    # pending penalty / q reduction
    # =========================================================

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        """
        pending points に近い候補点へ pointwise penalty を与える。

        Args:
            Xt:
                候補点。すでに `_apply_input_transform(raw_X)` を通した
                距離計算用 Tensor。shape は `(*batch, q_like, d)`。

        Returns:
            Tensor:
                pending penalty。shape は `(*batch, q_like)`。
        """
        Xt = self._ensure_q_batch(Xt)

        if self.pending_penalty_weight <= 0.0:
            return torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)

        Xp_t = self._transform_pending_like_candidate(
            getattr(self, "X_pending", None),
            ref=Xt,
        )
        if Xp_t is None or Xp_t.numel() == 0:
            return torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)

        d = Xt.shape[-1]
        X2d = Xt.reshape(-1, d)
        Xp2d = Xp_t.reshape(-1, Xp_t.shape[-1])

        if Xp2d.shape[-1] != d:
            raise RuntimeError(
                "X_pending feature dimension mismatch in pending penalty after transform: "
                f"Xt.shape={tuple(Xt.shape)}, X_pending_transformed.shape={tuple(Xp_t.shape)}."
            )

        dists = torch.cdist(X2d, Xp2d)
        min_dist = dists.min(dim=-1).values.reshape(*Xt.shape[:-1])

        return self.pending_penalty_weight * torch.exp(
            -self.pending_penalty_beta * min_dist
        )

    def _reduce_q(self, score: Tensor) -> Tensor:
        """
        score: (*batch, q)
        return: (*batch,)
        """
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        raise ValueError(f"Unknown reduction: {self.reduction}")

    # =========================================================
    # multi-output aggregation
    # =========================================================
    def _aggregate_outputs(
        self,
        score_per_output: Tensor,
        *,
        output_mode: MultiOutputMode,
        output_weights: Optional[Tensor] = None,
        probs_for_all_positive: Optional[Tensor] = None,
        score_type_for_all_positive: Optional[UncertaintyScoreType] = None,
    ) -> Tensor:
        """
        score_per_output: (*batch, q_like, m)
        return: (*batch, q_like)
        """
        if output_mode == "mean":
            return score_per_output.mean(dim=-1)

        if output_mode == "sum":
            return score_per_output.sum(dim=-1)

        if output_mode == "max":
            return score_per_output.max(dim=-1).values

        if output_mode == "min":
            return score_per_output.min(dim=-1).values

        if output_mode == "weighted_mean":
            if output_weights is None:
                raise ValueError("output_weights must be provided when output_mode='weighted_mean'.")
            w = output_weights.to(device=score_per_output.device, dtype=score_per_output.dtype)
            if w.ndim != 1 or w.numel() != score_per_output.shape[-1]:
                raise ValueError(
                    f"output_weights must have shape ({score_per_output.shape[-1]},), got {tuple(w.shape)}."
                )
            w = w / w.sum().clamp_min(self.eps)
            view_shape = (1,) * (score_per_output.ndim - 1) + (w.numel(),)
            return (score_per_output * w.view(*view_shape)).sum(dim=-1)

        if output_mode == "all_positive":
            if probs_for_all_positive is None:
                raise ValueError("probs_for_all_positive must be provided for output_mode='all_positive'.")
            if score_type_for_all_positive is None:
                raise ValueError("score_type_for_all_positive must be provided for output_mode='all_positive'.")

            log_p_all = probs_for_all_positive.log().sum(dim=-1)
            p_all = log_p_all.exp().clamp(self.eps, 1.0 - self.eps)
            return self._uncertainty_score_binary_event(p_all, score_type_for_all_positive)

        raise ValueError(f"Unknown output_mode: {output_mode}")

    def _uncertainty_score_binary_event(
        self,
        p: Tensor,
        score_type: UncertaintyScoreType,
    ) -> Tensor:
        """
        p: 任意 shape の binary probability
        """
        if score_type == "variance":
            return p * (1.0 - p)
        if score_type == "entropy":
            return self._binary_entropy(p, self.eps)
        if score_type == "least_confidence":
            return 1.0 - torch.maximum(p, 1.0 - p)
        raise ValueError(f"Unknown score_type: {score_type}")


class _MultiOutputUncertaintySamplingClassifierAcquisition(_MultiOutputBinaryClassificationAcqBase):
    """
    多出力 2値分類用 uncertainty sampling acquisition。

    objective:
        acquisition score に作用する objective。
        InputPerturbation 集約には MultiOutputClassificationScoreObjective を渡す。
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        score_type: UncertaintyScoreType = "variance",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        apply_sigmoid_if_needed: bool = False,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.score_type = score_type
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        self._set_eval_mode()

        raw_X = X
        original_batch_shape = raw_X.shape[:-2]

        # posterior は raw_X で評価する。shape 整合と penalty には expanded_X を使う。
        Xt = self._apply_input_transform(raw_X)
        posterior = self._get_probability_posterior(raw_X)

        probs = self._normalize_mean_shape(posterior.mean, Xt)
        probs = self._to_probability(
            probs,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            name="probability_posterior.mean",
        )

        score_per_output = self._uncertainty_score_binary_event(probs, self.score_type)
        score = self._aggregate_outputs(
            score_per_output,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
            probs_for_all_positive=probs,
            score_type_for_all_positive=self.score_type,
        )  # (*batch, q_like)

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="MultiOutputUncertaintySampling",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "MultiOutputUncertaintySampling")
        return out


class _BALDMultiOutputAcquisition(_MultiOutputBinaryClassificationAcqBase):
    """
    多出力 2値分類用 BALD acquisition。

    objective:
        acquisition score に作用する objective。
        InputPerturbation 集約には MultiOutputClassificationScoreObjective を渡す。
    """

    def __init__(
        self,
        model,
        num_samples: int = 16,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "all_positive",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.num_samples = int(num_samples)
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self._set_multioutput_classification_objective(objective)

    def _event_bald(self, p: Tensor) -> Tensor:
        """
        p: (S, *batch, q_like)
        return: (*batch, q_like)
        """
        entropy_conditional = self._binary_entropy(p, self.eps).mean(dim=0)
        mean_prob = p.mean(dim=0)
        mean_entropy = self._binary_entropy(mean_prob, self.eps)
        return mean_entropy - entropy_conditional

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        self._set_eval_mode()

        raw_X = X
        original_batch_shape = raw_X.shape[:-2]

        Xt = self._apply_input_transform(raw_X)

        if self.samples_are_probs:
            posterior = self._get_probability_posterior(raw_X)
            samples = posterior.rsample(torch.Size([self.num_samples]))
            probs = self._reshape_samples(samples, Xt, self.num_samples)
            probs = self._to_probability(
                probs,
                apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
                name="probability_posterior.rsample()",
            )
        else:
            posterior = self._get_latent_posterior(raw_X)
            latent_samples = posterior.rsample(torch.Size([self.num_samples]))
            latent_samples = self._reshape_samples(latent_samples, Xt, self.num_samples)
            probs = torch.sigmoid(latent_samples).clamp(self.eps, 1.0 - self.eps)

        if self.output_mode == "all_positive":
            log_p_all = probs.log().sum(dim=-1)  # (S, *batch, q_like)
            p_all = log_p_all.exp().clamp(self.eps, 1.0 - self.eps)
            score = self._event_bald(p_all)  # (*batch, q_like)
        else:
            score_per_output = self._event_bald(probs)  # (*batch, q_like, m)
            score = self._aggregate_outputs(
                score_per_output,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
            )  # (*batch, q_like)

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="BALDMultiOutput",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "BALDMultiOutput")
        return out





class qMultiOutputBinaryPredictiveEntropy(_MultiOutputUncertaintySamplingClassifierAcquisition):
    """multi-output classification 用 predictive entropy acquisition。予測分布の曖昧さが大きい点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        予測が曖昧な点を探索したい場合の基本的な active learning acquisition です。
    """

    def __init__(self, model, *args, **kwargs) -> None:
        kwargs.pop("score_type", None)
        super().__init__(model, *args, score_type="entropy", **kwargs)


class qMultiOutputBinaryProbabilityVariance(_MultiOutputUncertaintySamplingClassifierAcquisition):
    """multi-output classification 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(self, model, *args, **kwargs) -> None:
        kwargs.pop("score_type", None)
        super().__init__(model, *args, score_type="variance", **kwargs)


class qMultiOutputBinaryMarginUncertainty(_MultiOutputUncertaintySamplingClassifierAcquisition):
    """multi-output classification 用 margin uncertainty acquisition。決定境界または class 境界に近い点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(self, model, *args, **kwargs) -> None:
        kwargs.pop("score_type", None)
        super().__init__(model, *args, score_type="least_confidence", **kwargs)


class qMultiOutputBinaryBALD(_BALDMultiOutputAcquisition):
    """multi-output classification 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        BALD は predictive entropy から条件付き entropy を引いた情報利得として解釈できます。
    """

    pass


class qMultiOutputBinaryIntegratedPosteriorVarianceProxy(qMultiOutputBinaryProbabilityVariance):
    """multi-output classification 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    pass

__all__ = [
    "qMultiOutputBinaryPredictiveEntropy",
    "qMultiOutputBinaryProbabilityVariance",
    "qMultiOutputBinaryMarginUncertainty",
    "qMultiOutputBinaryBALD",
    "qMultiOutputBinaryIntegratedPosteriorVarianceProxy",
]
