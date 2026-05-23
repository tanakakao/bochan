from __future__ import annotations

import math
from typing import Callable, List, Literal, Optional, Tuple

import torch
from botorch.models.model import ModelList
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.binary.base import (
    ReductionType,
    _BinaryClassificationAcqBase,
)


MultiOutputMode = Literal[
    "mean",
    "sum",
    "max",
    "min",
    "weighted_mean",
]

ROIWeightMode = Literal[
    "none",
    "threshold",
    "target_prob",
    "interval",
    "band",
]
ROICombineType = Literal["multiply", "add"]

NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "exp"]
NoiseCombineType = Literal["multiply", "add"]

RegionMode = Literal["independent", "all_positive", "any_positive"]


class _MultiOutputLatentStraddleBase(_BinaryClassificationAcqBase):
    """
    多出力 binary classification の latent straddle 系 acquisition 共通 base。

    対応する model 形態:
      1. MultiOutputClassificationModel のような wrapper
         - model.latent_posterior(X) が latent f の multi-output posterior を返す
         - model.probability_posterior(X) が P(y=1) の multi-output posterior を返す
      2. BoTorch ModelList / 独自 ModelList
         - model.models に single-output classification model 群を持つ
      3. single-output classification model
         - model.latent_posterior(X) / model.posterior(X) などを持つ

    重要:
      - wrapper の latent_posterior / probability_posterior には raw X を渡す。
        各 model 側で input_transform を適用する設計を想定するため。
      - inner latent GP を直接呼ぶ fallback の場合だけ、この acquisition 側で
        submodel.input_transform を適用する。
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        eps: float = 1e-6,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
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
        """
        X_pending を Tensor または None に正規化する。

        `optimize_acqf(sequential=True, q>1)` では、BoTorch 側から
        `set_X_pending(...)` が逐次呼ばれる。wrapper や acquisition の
        組み合わせによっては `X_pending` が Tensor ではなく list / tuple
        として入ることがあるため、pending penalty 計算前に Tensor に揃える。

        Args:
            X_pending:
                pending points。`None`, Tensor, list, tuple を許容する。
                list / tuple の中に Tensor が複数ある場合は、候補点方向
                `dim=-2` で結合する。
            ref:
                dtype / device を合わせるための参照 Tensor。

        Returns:
            Optional[Tensor]:
                Tensor に正規化された pending points。
                空の場合は None。
        """
        if X_pending is None:
            return None

        if torch.is_tensor(X_pending):
            out = X_pending

        elif isinstance(X_pending, (list, tuple)):
            tensors = []
            for item in X_pending:
                if item is None:
                    continue
                item_tensor = self._coerce_pending_to_tensor(item, ref=ref)
                if item_tensor is not None and item_tensor.numel() > 0:
                    tensors.append(item_tensor)

            if len(tensors) == 0:
                return None
            if len(tensors) == 1:
                out = tensors[0]
            else:
                try:
                    out = torch.cat(tensors, dim=-2)
                except RuntimeError:
                    out = torch.cat(
                        [t.reshape(-1, t.shape[-1]) for t in tensors],
                        dim=-2,
                    )
        else:
            raise TypeError(
                "X_pending must be None, Tensor, list, or tuple. "
                f"Got {type(X_pending)}."
            )

        if ref is not None:
            out = out.to(device=ref.device, dtype=ref.dtype)

        return out

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        """
        pending points を設定する。

        Args:
            X_pending:
                評価中または sequential optimization 中に既に選ばれた候補点。
                Tensor だけでなく list / tuple も受け取り、内部では Tensor に
                正規化して保持する。
        """
        self.X_pending = self._coerce_pending_to_tensor(X_pending)


    def _transform_pending_like_candidate(
        self,
        X_pending,
        *,
        ref: Tensor,
    ) -> Optional[Tensor]:
        """X_pending を candidate と同じ距離計算空間へ写す。

        `optimize_acqf(sequential=True, q>1)` から渡される `X_pending` は
        通常 raw input space の候補点です。一方、acquisition score 側は
        `_apply_input_transform_safe(raw_X)` 後の `Xt` に揃えているため、
        pending 側も同じ transform を通してから距離を計算する。
        """
        Xp = self._coerce_pending_to_tensor(X_pending, ref=ref)
        if Xp is None or Xp.numel() == 0:
            return None

        Xp_t = self._apply_input_transform_safe(Xp)
        Xp_t = self._ensure_q_batch(self._as_tensor(Xp_t))
        return Xp_t.to(device=ref.device, dtype=ref.dtype)


    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        """
        pending points に近い候補点へ pointwise penalty を与える。

        Args:
            Xt:
                候補点。すでに `_apply_input_transform_safe(raw_X)` を通した
                距離計算用 Tensor。shape は `(*batch, q_like, d)`。

        Returns:
            Tensor:
                pending penalty。shape は `(*batch, q_like)`。

        Notes:
            `X_pending` は raw input space で保持されることが多いため、
            この関数内で pending 側も `_apply_input_transform_safe` に通す。
            これにより candidate と pending の距離計算空間を揃える。
        """
        Xt = self._ensure_q_batch(self._as_tensor(Xt))

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

        MultiOutputClassificationInputPerturbationObjective は MCMultiOutputObjective
        を継承する qEHVI / qNEHVI 用 objective なので、score objective としては
        扱わない。
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
        pointwise score に objective を適用する。

        Args:
            score:
                (*batch, q_like) または (*batch, q_like, m)。
                InputPerturbation 使用時は q_like = q * n_w。
            raw_X:
                optimize_acqf から渡された元の X。shape = (*batch, q, d)。
            expanded_X:
                input_transform 後の X。shape = (*batch, q_like, d)。

        Notes:
            - MultiOutputClassificationScoreObjective は score をそのまま受ける。
            - MCMultiOutputObjective は samples 風に最後の output 次元を持つ形を期待する。
        """
        objective = getattr(self, "objective", None)
        if objective is None:
            return score

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
    # shape / transform utilities
    # =========================================================
    @staticmethod
    def _as_tensor(X) -> Tensor:
        if isinstance(X, tuple):
            return X[0]
        return X

    @staticmethod
    def _flatten_points(X: Tensor) -> Tensor:
        """(..., d) -> (N, d)"""
        X = _MultiOutputLatentStraddleBase._as_tensor(X)
        return X.reshape(-1, X.shape[-1])

    def _apply_input_transform_safe(self, X: Tensor) -> Tensor:
        """
        acquisition 側で shape 整合・pending penalty・objective に使う X を返す。

        MultiOutputClassificationModel 自体には input_transform が無いことがある。
        その場合でも各 submodel が InputPerturbation を持っていれば、
        posterior.mean は q * n_w に展開される。

        そのため、top-level transform が無い場合は first submodel の
        input_transform を fallback として使い、score / posterior の q_like と
        expanded_X.shape[-2] を一致させる。
        """
        X = self._as_tensor(X)

        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            return self._as_tensor(Xt)

        models = getattr(self.model, "models", None)
        if models is not None and len(models) > 0:
            first_model = models[0]
            it = getattr(first_model, "input_transform", None)
            if it is not None:
                Xt = it(X)
                return self._as_tensor(Xt)

        return X

    def _transform_and_flatten_reference(self, Xref: Optional[Tensor]) -> Optional[Tensor]:
        """参照点を距離 penalty 用の空間に写し、2D (N, d) に潰す。"""
        Xref = self._coerce_pending_to_tensor(Xref)
        if Xref is None or Xref.numel() == 0:
            return None
        Xt = self._apply_input_transform_safe(Xref)
        return self._flatten_points(Xt)

    def _normalize_multioutput_stat_shape(
        self,
        z: Tensor,
        X: Tensor,
        name: str,
    ) -> Tensor:
        """
        latent mean / variance / probability mean を (*batch, q, m) に整形する。

        想定:
          - single-output 相当: (*batch, q)
          - multi-output:       (*batch, q, m)
          - flatten 済みでも総要素数から復元
        """
        X = self._ensure_q_batch(self._as_tensor(X))
        expected_prefix = X.shape[:-1]  # (*batch, q)
        n_points = math.prod(expected_prefix)

        if z.shape == expected_prefix:
            return z.unsqueeze(-1)

        if z.ndim == X.ndim and z.shape[:-1] == expected_prefix:
            return z

        if z.numel() % n_points == 0:
            m = z.numel() // n_points
            if m <= 0:
                raise RuntimeError(f"Invalid inferred output dim for {name}: m={m}")
            return z.reshape(*expected_prefix, m)

        raise RuntimeError(
            f"Unexpected {name} shape: "
            f"X.shape={tuple(X.shape)}, {name}.shape={tuple(z.shape)}"
        )

    def _normalize_mean_shape(self, mean: Tensor, X: Tensor) -> Tensor:
        return self._normalize_multioutput_stat_shape(mean, X, name="mean")

    def _threshold_vector(self, thresholds: float | Tensor, m: int, device, dtype) -> Tensor:
        """scalar または shape (m,) の threshold を Tensor 化する。"""
        if isinstance(thresholds, (float, int)):
            return torch.full((m,), float(thresholds), device=device, dtype=dtype)

        thr = torch.as_tensor(thresholds, device=device, dtype=dtype)
        if thr.ndim != 1 or thr.numel() != m:
            raise ValueError(
                f"thresholds must be scalar or shape ({m},), got {tuple(thr.shape)}"
            )
        return thr

    @staticmethod
    def _expand_pending_to_batch(X_pending: Tensor, batch_shape: torch.Size) -> Tensor:
        """X_pending を (*batch_shape, m, d) に展開する。"""
        if X_pending.ndim == 2:
            m, d = X_pending.shape
            return X_pending.view(*([1] * len(batch_shape)), m, d).expand(
                *batch_shape, m, d
            )

        if X_pending.ndim >= 3:
            m, d = X_pending.shape[-2], X_pending.shape[-1]
            Xp = X_pending.reshape(*([1] * len(batch_shape)), m, d)
            return Xp.expand(*batch_shape, m, d)

        raise ValueError(f"Unexpected X_pending shape: {tuple(X_pending.shape)}")

    # =========================================================
    # posterior utilities
    # =========================================================
    def _extract_variance_from_posterior(self, posterior) -> Tensor:
        if hasattr(posterior, "variance"):
            return posterior.variance

        dist = getattr(posterior, "distribution", None)
        if dist is not None and hasattr(dist, "variance"):
            return dist.variance

        mvn = getattr(posterior, "mvn", None)
        if mvn is not None and hasattr(mvn, "variance"):
            return mvn.variance

        raise AttributeError(
            "Could not extract variance from latent posterior. "
            "Expected posterior.variance or posterior.distribution.variance."
        )

    def _get_submodels(self) -> List:
        """
        ModelList / 独自 wrapper の submodels を取得する。

        ただし MultiOutputClassificationModel のように latent_posterior を持つ wrapper は、
        原則として wrapper 自体の posterior を優先する。
        この関数は fallback として使う。
        """
        model = self.model
        if isinstance(model, ModelList):
            return list(model.models)
        if hasattr(model, "models"):
            return list(model.models)
        return [model]

    def _apply_input_transform_single(self, submodel, X: Tensor) -> Tensor:
        X = self._as_tensor(X)
        it = getattr(submodel, "input_transform", None)
        if it is not None:
            return self._as_tensor(it(X))
        return X

    def _map_X_for_single_model(self, submodel, X: Tensor) -> Tensor:
        """
        各 submodel が期待する内部特徴空間へ X を写す。

        優先順:
          1. decomposition / mixed wrapper の _to_internal
          2. PCA / REMBO wrapper などの _to_latent
          3. input_transform
        """
        X = self._as_tensor(X)

        if hasattr(submodel, "_to_internal"):
            return submodel._to_internal(X)

        if hasattr(submodel, "_to_latent"):
            return submodel._to_latent(X)

        return self._apply_input_transform_single(submodel, X)

    def _get_single_model_latent_posterior(self, submodel, X: Tensor):
        """
        single-output model から latent posterior を取得する。

        まず wrapper の latent posterior accessor を優先する。
        それが無い場合だけ inner latent GP を直接呼ぶ。
        """
        X = self._as_tensor(X)

        # 1. wrapper が latent posterior を提供している場合: raw X を渡す
        for name in ("latent_posterior", "posterior_latent", "posterior_f"):
            fn = getattr(submodel, name, None)
            if callable(fn):
                return fn(X)

        # 2. inner model が posterior を持つ場合
        inner_model = getattr(submodel, "model", None)
        if inner_model is not None and callable(getattr(inner_model, "posterior", None)):
            Xt = self._map_X_for_single_model(submodel, X)
            return inner_model.posterior(Xt)

        gp_model = getattr(submodel, "gp_model", None)
        if gp_model is not None and callable(getattr(gp_model, "posterior", None)):
            Xt = self._map_X_for_single_model(submodel, X)
            return gp_model.posterior(Xt)

        # 3. inner latent GP を直接 call する fallback
        latent_gp = getattr(submodel, "model", submodel)
        if callable(latent_gp):
            Xt = self._map_X_for_single_model(submodel, X)
            Xf = Xt.reshape(-1, Xt.shape[-1])
            latent_dist = latent_gp(Xf)
            return latent_dist

        raise AttributeError(
            f"Latent posterior accessor was not found for submodel {type(submodel).__name__}. "
            "Expected one of: latent_posterior / posterior_latent / posterior_f / "
            "model.posterior / gp_model.posterior / callable latent model."
        )

    def _call_single_latent(
        self,
        submodel,
        X: Tensor,
        shape_X: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """single-output submodel の latent mean / variance を (*batch, q_like) で返す。"""
        X = self._ensure_q_batch(self._as_tensor(X))
        shape_X = self._ensure_q_batch(self._as_tensor(shape_X)) if shape_X is not None else X
        latent_post = self._get_single_model_latent_posterior(submodel, X)

        mu = getattr(latent_post, "mean", None)
        if mu is None:
            dist = getattr(latent_post, "distribution", latent_post)
            mu = dist.mean

        var = self._extract_variance_from_posterior(latent_post)

        mu_i = self._normalize_multioutput_stat_shape(mu, shape_X, "latent mean")
        var_i = self._normalize_multioutput_stat_shape(var, shape_X, "latent variance")

        if mu_i.shape[-1] != 1:
            raise RuntimeError(
                "Each submodel must contribute one latent output. "
                f"Got latent mean shape {tuple(mu_i.shape)}."
            )
        if var_i.shape[-1] != 1:
            raise RuntimeError(
                "Each submodel must contribute one latent output. "
                f"Got latent variance shape {tuple(var_i.shape)}."
            )

        return mu_i.squeeze(-1), var_i.squeeze(-1)

    def _get_multioutput_latent_stats(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        latent mean / variance を (*batch, q, m) で返す。

        MultiOutputClassificationModel のように wrapper 自体が latent_posterior を持つ場合は
        それを最優先で使う。ModelList の場合は submodel ごとに latent posterior を取得する。
        """
        X = self._ensure_q_batch(self._as_tensor(X))
        shape_X = self._ensure_q_batch(self._apply_input_transform_safe(X))

        latent_fn = getattr(self.model, "latent_posterior", None)
        if callable(latent_fn):
            latent_post = latent_fn(X)
            mu = self._normalize_multioutput_stat_shape(
                latent_post.mean,
                shape_X,
                name="latent mean",
            )
            var = self._normalize_multioutput_stat_shape(
                self._extract_variance_from_posterior(latent_post),
                shape_X,
                name="latent variance",
            )
            return mu, var

        mus = []
        vars_ = []
        for submodel in self._get_submodels():
            mu_i, var_i = self._call_single_latent(submodel, X, shape_X=shape_X)
            mus.append(mu_i.unsqueeze(-1))
            vars_.append(var_i.unsqueeze(-1))

        return torch.cat(mus, dim=-1), torch.cat(vars_, dim=-1)

    def _get_multioutput_probability_mean(self, X: Tensor) -> Tensor:
        """
        P(y=1) の mean を (*batch, q, m) で返す。

        MultiOutputClassificationModel では probability_posterior(X) を優先する。
        ModelList では各 submodel.posterior(X) を cat する。
        """
        X = self._ensure_q_batch(self._as_tensor(X))
        shape_X = self._ensure_q_batch(self._apply_input_transform_safe(X))

        prob_fn = getattr(self.model, "probability_posterior", None)
        if callable(prob_fn):
            post = prob_fn(X)
            mean_prob = self._normalize_multioutput_stat_shape(
                post.mean,
                shape_X,
                name="probability mean",
            )
            return self._to_probability(
                mean_prob,
                apply_sigmoid_if_needed=False,
                name="probability_posterior.mean",
            )

        # ModelList / 独自 models list
        if isinstance(self.model, ModelList) or hasattr(self.model, "models"):
            probs = []
            for submodel in self._get_submodels():
                post_i = submodel.posterior(X)
                p_i = self._normalize_multioutput_stat_shape(
                    post_i.mean,
                    shape_X,
                    name="submodel posterior.mean",
                )
                if p_i.shape[-1] != 1:
                    raise RuntimeError(
                        "Each submodel posterior must be single-output probability. "
                        f"Got shape {tuple(p_i.shape)}."
                    )
                probs.append(p_i)
            mean_prob = torch.cat(probs, dim=-1)
            return self._to_probability(
                mean_prob,
                apply_sigmoid_if_needed=False,
                name="ModelList posterior.mean",
            )

        post = self.model.posterior(X)
        mean_prob = self._normalize_multioutput_stat_shape(
            post.mean,
            shape_X,
            name="posterior.mean",
        )
        return self._to_probability(
            mean_prob,
            apply_sigmoid_if_needed=False,
            name="posterior.mean",
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
            "Use probability_posterior(X) for multi-output wrappers, or set a "
            "probability-returning posterior on the classifier wrapper."
        )

    def _set_eval_mode(self) -> None:
        self.model.eval()
        like = getattr(self.model, "likelihood", None)
        if like is not None:
            like.eval()
        for submodel in self._get_submodels() if hasattr(self.model, "models") else []:
            submodel.eval()
            like_i = getattr(submodel, "likelihood", None)
            if like_i is not None:
                like_i.eval()

    # =========================================================
    # output aggregation
    # =========================================================
    def _aggregate_outputs(
        self,
        score_per_output: Tensor,
        *,
        output_mode: str,
        output_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        score_per_output: (*batch, q, m)
        return: (*batch, q)
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
                    f"output_weights must have shape ({score_per_output.shape[-1]},), "
                    f"got {tuple(w.shape)}."
                )
            w = w / w.sum().clamp_min(self.eps)
            view_shape = (1,) * (score_per_output.ndim - 1) + (w.numel(),)
            return (score_per_output * w.view(*view_shape)).sum(dim=-1)

        raise ValueError(f"Unknown output_mode: {output_mode}")


class _MultiOutputLatentStraddleAcquisition(_MultiOutputLatentStraddleBase):
    """
    多出力 2値分類用の pointwise latent straddle acquisition。

    各出力 j に対して
        score_j(x) = beta * sigma_j(x) - sqrt((mu_j(x) - threshold_j)^2 + smooth_abs_eps)
    を計算し、出力方向に集約してから q 方向を reduction で集約する。
    """

    def __init__(
        self,
        model,
        beta: float = 1.0,
        thresholds: float | Tensor = 0.0,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        smooth_abs_eps: float = 1e-8,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.beta = float(beta)
        self.thresholds = thresholds
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.smooth_abs_eps = float(smooth_abs_eps)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        raw_X = self._ensure_q_batch(self._as_tensor(X))
        original_batch_shape = raw_X.shape[:-2]
        self._set_eval_mode()

        Xt = self._ensure_q_batch(self._apply_input_transform_safe(raw_X))
        mu, var = self._get_multioutput_latent_stats(raw_X)
        var = var.clamp_min(self.eps)
        sigma = var.sqrt()

        m = mu.shape[-1]
        thr = self._threshold_vector(self.thresholds, m, mu.device, mu.dtype)
        thr = thr.view(*((1,) * (mu.ndim - 1)), m)

        score_per_output = self.beta * sigma - torch.sqrt(
            (mu - thr).pow(2) + self.smooth_abs_eps
        )  # (*batch, q, m)

        score = self._aggregate_outputs(
            score_per_output,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
        )  # (*batch, q)

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="MultiOutputLatentStraddle",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "MultiOutputLatentStraddle")
        return out


class _JointMultiOutputLatentStraddleAcquisition(_MultiOutputLatentStraddleBase):
    """
    多出力 2値分類用の joint latent straddle acquisition。

    q 点 × m 出力を 1 つの joint latent vector とみなし、
        score(X) = beta * U(Cov[f(X)]) - D(E[f(X)], thresholds) - repulsion(X)
    を評価する。

    注意:
        MultiOutputClassificationModel の latent_posterior に covariance_matrix があれば使う。
        ModelList fallback では diagonal covariance 近似を使う。
    """

    def __init__(
        self,
        model,
        beta: float = 2.0,
        thresholds: float | Tensor = 0.0,
        uncertainty_mode: str = "logdet1p",   # "logdet1p", "logdet", "sqrt_trace"
        boundary_mode: str = "l2_mean",       # "mean_abs", "l2_mean", "max_abs"
        tau: float = 1.0,
        jitter: float = 1e-6,
        eps: float = 1e-10,
        marginalize_pending: bool = True,
        same_batch_penalty_weight: float = 0.1,
        pending_penalty_weight: float = 0.1,
        observed_penalty_weight: float = 0.0,
        distance_beta: float = 20.0,
        duplicate_tol: float = 1e-6,
        hard_duplicate_penalty: float = 1e6,
        X_observed: Optional[Tensor] = None,
    ):
        super().__init__(
            model=model,
            reduction="sum",
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=distance_beta,
            eps=eps,
        )
        self.beta = float(beta)
        self.thresholds = thresholds
        self.uncertainty_mode = uncertainty_mode
        self.boundary_mode = boundary_mode
        self.tau = float(tau)
        self.jitter = float(jitter)
        self.marginalize_pending = bool(marginalize_pending)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.distance_beta = float(distance_beta)
        self.duplicate_tol = float(duplicate_tol)
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)
        self.X_observed = X_observed

    def set_X_observed(self, X_observed: Optional[Tensor]) -> None:
        self.X_observed = X_observed

    def _normalize_joint_cov_shape(self, cov: Tensor, X: Tensor, m: int) -> Tensor:
        """cov -> (*batch, q*m, q*m)"""
        X = self._ensure_q_batch(self._as_tensor(X))
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        qm = q * m
        expected = (*batch_shape, qm, qm)

        if cov.shape == expected:
            return cov

        if cov.numel() == math.prod(expected):
            return cov.reshape(*expected)

        raise RuntimeError(
            f"Unexpected latent covariance shape: "
            f"X.shape={tuple(X.shape)}, cov.shape={tuple(cov.shape)}"
        )

    def _latent_mean_and_cov(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """
        X: (*batch, q, d)
        return:
          mu:  (*batch, q, m)
          cov: (*batch, q*m, q*m)
        """
        self._prepare_eval()
        X = self._ensure_q_batch(self._as_tensor(X))
        shape_X = self._ensure_q_batch(self._apply_input_transform_safe(X))
        self._set_eval_mode()

        latent_fn = getattr(self.model, "latent_posterior", None)
        if callable(latent_fn):
            latent_post = latent_fn(X)
            mu = self._normalize_multioutput_stat_shape(
                latent_post.mean,
                shape_X,
                name="latent mean",
            )
            m = mu.shape[-1]

            dist = getattr(latent_post, "distribution", None)
            cov_raw = None
            if dist is not None and hasattr(dist, "covariance_matrix"):
                cov_raw = dist.covariance_matrix
            elif hasattr(latent_post, "covariance_matrix"):
                cov_raw = latent_post.covariance_matrix

            if cov_raw is not None:
                cov = self._normalize_joint_cov_shape(cov_raw, shape_X, m)
            else:
                var = self._normalize_multioutput_stat_shape(
                    self._extract_variance_from_posterior(latent_post),
                    shape_X,
                    name="latent variance",
                ).clamp_min(self.eps)
                flat_var = var.reshape(*var.shape[:-2], -1)
                cov = torch.diag_embed(flat_var)
        else:
            mu, var = self._get_multioutput_latent_stats(X)
            var = var.clamp_min(self.eps)
            flat_var = var.reshape(*var.shape[:-2], -1)
            cov = torch.diag_embed(flat_var)

        qm = cov.shape[-1]
        eye = torch.eye(qm, dtype=cov.dtype, device=cov.device)
        cov = cov + self.jitter * eye
        return mu, cov

    def _joint_uncertainty(self, cov: Tensor) -> Tensor:
        """cov: (*batch, q*m, q*m) -> (*batch,)"""
        qm = cov.shape[-1]
        eye = torch.eye(qm, dtype=cov.dtype, device=cov.device)

        if self.uncertainty_mode == "logdet1p":
            tau2 = max(self.tau ** 2, self.eps)
            mat = eye + cov / tau2
            sign, logabsdet = torch.linalg.slogdet(mat)
            if not torch.all(sign > 0):
                raise RuntimeError("Non-positive definite matrix encountered in logdet1p.")
            return 0.5 * logabsdet

        if self.uncertainty_mode == "logdet":
            sign, logabsdet = torch.linalg.slogdet(cov)
            if not torch.all(sign > 0):
                raise RuntimeError("Non-positive definite covariance encountered in logdet.")
            return 0.5 * logabsdet

        if self.uncertainty_mode == "sqrt_trace":
            tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(self.eps)
            return tr.sqrt()

        raise ValueError(f"Unknown uncertainty_mode: {self.uncertainty_mode}")

    def _boundary_distance(self, mu: Tensor) -> Tensor:
        """mu: (*batch, q, m) -> (*batch,)"""
        m = mu.shape[-1]
        thr = self._threshold_vector(self.thresholds, m, mu.device, mu.dtype)
        thr = thr.view(*((1,) * (mu.ndim - 1)), m)
        diff = mu - thr

        if self.boundary_mode == "mean_abs":
            return diff.abs().mean(dim=(-2, -1))

        if self.boundary_mode == "l2_mean":
            return diff.pow(2).mean(dim=(-2, -1)).sqrt()

        if self.boundary_mode == "max_abs":
            return diff.abs().amax(dim=(-2, -1))

        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode}")

    def _joint_straddle_score(self, X: Tensor) -> Tensor:
        mu, cov = self._latent_mean_and_cov(X)
        uncertainty = self._joint_uncertainty(cov)
        boundary = self._boundary_distance(mu)
        return self.beta * uncertainty - boundary

    def _same_batch_repulsion(self, Xt: Tensor) -> Tensor:
        """同一 q-batch 内の近接・重複を抑制する。"""
        Xt = self._ensure_q_batch(self._as_tensor(Xt))
        batch_shape = Xt.shape[:-2]
        q = Xt.shape[-2]
        d = Xt.shape[-1]

        if q <= 1 or self.same_batch_penalty_weight <= 0.0:
            return torch.zeros(batch_shape, device=Xt.device, dtype=Xt.dtype)

        Xb = Xt.reshape(-1, q, d)
        dmat = torch.cdist(Xb, Xb)

        eye_mask = torch.eye(q, device=Xt.device, dtype=torch.bool).unsqueeze(0)
        dmat = dmat.masked_fill(eye_mask, float("inf"))

        nearest = dmat.min(dim=-1).values
        soft_pen = torch.exp(-self.pending_penalty_beta * nearest).sum(dim=-1)
        hard_hits = (nearest <= self.duplicate_tol).to(Xt.dtype).sum(dim=-1)

        total = (
            self.same_batch_penalty_weight * soft_pen
            + self.hard_duplicate_penalty * hard_hits
        )
        return total.reshape(*batch_shape)

    def _reference_repulsion(
        self,
        Xt: Tensor,
        Xref: Optional[Tensor],
        weight: float,
    ) -> Tensor:
        """候補 Xt と参照点 Xref の近接・重複を抑制する。"""
        Xt = self._ensure_q_batch(self._as_tensor(Xt))
        batch_shape = Xt.shape[:-2]
        q = Xt.shape[-2]
        d = Xt.shape[-1]

        if weight <= 0.0:
            return torch.zeros(batch_shape, device=Xt.device, dtype=Xt.dtype)

        Xref2d = self._transform_and_flatten_reference(Xref)
        if Xref2d is None or Xref2d.numel() == 0:
            return torch.zeros(batch_shape, device=Xt.device, dtype=Xt.dtype)

        Xb = Xt.reshape(-1, q, d)
        dists = torch.cdist(Xb.reshape(-1, d), Xref2d)
        nearest = dists.min(dim=-1).values.reshape(-1, q)

        soft_pen = torch.exp(-self.pending_penalty_beta * nearest).sum(dim=-1)
        hard_hits = (nearest <= self.duplicate_tol).to(Xt.dtype).sum(dim=-1)

        total = weight * soft_pen + self.hard_duplicate_penalty * hard_hits
        return total.reshape(*batch_shape)

    def _repulsion_penalty(self, X: Tensor) -> Tensor:
        Xt = self._apply_input_transform_safe(X)
        penalty = self._same_batch_repulsion(Xt)
        penalty = penalty + self._reference_repulsion(
            Xt,
            getattr(self, "X_pending", None),
            self.pending_penalty_weight,
        )
        penalty = penalty + self._reference_repulsion(
            Xt,
            self.X_observed,
            self.observed_penalty_weight,
        )
        return penalty

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X = self._ensure_q_batch(self._as_tensor(X))
        batch_shape = X.shape[:-2]
        Xp = self._coerce_pending_to_tensor(getattr(self, "X_pending", None), ref=X)

        if Xp is None or Xp.numel() == 0 or not self.marginalize_pending:
            out = self._joint_straddle_score(X)
            out = out - self._repulsion_penalty(X)
            self._check_output_shape(out, batch_shape, "JointMultiOutputLatentStraddle")
            return out

        Xp_batch = self._expand_pending_to_batch(Xp, batch_shape)
        score_pending = self._joint_straddle_score(Xp_batch)

        X_all = torch.cat([Xp_batch, X], dim=-2)
        score_all = self._joint_straddle_score(X_all)

        out = score_all - score_pending
        out = out - self._repulsion_penalty(X)

        self._check_output_shape(out, batch_shape, "JointMultiOutputLatentStraddle")
        return out


class _HeteroLatentStraddleMultiOutputAcquisition(_MultiOutputLatentStraddleBase):
    """
    multi-output heteroscedastic binary classification 用 latent straddle。

    latent 側は latent_posterior(X)、ROI 確率側は probability_posterior(X) を優先する。
    ModelList が渡された場合は submodel ごとに posterior を取得して cat する。
    """

    def __init__(
        self,
        model,
        beta: float = 2.0,
        threshold: float = 0.0,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        region_mode: RegionMode = "independent",
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
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # noise
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.beta = float(beta)
        self.threshold = float(threshold)

        self.output_mode = output_mode
        self.output_weights = output_weights
        self.region_mode = region_mode

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
        self.roi_weight_fn = roi_weight_fn

        # noise
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_model_outputs_log_var = bool(noise_model_outputs_log_var)
        self.noise_weight_fn = noise_weight_fn
        self._set_multioutput_classification_objective(objective)

    # =========================================================
    # latent / probability helpers
    # =========================================================
    def _latent_stats_and_mean_prob(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        return:
            mu_f      : (*batch, q, m)
            var_f     : (*batch, q, m)
            mean_prob : (*batch, q, m)
            Xt        : penalty/weight 用の top-level transformed X
        """
        X = self._ensure_q_batch(self._as_tensor(X))
        Xt = self._apply_input_transform_safe(X)

        mu_f, var_f = self._get_multioutput_latent_stats(X)
        var_f = var_f.clamp_min(self.eps)
        mean_prob = self._get_multioutput_probability_mean(X)

        return mu_f, var_f, mean_prob, Xt

    # =========================================================
    # ROI
    # =========================================================
    def _build_roi_weight(self, prob: Tensor, X: Optional[Tensor]) -> Tensor:
        """
        prob:
            (*batch, q, m) あるいは (*batch, q)
        return:
            同 shape の weight
        """
        if self.roi_weight_fn is not None:
            w = self.roi_weight_fn(prob, X)
            return w.to(device=prob.device, dtype=prob.dtype)

        if self.roi_mode == "none":
            w = torch.ones_like(prob)

        elif self.roi_mode == "threshold":
            w = torch.sigmoid(self.roi_beta * (prob - self.roi_threshold))

        elif self.roi_mode == "target_prob":
            bw = max(self.roi_bandwidth, self.eps)
            z = (prob - self.roi_target_prob) / bw
            w = torch.exp(-0.5 * z * z)

        elif self.roi_mode == "interval":
            if self.roi_interval is None:
                raise ValueError("roi_interval must be provided when roi_mode='interval'.")
            lo, hi = self.roi_interval
            left = torch.sigmoid(self.roi_beta * (prob - float(lo)))
            right = torch.sigmoid(self.roi_beta * (float(hi) - prob))
            w = left * right

        elif self.roi_mode == "band":
            bw = max(self.roi_bandwidth, self.eps)
            z = (prob - self.roi_target_prob) / bw
            w = torch.exp(-0.5 * z * z)

        else:
            raise ValueError(f"Unknown roi_mode: {self.roi_mode}")

        if self.roi_min_weight > 0.0:
            w = self.roi_min_weight + (1.0 - self.roi_min_weight) * w

        if self.roi_weight_scale != 1.0:
            w = self.roi_weight_scale * w

        return w

    def _combine_score_and_weight(self, score: Tensor, weight: Tensor, combine: str) -> Tensor:
        if combine == "multiply":
            return score * weight
        if combine == "add":
            return score + weight
        raise ValueError(f"Unknown combine mode: {combine}")

    def _apply_roi_weight(self, score: Tensor, prob: Tensor, X: Optional[Tensor]) -> Tensor:
        w = self._build_roi_weight(prob, X)
        return self._combine_score_and_weight(score, w, self.roi_combine)

    # =========================================================
    # noise
    # =========================================================
    def _noise_post_matches_shape(self, noise_post, shape_X: Tensor) -> bool:
        """
        noise posterior の mean が shape_X の q_like と整合するか確認する。

        shape_X:
            (*batch, q_like, d)
        """
        try:
            _ = self._normalize_multioutput_stat_shape(
                noise_post.mean,
                shape_X,
                name="noise mean shape check",
            )
            return True
        except Exception:
            return False

    def _get_single_model_noise_posterior(
        self,
        submodel,
        X: Tensor,
        shape_X: Optional[Tensor] = None,
    ):
        """
        単一モデルから noise posterior を取得する。

        重要:
            heteroscedastic model の noise_model は、学習時には Normalize のみを
            かけた入力で fit されていることが多い。一方、acquisition 側では
            InputPerturbation により q -> q * n_w に展開された点ごとに
            noise を評価したい。

            そのため、raw X で q が合わない場合は、expanded/transformed な
            shape_X を使って noise_model.posterior(...) を再評価する。
        """
        X = self._ensure_q_batch(self._as_tensor(X))
        shape_X = self._ensure_q_batch(self._as_tensor(shape_X)) if shape_X is not None else X

        # ------------------------------------------------------------
        # 1. accessor がある場合
        # ------------------------------------------------------------
        for name in ("posterior_noise", "noise_posterior"):
            fn = getattr(submodel, name, None)
            if callable(fn):
                # まず raw X で試す。q_like が合わなければ shape_X で再評価。
                try:
                    post = fn(X)
                    if self._noise_post_matches_shape(post, shape_X):
                        return post
                except Exception:
                    pass

                try:
                    post = fn(shape_X)
                    if self._noise_post_matches_shape(post, shape_X):
                        return post
                except Exception:
                    pass

        # ------------------------------------------------------------
        # 2. noise_model を直接取得
        # ------------------------------------------------------------
        noise_model = getattr(submodel, "noise_model", None)
        if noise_model is None:
            inner_model = getattr(submodel, "model", None)
            if inner_model is not None:
                noise_model = getattr(inner_model, "noise_model", None)

        if noise_model is None:
            raise AttributeError(
                f"Noise posterior was not found for submodel {type(submodel).__name__}. "
                "Expected one of: posterior_noise / noise_posterior / noise_model.posterior."
            )

        # 候補順:
        #   1. shape_X:
        #        q*n_w に展開済み。noise_model を直接呼ぶ場合はこれが最も必要。
        #   2. submodel._transform_noise_inputs(shape_X):
        #        shape_X が raw expanded の場合の保険。
        #   3. submodel._transform_noise_inputs(X):
        #        非 InputPerturbation / raw q の場合の従来経路。
        #   4. _map_X_for_single_model(submodel, X):
        #   5. X:
        #        最後の fallback。
        candidates = []

        candidates.append(shape_X)

        transform_noise_inputs = getattr(submodel, "_transform_noise_inputs", None)
        if callable(transform_noise_inputs):
            try:
                candidates.append(transform_noise_inputs(shape_X))
            except Exception:
                pass
            try:
                candidates.append(transform_noise_inputs(X))
            except Exception:
                pass

        try:
            candidates.append(self._map_X_for_single_model(submodel, X))
        except Exception:
            pass

        candidates.append(X)

        # 重複候補を避けつつ、shape_X と q_like が合う posterior を採用。
        seen = set()
        last_error = None
        for noise_in in candidates:
            if not torch.is_tensor(noise_in):
                continue

            key = (tuple(noise_in.shape), noise_in.data_ptr() if noise_in.numel() > 0 else 0)
            if key in seen:
                continue
            seen.add(key)

            try:
                post = noise_model.posterior(noise_in)
                if self._noise_post_matches_shape(post, shape_X):
                    return post
            except Exception as err:
                last_error = err
                continue

        if last_error is not None:
            raise RuntimeError(
                "Could not obtain a noise posterior whose q dimension matches shape_X. "
                f"raw X.shape={tuple(X.shape)}, shape_X.shape={tuple(shape_X.shape)}. "
                f"Last error: {last_error}"
            )

        raise RuntimeError(
            "Could not obtain a noise posterior whose q dimension matches shape_X. "
            f"raw X.shape={tuple(X.shape)}, shape_X.shape={tuple(shape_X.shape)}."
        )

    def _get_noise_values(self, X: Tensor) -> Tensor:
        """noise: (*batch, q, m) を返す。"""
        X = self._ensure_q_batch(self._as_tensor(X))
        shape_X = self._ensure_q_batch(self._apply_input_transform_safe(X))

        if self.noise_weight_fn is not None:
            v = self.noise_weight_fn(None, X)
            return v.to(device=X.device, dtype=X.dtype)

        # multi-output wrapper 自体が noise posterior を提供している場合
        for name in ("posterior_noise", "noise_posterior"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                errors = []
                for noise_X in (X, shape_X):
                    try:
                        noise_post = fn(noise_X)
                        noise_mean = self._normalize_multioutput_stat_shape(
                            noise_post.mean,
                            shape_X,
                            name="noise mean",
                        )
                        return self._postprocess_noise(noise_mean)
                    except Exception as err:
                        errors.append(err)
                        continue
                if errors:
                    raise RuntimeError(
                        f"{name} exists, but its output could not be aligned to shape_X. "
                        f"X.shape={tuple(X.shape)}, shape_X.shape={tuple(shape_X.shape)}. "
                        f"Last error: {errors[-1]}"
                    )

        # ModelList / models list の場合
        if isinstance(self.model, ModelList) or hasattr(self.model, "models"):
            noise_list = []
            for submodel in self._get_submodels():
                noise_post = self._get_single_model_noise_posterior(submodel, X, shape_X=shape_X)
                noise_i = self._normalize_multioutput_stat_shape(
                    noise_post.mean,
                    shape_X,
                    name="submodel noise mean",
                )
                if noise_i.shape[-1] != 1:
                    raise RuntimeError(
                        "Each submodel must contribute one noise output. "
                        f"Got noise mean shape {tuple(noise_i.shape)}."
                    )
                noise_list.append(self._postprocess_noise(noise_i))
            return torch.cat(noise_list, dim=-1)

        noise_post = self._get_single_model_noise_posterior(self.model, X, shape_X=shape_X)
        noise_mean = self._normalize_multioutput_stat_shape(
            noise_post.mean,
            shape_X,
            name="noise mean",
        )
        return self._postprocess_noise(noise_mean)

    def _postprocess_noise(self, noise_mean: Tensor) -> Tensor:
        if self.noise_model_outputs_log_var:
            return torch.exp(noise_mean.clamp(min=math.log(self.eps), max=30.0))
        return noise_mean.clamp_min(self.eps)

    def _noise_to_weight(self, noise: Tensor) -> Tensor:
        lam = self.noise_penalty_lambda

        if self.noise_mode == "none":
            w = torch.ones_like(noise)

        elif self.noise_mode == "inverse_linear":
            w = 1.0 / (1.0 + lam * noise)

        elif self.noise_mode == "inverse_sqrt":
            w = 1.0 / torch.sqrt(1.0 + lam * noise)

        elif self.noise_mode == "exp":
            w = torch.exp(-lam * noise)

        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

        if self.noise_min_weight > 0.0:
            w = self.noise_min_weight + (1.0 - self.noise_min_weight) * w

        if self.noise_weight_scale != 1.0:
            w = self.noise_weight_scale * w

        return w

    def _apply_noise_weight(self, score: Tensor, weight: Tensor) -> Tensor:
        if self.noise_combine == "multiply":
            return score * weight
        if self.noise_combine == "add":
            return score - (1.0 - weight)
        raise ValueError(f"Unknown noise_combine: {self.noise_combine}")

    # =========================================================
    # region helpers
    # =========================================================
    def _event_prob(self, mean_prob: Tensor) -> Tensor:
        """mean_prob: (*batch, q, m) -> (*batch, q)"""
        if self.region_mode == "all_positive":
            return mean_prob.prod(dim=-1)

        if self.region_mode == "any_positive":
            return 1.0 - (1.0 - mean_prob).prod(dim=-1)

        raise ValueError("event_prob is only valid for region_mode != 'independent'.")

    def _aggregate_score_for_region(self, score_per_output: Tensor) -> Tensor:
        """score_per_output: (*batch, q, m) -> (*batch, q)"""
        if self.region_mode == "independent":
            return self._aggregate_outputs(
                score_per_output,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
            )

        if self.region_mode == "all_positive":
            return score_per_output.min(dim=-1).values

        if self.region_mode == "any_positive":
            return score_per_output.max(dim=-1).values

        raise ValueError(f"Unknown region_mode: {self.region_mode}")

    def _aggregate_noise_weight_for_region(self, noise_weight_per_output: Tensor) -> Tensor:
        """noise_weight_per_output: (*batch, q, m) -> (*batch, q)"""
        if self.region_mode == "independent":
            return self._aggregate_outputs(
                noise_weight_per_output,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
            )

        if self.region_mode == "all_positive":
            return noise_weight_per_output.min(dim=-1).values

        if self.region_mode == "any_positive":
            return noise_weight_per_output.max(dim=-1).values

        raise ValueError(f"Unknown region_mode: {self.region_mode}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        raw_X = self._ensure_q_batch(self._as_tensor(X))
        original_batch_shape = raw_X.shape[:-2]
        mu_f, var_f, mean_prob, Xt = self._latent_stats_and_mean_prob(raw_X)
        sigma_f = var_f.sqrt()

        # 各出力ごとの latent straddle
        score_per_output = (self.beta ** 0.5) * sigma_f - (mu_f - self.threshold).abs()
        # shape: (*batch, q, m)

        if self.region_mode == "independent":
            # ROI は出力ごと
            score_per_output = self._apply_roi_weight(score_per_output, mean_prob, Xt)

            # hetero noise penalty も出力ごと
            noise = self._get_noise_values(X)
            noise_weight = self._noise_to_weight(noise)
            score_per_output = self._apply_noise_weight(score_per_output, noise_weight)

            score = self._aggregate_score_for_region(score_per_output)

        else:
            # region-level score
            score = self._aggregate_score_for_region(score_per_output)  # (*batch, q)

            # ROI は event probability ベース
            event_prob = self._event_prob(mean_prob)  # (*batch, q)
            score = self._apply_roi_weight(score, event_prob, Xt)

            # noise は region-level に集約してから適用
            noise = self._get_noise_values(X)                    # (*batch, q, m)
            noise_weight = self._noise_to_weight(noise)           # (*batch, q, m)
            event_noise_weight = self._aggregate_noise_weight_for_region(noise_weight)
            score = self._apply_noise_weight(score, event_noise_weight)

        # pending penalty
        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="HeteroLatentStraddleMultiOutput",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "HeteroLatentStraddleMultiOutput")
        return out



# =========================================================




class qHeteroMultiOutputBinaryClassEntropyAcquisition(_HeteroLatentStraddleMultiOutputAcquisition):
    """
    Heteroscedastic multi-output binary class entropy acquisition.

    各出力の positive-class probability に対して binary entropy を計算し、
    ROI weight と heteroscedastic noise weight を適用する
    level-set estimation 用 acquisition。

    通常版の `qMultiOutputBinaryClassEntropyAcquisition` に対応する hetero 版。
    probability boundary `p=0.5` 付近を重視しつつ、ノイズが大きい領域を
    `noise_mode` / `noise_penalty_lambda` により抑制できる。

    Args:
        model:
            BoTorch 互換の multi-output binary classification model。
            `latent_posterior(X)`, `probability_posterior(X)`, noise posterior
            などを必要に応じて参照する。
        reduction:
            q-batch 内の pointwise score の集約方法。
            `"mean"` または `"sum"`。
        output_mode:
            `region_mode="independent"` のときの出力方向集約。
            `"mean"`, `"sum"`, `"max"`, `"min"`, `"weighted_mean"`。
        output_weights:
            `output_mode="weighted_mean"` のときの出力重み。
        region_mode:
            multi-output の領域イベントの扱い。
            `"independent"` は出力ごとに score を計算して集約する。
            `"all_positive"` は全出力 positive のイベントを重視する。
            `"any_positive"` は少なくとも一つの出力が positive のイベントを重視する。
        pending_penalty_weight:
            `X_pending` に近い候補点を避けるためのペナルティ係数。
        pending_penalty_beta:
            pending penalty の距離減衰係数。
        eps:
            数値安定化パラメータ。
        roi_mode:
            ROI weighting の方法。`"none"`, `"threshold"`, `"target_prob"`,
            `"interval"`, `"band"`。
        roi_combine:
            ROI weight と score の結合方法。`"multiply"` または `"add"`。
        roi_threshold:
            `roi_mode="threshold"` の閾値。
        roi_target_prob:
            `roi_mode="target_prob"` / `"band"` の目標確率。
        roi_interval:
            `roi_mode="interval"` の `(lo, hi)`。
        roi_beta:
            ROI sigmoid weight の鋭さ。
        roi_bandwidth:
            target probability / band weighting の幅。
        roi_min_weight:
            ROI weight の下限。
        roi_weight_scale:
            ROI weight のスケール係数。
        roi_weight_fn:
            カスタム ROI weight 関数。
        noise_mode:
            noise weight の作り方。`"none"`, `"inverse_linear"`,
            `"inverse_sqrt"`, `"exp"`。
        noise_combine:
            noise weight と score の結合方法。`"multiply"` または `"add"`。
        noise_penalty_lambda:
            noise penalty の強さ。
        noise_min_weight:
            noise weight の下限。
        noise_weight_scale:
            noise weight のスケール係数。
        noise_model_outputs_log_var:
            noise model の出力を log variance とみなすかどうか。
        noise_weight_fn:
            カスタム noise weight 関数。
        objective:
            acquisition が計算した pointwise score に作用する objective。
            InputPerturbation 集約に使う。

    Forward Args:
        X:
            候補点。shape は通常 `batch_shape x q x d`。

    Returns:
        Tensor:
            shape `batch_shape` の acquisition value。

    Notes:
        entropy score は `H(p) = -p log(p) - (1-p) log(1-p)`。
        `p=0.5` で最大となるため、binary classification boundary 付近を
        重点的に選ぶ。
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        region_mode: RegionMode = "independent",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            beta=1.0,
            threshold=0.0,
            reduction=reduction,
            output_mode=output_mode,
            output_weights=output_weights,
            region_mode=region_mode,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            roi_mode=roi_mode,
            roi_combine=roi_combine,
            roi_threshold=roi_threshold,
            roi_target_prob=roi_target_prob,
            roi_interval=roi_interval,
            roi_beta=roi_beta,
            roi_bandwidth=roi_bandwidth,
            roi_min_weight=roi_min_weight,
            roi_weight_scale=roi_weight_scale,
            roi_weight_fn=roi_weight_fn,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            noise_weight_fn=noise_weight_fn,
            objective=objective,
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        raw_X = self._ensure_q_batch(self._as_tensor(X))
        original_batch_shape = raw_X.shape[:-2]
        _, _, mean_prob, Xt = self._latent_stats_and_mean_prob(raw_X)

        p = mean_prob.clamp(self.eps, 1.0 - self.eps)
        score_per_output = -(p * p.log() + (1.0 - p) * (1.0 - p).log())

        if self.region_mode == "independent":
            score_per_output = self._apply_roi_weight(score_per_output, mean_prob, Xt)

            noise = self._get_noise_values(raw_X)
            noise_weight = self._noise_to_weight(noise)
            score_per_output = self._apply_noise_weight(score_per_output, noise_weight)

            score = self._aggregate_score_for_region(score_per_output)

        else:
            score = self._aggregate_score_for_region(score_per_output)

            event_prob = self._event_prob(mean_prob)
            score = self._apply_roi_weight(score, event_prob, Xt)

            noise = self._get_noise_values(raw_X)
            noise_weight = self._noise_to_weight(noise)
            event_noise_weight = self._aggregate_noise_weight_for_region(noise_weight)
            score = self._apply_noise_weight(score, event_noise_weight)

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="qHeteroMultiOutputBinaryClassEntropyAcquisition",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroMultiOutputBinaryClassEntropyAcquisition")
        return out


class qHeteroMultiOutputBinaryICUAcquisition(_HeteroLatentStraddleMultiOutputAcquisition):
    """
    Heteroscedastic multi-output binary ICU acquisition.

    binary classification の contour uncertainty として
    `4 * p * (1 - p)` を評価し、ROI weight と heteroscedastic noise weight を
    適用する level-set estimation 用 acquisition。

    通常版の `qMultiOutputBinaryICUAcquisition` に対応する hetero 版。
    ノイズが大きい領域を抑制しながら、`p=0.5` 付近の境界不確実性を探索する。

    Args:
        model:
            BoTorch 互換の multi-output binary classification model。
        reduction:
            q-batch 内の pointwise score の集約方法。
        output_mode:
            `region_mode="independent"` のときの出力方向集約。
        output_weights:
            `output_mode="weighted_mean"` のときの出力重み。
        region_mode:
            `"independent"`, `"all_positive"`, `"any_positive"`。
        pending_penalty_weight:
            `X_pending` に近い候補点を避けるためのペナルティ係数。
        pending_penalty_beta:
            pending penalty の距離減衰係数。
        eps:
            数値安定化パラメータ。
        roi_mode:
            ROI weighting の方法。
        roi_combine:
            ROI weight と score の結合方法。
        roi_threshold:
            `roi_mode="threshold"` の閾値。
        roi_target_prob:
            `roi_mode="target_prob"` / `"band"` の目標確率。
        roi_interval:
            `roi_mode="interval"` の `(lo, hi)`。
        roi_beta:
            ROI sigmoid weight の鋭さ。
        roi_bandwidth:
            target probability / band weighting の幅。
        roi_min_weight:
            ROI weight の下限。
        roi_weight_scale:
            ROI weight のスケール係数。
        roi_weight_fn:
            カスタム ROI weight 関数。
        noise_mode:
            noise weight の作り方。
        noise_combine:
            noise weight と score の結合方法。
        noise_penalty_lambda:
            noise penalty の強さ。
        noise_min_weight:
            noise weight の下限。
        noise_weight_scale:
            noise weight のスケール係数。
        noise_model_outputs_log_var:
            noise model の出力を log variance とみなすかどうか。
        noise_weight_fn:
            カスタム noise weight 関数。
        objective:
            pointwise score に作用する objective。

    Forward Args:
        X:
            候補点。shape は通常 `batch_shape x q x d`。

    Returns:
        Tensor:
            shape `batch_shape` の acquisition value。

    Notes:
        score は `4p(1-p)`。最大値は `p=0.5` で 1。
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        region_mode: RegionMode = "independent",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            beta=1.0,
            threshold=0.0,
            reduction=reduction,
            output_mode=output_mode,
            output_weights=output_weights,
            region_mode=region_mode,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            roi_mode=roi_mode,
            roi_combine=roi_combine,
            roi_threshold=roi_threshold,
            roi_target_prob=roi_target_prob,
            roi_interval=roi_interval,
            roi_beta=roi_beta,
            roi_bandwidth=roi_bandwidth,
            roi_min_weight=roi_min_weight,
            roi_weight_scale=roi_weight_scale,
            roi_weight_fn=roi_weight_fn,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            noise_weight_fn=noise_weight_fn,
            objective=objective,
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        raw_X = self._ensure_q_batch(self._as_tensor(X))
        original_batch_shape = raw_X.shape[:-2]
        _, _, mean_prob, Xt = self._latent_stats_and_mean_prob(raw_X)

        score_per_output = 4.0 * mean_prob * (1.0 - mean_prob)

        if self.region_mode == "independent":
            score_per_output = self._apply_roi_weight(score_per_output, mean_prob, Xt)

            noise = self._get_noise_values(raw_X)
            noise_weight = self._noise_to_weight(noise)
            score_per_output = self._apply_noise_weight(score_per_output, noise_weight)

            score = self._aggregate_score_for_region(score_per_output)

        else:
            score = self._aggregate_score_for_region(score_per_output)

            event_prob = self._event_prob(mean_prob)
            event_icu = 4.0 * event_prob * (1.0 - event_prob)
            score = 0.5 * (score + event_icu)

            score = self._apply_roi_weight(score, event_prob, Xt)

            noise = self._get_noise_values(raw_X)
            noise_weight = self._noise_to_weight(noise)
            event_noise_weight = self._aggregate_noise_weight_for_region(noise_weight)
            score = self._apply_noise_weight(score, event_noise_weight)

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="qHeteroMultiOutputBinaryICUAcquisition",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroMultiOutputBinaryICUAcquisition")
        return out


class qHeteroMultiOutputBinaryBoundaryVarianceAcquisition(_HeteroLatentStraddleMultiOutputAcquisition):
    """
    Heteroscedastic multi-output binary boundary variance acquisition.

    latent posterior の mean / variance を使い、binary boundary 近傍の
    posterior variance を評価し、ROI weight と heteroscedastic noise weight を
    適用する level-set estimation 用 acquisition。

    各出力 j について次の score を計算する。

        `score_j(x) = Var[f_j(x)] * exp(-0.5 * ((E[f_j(x)] - threshold) / tau)^2)`

    Args:
        model:
            BoTorch 互換の multi-output binary classification model。
        threshold:
            latent boundary の閾値。binary sigmoid classifier では通常 `0.0`。
        tau:
            boundary 近傍をどれくらい広く見るかを決める bandwidth。
        reduction:
            q-batch 内の pointwise score の集約方法。
        output_mode:
            `region_mode="independent"` のときの出力方向集約。
        output_weights:
            `output_mode="weighted_mean"` のときの出力重み。
        region_mode:
            `"independent"`, `"all_positive"`, `"any_positive"`。
        pending_penalty_weight:
            `X_pending` に近い候補点を避けるためのペナルティ係数。
        pending_penalty_beta:
            pending penalty の距離減衰係数。
        eps:
            数値安定化パラメータ。
        roi_mode:
            ROI weighting の方法。
        roi_combine:
            ROI weight と score の結合方法。
        roi_threshold:
            `roi_mode="threshold"` の閾値。
        roi_target_prob:
            `roi_mode="target_prob"` / `"band"` の目標確率。
        roi_interval:
            `roi_mode="interval"` の `(lo, hi)`。
        roi_beta:
            ROI sigmoid weight の鋭さ。
        roi_bandwidth:
            target probability / band weighting の幅。
        roi_min_weight:
            ROI weight の下限。
        roi_weight_scale:
            ROI weight のスケール係数。
        roi_weight_fn:
            カスタム ROI weight 関数。
        noise_mode:
            noise weight の作り方。
        noise_combine:
            noise weight と score の結合方法。
        noise_penalty_lambda:
            noise penalty の強さ。
        noise_min_weight:
            noise weight の下限。
        noise_weight_scale:
            noise weight のスケール係数。
        noise_model_outputs_log_var:
            noise model の出力を log variance とみなすかどうか。
        noise_weight_fn:
            カスタム noise weight 関数。
        objective:
            pointwise score に作用する objective。

    Forward Args:
        X:
            候補点。shape は通常 `batch_shape x q x d`。

    Returns:
        Tensor:
            shape `batch_shape` の acquisition value。

    Notes:
        `qHeteroMultiOutputBinaryICUAcquisition` と
        `qHeteroMultiOutputBinaryClassEntropyAcquisition` は probability scale の
        境界不確実性を使う。一方、この acquisition は latent f の分散を使うため、
        GP latent posterior の不確実性をより直接的に反映する。
    """

    def __init__(
        self,
        model,
        threshold: float = 0.0,
        tau: float = 1.0,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        region_mode: RegionMode = "independent",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            beta=1.0,
            threshold=threshold,
            reduction=reduction,
            output_mode=output_mode,
            output_weights=output_weights,
            region_mode=region_mode,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            roi_mode=roi_mode,
            roi_combine=roi_combine,
            roi_threshold=roi_threshold,
            roi_target_prob=roi_target_prob,
            roi_interval=roi_interval,
            roi_beta=roi_beta,
            roi_bandwidth=roi_bandwidth,
            roi_min_weight=roi_min_weight,
            roi_weight_scale=roi_weight_scale,
            roi_weight_fn=roi_weight_fn,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            noise_weight_fn=noise_weight_fn,
            objective=objective,
        )
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        raw_X = self._ensure_q_batch(self._as_tensor(X))
        original_batch_shape = raw_X.shape[:-2]

        mu_f, var_f, mean_prob, Xt = self._latent_stats_and_mean_prob(raw_X)
        var_f = var_f.clamp_min(self.eps)

        tau = torch.as_tensor(
            self.tau,
            device=mu_f.device,
            dtype=mu_f.dtype,
        ).clamp_min(self.eps)

        threshold = torch.as_tensor(
            self.threshold,
            device=mu_f.device,
            dtype=mu_f.dtype,
        )

        score_per_output = var_f * torch.exp(
            -0.5 * ((mu_f - threshold) / tau).pow(2)
        )

        if self.region_mode == "independent":
            score_per_output = self._apply_roi_weight(score_per_output, mean_prob, Xt)

            noise = self._get_noise_values(raw_X)
            noise_weight = self._noise_to_weight(noise)
            score_per_output = self._apply_noise_weight(score_per_output, noise_weight)

            score = self._aggregate_score_for_region(score_per_output)

        else:
            score = self._aggregate_score_for_region(score_per_output)

            event_prob = self._event_prob(mean_prob)
            score = self._apply_roi_weight(score, event_prob, Xt)

            noise = self._get_noise_values(raw_X)
            noise_weight = self._noise_to_weight(noise)
            event_noise_weight = self._aggregate_noise_weight_for_region(noise_weight)
            score = self._apply_noise_weight(score, event_noise_weight)

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="qHeteroMultiOutputBinaryBoundaryVarianceAcquisition",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroMultiOutputBinaryBoundaryVarianceAcquisition")
        return out


class qHeteroMultiOutputBinaryLatentStraddleAcquisition(_HeteroLatentStraddleMultiOutputAcquisition):
    """qHeteroMultiOutputBinaryLatentStraddleAcquisition の Google スタイル API docstring。
    
    Latent Straddle は latent mean が境界に近く、かつ不確実な点を選びます。
    hetero 版なので、通常版に比べて noise posterior / noise penalty を使い、ノイズが大きい領域を避ける設計です。 multi-output 版なので、出力方向の集約方法や weights の設定が重要です。 binary classification 系では、posterior が probability 空間か latent 空間かを確認してください。
    
    Args:
        model: BoTorch 互換のモデル。少なくとも `posterior(X)` を持つ必要があります。classification / ordinal では `probability_posterior`、`latent_posterior`、`ordinal_likelihood` などを参照する実装があります。
        beta: UCB / Straddle / hetero sample adjustment で不確実性の重みを決める係数。大きいほど exploration 寄りになります。
        threshold: classification / PoF / boundary 判定の閾値。binary latent classification では通常 `0.0` が `p=0.5` に対応します。
        reduction: q-batch 内の点ごとの score をどう集約するか。典型は `mean` または `sum`。
        output_mode: multi-output classification の出力集約方法。`mean`, `sum`, `max`, `min`, `weighted_mean`, `all_positive` など。
        output_weights: multi-output の出力ごとの重み。`weighted_mean` / `weighted_sum` などで使います。
        region_mode: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        pending_penalty_weight: X_pending に近い候補を避けるペナルティの強さ。
        pending_penalty_beta: X_pending ペナルティの距離減衰係数。大きいほど近接点だけを強く避けます。
        eps: 数値安定化用の微小値。
        roi_mode: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_combine: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_threshold: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_target_prob: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_interval: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_beta: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_bandwidth: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_min_weight: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_weight_scale: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        roi_weight_fn: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        noise_mode: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        noise_combine: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        noise_penalty_lambda: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        noise_min_weight: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        noise_weight_scale: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        noise_model_outputs_log_var: noise model 出力が log variance かどうか。classification hetero 系で使います。
        noise_weight_fn: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        objective: acquisition に渡す objective。BO では posterior samples に作用し、active learning / level-set では計算済み score に作用する場合があります。
    
    Forward Args:
        X: 候補点。通常は shape `batch_shape x q x d` です。`q=1` の場合も `optimize_acqf` では q 次元を持つ形で渡されます。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。BoTorch の `optimize_acqf` はこの値を最大化します。
    
    Notes:
        InputPerturbation を使う場合は、`n_w` または `input_perturbation_n_w` と objective の設定を一致させてください。
        hetero 版では noise model の出力が分散なのか log variance なのかを `noise_is_log_var` / `noise_model_outputs_log_var` で確認してください。
        binary classification では posterior が probability を返すか latent f を返すかで `samples_are_probs` や `apply_sigmoid_if_needed` の指定が変わります。"""
    pass


class qHeteroMultiOutputBinaryJointLatentStraddleAcquisition(_JointMultiOutputLatentStraddleAcquisition):
    """qHeteroMultiOutputBinaryJointLatentStraddleAcquisition の Google スタイル API docstring。
    
    Joint Latent Straddle は q-batch 全体の joint uncertainty を使う境界探索 acquisition です。
    hetero 版なので、通常版に比べて noise posterior / noise penalty を使い、ノイズが大きい領域を避ける設計です。 multi-output 版なので、出力方向の集約方法や weights の設定が重要です。 binary classification 系では、posterior が probability 空間か latent 空間かを確認してください。
    
    Args:
        model: BoTorch 互換のモデル。少なくとも `posterior(X)` を持つ必要があります。classification / ordinal では `probability_posterior`、`latent_posterior`、`ordinal_likelihood` などを参照する実装があります。
        beta: UCB / Straddle / hetero sample adjustment で不確実性の重みを決める係数。大きいほど exploration 寄りになります。
        thresholds: multi-output で出力ごとに使う閾値。出力数と長さを一致させます。
        uncertainty_mode: multi-output straddle で不確実性をどう扱うか。
        boundary_mode: multi-output boundary の集約方法。
        tau: PI の smoothing、boundary kernel、logdet スケールなどに使う温度・幅パラメータ。小さいほど鋭い判定になります。
        jitter: 共分散行列の対角に加える安定化項。
        eps: 数値安定化用の微小値。
        marginalize_pending: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        same_batch_penalty_weight: 同じ q-batch 内で近すぎる候補を避けるペナルティの強さ。
        pending_penalty_weight: X_pending に近い候補を避けるペナルティの強さ。
        observed_penalty_weight: X_observed に近い候補を避けるペナルティの強さ。
        distance_beta: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        duplicate_tol: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        hard_duplicate_penalty: 完全一致または近接重複に対する強いペナルティ。
        X_observed: 既存観測点。active learning / level-set で観測済み点への近接ペナルティに使います。
        threshold: classification / PoF / boundary 判定の閾値。binary latent classification では通常 `0.0` が `p=0.5` に対応します。
        reduction: q-batch 内の点ごとの score をどう集約するか。典型は `mean` または `sum`。
        pending_penalty_beta: X_pending ペナルティの距離減衰係数。大きいほど近接点だけを強く避けます。
        objective: acquisition に渡す objective。BO では posterior samples に作用し、active learning / level-set では計算済み score に作用する場合があります。
        uncertainty_measure: joint uncertainty の評価方法。`logdet` や `trace` など。
        penalty_lengthscale: level-set joint penalty などで距離ペナルティの幅を指定します。
    
    Forward Args:
        X: 候補点。通常は shape `batch_shape x q x d` です。`q=1` の場合も `optimize_acqf` では q 次元を持つ形で渡されます。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。BoTorch の `optimize_acqf` はこの値を最大化します。
    
    Notes:
        InputPerturbation を使う場合は、`n_w` または `input_perturbation_n_w` と objective の設定を一致させてください。
        hetero 版では noise model の出力が分散なのか log variance なのかを `noise_is_log_var` / `noise_model_outputs_log_var` で確認してください。
        binary classification では posterior が probability を返すか latent f を返すかで `samples_are_probs` や `apply_sigmoid_if_needed` の指定が変わります。"""
    pass

__all__ = [
    "qHeteroMultiOutputBinaryClassEntropyAcquisition",
    "qHeteroMultiOutputBinaryICUAcquisition",
    "qHeteroMultiOutputBinaryBoundaryVarianceAcquisition",
    "qHeteroMultiOutputBinaryLatentStraddleAcquisition",
    "qHeteroMultiOutputBinaryJointLatentStraddleAcquisition",
]
