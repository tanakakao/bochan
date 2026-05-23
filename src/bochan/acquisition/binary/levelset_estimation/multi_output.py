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

        # optimize_acqf(sequential=True) では直前候補が X_pending に入る。
        # pending / observed は最適化対象ではないため、古い計算グラフを保持しない。
        return out.detach()

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

    def _get_multioutput_probability_mean(
        self,
        X: Tensor,
        *,
        apply_sigmoid_if_needed: bool = False,
    ) -> Tensor:
        """
        P(y=1) の mean を (*batch, q, m) で返す。

        MultiOutputClassificationModel では probability_posterior(X) を優先する。
        ModelList では各 submodel.posterior(X) を cat する。

        apply_sigmoid_if_needed=True の場合、posterior.mean が latent f を返す
        wrapper / submodel でも sigmoid で probability に変換する。
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
                apply_sigmoid_if_needed=apply_sigmoid_if_needed,
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
                apply_sigmoid_if_needed=apply_sigmoid_if_needed,
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
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
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
            "Use probability_posterior(X), set apply_sigmoid_if_needed=True, "
            "or make the classifier wrapper return probability posterior.mean."
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


# =========================================================





class qMultiOutputBinaryClassEntropyAcquisition(_MultiOutputLatentStraddleBase):
    """
    Multi-output binary classification class entropy acquisition.

    各出力の positive-class probability に対して binary entropy を計算し、
    出力方向に集約する level-set estimation 用 acquisition。

    Active Learning の predictive entropy と数式的には同じだが、
    このクラスは classification boundary / level-set estimation family として
    使うための名前にしている。

    Args:
        model:
            BoTorch 互換の multi-output binary classification model。
            `model.probability_posterior(X)` があれば優先して使う。
            なければ `model.posterior(X)` または `model.models` の各 submodel
            posterior を stack して使う。
        reduction:
            q-batch 内の pointwise score の集約方法。
            `"mean"` または `"sum"`。
        output_mode:
            出力方向の集約方法。
            `"mean"`, `"sum"`, `"max"`, `"min"`, `"weighted_mean"`。
        output_weights:
            `output_mode="weighted_mean"` のときの出力重み。
            shape は `(m,)` を想定する。
        pending_penalty_weight:
            `X_pending` に近い候補点を避けるためのペナルティ係数。
        pending_penalty_beta:
            pending penalty の距離減衰係数。
        apply_sigmoid_if_needed:
            probability mean が `[0, 1]` の範囲外の場合に sigmoid で
            probability に変換するかどうか。
            latent f を probability として返してしまう wrapper では True にする。
        eps:
            数値安定化パラメータ。
        objective:
            acquisition が計算した pointwise score に作用する objective。
            InputPerturbation の `q * n_w -> q` 集約には
            `MultiOutputClassificationScoreObjective` を渡す。

    Forward Args:
        X:
            候補点。shape は通常 `batch_shape x q x d`。

    Returns:
        Tensor:
            shape `batch_shape` の acquisition value。

    Notes:
        score は `H(p) = -p log(p) - (1-p) log(1-p)`。
        `p=0.5` で最大となるため、binary classification boundary 付近を
        重点的に選ぶ。
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        apply_sigmoid_if_needed: bool = False,
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
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        raw_X = self._ensure_q_batch(self._as_tensor(X))
        original_batch_shape = raw_X.shape[:-2]
        self._set_eval_mode()

        Xt = self._ensure_q_batch(self._apply_input_transform_safe(raw_X))
        mean_prob = self._get_multioutput_probability_mean(
            raw_X,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
        )

        probs = self._to_probability(
            mean_prob,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            name="probability mean",
        )

        score_per_output = -(
            probs * probs.clamp_min(self.eps).log()
            + (1.0 - probs) * (1.0 - probs).clamp_min(self.eps).log()
        )

        score = self._aggregate_outputs(
            score_per_output,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
        )

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="qMultiOutputBinaryClassEntropyAcquisition",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qMultiOutputBinaryClassEntropyAcquisition")
        return out


class qMultiOutputBinaryICUAcquisition(_MultiOutputLatentStraddleBase):
    """
    Multi-output binary classification ICU acquisition.

    binary classification の contour uncertainty として
    `4 * p * (1 - p)` を評価する level-set estimation 用 acquisition。

    ordinal の `qMultiOutputOrdinalICUAcquisition` に対応する binary 版。
    binary では境界が `p=0.5` の1本なので、各出力に対して
    boundary uncertainty を計算し、出力方向に集約する。

    Args:
        model:
            BoTorch 互換の multi-output binary classification model。
            `model.probability_posterior(X)` があれば優先して使う。
        reduction:
            q-batch 内の pointwise score の集約方法。
            `"mean"` または `"sum"`。
        output_mode:
            出力方向の集約方法。
            `"mean"`, `"sum"`, `"max"`, `"min"`, `"weighted_mean"`。
        output_weights:
            `output_mode="weighted_mean"` のときの出力重み。
        pending_penalty_weight:
            `X_pending` に近い候補点を避けるためのペナルティ係数。
        pending_penalty_beta:
            pending penalty の距離減衰係数。
        apply_sigmoid_if_needed:
            probability mean が `[0, 1]` の範囲外の場合に sigmoid 変換するか。
        eps:
            数値安定化パラメータ。
        objective:
            pointwise score に作用する objective。
            InputPerturbation 集約に使う。

    Forward Args:
        X:
            候補点。shape は通常 `batch_shape x q x d`。

    Returns:
        Tensor:
            shape `batch_shape` の acquisition value。

    Notes:
        `qMultiOutputBinaryProbabilityVariance` 相当の `p(1-p)` に対して、
        本クラスは ICU / contour uncertainty として説明しやすいように
        `4p(1-p)` を使う。最大値は `p=0.5` で 1 になる。
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        apply_sigmoid_if_needed: bool = False,
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
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        raw_X = self._ensure_q_batch(self._as_tensor(X))
        original_batch_shape = raw_X.shape[:-2]
        self._set_eval_mode()

        Xt = self._ensure_q_batch(self._apply_input_transform_safe(raw_X))
        mean_prob = self._get_multioutput_probability_mean(
            raw_X,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
        )

        probs = self._to_probability(
            mean_prob,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            name="probability mean",
        )

        score_per_output = 4.0 * probs * (1.0 - probs)

        score = self._aggregate_outputs(
            score_per_output,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
        )

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="qMultiOutputBinaryICUAcquisition",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qMultiOutputBinaryICUAcquisition")
        return out


class qMultiOutputBinaryBoundaryVarianceAcquisition(_MultiOutputLatentStraddleBase):
    """
    Multi-output binary classification boundary variance acquisition.

    latent posterior の mean / variance を使い、binary boundary 近傍の
    posterior variance を評価する level-set estimation 用 acquisition。

    各出力 j に対して次の score を計算する。

        `score_j(x) = Var[f_j(x)] * exp(-0.5 * ((E[f_j(x)] - threshold_j) / tau)^2)`

    つまり、latent mean が boundary に近く、かつ latent variance が大きい点を
    高く評価する。

    Args:
        model:
            BoTorch 互換の multi-output binary classification model。
            latent posterior を取得するため、`latent_posterior(X)`,
            `posterior_latent(X)`, `posterior_f(X)` があれば優先する。
            なければ `posterior(X)` や `model.models` の submodel posterior を使う。
        thresholds:
            latent boundary の閾値。
            scalar または shape `(m,)` の Tensor を指定できる。
            sigmoid-Bernoulli binary classifier では通常 `0.0` が `p=0.5`
            に対応する。
        tau:
            boundary 近傍をどれくらい広く見るかを決める bandwidth。
            小さいほど boundary 近傍だけを強く評価する。
        reduction:
            q-batch 内の pointwise score の集約方法。
            `"mean"` または `"sum"`。
        output_mode:
            出力方向の集約方法。
            `"mean"`, `"sum"`, `"max"`, `"min"`, `"weighted_mean"`。
        output_weights:
            `output_mode="weighted_mean"` のときの出力重み。
        pending_penalty_weight:
            `X_pending` に近い候補点を避けるためのペナルティ係数。
        pending_penalty_beta:
            pending penalty の距離減衰係数。
        eps:
            数値安定化パラメータ。
        objective:
            pointwise score に作用する objective。
            InputPerturbation の `q * n_w -> q` 集約に使う。

    Forward Args:
        X:
            候補点。shape は通常 `batch_shape x q x d`。

    Returns:
        Tensor:
            shape `batch_shape` の acquisition value。

    Notes:
        `qMultiOutputBinaryICUAcquisition` と
        `qMultiOutputBinaryClassEntropyAcquisition` は probability scale の
        境界不確実性を使う。一方、この acquisition は latent f の分散を使うため、
        GP latent posterior の不確実性をより直接的に反映する。
    """

    def __init__(
        self,
        model,
        thresholds: float | Tensor = 0.0,
        tau: float = 1.0,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
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
        self.thresholds = thresholds
        self.tau = float(tau)
        self.output_mode = output_mode
        self.output_weights = output_weights
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

        m = mu.shape[-1]
        thr = self._threshold_vector(self.thresholds, m, mu.device, mu.dtype)
        thr = thr.view(*((1,) * (mu.ndim - 1)), m)

        tau = torch.as_tensor(
            self.tau,
            device=mu.device,
            dtype=mu.dtype,
        ).clamp_min(self.eps)

        boundary_weight = torch.exp(-0.5 * ((mu - thr) / tau).pow(2))
        score_per_output = var * boundary_weight

        score = self._aggregate_outputs(
            score_per_output,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
        )

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="qMultiOutputBinaryBoundaryVarianceAcquisition",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qMultiOutputBinaryBoundaryVarianceAcquisition")
        return out


class qMultiOutputBinaryLatentStraddleAcquisition(_MultiOutputLatentStraddleAcquisition):
    """qMultiOutputBinaryLatentStraddleAcquisition の Google スタイル API docstring。
    
    Latent Straddle は latent mean が境界に近く、かつ不確実な点を選びます。
    multi-output 版なので、出力方向の集約方法や weights の設定が重要です。 binary classification 系では、posterior が probability 空間か latent 空間かを確認してください。
    
    Args:
        model: BoTorch 互換のモデル。少なくとも `posterior(X)` を持つ必要があります。classification / ordinal では `probability_posterior`、`latent_posterior`、`ordinal_likelihood` などを参照する実装があります。
        beta: UCB / Straddle / hetero sample adjustment で不確実性の重みを決める係数。大きいほど exploration 寄りになります。
        thresholds: multi-output で出力ごとに使う閾値。出力数と長さを一致させます。
        reduction: q-batch 内の点ごとの score をどう集約するか。典型は `mean` または `sum`。
        output_mode: multi-output classification の出力集約方法。`mean`, `sum`, `max`, `min`, `weighted_mean`, `all_positive` など。
        output_weights: multi-output の出力ごとの重み。`weighted_mean` / `weighted_sum` などで使います。
        pending_penalty_weight: X_pending に近い候補を避けるペナルティの強さ。
        pending_penalty_beta: X_pending ペナルティの距離減衰係数。大きいほど近接点だけを強く避けます。
        smooth_abs_eps: この class 固有、または親 class / 内部 acquisition に渡される引数です。
        eps: 数値安定化用の微小値。
        objective: acquisition に渡す objective。BO では posterior samples に作用し、active learning / level-set では計算済み score に作用する場合があります。
        threshold: classification / PoF / boundary 判定の閾値。binary latent classification では通常 `0.0` が `p=0.5` に対応します。
        objective_weights: multi-objective / hetero ordinal で出力・目的を集約する重み。
        aggregation: multi-output regression / deep acquisition で出力次元を集約する方法。
        aggregate: hetero ordinal multi-output で目的方向を集約する方法。`mean`, `weighted_sum`, `product` など。
    
    Forward Args:
        X: 候補点。通常は shape `batch_shape x q x d` です。`q=1` の場合も `optimize_acqf` では q 次元を持つ形で渡されます。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。BoTorch の `optimize_acqf` はこの値を最大化します。
    
    Notes:
        InputPerturbation を使う場合は、`n_w` または `input_perturbation_n_w` と objective の設定を一致させてください。
        binary classification では posterior が probability を返すか latent f を返すかで `samples_are_probs` や `apply_sigmoid_if_needed` の指定が変わります。"""
    pass


class qMultiOutputBinaryJointLatentStraddleAcquisition(_JointMultiOutputLatentStraddleAcquisition):
    """qMultiOutputBinaryJointLatentStraddleAcquisition の Google スタイル API docstring。
    
    Joint Latent Straddle は q-batch 全体の joint uncertainty を使う境界探索 acquisition です。
    multi-output 版なので、出力方向の集約方法や weights の設定が重要です。 binary classification 系では、posterior が probability 空間か latent 空間かを確認してください。
    
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
        binary classification では posterior が probability を返すか latent f を返すかで `samples_are_probs` や `apply_sigmoid_if_needed` の指定が変わります。"""
    pass

__all__ = [
    "qMultiOutputBinaryClassEntropyAcquisition",
    "qMultiOutputBinaryICUAcquisition",
    "qMultiOutputBinaryBoundaryVarianceAcquisition",
    "qMultiOutputBinaryLatentStraddleAcquisition",
    "qMultiOutputBinaryJointLatentStraddleAcquisition",
]
