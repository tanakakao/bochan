from __future__ import annotations

from .common import read, replace_once, replace_regex_once, write


def _replace_first(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: expected at least one match")
    return text.replace(old, new, 1)


def patch_ordinal_single() -> None:
    path = "src/bochan/acquisition/ordinal/active_learning/single_output.py"
    text = read(path)
    text = replace_once(
        text,
        "from bochan.likelihoods.ordinal import OrdinalLogitLikelihood\n",
        "from bochan.acquisition._duplicate_exclusion import (\n"
        "    hard_reference_duplicate_penalty_per_point,\n"
        "    hard_same_batch_duplicate_penalty_per_point,\n"
        ")\n"
        "from bochan.likelihoods.ordinal import OrdinalLogitLikelihood\n",
        label="ordinal single imports",
    )
    text = _replace_first(
        text,
        "        observed_penalty_weight: float = 0.0,\n"
        "        observed_penalty_beta: float = 10.0,\n"
        "        X_pending: Optional[Tensor] = None,\n",
        "        observed_penalty_weight: float = 0.0,\n"
        "        observed_penalty_beta: float = 10.0,\n"
        "        hard_duplicate_tol: float = 1e-8,\n"
        "        exclude_same_batch_duplicates: bool = True,\n"
        "        exclude_pending_duplicates: bool = True,\n"
        "        exclude_observed_duplicates: bool = False,\n"
        "        X_pending: Optional[Tensor] = None,\n",
        label="ordinal single signature",
    )
    text = _replace_first(
        text,
        "        self.observed_penalty_weight = float(observed_penalty_weight)\n"
        "        self.observed_penalty_beta = float(observed_penalty_beta)\n"
        "        self.eps = float(eps)\n",
        "        self.observed_penalty_weight = float(observed_penalty_weight)\n"
        "        self.observed_penalty_beta = float(observed_penalty_beta)\n"
        "        self.hard_duplicate_tol = float(hard_duplicate_tol)\n"
        "        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n"
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n"
        "        if self.hard_duplicate_tol < 0.0:\n"
        "            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n"
        "        self.eps = float(eps)\n",
        label="ordinal single attributes",
    )
    new_method = '''    def _pointwise_reference_penalty(self, Xt: Tensor) -> Tensor:
        penalty = hard_same_batch_duplicate_penalty_per_point(
            Xt,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        if self.pending_penalty_weight > 0.0:
            penalty = penalty + self.pending_penalty_weight * _rbf_reference_penalty_per_point(
                X=Xt,
                X_ref=Xp_t,
                beta=self.pending_penalty_beta,
            )
        penalty = penalty + hard_reference_duplicate_penalty_per_point(
            Xt,
            Xp_t,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
        if self.observed_penalty_weight > 0.0:
            penalty = penalty + self.observed_penalty_weight * _rbf_reference_penalty_per_point(
                X=Xt,
                X_ref=Xobs_t,
                beta=self.observed_penalty_beta,
            )
        penalty = penalty + hard_reference_duplicate_penalty_per_point(
            Xt,
            Xobs_t,
            enabled=self.exclude_observed_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return penalty

'''
    text = replace_regex_once(
        text,
        r"    def _pointwise_reference_penalty\(self, Xt: Tensor\) -> Tensor:\n.*?(?=    def _finalize_pointwise_score)",
        new_method,
        label="ordinal single penalty",
    )
    write(path, text)


def patch_ordinal_multi() -> None:
    path = "src/bochan/acquisition/ordinal/active_learning/multi_output.py"
    text = read(path)
    text = replace_once(
        text,
        "from bochan.likelihoods.ordinal import OrdinalLogitLikelihood\n",
        "from bochan.acquisition._duplicate_exclusion import (\n"
        "    hard_reference_duplicate_penalty_per_point,\n"
        "    hard_same_batch_duplicate_penalty_per_point,\n"
        ")\n"
        "from bochan.likelihoods.ordinal import OrdinalLogitLikelihood\n",
        label="ordinal multi imports",
    )
    text = _replace_first(
        text,
        "        same_batch_penalty_weight: float = 0.0,\n"
        "        same_batch_penalty_beta: float = 10.0,\n"
        "        X_pending: Optional[Tensor] = None,\n",
        "        same_batch_penalty_weight: float = 0.0,\n"
        "        same_batch_penalty_beta: float = 10.0,\n"
        "        hard_duplicate_tol: float = 1e-8,\n"
        "        exclude_same_batch_duplicates: bool = True,\n"
        "        exclude_pending_duplicates: bool = True,\n"
        "        X_pending: Optional[Tensor] = None,\n",
        label="ordinal multi signature",
    )
    text = _replace_first(
        text,
        "        self.same_batch_penalty_weight = float(same_batch_penalty_weight)\n"
        "        self.same_batch_penalty_beta = float(same_batch_penalty_beta)\n"
        "        self.cat_dims = _resolve_cat_dims(model)\n",
        "        self.same_batch_penalty_weight = float(same_batch_penalty_weight)\n"
        "        self.same_batch_penalty_beta = float(same_batch_penalty_beta)\n"
        "        self.hard_duplicate_tol = float(hard_duplicate_tol)\n"
        "        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n"
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        if self.hard_duplicate_tol < 0.0:\n"
        "            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n"
        "        self.cat_dims = _resolve_cat_dims(model)\n",
        label="ordinal multi attributes",
    )
    old_pending = '''    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        return _reference_penalty_per_point(
            Xt,
            Xp_t,
            beta=self.pending_penalty_beta,
            weight=self.pending_penalty_weight,
            cat_dims=self.cat_dims,
        )
'''
    new_pending = '''    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        soft = _reference_penalty_per_point(
            Xt,
            Xp_t,
            beta=self.pending_penalty_beta,
            weight=self.pending_penalty_weight,
            cat_dims=self.cat_dims,
        )
        hard = hard_reference_duplicate_penalty_per_point(
            Xt,
            Xp_t,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return soft + hard
'''
    text = replace_once(text, old_pending, new_pending, label="ordinal multi pending")
    old_same = '''    def _same_batch_penalty_per_point(self, Xt: Tensor) -> Tensor:
        return _same_batch_penalty_per_point(
            Xt,
            beta=self.same_batch_penalty_beta,
            weight=self.same_batch_penalty_weight,
            cat_dims=self.cat_dims,
        )
'''
    new_same = '''    def _same_batch_penalty_per_point(self, Xt: Tensor) -> Tensor:
        soft = _same_batch_penalty_per_point(
            Xt,
            beta=self.same_batch_penalty_beta,
            weight=self.same_batch_penalty_weight,
            cat_dims=self.cat_dims,
        )
        hard = hard_same_batch_duplicate_penalty_per_point(
            Xt,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return soft + hard
'''
    text = replace_once(text, old_same, new_same, label="ordinal multi same batch")
    write(path, text)


def patch_ordinal_hetero(path: str, *, multi_output: bool) -> None:
    text = read(path)
    import_anchor = "from ..hetero_utils import " if multi_output else "from ..hetero_utils import get_hetero_ordinal_summary, get_noise_sigma\n"
    if multi_output:
        marker = "from ..hetero_utils import (\n"
        text = replace_once(
            text,
            marker,
            "from bochan.acquisition._duplicate_exclusion import (\n"
            "    hard_reference_duplicate_penalty_per_point,\n"
            "    hard_same_batch_duplicate_penalty_per_point,\n"
            ")\n\n"
            + marker,
            label=f"{path} imports",
        )
    else:
        text = replace_once(
            text,
            import_anchor,
            "from bochan.acquisition._duplicate_exclusion import (\n"
            "    hard_reference_duplicate_penalty_per_point,\n"
            "    hard_same_batch_duplicate_penalty_per_point,\n"
            ")\n"
            + import_anchor,
            label=f"{path} imports",
        )
    text = _replace_first(
        text,
        "        pending_penalty_weight: float = 0.0,\n"
        "        pending_penalty_beta: float = 10.0,\n"
        "        X_pending: Tensor | None = None,\n",
        "        pending_penalty_weight: float = 0.0,\n"
        "        pending_penalty_beta: float = 10.0,\n"
        "        hard_duplicate_tol: float = 1e-8,\n"
        "        exclude_same_batch_duplicates: bool = True,\n"
        "        exclude_pending_duplicates: bool = True,\n"
        "        X_pending: Tensor | None = None,\n",
        label=f"{path} signature",
    )
    text = _replace_first(
        text,
        "        self.pending_penalty_weight = float(pending_penalty_weight)\n"
        "        self.pending_penalty_beta = float(pending_penalty_beta)\n",
        "        self.pending_penalty_weight = float(pending_penalty_weight)\n"
        "        self.pending_penalty_beta = float(pending_penalty_beta)\n"
        "        self.hard_duplicate_tol = float(hard_duplicate_tol)\n"
        "        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n"
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        if self.hard_duplicate_tol < 0.0:\n"
        "            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n",
        label=f"{path} attributes",
    )
    new_method = '''    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        Xp_t = self._transform_reference_like_candidate(self.X_pending, ref=Xt)
        zeros = Xt.new_zeros(Xt.shape[:-1])
        if Xp_t is None or Xp_t.numel() == 0:
            pending = zeros
        else:
            d = Xt.shape[-1]
            X2d = Xt.reshape(-1, d)
            Xp2d = Xp_t.reshape(-1, Xp_t.shape[-1])
            if Xp2d.shape[-1] != d:
                raise RuntimeError(
                    "X_pending feature dimension mismatch after transform: "
                    f"Xt.shape={tuple(Xt.shape)}, X_pending_transformed.shape={tuple(Xp_t.shape)}."
                )
            dist = torch.cdist(X2d, Xp2d).min(dim=-1).values.reshape(*Xt.shape[:-1])
            pending = (
                self.pending_penalty_weight
                * torch.exp(-self.pending_penalty_beta * dist)
                if self.pending_penalty_weight > 0.0
                else zeros
            )
            pending = pending + hard_reference_duplicate_penalty_per_point(
                Xt,
                Xp_t,
                enabled=self.exclude_pending_duplicates,
                tolerance=self.hard_duplicate_tol,
            )
        return pending + hard_same_batch_duplicate_penalty_per_point(
            Xt,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

'''
    text = replace_regex_once(
        text,
        r"    def _pending_penalty_per_point\(self, Xt: Tensor\) -> Tensor:\n.*?(?=    def _summary)",
        new_method,
        label=f"{path} pending method",
    )
    write(path, text)
