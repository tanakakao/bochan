from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    file_path = ROOT / path
    text = file_path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:120]!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_before(path: str, marker: str, addition: str) -> None:
    file_path = ROOT / path
    text = file_path.read_text(encoding="utf-8")
    if addition in text:
        return
    if marker not in text:
        raise RuntimeError(f"Marker not found in {path}: {marker!r}")
    file_path.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")


def write(path: str, content: str) -> None:
    file_path = ROOT / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Web active-learning kwargs: split single/multi-output and make NIPV task-aware
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''def _set_active_learning_reference_kwargs(
    acqf_kwargs: dict[str, Any],
    *,
    acq_key: str,
    train_x: Any,
) -> None:
    """Attach only the reference inputs supported by the selected AL acquisition.

    True NIPV consumes ``mc_points`` as its integration set and has no
    ``X_observed`` argument. Pointwise uncertainty acquisitions use
    ``X_observed`` for optional observed-point penalties / exclusion.
    """
    if acq_key in {"nipv", "qnipv"}:
        acqf_kwargs.setdefault("mc_points", train_x)
        return
    acqf_kwargs.setdefault("X_observed", train_x)


''',
    '''def _set_active_learning_output_kwargs(
    acqf_kwargs: dict[str, Any],
    *,
    task_type: str,
    multi_output: bool,
    output_weights: Any,
) -> None:
    """Attach output aggregation kwargs only to true multi-output AL acquisitions."""
    if not multi_output:
        return

    acqf_kwargs.setdefault("output_weights", output_weights)
    if task_type in {"binary", "multiclass", "ordinal"}:
        acqf_kwargs.setdefault("output_mode", "weighted_mean")
    else:
        acqf_kwargs.setdefault("output_reduction", "weighted_mean")


def _set_active_learning_reference_kwargs(
    acqf_kwargs: dict[str, Any],
    *,
    acq_key: str,
    train_x: Any,
    task_type: str = "regression",
) -> None:
    """Attach task-appropriate observed / integration references for AL.

    Gaussian regression true NIPV accepts ``mc_points`` but not ``X_observed``.
    Binary, multiclass, and ordinal NIPV implementations use integration points
    while retaining the normal observed-point duplicate / repulsion controls.
    Other pointwise active-learning criteria only need ``X_observed``.
    """
    if acq_key in {"nipv", "qnipv"}:
        acqf_kwargs.setdefault("mc_points", train_x)
        if task_type in {"binary", "multiclass", "ordinal"}:
            acqf_kwargs.setdefault("X_observed", train_x)
        return
    acqf_kwargs.setdefault("X_observed", train_x)


''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''        objective_config = None
        acqf_kwargs.setdefault(
            "output_weights",
            objective_weights(
                target_columns=target_columns,
                objective_targets=objective_targets,
            ),
        )
        acqf_kwargs.setdefault("output_reduction", "weighted_mean")
        _set_active_learning_reference_kwargs(
            acqf_kwargs,
            acq_key=acq_key,
            train_x=train_x,
        )
''',
    '''        objective_config = None
        active_learning_task = (
            internal_tasks[0]
            if internal_tasks and all(task == internal_tasks[0] for task in internal_tasks)
            else "hybrid"
        )
        _set_active_learning_output_kwargs(
            acqf_kwargs,
            task_type=active_learning_task,
            multi_output=len(target_columns) > 1,
            output_weights=objective_weights(
                target_columns=target_columns,
                objective_targets=objective_targets,
            ),
        )
        _set_active_learning_reference_kwargs(
            acqf_kwargs,
            acq_key=acq_key,
            train_x=train_x,
            task_type=active_learning_task,
        )
''',
)


# ---------------------------------------------------------------------------
# One-output Hybrid: use the sole submodel for single-output acquisition classes
# ---------------------------------------------------------------------------
insert_before(
    "src/bochan/api/factory.py",
    "def build_acquisition(bundle: ModelBundle, config: AcquisitionConfig, data_context: DataContext | None = None) -> Any:\n",
    '''def _model_for_acquisition(bundle: ModelBundle, acqf_cls: Any) -> Any:
    """Return the effective model expected by the resolved acquisition class.

    A one-output Hybrid wrapper is retained for prediction / objective plumbing,
    but task-specific single-output acquisitions need the underlying classifier
    or ordinal model in order to access latent posteriors and likelihoods.
    True multi-output acquisitions continue to receive the Hybrid wrapper.
    """
    model = bundle.model
    if str(bundle.task_type) != "hybrid" or acqf_cls is None:
        return model

    acqf_name = str(getattr(acqf_cls, "__name__", ""))
    if "MultiOutput" in acqf_name:
        return model

    metadata = getattr(bundle, "metadata", {}) or {}
    sub_bundles = list(metadata.get("sub_bundles") or [])
    if len(sub_bundles) == 1:
        submodel = getattr(sub_bundles[0], "model", None)
        if submodel is not None:
            return submodel

    specs = getattr(model, "specs", None)
    if specs is not None and len(specs) == 1:
        submodel = getattr(specs[0], "model", None)
        if submodel is not None:
            return submodel

    models = getattr(model, "models", None)
    if models is not None and len(models) == 1:
        return models[0]

    return model


''',
)
replace_once(
    "src/bochan/api/factory.py",
    '    kwargs = {"model": bundle.model}\n',
    '    kwargs = {"model": _model_for_acquisition(bundle, config.acqf_cls)}\n',
)


# ---------------------------------------------------------------------------
# Binary NIPV proxy: expose the same reference / duplicate controls as pointwise AL
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/acquisition/binary/active_learning/integrated_posterior_variance.py",
    '''        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            num_samples=num_epistemic_samples,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
            objective=objective,
        )
''',
    '''        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            num_samples=num_epistemic_samples,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            X_pending=X_pending,
            X_observed=X_observed,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
            objective=objective,
        )
''',
)


# ---------------------------------------------------------------------------
# Multiclass public reference handling: sequences + shared observed-X resolver
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py",
    '''from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)
''',
    '''from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
    resolve_observed_X,
)
''',
)
replace_once(
    "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py",
    '''def _resolve_observed_X(model: Model, X_observed: Tensor | None = None) -> Tensor | None:
    if X_observed is not None:
        return X_observed
    for attr in ("train_X_original", "train_X", "train_inputs_raw"):
        x = getattr(model, attr, None)
        if x is not None:
            return x[0] if isinstance(x, tuple) else x
    x = getattr(model, "train_inputs", None)
    if isinstance(x, tuple) and len(x) > 0:
        return x[0]
    return None
''',
    '''def _resolve_observed_X(model: Model, X_observed: Tensor | None = None) -> Tensor | None:
    return resolve_observed_X(model, X_observed)
''',
)
replace_once(
    "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py",
    '''    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = None if X_pending is None else torch.as_tensor(X_pending).detach()

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        resolved = _resolve_observed_X(self.model, X_observed)
        self.X_observed = None if resolved is None else torch.as_tensor(resolved).detach()
''',
    '''    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = _coerce_reference_tensor(X_pending)

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        self.X_observed = _coerce_reference_tensor(
            _resolve_observed_X(self.model, X_observed)
        )
''',
)

replace_once(
    "src/bochan/acquisition/multiclass/active_learning/multi_output.py",
    '''from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
    resolve_observed_X,
)
''',
    '''from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
    resolve_observed_X,
)
from bochan.acquisition.multiclass.bayesian_optimization.single_output import (
    _coerce_reference_tensor,
)
''',
)
replace_once(
    "src/bochan/acquisition/multiclass/active_learning/multi_output.py",
    '''    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = None if X_pending is None else torch.as_tensor(X_pending).detach()

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        resolved = resolve_observed_X(self.model, X_observed)
        self.X_observed = None if resolved is None else torch.as_tensor(resolved).detach()
''',
    '''    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = _coerce_reference_tensor(X_pending)

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        self.X_observed = _coerce_reference_tensor(
            resolve_observed_X(self.model, X_observed)
        )
''',
)


# ---------------------------------------------------------------------------
# Ordinal single-output duplicate controls and differentiable NIPV proxy
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    '''from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)
''',
    '''from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
    resolve_observed_X,
)
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    '''def _resolve_observed_X(
    model: Model,
    X_observed: Optional[Tensor] = None,
) -> Optional[Tensor]:
    if X_observed is not None:
        return X_observed

    for attr in ("train_X_original", "train_X", "train_inputs_raw"):
        x = getattr(model, attr, None)
        if x is not None:
            return x

    x = getattr(model, "train_inputs", None)
    if isinstance(x, tuple) and len(x) > 0:
        return x[0]

    return None
''',
    '''def _resolve_observed_X(
    model: Model,
    X_observed: Optional[Tensor] = None,
) -> Optional[Tensor]:
    return resolve_observed_X(model, X_observed)
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    '        exclude_observed_duplicates: bool = False,\n',
    '        exclude_observed_duplicates: bool = True,\n',
)

# qOrdinalBALD public duplicate controls.
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    '''        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        if sampler is None:
''',
    '''        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        if sampler is None:
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    '''            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
            objective=objective,
        )
        self.num_samples = int(num_samples)
''',
    '''            X_pending=X_pending,
            X_observed=X_observed,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            eps=eps,
            objective=objective,
        )
        self.num_samples = int(num_samples)
''',
)

# qOrdinalUtilityVariance has the same public duplicate controls.
utility_marker = '''class qOrdinalUtilityVariance(_qOrdinalActiveLearningBase):'''
ordinal_path = ROOT / "src/bochan/acquisition/ordinal/active_learning/single_output.py"
ordinal_text = ordinal_path.read_text(encoding="utf-8")
start = ordinal_text.index(utility_marker)
end = ordinal_text.index("class qOrdinalMarginUncertainty", start)
utility_block = ordinal_text[start:end]
if "exclude_observed_duplicates: bool = True" not in utility_block:
    utility_block = utility_block.replace(
        '''        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
''',
        '''        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        eps: float = 1e-6,
''',
        1,
    )
    utility_block = utility_block.replace(
        '''            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
''',
        '''            X_pending=X_pending,
            X_observed=X_observed,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            eps=eps,
''',
        1,
    )
    ordinal_text = ordinal_text[:start] + utility_block + ordinal_text[end:]
    ordinal_path.write_text(ordinal_text, encoding="utf-8")

insert_before(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    "# =========================================================\n# Expensive / true-ish NIPV acquisition\n# =========================================================\n",
    '''class qOrdinalIntegratedPosteriorVarianceProxy(qOrdinalUtilityVariance):
    """Differentiable ordinal integrated-utility-variance proxy.

    This mirrors the Binary / Multiclass differentiable NIPV proxies used with
    ``optimize_acqf``. It scores candidates by coverage of high utility-variance
    integration points and retains the standard pending / observed / duplicate
    controls from :class:`qOrdinalUtilityVariance`.
    """

    def __init__(
        self,
        model: Model,
        *,
        mc_points: Optional[Tensor] = None,
        integration_beta: float = 25.0,
        local_weight: Optional[float] = None,
        integrated_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if integration_beta <= 0.0:
            raise ValueError("integration_beta must be positive.")
        if mc_points is not None and mc_points.ndim != 2:
            raise ValueError(
                "mc_points must have shape [n_mc, d]. "
                f"Got {tuple(mc_points.shape)}."
            )
        self.mc_points = None if mc_points is None else mc_points.detach().clone()
        self.integration_beta = float(integration_beta)
        self.local_weight = (
            1.0 if local_weight is None and mc_points is None else float(local_weight or 0.0)
        )
        self.integrated_weight = float(integrated_weight)

    def _utility_variance_score(self, X: Tensor) -> Tensor:
        probs = self._class_probs_from_posterior(X)
        utilities = _utility_values_tensor(
            self.utility_values,
            probs.shape[-1],
            device=probs.device,
            dtype=probs.dtype,
        )
        mean_u = (probs * utilities).sum(dim=-1)
        second_u = (probs * utilities.pow(2)).sum(dim=-1)
        return (second_u - mean_u.pow(2)).clamp_min(0.0)

    def _integrated_score(self, Xt: Tensor) -> Tensor:
        if self.mc_points is None:
            return Xt.new_zeros(Xt.shape[:-1])

        mc_raw = self.mc_points.to(device=Xt.device, dtype=Xt.dtype)
        mc_score = self._utility_variance_score(mc_raw).reshape(-1)
        mc_t = _apply_input_transform_for_reference(self.model, mc_raw).reshape(
            -1, Xt.shape[-1]
        )
        if mc_score.numel() != mc_t.shape[-2]:
            if mc_t.shape[-2] % mc_score.numel() == 0:
                mc_score = mc_score.repeat_interleave(mc_t.shape[-2] // mc_score.numel())
            elif mc_score.numel() % mc_t.shape[-2] == 0:
                mc_score = mc_score.reshape(mc_t.shape[-2], -1).mean(dim=-1)
            else:
                raise RuntimeError("Could not align ordinal mc_points and utility variance.")

        d2 = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), mc_t.detach()).pow(2)
        weights = torch.exp(-self.integration_beta * d2)
        score = (weights * mc_score.detach().reshape(1, -1)).sum(dim=-1)
        score = score / weights.sum(dim=-1).clamp_min(self.eps)
        return score.reshape(*Xt.shape[:-1])

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = _ensure_q_batch(X)
        Xt = _apply_input_transform_for_reference(self.model, raw_X)
        local_score = self._utility_variance_score(raw_X)
        local_score = _align_pointwise_score_to_X(
            local_score,
            Xt,
            name="qOrdinalIntegratedPosteriorVarianceProxy local score",
        )
        integrated_score = self._integrated_score(Xt)
        score = self.local_weight * local_score + self.integrated_weight * integrated_score
        return self._finalize_pointwise_score(
            score,
            raw_X,
            name="qOrdinalIntegratedPosteriorVarianceProxy",
        )


''',
)

# Explicit fantasy implementation also exposes hard duplicate controls.
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    '''        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        if mc_points.ndim != 2:
''',
    '''        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        if mc_points.ndim != 2:
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    '''        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.eps = float(eps)
''',
    '''        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.hard_duplicate_tol = float(hard_duplicate_tol)
        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)
        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)
        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)
        if self.hard_duplicate_tol < 0.0:
            raise ValueError("hard_duplicate_tol must be non-negative.")
        self.eps = float(eps)
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    '''    def _aggregated_reference_penalty(self, X: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_reference(self.model, X)
        penalty = torch.zeros(Xt.shape[:-2], device=Xt.device, dtype=Xt.dtype)

        if self.pending_penalty_weight > 0.0:
            Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
            penalty = penalty + self.pending_penalty_weight * _rbf_reference_penalty_aggregated(
                X=Xt,
                X_ref=Xp_t,
                beta=self.pending_penalty_beta,
                reduction="sum",
            )

        if self.observed_penalty_weight > 0.0:
            Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
            penalty = penalty + self.observed_penalty_weight * _rbf_reference_penalty_aggregated(
                X=Xt,
                X_ref=Xobs_t,
                beta=self.observed_penalty_beta,
                reduction="sum",
            )

        return penalty
''',
    '''    def _aggregated_reference_penalty(self, X: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_reference(self.model, X)
        pointwise = hard_same_batch_duplicate_penalty_per_point(
            Xt,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        if self.pending_penalty_weight > 0.0:
            pointwise = pointwise + self.pending_penalty_weight * _rbf_reference_penalty_per_point(
                X=Xt,
                X_ref=Xp_t,
                beta=self.pending_penalty_beta,
            )
        pointwise = pointwise + hard_reference_duplicate_penalty_per_point(
            Xt,
            Xp_t,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
        if self.observed_penalty_weight > 0.0:
            pointwise = pointwise + self.observed_penalty_weight * _rbf_reference_penalty_per_point(
                X=Xt,
                X_ref=Xobs_t,
                beta=self.observed_penalty_beta,
            )
        pointwise = pointwise + hard_reference_duplicate_penalty_per_point(
            Xt,
            Xobs_t,
            enabled=self.exclude_observed_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return pointwise.sum(dim=-1)
''',
)


# ---------------------------------------------------------------------------
# Ordinal multi-output: hard observed exclusion matches Binary / Multiclass
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/multi_output.py",
    '''from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)
''',
    '''from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
    resolve_observed_X,
)
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/multi_output.py",
    '''def _resolve_observed_X(model: Model, X_observed: Optional[Tensor] = None) -> Optional[Tensor]:
    if X_observed is not None:
        return X_observed
    for attr in ("train_X_original", "train_X", "train_inputs_raw"):
        x = getattr(model, attr, None)
        if x is not None:
            return x
    train_inputs = getattr(model, "train_inputs", None)
    if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
        return train_inputs[0]
    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        sm = models[0]
        for attr in ("train_X_original", "train_X", "train_inputs_raw"):
            x = getattr(sm, attr, None)
            if x is not None:
                return x
        train_inputs = getattr(sm, "train_inputs", None)
        if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
            return train_inputs[0]
    return None
''',
    '''def _resolve_observed_X(model: Model, X_observed: Optional[Tensor] = None) -> Optional[Tensor]:
    return resolve_observed_X(model, X_observed)
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/multi_output.py",
    '''        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
''',
    '''        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/multi_output.py",
    '''        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)
        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)
        if self.hard_duplicate_tol < 0.0:
''',
    '''        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)
        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)
        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)
        if self.hard_duplicate_tol < 0.0:
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/multi_output.py",
    '''    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
        return _reference_penalty_per_point(
            Xt,
            Xobs_t,
            beta=self.observed_penalty_beta,
            weight=self.observed_penalty_weight,
            cat_dims=self.cat_dims,
        )
''',
    '''    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
        soft = _reference_penalty_per_point(
            Xt,
            Xobs_t,
            beta=self.observed_penalty_beta,
            weight=self.observed_penalty_weight,
            cat_dims=self.cat_dims,
        )
        hard = hard_reference_duplicate_penalty_per_point(
            Xt,
            Xobs_t,
            enabled=self.exclude_observed_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return soft + hard
''',
)


# ---------------------------------------------------------------------------
# Ordinal package/registry: default NIPV is differentiable for optimize_acqf
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/__init__.py",
    '''from .single_output import (
    qOrdinalBALD,
    qOrdinalFantasyNegIntegratedPosteriorVariance,
    qOrdinalMarginUncertainty,
    qOrdinalPredictiveEntropy,
    qOrdinalUtilityVariance,
)
''',
    '''from .single_output import (
    qOrdinalBALD,
    qOrdinalFantasyNegIntegratedPosteriorVariance as qOrdinalFantasyNegIntegratedPosteriorVarianceEvo,
    qOrdinalIntegratedPosteriorVarianceProxy,
    qOrdinalMarginUncertainty,
    qOrdinalPredictiveEntropy,
    qOrdinalUtilityVariance,
)

# The contextual short name ``NIPV`` is optimized with gradient-based
# ``optimize_acqf`` in the Web workbench. Keep the refit/fantasy implementation
# explicit as ``Evo`` and expose the differentiable integrated-variance proxy as
# the standard package-level NIPV target, matching the Binary API.
qOrdinalFantasyNegIntegratedPosteriorVariance = qOrdinalIntegratedPosteriorVarianceProxy
''',
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/__init__.py",
    '''    "qOrdinalMarginUncertainty",
    "qOrdinalFantasyNegIntegratedPosteriorVariance",
]
''',
    '''    "qOrdinalMarginUncertainty",
    "qOrdinalIntegratedPosteriorVarianceProxy",
    "qOrdinalFantasyNegIntegratedPosteriorVariance",
    "qOrdinalFantasyNegIntegratedPosteriorVarianceEvo",
]
''',
)
replace_once(
    "src/bochan/api/acquisition_registry.py",
    '''        "qOrdinalMarginUncertainty",
        "qOrdinalFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputOrdinalPredictiveEntropy",
''',
    '''        "qOrdinalMarginUncertainty",
        "qOrdinalIntegratedPosteriorVarianceProxy",
        "qOrdinalFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputOrdinalPredictiveEntropy",
''',
)

# Existing routing test should reflect the public differentiable Ordinal NIPV.
replace_once(
    "tests/test_single_output_hybrid_acquisition_routing.py",
    '("ordinal", "NIPV", "qOrdinalFantasyNegIntegratedPosteriorVariance"),\n',
    '("ordinal", "NIPV", "qOrdinalIntegratedPosteriorVarianceProxy"),\n',
)


# ---------------------------------------------------------------------------
# Focused tests
# ---------------------------------------------------------------------------
write(
    "tests/test_classification_active_learning_web_kwargs.py",
    '''from __future__ import annotations

import pytest
import torch

from bochan.serving.webapp.workflows_tabular import (
    _set_active_learning_output_kwargs,
    _set_active_learning_reference_kwargs,
)


@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass", "ordinal"])
def test_single_output_web_active_learning_does_not_attach_output_reduction(task_type: str) -> None:
    kwargs: dict[str, object] = {}
    _set_active_learning_output_kwargs(
        kwargs,
        task_type=task_type,
        multi_output=False,
        output_weights=[1.0],
    )
    assert "output_weights" not in kwargs
    assert "output_mode" not in kwargs
    assert "output_reduction" not in kwargs


@pytest.mark.parametrize("task_type", ["binary", "multiclass", "ordinal"])
def test_multi_output_classification_web_active_learning_uses_output_mode(task_type: str) -> None:
    kwargs: dict[str, object] = {}
    _set_active_learning_output_kwargs(
        kwargs,
        task_type=task_type,
        multi_output=True,
        output_weights=[0.25, 0.75],
    )
    assert kwargs["output_weights"] == [0.25, 0.75]
    assert kwargs["output_mode"] == "weighted_mean"
    assert "output_reduction" not in kwargs


def test_multi_output_regression_web_active_learning_keeps_output_reduction() -> None:
    kwargs: dict[str, object] = {}
    _set_active_learning_output_kwargs(
        kwargs,
        task_type="regression",
        multi_output=True,
        output_weights=[0.25, 0.75],
    )
    assert kwargs["output_weights"] == [0.25, 0.75]
    assert kwargs["output_reduction"] == "weighted_mean"
    assert "output_mode" not in kwargs


@pytest.mark.parametrize("task_type", ["binary", "multiclass", "ordinal"])
def test_classification_nipv_web_kwargs_include_mc_points_and_observed(task_type: str) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {}
    _set_active_learning_reference_kwargs(
        kwargs,
        acq_key="nipv",
        train_x=train_x,
        task_type=task_type,
    )
    assert kwargs["mc_points"] is train_x
    assert kwargs["X_observed"] is train_x


def test_regression_true_nipv_web_kwargs_still_omit_x_observed() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {}
    _set_active_learning_reference_kwargs(
        kwargs,
        acq_key="nipv",
        train_x=train_x,
        task_type="regression",
    )
    assert kwargs["mc_points"] is train_x
    assert "X_observed" not in kwargs


@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass", "ordinal"])
def test_pointwise_web_active_learning_uses_x_observed(task_type: str) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {}
    _set_active_learning_reference_kwargs(
        kwargs,
        acq_key="bald",
        train_x=train_x,
        task_type=task_type,
    )
    assert kwargs["X_observed"] is train_x
    assert "mc_points" not in kwargs
''',
)

write(
    "tests/test_classification_active_learning_hybrid_e2e.py",
    '''from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.optim import optimize_acqf

from bochan.api import AcquisitionConfig, ModelConfig
from bochan.api.configs import DataContext, ModelBundle
from bochan.api.engine import BayesianOptimizer
from bochan.api.factory import build_acquisition
from bochan.models.classification.binary.base import BinaryClassificationGPModel
from bochan.models.classification.multiclass.base import MulticlassClassificationGPModel
from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec
from bochan.models.ordinal.base import OrdinalGPModel


def _training_data(task_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.tensor(
        [[0.0], [0.18], [0.36], [0.55], [0.73], [0.91]],
        dtype=torch.double,
    )
    if task_type == "binary":
        train_y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.double)
    elif task_type in {"multiclass", "ordinal"}:
        train_y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    else:
        raise ValueError(task_type)
    return train_x, train_y


def _submodel(task_type: str, train_x: torch.Tensor, train_y: torch.Tensor):
    if task_type == "binary":
        return BinaryClassificationGPModel(train_X=train_x, train_Y=train_y, num_inducing_points=6)
    if task_type == "multiclass":
        return MulticlassClassificationGPModel(
            train_X=train_x,
            train_Y=train_y,
            num_classes=3,
            num_inducing_points=6,
        )
    if task_type == "ordinal":
        return OrdinalGPModel(train_X=train_x, train_Y=train_y, num_classes=3)
    raise ValueError(task_type)


def _hybrid_bundle(task_type: str) -> tuple[ModelBundle, object, torch.Tensor]:
    train_x, train_y = _training_data(task_type)
    submodel = _submodel(task_type, train_x, train_y)
    hybrid = HybridMultiOutputModel(
        [OutputSpec(name="target", task_type=task_type, model=submodel)]
    )
    sub_bundle = SimpleNamespace(
        task_type=task_type,
        model_type="base",
        model=submodel,
    )
    bundle = ModelBundle(
        model=hybrid,
        train_X=train_x,
        train_Y=train_y,
        model_config=ModelConfig(task_type="hybrid", model_type="base"),
        task_type="hybrid",
        model_type="base",
        metadata={"multi_output": True, "sub_bundles": [sub_bundle]},
    )
    return bundle, submodel, train_x


def _resolved_acquisition(bundle: ModelBundle, name: str, train_x: torch.Tensor):
    optimizer = BayesianOptimizer.__new__(BayesianOptimizer)
    optimizer.bundle = bundle
    optimizer.model = bundle.model
    optimizer.acquisition_registry = None

    kwargs: dict[str, object] = {
        "X_observed": train_x,
        "exclude_observed_duplicates": False,
    }
    if name.lower() == "nipv":
        kwargs["mc_points"] = train_x

    config = optimizer._resolve_acquisition_config(
        AcquisitionConfig(name=name, acqf_kwargs=kwargs)
    )
    acquisition = build_acquisition(
        bundle=bundle,
        config=config,
        data_context=DataContext(X_baseline=train_x),
    )
    return config, acquisition


@pytest.mark.parametrize("task_type", ["binary", "multiclass", "ordinal"])
@pytest.mark.parametrize("name", ["variance", "predictive_entropy", "BALD", "NIPV"])
def test_one_output_hybrid_classification_active_learning_runs_optimize_acqf(
    task_type: str,
    name: str,
) -> None:
    torch.manual_seed(0)
    bundle, submodel, train_x = _hybrid_bundle(task_type)
    config, acquisition = _resolved_acquisition(bundle, name, train_x)

    assert config.acqf_cls is not None
    assert "MultiOutput" not in config.acqf_cls.__name__
    assert acquisition.model is submodel

    candidates, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=12,
        options={"maxiter": 30},
    )

    assert candidates.shape == torch.Size([1, 1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(torch.as_tensor(value)).all()
''',
)

write(
    "tests/test_classification_active_learning_input_perturbation_contract.py",
    '''from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.acquisition.binary.active_learning import qBinaryPredictiveEntropy
from bochan.acquisition.multiclass.active_learning import qMulticlassPredictiveEntropy
from bochan.acquisition.ordinal.active_learning import qOrdinalPredictiveEntropy


class _RecordingExpandTransform(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.inputs.append(X.detach().clone())
        offset = torch.full_like(X, 0.03)
        out = torch.stack([X, X + offset], dim=-2).flatten(-3, -2)
        self.outputs.append(out.detach().clone())
        return out


class _RecordingObjective(torch.nn.Module):
    def __init__(self, n_w: int = 2) -> None:
        super().__init__()
        self.n_w = int(n_w)
        self.seen_X: torch.Tensor | None = None

    def forward(self, score: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        assert X is not None
        self.seen_X = X.detach().clone()
        q = int(X.shape[-2])
        return score.reshape(*score.shape[:-1], q, self.n_w).mean(dim=-1)


class _BinaryPosteriorModel(torch.nn.Module):
    def __init__(self, transform: _RecordingExpandTransform, train_x: torch.Tensor) -> None:
        super().__init__()
        self.input_transform = transform
        self.train_inputs_raw = (train_x,)

    def posterior(self, X: torch.Tensor):
        Xt = self.input_transform(X)
        prob = torch.sigmoid(4.0 * (Xt[..., :1] - 0.5))
        return SimpleNamespace(mean=prob)


class _MulticlassPosteriorModel(torch.nn.Module):
    num_classes = 3

    def __init__(self, transform: _RecordingExpandTransform, train_x: torch.Tensor) -> None:
        super().__init__()
        self.input_transform = transform
        self.train_inputs_raw = (train_x,)

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        Xt = self.input_transform(X)
        z = Xt[..., 0]
        logits = torch.stack([-(z - 0.15).square(), -(z - 0.5).square(), -(z - 0.85).square()], dim=-1)
        return torch.softmax(8.0 * logits, dim=-1)


class _FakeOrdinalLikelihood(torch.nn.Module):
    num_classes = 3

    def marginal_class_probs(self, distribution) -> torch.Tensor:
        return distribution.probs

    def class_probs_from_f(self, f: torch.Tensor) -> torch.Tensor:
        z = f.squeeze(-1) if f.ndim > 0 and f.shape[-1] == 1 else f
        logits = torch.stack([-(z + 0.7).square(), -z.square(), -(z - 0.7).square()], dim=-1)
        return torch.softmax(logits, dim=-1)


class _OrdinalPosteriorModel(torch.nn.Module):
    def __init__(self, transform: _RecordingExpandTransform, train_x: torch.Tensor) -> None:
        super().__init__()
        self.input_transform = transform
        self.train_inputs_raw = (train_x,)
        self.ordinal_likelihood = _FakeOrdinalLikelihood()
        self.likelihood = self.ordinal_likelihood

    def posterior(self, X: torch.Tensor):
        Xt = self.input_transform(X)
        z = 4.0 * (Xt[..., 0] - 0.5)
        logits = torch.stack([-(z + 1.0).square(), -z.square(), -(z - 1.0).square()], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return SimpleNamespace(distribution=SimpleNamespace(probs=probs))


def _build_case(task_type: str, train_x: torch.Tensor, objective: _RecordingObjective):
    transform = _RecordingExpandTransform()
    if task_type == "binary":
        model = _BinaryPosteriorModel(transform, train_x)
        acquisition = qBinaryPredictiveEntropy(
            model=model,
            objective=objective,
            X_pending=train_x[:1],
            pending_penalty_weight=0.2,
            exclude_same_batch_duplicates=False,
            exclude_pending_duplicates=False,
            exclude_observed_duplicates=False,
        )
    elif task_type == "multiclass":
        model = _MulticlassPosteriorModel(transform, train_x)
        acquisition = qMulticlassPredictiveEntropy(
            model=model,
            objective=objective,
            X_pending=train_x[:1],
            pending_penalty_weight=0.2,
            exclude_same_batch_duplicates=False,
            exclude_pending_duplicates=False,
            exclude_observed_duplicates=False,
        )
    elif task_type == "ordinal":
        model = _OrdinalPosteriorModel(transform, train_x)
        acquisition = qOrdinalPredictiveEntropy(
            model=model,
            objective=objective,
            X_pending=train_x[:1],
            pending_penalty_weight=0.2,
            exclude_same_batch_duplicates=False,
            exclude_pending_duplicates=False,
            exclude_observed_duplicates=False,
        )
    else:
        raise ValueError(task_type)
    return transform, acquisition


@pytest.mark.parametrize("task_type", ["binary", "multiclass", "ordinal"])
def test_input_perturbation_contract_uses_raw_x_for_objective_and_transformed_x_for_distance(
    task_type: str,
) -> None:
    train_x = torch.tensor([[0.1], [0.9]], dtype=torch.double)
    raw_x = torch.tensor([[[0.25], [0.7]]], dtype=torch.double)
    objective = _RecordingObjective(n_w=2)
    transform, acquisition = _build_case(task_type, train_x, objective)

    value = acquisition(raw_x)

    assert torch.isfinite(value).all()
    assert objective.seen_X is not None
    assert torch.allclose(objective.seen_X, raw_x)
    assert objective.seen_X.shape[-2] == 2
    assert any(output.shape[-2] == 4 for output in transform.outputs)
    assert any(
        input_tensor.shape[-2] == 1 and torch.allclose(input_tensor.reshape(-1, 1), train_x[:1])
        for input_tensor in transform.inputs
    )
''',
)
