from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"pattern not found in {path}: {old[:120]!r}")
    if text.count(old) != 1:
        raise RuntimeError(f"pattern not unique in {path}: count={text.count(old)}")
    p.write_text(text.replace(old, new), encoding="utf-8")


# Shared one-output wrapper unwrapping helper.
p = Path("src/bochan/acquisition/_duplicate_exclusion.py")
text = p.read_text(encoding="utf-8")
if "def unwrap_single_output_model(" not in text:
    text += '''\n\ndef unwrap_single_output_model(model):\n    """Return the sole submodel from a one-output wrapper.\n\n    One-output ``HybridMultiOutputModel`` instances are retained as model\n    containers by the Web/API layer, while acquisition routing intentionally\n    resolves them to single-output acquisition classes.  Classification and\n    ordinal single-output acquisitions need the task-native submodel (likelihood,\n    class probabilities, latent posterior), not the wrapper's scalar objective\n    posterior.\n    """\n    specs = getattr(model, "specs", None)\n    models = getattr(model, "models", None)\n    if specs is not None and models is not None and len(specs) == 1 and len(models) == 1:\n        return models[0]\n    return model\n'''
    p.write_text(text, encoding="utf-8")

# Binary single-output base: unwrap one-output Hybrid before all model-specific logic.
replace_once(
    "src/bochan/acquisition/binary/base.py",
    "    resolve_observed_X,\n)",
    "    resolve_observed_X,\n    unwrap_single_output_model,\n)",
)
replace_once(
    "src/bochan/acquisition/binary/base.py",
    "    ):\n        if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):\n            model = model.models[0]\n\n        super().__init__(model)",
    "    ):\n        model = unwrap_single_output_model(model)\n        if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):\n            model = model.models[0]\n\n        super().__init__(model)",
)

# Binary IPV/NIPV proxy: expose the same public reference/duplicate controls.
replace_once(
    "src/bochan/acquisition/binary/active_learning/integrated_posterior_variance.py",
    "        pending_penalty_beta: float = 10.0,\n        apply_sigmoid_if_needed: bool = True,\n        eps: float = 1e-6,\n        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,\n",
    "        pending_penalty_beta: float = 10.0,\n        observed_penalty_weight: float = 0.0,\n        observed_penalty_beta: float = 10.0,\n        hard_duplicate_tol: float = 1e-8,\n        exclude_same_batch_duplicates: bool = True,\n        exclude_pending_duplicates: bool = True,\n        exclude_observed_duplicates: bool = True,\n        X_pending: Optional[Tensor] = None,\n        X_observed: Optional[Tensor] = None,\n        apply_sigmoid_if_needed: bool = True,\n        eps: float = 1e-6,\n        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,\n",
)
replace_once(
    "src/bochan/acquisition/binary/active_learning/integrated_posterior_variance.py",
    "            pending_penalty_beta=pending_penalty_beta,\n            apply_sigmoid_if_needed=apply_sigmoid_if_needed,\n            eps=eps,\n            objective=objective,\n",
    "            pending_penalty_beta=pending_penalty_beta,\n            observed_penalty_weight=observed_penalty_weight,\n            observed_penalty_beta=observed_penalty_beta,\n            hard_duplicate_tol=hard_duplicate_tol,\n            exclude_same_batch_duplicates=exclude_same_batch_duplicates,\n            exclude_pending_duplicates=exclude_pending_duplicates,\n            exclude_observed_duplicates=exclude_observed_duplicates,\n            X_pending=X_pending,\n            X_observed=X_observed,\n            apply_sigmoid_if_needed=apply_sigmoid_if_needed,\n            eps=eps,\n            objective=objective,\n",
)

# Multiclass single-output base: unwrap one-output Hybrid; use shared observed-X resolution.
replace_once(
    "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py",
    "    hard_same_batch_duplicate_penalty_per_point,\n)",
    "    hard_same_batch_duplicate_penalty_per_point,\n    resolve_observed_X,\n    unwrap_single_output_model,\n)",
)
replace_once(
    "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py",
    "def _resolve_observed_X(model: Model, X_observed: Tensor | None = None) -> Tensor | None:\n    if X_observed is not None:\n        return X_observed\n    for attr in (\"train_X_original\", \"train_X\", \"train_inputs_raw\"):\n        x = getattr(model, attr, None)\n        if x is not None:\n            return x[0] if isinstance(x, tuple) else x\n    x = getattr(model, \"train_inputs\", None)\n    if isinstance(x, tuple) and len(x) > 0:\n        return x[0]\n    return None\n\n\n",
    "",
)
replace_once(
    "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py",
    "    ) -> None:\n        if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):\n            model = model.models[0]\n",
    "    ) -> None:\n        model = unwrap_single_output_model(model)\n        if isinstance(model, (ModelListGP, ModelListGPyTorchModel)):\n            model = model.models[0]\n",
)
replace_once(
    "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py",
    "        resolved = _resolve_observed_X(self.model, X_observed)",
    "        resolved = resolve_observed_X(self.model, X_observed)",
)

# Multiclass IPV/NIPV proxy: make public controls explicit rather than hidden in **kwargs.
replace_once(
    "src/bochan/acquisition/multiclass/active_learning/single_output.py",
    "        local_weight: float | None = None,\n        integrated_weight: float = 1.0,\n        **kwargs,\n    ) -> None:\n        super().__init__(model=model, **kwargs)",
    "        local_weight: float | None = None,\n        integrated_weight: float = 1.0,\n        pending_penalty_weight: float = 0.0,\n        pending_penalty_beta: float = 10.0,\n        observed_penalty_weight: float = 0.0,\n        observed_penalty_beta: float = 10.0,\n        same_batch_penalty_weight: float = 0.0,\n        same_batch_penalty_beta: float = 10.0,\n        hard_duplicate_tol: float = 1e-8,\n        exclude_same_batch_duplicates: bool = True,\n        exclude_pending_duplicates: bool = True,\n        exclude_observed_duplicates: bool = True,\n        X_pending: Tensor | None = None,\n        X_observed: Tensor | None = None,\n        **kwargs,\n    ) -> None:\n        super().__init__(\n            model=model,\n            pending_penalty_weight=pending_penalty_weight,\n            pending_penalty_beta=pending_penalty_beta,\n            observed_penalty_weight=observed_penalty_weight,\n            observed_penalty_beta=observed_penalty_beta,\n            same_batch_penalty_weight=same_batch_penalty_weight,\n            same_batch_penalty_beta=same_batch_penalty_beta,\n            hard_duplicate_tol=hard_duplicate_tol,\n            exclude_same_batch_duplicates=exclude_same_batch_duplicates,\n            exclude_pending_duplicates=exclude_pending_duplicates,\n            exclude_observed_duplicates=exclude_observed_duplicates,\n            X_pending=X_pending,\n            X_observed=X_observed,\n            **kwargs,\n        )",
)

# Ordinal single-output: shared observed-X, unwrap Hybrid, latent posterior, public duplicate controls.
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    "    hard_same_batch_duplicate_penalty_per_point,\n)",
    "    hard_same_batch_duplicate_penalty_per_point,\n    resolve_observed_X,\n    unwrap_single_output_model,\n)",
)
start = '''def _resolve_observed_X(\n    model: Model,\n    X_observed: Optional[Tensor] = None,\n) -> Optional[Tensor]:\n    if X_observed is not None:\n        return X_observed\n\n    for attr in ("train_X_original", "train_X", "train_inputs_raw"):\n        x = getattr(model, attr, None)\n        if x is not None:\n            return x\n\n    x = getattr(model, "train_inputs", None)\n    if isinstance(x, tuple) and len(x) > 0:\n        return x[0]\n\n    return None\n\n\n'''
replace_once("src/bochan/acquisition/ordinal/active_learning/single_output.py", start, "")
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    "    ) -> None:\n        super().__init__(model=model)\n\n        if reduction not in (\"mean\", \"sum\"):",
    "    ) -> None:\n        model = unwrap_single_output_model(model)\n        super().__init__(model=model)\n\n        if reduction not in (\"mean\", \"sum\"):",
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    "        exclude_observed_duplicates: bool = False,",
    "        exclude_observed_duplicates: bool = True,",
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    "            _resolve_observed_X(self.model, X_observed)\n",
    "            resolve_observed_X(self.model, X_observed)\n",
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    "    def _posterior(self, X: Tensor):\n        return self.model.posterior(X)\n",
    "    def _posterior(self, X: Tensor):\n        latent_posterior = getattr(self.model, \"latent_posterior\", None)\n        if callable(latent_posterior):\n            return latent_posterior(X)\n        return self.model.posterior(X)\n",
)
# qOrdinalBALD duplicate args.
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    "        observed_penalty_beta: float = 10.0,\n        X_pending: Optional[Tensor] = None,\n        X_observed: Optional[Tensor] = None,\n        eps: float = 1e-6,\n        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,\n    ) -> None:\n        if sampler is None:\n",
    "        observed_penalty_beta: float = 10.0,\n        hard_duplicate_tol: float = 1e-8,\n        exclude_same_batch_duplicates: bool = True,\n        exclude_pending_duplicates: bool = True,\n        exclude_observed_duplicates: bool = True,\n        X_pending: Optional[Tensor] = None,\n        X_observed: Optional[Tensor] = None,\n        eps: float = 1e-6,\n        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,\n    ) -> None:\n        if sampler is None:\n",
)
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/single_output.py",
    "            observed_penalty_beta=observed_penalty_beta,\n            X_pending=X_pending,\n            X_observed=X_observed,\n            eps=eps,\n            objective=objective,\n        )\n        self.num_samples = int(num_samples)",
    "            observed_penalty_beta=observed_penalty_beta,\n            hard_duplicate_tol=hard_duplicate_tol,\n            exclude_same_batch_duplicates=exclude_same_batch_duplicates,\n            exclude_pending_duplicates=exclude_pending_duplicates,\n            exclude_observed_duplicates=exclude_observed_duplicates,\n            X_pending=X_pending,\n            X_observed=X_observed,\n            eps=eps,\n            objective=objective,\n        )\n        self.num_samples = int(num_samples)",
)
# UtilityVariance duplicate args (same signature pattern occurs once after class marker via targeted segment).
p = Path("src/bochan/acquisition/ordinal/active_learning/single_output.py")
text = p.read_text(encoding="utf-8")
marker = "class qOrdinalUtilityVariance(_qOrdinalActiveLearningBase):"
pos = text.index(marker)
tail = text[pos:]
old = "        observed_penalty_beta: float = 10.0,\n        X_pending: Optional[Tensor] = None,"
new = "        observed_penalty_beta: float = 10.0,\n        hard_duplicate_tol: float = 1e-8,\n        exclude_same_batch_duplicates: bool = True,\n        exclude_pending_duplicates: bool = True,\n        exclude_observed_duplicates: bool = True,\n        X_pending: Optional[Tensor] = None,"
if old not in tail:
    raise RuntimeError("qOrdinalUtilityVariance signature pattern missing")
tail = tail.replace(old, new, 1)
old2 = "            observed_penalty_beta=observed_penalty_beta,\n            X_pending=X_pending,"
new2 = "            observed_penalty_beta=observed_penalty_beta,\n            hard_duplicate_tol=hard_duplicate_tol,\n            exclude_same_batch_duplicates=exclude_same_batch_duplicates,\n            exclude_pending_duplicates=exclude_pending_duplicates,\n            exclude_observed_duplicates=exclude_observed_duplicates,\n            X_pending=X_pending,"
if old2 not in tail:
    raise RuntimeError("qOrdinalUtilityVariance super pattern missing")
tail = tail.replace(old2, new2, 1)
p.write_text(text[:pos] + tail, encoding="utf-8")

# Ordinal fantasy NIPV: unwrap Hybrid + hard duplicate controls + latent accessors.
p = Path("src/bochan/acquisition/ordinal/active_learning/single_output.py")
text = p.read_text(encoding="utf-8")
marker = "class qOrdinalFantasyNegIntegratedPosteriorVariance(AcquisitionFunction):"
pos = text.index(marker)
head, tail = text[:pos], text[pos:]
tail = tail.replace(
    "        observed_penalty_beta: float = 10.0,\n        X_pending: Optional[Tensor] = None,",
    "        observed_penalty_beta: float = 10.0,\n        hard_duplicate_tol: float = 1e-8,\n        exclude_same_batch_duplicates: bool = True,\n        exclude_pending_duplicates: bool = True,\n        exclude_observed_duplicates: bool = True,\n        X_pending: Optional[Tensor] = None,",
    1,
)
tail = tail.replace(
    "    ) -> None:\n        super().__init__(model=model)\n\n        if mc_points.ndim != 2:",
    "    ) -> None:\n        model = unwrap_single_output_model(model)\n        super().__init__(model=model)\n\n        if mc_points.ndim != 2:",
    1,
)
tail = tail.replace(
    "        self.observed_penalty_beta = float(observed_penalty_beta)\n        self.eps = float(eps)",
    "        self.observed_penalty_beta = float(observed_penalty_beta)\n        self.hard_duplicate_tol = float(hard_duplicate_tol)\n        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n        if self.hard_duplicate_tol < 0.0:\n            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n        self.eps = float(eps)",
    1,
)
tail = tail.replace("            _resolve_observed_X(self.model, X_observed)\n", "            resolve_observed_X(self.model, X_observed)\n", 1)
tail = tail.replace(
    "        posterior = self.model.posterior(X)\n        latent_samples = posterior.rsample",
    "        latent_posterior = getattr(self.model, \"latent_posterior\", None)\n        posterior = latent_posterior(X) if callable(latent_posterior) else self.model.posterior(X)\n        latent_samples = posterior.rsample",
    1,
)
tail = tail.replace(
    "        posterior = fantasy_model.posterior(self.mc_points)\n        return posterior.variance.mean()",
    "        latent_posterior = getattr(fantasy_model, \"latent_posterior\", None)\n        posterior = (\n            latent_posterior(self.mc_points)\n            if callable(latent_posterior)\n            else fantasy_model.posterior(self.mc_points)\n        )\n        return posterior.variance.mean()",
    1,
)
old_pen = '''    def _aggregated_reference_penalty(self, X: Tensor) -> Tensor:\n        Xt = _apply_input_transform_for_reference(self.model, X)\n        penalty = torch.zeros(Xt.shape[:-2], device=Xt.device, dtype=Xt.dtype)\n\n        if self.pending_penalty_weight > 0.0:\n            Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)\n            penalty = penalty + self.pending_penalty_weight * _rbf_reference_penalty_aggregated(\n                X=Xt,\n                X_ref=Xp_t,\n                beta=self.pending_penalty_beta,\n                reduction="sum",\n            )\n\n        if self.observed_penalty_weight > 0.0:\n            Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)\n            penalty = penalty + self.observed_penalty_weight * _rbf_reference_penalty_aggregated(\n                X=Xt,\n                X_ref=Xobs_t,\n                beta=self.observed_penalty_beta,\n                reduction="sum",\n            )\n\n        return penalty\n'''
new_pen = '''    def _aggregated_reference_penalty(self, X: Tensor) -> Tensor:\n        Xt = _apply_input_transform_for_reference(self.model, X)\n        penalty = hard_same_batch_duplicate_penalty_per_point(\n            Xt,\n            enabled=self.exclude_same_batch_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        ).sum(dim=-1)\n\n        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)\n        if self.pending_penalty_weight > 0.0:\n            penalty = penalty + self.pending_penalty_weight * _rbf_reference_penalty_aggregated(\n                X=Xt,\n                X_ref=Xp_t,\n                beta=self.pending_penalty_beta,\n                reduction="sum",\n            )\n        penalty = penalty + hard_reference_duplicate_penalty_per_point(\n            Xt,\n            Xp_t,\n            enabled=self.exclude_pending_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        ).sum(dim=-1)\n\n        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)\n        if self.observed_penalty_weight > 0.0:\n            penalty = penalty + self.observed_penalty_weight * _rbf_reference_penalty_aggregated(\n                X=Xt,\n                X_ref=Xobs_t,\n                beta=self.observed_penalty_beta,\n                reduction="sum",\n            )\n        penalty = penalty + hard_reference_duplicate_penalty_per_point(\n            Xt,\n            Xobs_t,\n            enabled=self.exclude_observed_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        ).sum(dim=-1)\n        return penalty\n'''
if old_pen not in tail:
    raise RuntimeError("Ordinal fantasy penalty block missing")
tail = tail.replace(old_pen, new_pen, 1)
p.write_text(head + tail, encoding="utf-8")

# Ordinal multi-output: shared observed resolver and hard observed exclusion.
replace_once(
    "src/bochan/acquisition/ordinal/active_learning/multi_output.py",
    "    hard_same_batch_duplicate_penalty_per_point,\n)",
    "    hard_same_batch_duplicate_penalty_per_point,\n    resolve_observed_X,\n)",
)
p = Path("src/bochan/acquisition/ordinal/active_learning/multi_output.py")
text = p.read_text(encoding="utf-8")
start = text.index("def _resolve_observed_X(")
end = text.index("\n\ndef _broadcast_reference_to_batch", start)
text = text[:start] + text[end + 2:]
text = text.replace("_resolve_observed_X(self.model, X_observed)", "resolve_observed_X(self.model, X_observed)")
text = text.replace(
    "        exclude_pending_duplicates: bool = True,\n        X_pending: Optional[Tensor] = None,",
    "        exclude_pending_duplicates: bool = True,\n        exclude_observed_duplicates: bool = True,\n        X_pending: Optional[Tensor] = None,",
    1,
)
text = text.replace(
    "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n        if self.hard_duplicate_tol < 0.0:",
    "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n        if self.hard_duplicate_tol < 0.0:",
    1,
)
old_obs = '''    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)\n        return _reference_penalty_per_point(\n            Xt,\n            Xobs_t,\n            beta=self.observed_penalty_beta,\n            weight=self.observed_penalty_weight,\n            cat_dims=self.cat_dims,\n        )\n'''
new_obs = '''    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)\n        soft = _reference_penalty_per_point(\n            Xt,\n            Xobs_t,\n            beta=self.observed_penalty_beta,\n            weight=self.observed_penalty_weight,\n            cat_dims=self.cat_dims,\n        )\n        hard = hard_reference_duplicate_penalty_per_point(\n            Xt,\n            Xobs_t,\n            enabled=self.exclude_observed_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        )\n        return soft + hard\n'''
if old_obs not in text:
    raise RuntimeError("multi-output ordinal observed penalty missing")
text = text.replace(old_obs, new_obs, 1)
# Public qMultiOutputOrdinalUtilityVariance: forward all duplicate controls.
marker = "class qMultiOutputOrdinalUtilityVariance(_qMultiOutputOrdinalActiveLearningBase):"
pos = text.index(marker)
head, tail = text[:pos], text[pos:]
tail = tail.replace(
    "        same_batch_penalty_beta: float = 10.0,\n        X_pending: Optional[Tensor] = None,",
    "        same_batch_penalty_beta: float = 10.0,\n        hard_duplicate_tol: float = 1e-8,\n        exclude_same_batch_duplicates: bool = True,\n        exclude_pending_duplicates: bool = True,\n        exclude_observed_duplicates: bool = True,\n        X_pending: Optional[Tensor] = None,",
    1,
)
tail = tail.replace(
    "            same_batch_penalty_beta=same_batch_penalty_beta,\n            X_pending=X_pending,",
    "            same_batch_penalty_beta=same_batch_penalty_beta,\n            hard_duplicate_tol=hard_duplicate_tol,\n            exclude_same_batch_duplicates=exclude_same_batch_duplicates,\n            exclude_pending_duplicates=exclude_pending_duplicates,\n            exclude_observed_duplicates=exclude_observed_duplicates,\n            X_pending=X_pending,",
    1,
)
text = head + tail
p.write_text(text, encoding="utf-8")

# Web: task/output-aware active-learning kwargs.
replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''def _set_active_learning_reference_kwargs(\n    acqf_kwargs: dict[str, Any],\n    *,\n    acq_key: str,\n    train_x: Any,\n) -> None:\n    """Attach only the reference inputs supported by the selected AL acquisition.\n\n    True NIPV consumes ``mc_points`` as its integration set and has no\n    ``X_observed`` argument. Pointwise uncertainty acquisitions use\n    ``X_observed`` for optional observed-point penalties / exclusion.\n    """\n    if acq_key in {"nipv", "qnipv"}:\n        acqf_kwargs.setdefault("mc_points", train_x)\n        return\n    acqf_kwargs.setdefault("X_observed", train_x)\n''',
    '''def _set_active_learning_kwargs(\n    acqf_kwargs: dict[str, Any],\n    *,\n    acq_key: str,\n    train_x: Any,\n    task_type: str,\n    multi_output: bool,\n    output_weights: Any | None = None,\n) -> None:\n    """Attach task/output-aware Active Learning constructor arguments."""\n    task = str(task_type).lower()\n    if multi_output:\n        if output_weights is not None:\n            acqf_kwargs.setdefault("output_weights", output_weights)\n        if task == "regression":\n            acqf_kwargs.setdefault("output_reduction", "weighted_mean")\n        else:\n            acqf_kwargs.setdefault("output_mode", "weighted_mean")\n\n    if acq_key in {"nipv", "qnipv"}:\n        acqf_kwargs.setdefault("mc_points", train_x)\n        # Classification / ordinal NIPV implementations also expose observed\n        # exclusion. True Gaussian regression NIPV does not accept X_observed.\n        if task != "regression":\n            acqf_kwargs.setdefault("X_observed", train_x)\n        return\n\n    acqf_kwargs.setdefault("X_observed", train_x)\n''',
)
replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''        objective_config = None\n        acqf_kwargs.setdefault(\n            "output_weights",\n            objective_weights(\n                target_columns=target_columns,\n                objective_targets=objective_targets,\n            ),\n        )\n        acqf_kwargs.setdefault("output_reduction", "weighted_mean")\n        _set_active_learning_reference_kwargs(\n            acqf_kwargs,\n            acq_key=acq_key,\n            train_x=train_x,\n        )\n''',
    '''        objective_config = None\n        homogeneous_task = (\n            internal_tasks[0]\n            if internal_tasks and all(task == internal_tasks[0] for task in internal_tasks)\n            else "hybrid"\n        )\n        _set_active_learning_kwargs(\n            acqf_kwargs,\n            acq_key=acq_key,\n            train_x=train_x,\n            task_type=homogeneous_task,\n            multi_output=len(target_columns) > 1,\n            output_weights=objective_weights(\n                target_columns=target_columns,\n                objective_targets=objective_targets,\n            ),\n        )\n''',
)

# Engine: homogeneous multi-output Hybrid should route to task-native multi-output AL.
replace_once(
    "src/bochan/api/engine.py",
    '''        sub_bundles = list(bundle.metadata.get("sub_bundles") or [])\n        if len(sub_bundles) == 1:\n            sub_bundle = sub_bundles[0]\n            return (\n                str(sub_bundle.task_type),\n                str(sub_bundle.model_type),\n                False,\n            )\n\n        specs = list(getattr(bundle.model, "specs", None) or [])\n        if len(specs) == 1:\n            return str(specs[0].task_type), model_type, False\n\n        return task_type, model_type, multi_output\n''',
    '''        sub_bundles = list(bundle.metadata.get("sub_bundles") or [])\n        if len(sub_bundles) == 1:\n            sub_bundle = sub_bundles[0]\n            return (\n                str(sub_bundle.task_type),\n                str(sub_bundle.model_type),\n                False,\n            )\n        if len(sub_bundles) > 1:\n            tasks = {str(sub_bundle.task_type) for sub_bundle in sub_bundles}\n            model_types = {str(sub_bundle.model_type) for sub_bundle in sub_bundles}\n            if len(tasks) == 1:\n                resolved_model_type = next(iter(model_types)) if len(model_types) == 1 else model_type\n                return next(iter(tasks)), resolved_model_type, True\n\n        specs = list(getattr(bundle.model, "specs", None) or [])\n        if len(specs) == 1:\n            return str(specs[0].task_type), model_type, False\n        if len(specs) > 1:\n            tasks = {str(spec.task_type) for spec in specs}\n            if len(tasks) == 1:\n                return next(iter(tasks)), model_type, True\n\n        return task_type, model_type, multi_output\n''',
)

print("classification / ordinal AL source patch applied")
