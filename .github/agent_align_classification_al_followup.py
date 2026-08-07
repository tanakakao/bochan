from pathlib import Path

# Keep the legacy Web helper available for existing focused tests while the new
# implementation routes through the task/output-aware helper.
p = Path("src/bochan/serving/webapp/workflows_tabular.py")
text = p.read_text(encoding="utf-8")
needle = '''    acqf_kwargs.setdefault("X_observed", train_x)\n\n\ndef _request_with_constraints'''
replacement = '''    acqf_kwargs.setdefault("X_observed", train_x)\n\n\ndef _set_active_learning_reference_kwargs(\n    acqf_kwargs: dict[str, object],\n    *,\n    acq_key: str,\n    train_x: object,\n) -> None:\n    """Backward-compatible Regression single-output AL reference helper."""\n    _set_active_learning_kwargs(\n        acqf_kwargs,\n        acq_key=acq_key,\n        train_x=train_x,\n        task_type="regression",\n        multi_output=False,\n    )\n\n\ndef _request_with_constraints'''
if needle not in text:
    raise RuntimeError("new Web AL helper insertion point not found")
p.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

# Bring the fantasy multi-output Ordinal NIPV duplicate contract to parity with
# the other Binary / Multiclass / Ordinal active-learning acquisitions.
p = Path("src/bochan/acquisition/ordinal/active_learning/multi_output.py")
text = p.read_text(encoding="utf-8")
marker = "class qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance(AcquisitionFunction):"
pos = text.index(marker)
head, tail = text[:pos], text[pos:]

old = '''        same_batch_penalty_weight: float = 0.0,\n        same_batch_penalty_beta: float = 10.0,\n        X_pending: Optional[Tensor] = None,\n        X_observed: Optional[Tensor] = None,\n'''
new = '''        same_batch_penalty_weight: float = 0.0,\n        same_batch_penalty_beta: float = 10.0,\n        hard_duplicate_tol: float = 1e-8,\n        exclude_same_batch_duplicates: bool = True,\n        exclude_pending_duplicates: bool = True,\n        exclude_observed_duplicates: bool = True,\n        X_pending: Optional[Tensor] = None,\n        X_observed: Optional[Tensor] = None,\n'''
if old not in tail:
    raise RuntimeError("multi-output ordinal NIPV signature block missing")
tail = tail.replace(old, new, 1)

old = '''        self.same_batch_penalty_weight = float(same_batch_penalty_weight)\n        self.same_batch_penalty_beta = float(same_batch_penalty_beta)\n        self.cat_dims = _resolve_cat_dims(model)\n'''
new = '''        self.same_batch_penalty_weight = float(same_batch_penalty_weight)\n        self.same_batch_penalty_beta = float(same_batch_penalty_beta)\n        self.hard_duplicate_tol = float(hard_duplicate_tol)\n        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n        if self.hard_duplicate_tol < 0.0:\n            raise ValueError("hard_duplicate_tol must be non-negative.")\n        self.cat_dims = _resolve_cat_dims(model)\n'''
if old not in tail:
    raise RuntimeError("multi-output ordinal NIPV duplicate state block missing")
tail = tail.replace(old, new, 1)

old = '''    def _aggregated_repulsion_penalty(self, X: Tensor) -> Tensor:\n        Xt = _apply_input_transform_for_reference(self.model, X)\n        penalty = _same_batch_penalty_aggregated(\n            Xt,\n            beta=self.same_batch_penalty_beta,\n            weight=self.same_batch_penalty_weight,\n            cat_dims=self.cat_dims,\n        )\n        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)\n        penalty = penalty + _reference_penalty_aggregated(\n            Xt,\n            Xp_t,\n            beta=self.pending_penalty_beta,\n            weight=self.pending_penalty_weight,\n            cat_dims=self.cat_dims,\n            reduction="sum",\n        )\n        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)\n        penalty = penalty + _reference_penalty_aggregated(\n            Xt,\n            Xobs_t,\n            beta=self.observed_penalty_beta,\n            weight=self.observed_penalty_weight,\n            cat_dims=self.cat_dims,\n            reduction="sum",\n        )\n        return penalty\n'''
new = '''    def _aggregated_repulsion_penalty(self, X: Tensor) -> Tensor:\n        Xt = _apply_input_transform_for_reference(self.model, X)\n        penalty = _same_batch_penalty_aggregated(\n            Xt,\n            beta=self.same_batch_penalty_beta,\n            weight=self.same_batch_penalty_weight,\n            cat_dims=self.cat_dims,\n        )\n        penalty = penalty + hard_same_batch_duplicate_penalty_per_point(\n            Xt,\n            enabled=self.exclude_same_batch_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        ).sum(dim=-1)\n\n        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)\n        penalty = penalty + _reference_penalty_aggregated(\n            Xt,\n            Xp_t,\n            beta=self.pending_penalty_beta,\n            weight=self.pending_penalty_weight,\n            cat_dims=self.cat_dims,\n            reduction="sum",\n        )\n        penalty = penalty + hard_reference_duplicate_penalty_per_point(\n            Xt,\n            Xp_t,\n            enabled=self.exclude_pending_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        ).sum(dim=-1)\n\n        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)\n        penalty = penalty + _reference_penalty_aggregated(\n            Xt,\n            Xobs_t,\n            beta=self.observed_penalty_beta,\n            weight=self.observed_penalty_weight,\n            cat_dims=self.cat_dims,\n            reduction="sum",\n        )\n        penalty = penalty + hard_reference_duplicate_penalty_per_point(\n            Xt,\n            Xobs_t,\n            enabled=self.exclude_observed_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        ).sum(dim=-1)\n        return penalty\n'''
if old not in tail:
    raise RuntimeError("multi-output ordinal NIPV repulsion block missing")
tail = tail.replace(old, new, 1)

p.write_text(head + tail, encoding="utf-8")
print("follow-up compatibility and ordinal multi-output NIPV patch applied")
