# Pending penalty raw/transformed space fix for bochan/acquisition/classification/base.py
#
# Add these methods to _BinaryClassificationAcqBase, or replace existing
# set_X_pending / _pending_penalty_per_point / _get_pending_in_feature_space logic
# with this version.
#
# The key policy is:
#   - keep X_pending in raw input space
#   - when candidate uses transformed Xt, transform X_pending through the same input_transform
#   - compute distances in the same feature space

def _coerce_pending_to_tensor(self, X_pending, *, ref=None):
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
        raise TypeError(f"X_pending must be None, Tensor, list, or tuple. Got {type(X_pending)}.")
    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out

def set_X_pending(self, X_pending=None):
    self.X_pending = self._coerce_pending_to_tensor(X_pending)

def _transform_pending_like_candidate(self, X_pending, *, ref):
    Xp = self._coerce_pending_to_tensor(X_pending, ref=ref)
    if Xp is None or Xp.numel() == 0:
        return None

    # If the class already has a feature-space transform helper, use it.
    for name in ("_apply_input_transform_safe", "_apply_input_transform", "_shape_X_for_model"):
        fn = getattr(self, name, None)
        if callable(fn):
            Xp_t = fn(Xp)
            if isinstance(Xp_t, tuple):
                Xp_t = Xp_t[0]
            if Xp_t.ndim == 2:
                Xp_t = Xp_t.unsqueeze(-2)
            return Xp_t.to(device=ref.device, dtype=ref.dtype)

    # Fallback: top-level model input_transform or first submodel input_transform.
    model = getattr(self, "model", None)
    it = getattr(model, "input_transform", None)
    if it is None and hasattr(model, "models") and len(model.models) > 0:
        it = getattr(model.models[0], "input_transform", None)
    Xp_t = it(Xp) if it is not None else Xp
    if isinstance(Xp_t, tuple):
        Xp_t = Xp_t[0]
    if Xp_t.ndim == 2:
        Xp_t = Xp_t.unsqueeze(-2)
    return Xp_t.to(device=ref.device, dtype=ref.dtype)

def _pending_penalty_per_point(self, Xt):
    if Xt.ndim == 2:
        Xt = Xt.unsqueeze(-2)

    if getattr(self, "pending_penalty_weight", 0.0) <= 0.0:
        return torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)

    Xp_t = self._transform_pending_like_candidate(getattr(self, "X_pending", None), ref=Xt)
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
    dist = torch.cdist(X2d, Xp2d).min(dim=-1).values.reshape(*Xt.shape[:-1])
    return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist)
