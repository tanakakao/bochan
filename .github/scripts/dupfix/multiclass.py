from __future__ import annotations

from .common import read, replace_once, write


def patch_multiclass_single() -> None:
    path = "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py"
    text = read(path)
    text = replace_once(
        text,
        "from bochan.acquisition.multiclass.base import ClassReductionType, ReductionType\n",
        "from bochan.acquisition._duplicate_exclusion import (\n"
        "    hard_reference_duplicate_penalty_per_point,\n"
        "    hard_same_batch_duplicate_penalty_per_point,\n"
        ")\n"
        "from bochan.acquisition.multiclass.base import ClassReductionType, ReductionType\n",
        label="multiclass single imports",
    )
    text = replace_once(
        text,
        "        same_batch_penalty_weight: float = 0.0,\n"
        "        same_batch_penalty_beta: float = 10.0,\n"
        "        observed_penalty_weight: float = 0.0,\n",
        "        same_batch_penalty_weight: float = 0.0,\n"
        "        same_batch_penalty_beta: float = 10.0,\n"
        "        hard_duplicate_tol: float = 1e-8,\n"
        "        exclude_same_batch_duplicates: bool = True,\n"
        "        exclude_pending_duplicates: bool = True,\n"
        "        observed_penalty_weight: float = 0.0,\n",
        label="multiclass single signature",
    )
    text = replace_once(
        text,
        "        self.same_batch_penalty_weight = float(same_batch_penalty_weight)\n"
        "        self.same_batch_penalty_beta = float(same_batch_penalty_beta)\n",
        "        self.same_batch_penalty_weight = float(same_batch_penalty_weight)\n"
        "        self.same_batch_penalty_beta = float(same_batch_penalty_beta)\n"
        "        self.hard_duplicate_tol = float(hard_duplicate_tol)\n"
        "        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n"
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        if self.hard_duplicate_tol < 0.0:\n"
        "            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n",
        label="multiclass single attributes",
    )
    old_pending = '''    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        if self.pending_penalty_weight <= 0:
            return Xt.new_zeros(Xt.shape[:-1])
        Xp = self._reference_points_transformed(getattr(self, "X_pending", None), ref=Xt)
        if Xp is None:
            return Xt.new_zeros(Xt.shape[:-1])
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xp).min(dim=-1).values
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist.reshape(Xt.shape[:-1]))
'''
    new_pending = '''    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        zeros = Xt.new_zeros(Xt.shape[:-1])
        Xp = self._reference_points_transformed(getattr(self, "X_pending", None), ref=Xt)
        if Xp is None:
            return zeros
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xp).min(dim=-1).values
        soft = (
            self.pending_penalty_weight
            * torch.exp(-self.pending_penalty_beta * dist.reshape(Xt.shape[:-1]))
            if self.pending_penalty_weight > 0.0
            else zeros
        )
        hard = hard_reference_duplicate_penalty_per_point(
            Xt,
            Xp,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return soft + hard
'''
    text = replace_once(text, old_pending, new_pending, label="multiclass single pending")
    old_same = '''    def _same_batch_penalty(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        if self.same_batch_penalty_weight <= 0 or Xt.shape[-2] <= 1:
            return Xt.new_zeros(Xt.shape[:-2])
        batch_shape = Xt.shape[:-2]
        q = Xt.shape[-2]
        Xb = Xt.reshape(-1, q, Xt.shape[-1])
        d = torch.cdist(Xb, Xb)
        eye = torch.eye(q, device=Xt.device, dtype=torch.bool).unsqueeze(0)
        d = d.masked_fill(eye, float("inf"))
        penalty = 0.5 * self.same_batch_penalty_weight * torch.exp(-self.same_batch_penalty_beta * d).sum(dim=(-1, -2))
        return penalty.reshape(*batch_shape)
'''
    new_same = '''    def _same_batch_penalty(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        batch_shape = Xt.shape[:-2]
        if self.same_batch_penalty_weight > 0 and Xt.shape[-2] > 1:
            q = Xt.shape[-2]
            Xb = Xt.reshape(-1, q, Xt.shape[-1])
            d = torch.cdist(Xb, Xb)
            eye = torch.eye(q, device=Xt.device, dtype=torch.bool).unsqueeze(0)
            d = d.masked_fill(eye, float("inf"))
            soft = 0.5 * self.same_batch_penalty_weight * torch.exp(
                -self.same_batch_penalty_beta * d
            ).sum(dim=(-1, -2))
            soft = soft.reshape(*batch_shape)
        else:
            soft = Xt.new_zeros(batch_shape)
        hard = hard_same_batch_duplicate_penalty_per_point(
            Xt,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        ).amax(dim=-1)
        return soft + hard
'''
    text = replace_once(text, old_same, new_same, label="multiclass single same batch")
    write(path, text)


def patch_multiclass_multi() -> None:
    path = "src/bochan/acquisition/multiclass/active_learning/multi_output.py"
    text = read(path)
    text = replace_once(
        text,
        "from torch import Tensor\n\nReductionType = Literal",
        "from torch import Tensor\n\n"
        "from bochan.acquisition._duplicate_exclusion import (\n"
        "    hard_reference_duplicate_penalty_per_point,\n"
        "    hard_same_batch_duplicate_penalty_per_point,\n"
        ")\n\n"
        "ReductionType = Literal",
        label="multiclass multi imports",
    )
    text = replace_once(
        text,
        "        same_batch_penalty_weight: float = 0.0,\n"
        "        same_batch_penalty_beta: float = 10.0,\n"
        "        X_observed: Tensor | None = None,\n",
        "        same_batch_penalty_weight: float = 0.0,\n"
        "        same_batch_penalty_beta: float = 10.0,\n"
        "        hard_duplicate_tol: float = 1e-8,\n"
        "        exclude_same_batch_duplicates: bool = True,\n"
        "        exclude_pending_duplicates: bool = True,\n"
        "        X_observed: Tensor | None = None,\n",
        label="multiclass multi signature",
    )
    text = replace_once(
        text,
        "        self.same_batch_penalty_weight = float(same_batch_penalty_weight)\n"
        "        self.same_batch_penalty_beta = float(same_batch_penalty_beta)\n"
        "        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()\n",
        "        self.same_batch_penalty_weight = float(same_batch_penalty_weight)\n"
        "        self.same_batch_penalty_beta = float(same_batch_penalty_beta)\n"
        "        self.hard_duplicate_tol = float(hard_duplicate_tol)\n"
        "        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n"
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        if self.hard_duplicate_tol < 0.0:\n"
        "            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n"
        "        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()\n",
        label="multiclass multi attributes",
    )
    old_pending = '''    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        if self.pending_penalty_weight <= 0:
            return Xt.new_zeros(Xt.shape[:-1])
        Xp = self._reference_points_transformed(getattr(self, "X_pending", None), ref=Xt)
        if Xp is None:
            return Xt.new_zeros(Xt.shape[:-1])
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xp).min(dim=-1).values
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist.reshape(Xt.shape[:-1]))
'''
    new_pending = '''    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        zeros = Xt.new_zeros(Xt.shape[:-1])
        Xp = self._reference_points_transformed(getattr(self, "X_pending", None), ref=Xt)
        if Xp is None:
            return zeros
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xp).min(dim=-1).values
        soft = (
            self.pending_penalty_weight
            * torch.exp(-self.pending_penalty_beta * dist.reshape(Xt.shape[:-1]))
            if self.pending_penalty_weight > 0.0
            else zeros
        )
        hard = hard_reference_duplicate_penalty_per_point(
            Xt,
            Xp,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return soft + hard
'''
    text = replace_once(text, old_pending, new_pending, label="multiclass multi pending")
    old_same = '''    def _same_batch_penalty(self, Xt: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        if self.same_batch_penalty_weight <= 0 or Xt.shape[-2] <= 1:
            return Xt.new_zeros(Xt.shape[:-2])
        Xb = Xt.reshape(-1, Xt.shape[-2], Xt.shape[-1])
        d = torch.cdist(Xb, Xb)
        q = Xt.shape[-2]
        eye = torch.eye(q, device=Xt.device, dtype=torch.bool).unsqueeze(0)
        d = d.masked_fill(eye, float("inf"))
        penalty = 0.5 * self.same_batch_penalty_weight * torch.exp(-self.same_batch_penalty_beta * d).sum(dim=(-1, -2))
        return penalty.reshape(*Xt.shape[:-2])
'''
    new_same = '''    def _same_batch_penalty(self, Xt: Tensor) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        if self.same_batch_penalty_weight > 0 and Xt.shape[-2] > 1:
            Xb = Xt.reshape(-1, Xt.shape[-2], Xt.shape[-1])
            d = torch.cdist(Xb, Xb)
            q = Xt.shape[-2]
            eye = torch.eye(q, device=Xt.device, dtype=torch.bool).unsqueeze(0)
            d = d.masked_fill(eye, float("inf"))
            soft = 0.5 * self.same_batch_penalty_weight * torch.exp(
                -self.same_batch_penalty_beta * d
            ).sum(dim=(-1, -2))
            soft = soft.reshape(*Xt.shape[:-2])
        else:
            soft = Xt.new_zeros(Xt.shape[:-2])
        hard = hard_same_batch_duplicate_penalty_per_point(
            Xt,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        ).amax(dim=-1)
        return soft + hard
'''
    text = replace_once(text, old_same, new_same, label="multiclass multi same batch")
    write(path, text)
