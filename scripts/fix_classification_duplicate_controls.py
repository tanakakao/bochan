from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_at_least_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"{label}: expected at least one match")
    return text.replace(old, new)


def class_block(text: str, class_name: str) -> tuple[int, int, str]:
    match = re.search(rf"^class {re.escape(class_name)}\b", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"class not found: {class_name}")
    next_match = re.search(r"^class \w+", text[match.end() :], flags=re.MULTILINE)
    end = len(text) if next_match is None else match.end() + next_match.start()
    return match.start(), end, text[match.start() : end]


def replace_in_class(
    text: str,
    class_name: str,
    old: str,
    new: str,
    *,
    label: str,
) -> str:
    start, end, block = class_block(text, class_name)
    block = replace_once(block, old, new, label=f"{class_name}: {label}")
    return text[:start] + block + text[end:]


def add_kwargs_passthrough(path: str) -> None:
    """Expose future/base duplicate controls on ordinal public constructors."""

    text = read(path)
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits: list[tuple[int, str]] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if "Ordinal" not in node.name:
            continue
        init = next(
            (
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "__init__"
            ),
            None,
        )
        if init is None or init.args.kwarg is not None:
            continue
        arg_names = {
            arg.arg
            for arg in [
                *init.args.posonlyargs,
                *init.args.args,
                *init.args.kwonlyargs,
            ]
        }
        duplicate_names = {
            "hard_duplicate_tol",
            "exclude_same_batch_duplicates",
            "exclude_pending_duplicates",
            "exclude_observed_duplicates",
        }
        if duplicate_names.issubset(arg_names):
            continue

        signature_end = init.body[0].lineno - 1
        close_index = None
        for index in range(init.lineno - 1, signature_end):
            if re.match(r"^\s*\)\s*(?:->\s*[^:]+)?\s*:\s*$", lines[index]):
                close_index = index
                break
        if close_index is None:
            continue
        edits.append((close_index, "        **kwargs,\n"))

        super_call = None
        for child in ast.walk(init):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "__init__"
                and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"
            ):
                super_call = child
                break
        if super_call is not None and super_call.end_lineno is not None:
            close_line = super_call.end_lineno - 1
            if lines[close_line].strip() == ")":
                edits.append((close_line, "            **kwargs,\n"))

    if not edits:
        return
    for index, insertion in sorted(edits, reverse=True):
        lines.insert(index, insertion)
    write(path, "".join(lines))


def patch_duplicate_helpers() -> None:
    path = "src/bochan/acquisition/_duplicate_exclusion.py"
    text = read(path)
    text = replace_once(
        text,
        "duplicate_pairs = (~eye) & (d2 <= tolerance)",
        "duplicate_pairs = (~eye) & (d2 <= tolerance**2)",
        label="same-batch tolerance",
    )
    text = replace_once(
        text,
        "duplicate_batch = (d2 <= tolerance).any(dim=-1).any(dim=-1, keepdim=True)",
        "duplicate_batch = (d2 <= tolerance**2).any(dim=-1).any(dim=-1, keepdim=True)",
        label="reference tolerance",
    )
    helper = '''\n\ndef resolve_observed_X(model, X_observed: Tensor | None = None) -> Tensor | None:\n    """Resolve observed inputs consistently across classification acquisitions."""\n\n    if X_observed is not None:\n        return X_observed\n\n    for attr in ("train_X_original", "train_X", "train_inputs_raw"):\n        value = getattr(model, attr, None)\n        if value is not None:\n            return value\n\n    train_inputs = getattr(model, "train_inputs", None)\n    if isinstance(train_inputs, tuple) and len(train_inputs) > 0:\n        return train_inputs[0]\n\n    submodels = getattr(model, "models", None) or getattr(model, "submodels", None)\n    if submodels is not None and len(submodels) > 0:\n        return resolve_observed_X(submodels[0], None)\n    return None\n'''
    text = replace_once(
        text,
        "\n\n__all__ = [\n",
        helper + "\n\n__all__ = [\n",
        label="observed resolver insertion",
    )
    text = replace_once(
        text,
        '    "hard_same_batch_duplicate_penalty_per_point",\n]',
        '    "hard_same_batch_duplicate_penalty_per_point",\n    "resolve_observed_X",\n]',
        label="observed resolver export",
    )
    write(path, text)


def patch_binary_base() -> None:
    path = "src/bochan/acquisition/binary/base.py"
    text = read(path)
    text = replace_once(
        text,
        "    hard_same_batch_duplicate_penalty_per_point,\n)",
        "    hard_same_batch_duplicate_penalty_per_point,\n    resolve_observed_X,\n)",
        label="binary base helper import",
    )
    text = replace_once(
        text,
        "        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,\n    ):",
        "        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,\n        X_pending: Optional[Tensor] = None,\n        X_observed: Optional[Tensor] = None,\n        observed_penalty_weight: float = 0.0,\n        observed_penalty_beta: float = 10.0,\n        exclude_observed_duplicates: bool = True,\n    ):",
        label="binary base signature",
    )
    text = replace_once(
        text,
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n        if self.hard_duplicate_tol < 0.0:",
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n        self.observed_penalty_weight = float(observed_penalty_weight)\n        self.observed_penalty_beta = float(observed_penalty_beta)\n        if self.hard_duplicate_tol < 0.0:",
        label="binary base observed fields",
    )
    text = replace_once(
        text,
        "        self.set_X_pending(None)\n",
        "        self.set_X_pending(X_pending)\n        self.set_X_observed(X_observed)\n",
        label="binary base reference init",
    )
    reference_methods = '''\n    def _coerce_reference_to_tensor(\n        self,\n        X_ref,\n        *,\n        ref: Optional[Tensor] = None,\n    ) -> Optional[Tensor]:\n        if X_ref is None:\n            return None\n        if torch.is_tensor(X_ref):\n            out = X_ref\n        elif isinstance(X_ref, (list, tuple)):\n            tensors = [\n                tensor\n                for item in X_ref\n                if (tensor := self._coerce_reference_to_tensor(item, ref=ref)) is not None\n                and tensor.numel() > 0\n            ]\n            if not tensors:\n                return None\n            out = torch.cat(\n                [tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors],\n                dim=-2,\n            )\n        else:\n            raise TypeError(\n                "Reference points must be None, Tensor, list, or tuple. "\n                f"Got {type(X_ref)}."\n            )\n        if ref is not None:\n            out = out.to(device=ref.device, dtype=ref.dtype)\n        return out.detach()\n\n    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:\n        self.X_pending = self._coerce_reference_to_tensor(X_pending)\n\n    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:\n        self.X_observed = self._coerce_reference_to_tensor(\n            resolve_observed_X(self.model, X_observed)\n        )\n'''
    text = replace_once(
        text,
        "    def _map_to_training_feature_space(self, X: Tensor) -> Tensor:\n",
        reference_methods + "\n    def _map_to_training_feature_space(self, X: Tensor) -> Tensor:\n",
        label="binary base reference methods",
    )
    text = replace_once(
        text,
        '''    def _get_pending_in_feature_space(self) -> Optional[Tensor]:\n        """X_pending を現在の candidate と同じ feature space に写す。"""\n        Xp = getattr(self, "X_pending", None)\n        if Xp is None or Xp.numel() == 0:\n            return None\n        return self._apply_input_transform(Xp)\n''',
        '''    def _get_reference_in_feature_space(self, X_ref) -> Optional[Tensor]:\n        X_ref = self._coerce_reference_to_tensor(X_ref)\n        if X_ref is None or X_ref.numel() == 0:\n            return None\n        transformed = self._apply_input_transform(X_ref)\n        if isinstance(transformed, list):\n            transformed = transformed[0]\n        return self._ensure_q_batch(transformed)\n\n    def _get_pending_in_feature_space(self) -> Optional[Tensor]:\n        """X_pending を現在の candidate と同じ feature space に写す。"""\n        return self._get_reference_in_feature_space(getattr(self, "X_pending", None))\n\n    def _get_observed_in_feature_space(self) -> Optional[Tensor]:\n        """X_observed を現在の candidate と同じ feature space に写す。"""\n        return self._get_reference_in_feature_space(getattr(self, "X_observed", None))\n''',
        label="binary base reference transform",
    )
    observed_method = '''\n    def _observed_penalty_per_point(self, X: Tensor) -> Tensor:\n        """Return soft observed repulsion plus scale-independent hard exclusion."""\n        X = self._ensure_q_batch(X)\n        zeros = X.new_zeros(X.shape[:-1])\n        Xobs = self._get_observed_in_feature_space()\n        if Xobs is None or Xobs.numel() == 0:\n            return zeros\n\n        Xobs2d = Xobs.reshape(-1, Xobs.shape[-1])\n        min_dist = torch.cdist(\n            X.reshape(-1, X.shape[-1]),\n            Xobs2d,\n        ).min(dim=-1).values.reshape(*X.shape[:-1])\n        soft = (\n            self.observed_penalty_weight\n            * torch.exp(-self.observed_penalty_beta * min_dist)\n            if self.observed_penalty_weight > 0.0\n            else zeros\n        )\n        hard = hard_reference_duplicate_penalty_per_point(\n            X,\n            Xobs2d,\n            enabled=self.exclude_observed_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        )\n        return soft + hard\n'''
    text = replace_once(
        text,
        "    def _candidate_penalty_per_point(self, X: Tensor) -> Tensor:\n",
        observed_method + "\n    def _candidate_penalty_per_point(self, X: Tensor) -> Tensor:\n",
        label="binary base observed penalty",
    )
    text = replace_once(
        text,
        "            self._pending_penalty_per_point(X)\n            + self._same_batch_duplicate_penalty_per_point(X)\n",
        "            self._pending_penalty_per_point(X)\n            + self._observed_penalty_per_point(X)\n            + self._same_batch_duplicate_penalty_per_point(X)\n",
        label="binary base candidate penalty",
    )
    write(path, text)


def duplicate_parameter_block(indent: str = "        ") -> str:
    return (
        f"{indent}X_pending: Optional[Tensor] = None,\n"
        f"{indent}X_observed: Optional[Tensor] = None,\n"
        f"{indent}observed_penalty_weight: float = 0.0,\n"
        f"{indent}observed_penalty_beta: float = 10.0,\n"
        f"{indent}hard_duplicate_tol: float = 1e-8,\n"
        f"{indent}exclude_same_batch_duplicates: bool = True,\n"
        f"{indent}exclude_pending_duplicates: bool = True,\n"
        f"{indent}exclude_observed_duplicates: bool = True,\n"
    )


def duplicate_super_block(indent: str = "            ") -> str:
    return (
        f"{indent}X_pending=X_pending,\n"
        f"{indent}X_observed=X_observed,\n"
        f"{indent}observed_penalty_weight=observed_penalty_weight,\n"
        f"{indent}observed_penalty_beta=observed_penalty_beta,\n"
        f"{indent}hard_duplicate_tol=hard_duplicate_tol,\n"
        f"{indent}exclude_same_batch_duplicates=exclude_same_batch_duplicates,\n"
        f"{indent}exclude_pending_duplicates=exclude_pending_duplicates,\n"
        f"{indent}exclude_observed_duplicates=exclude_observed_duplicates,\n"
    )


def patch_binary_single_output() -> None:
    path = "src/bochan/acquisition/binary/active_learning/single_output.py"
    text = read(path)
    classes = [
        "_BALDAcquisition",
        "_JointQBALDAcquisitionBinary",
        "_GreedyJointQBALDAcquisitionBinary",
        "_UncertaintySamplingClassifierAcquisition",
    ]
    for name in classes:
        start, end, block = class_block(text, name)
        block = replace_once(
            block,
            "        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,\n    ):",
            "        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,\n"
            + duplicate_parameter_block()
            + "    ):",
            label=f"{name} signature",
        )
        super_anchor = "            eps=eps,\n"
        block = replace_once(
            block,
            super_anchor,
            super_anchor + duplicate_super_block(),
            label=f"{name} super kwargs",
        )
        text = text[:start] + block + text[end:]
    write(path, text)


def patch_binary_multi_output_active_learning() -> None:
    path = "src/bochan/acquisition/binary/active_learning/multi_output.py"
    text = read(path)
    text = replace_once(
        text,
        "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n",
        "from bochan.acquisition._duplicate_exclusion import (\n"
        "    hard_reference_duplicate_penalty_per_point,\n"
        "    hard_same_batch_duplicate_penalty_per_point,\n"
        "    resolve_observed_X,\n"
        ")\n"
        "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n",
        label="multi-output binary imports",
    )
    text = replace_once(
        text,
        "        pending_penalty_beta: float = 10.0,\n        eps: float = 1e-6,\n",
        "        pending_penalty_beta: float = 10.0,\n"
        "        observed_penalty_weight: float = 0.0,\n"
        "        observed_penalty_beta: float = 10.0,\n"
        "        hard_duplicate_tol: float = 1e-8,\n"
        "        exclude_same_batch_duplicates: bool = True,\n"
        "        exclude_pending_duplicates: bool = True,\n"
        "        exclude_observed_duplicates: bool = True,\n"
        "        X_pending: Optional[Tensor] = None,\n"
        "        X_observed: Optional[Tensor] = None,\n"
        "        eps: float = 1e-6,\n",
        label="multi-output binary base signature",
    )
    text = replace_once(
        text,
        "        self.pending_penalty_beta = float(pending_penalty_beta)\n        self.eps = float(eps)\n",
        "        self.pending_penalty_beta = float(pending_penalty_beta)\n"
        "        self.observed_penalty_weight = float(observed_penalty_weight)\n"
        "        self.observed_penalty_beta = float(observed_penalty_beta)\n"
        "        self.hard_duplicate_tol = float(hard_duplicate_tol)\n"
        "        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n"
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n"
        "        if self.hard_duplicate_tol < 0.0:\n"
        "            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n"
        "        self.eps = float(eps)\n",
        label="multi-output binary base fields",
    )
    text = replace_once(
        text,
        "        self.set_X_pending(None)\n",
        "        self.set_X_pending(X_pending)\n        self.set_X_observed(X_observed)\n",
        label="multi-output binary base references",
    )
    text = replace_once(
        text,
        '''    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:\n        """pending points を raw input space の値として保持する。"""\n        self.X_pending = self._coerce_pending_to_tensor(X_pending)\n\n    def _transform_pending_like_candidate(\n        self,\n        X_pending,\n        *,\n        ref: Tensor,\n    ) -> Optional[Tensor]:\n        """X_pending を candidate と同じ距離計算空間へ写す。"""\n        Xp = self._coerce_pending_to_tensor(X_pending, ref=ref)\n        if Xp is None or Xp.numel() == 0:\n            return None\n        Xp_t = self._apply_input_transform(Xp)\n        Xp_t = self._ensure_q_batch(Xp_t)\n        return Xp_t.to(device=ref.device, dtype=ref.dtype)\n''',
        '''    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:\n        """pending points を raw input space の値として保持する。"""\n        self.X_pending = self._coerce_pending_to_tensor(X_pending)\n\n    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:\n        """observed points を raw input space の値として保持する。"""\n        self.X_observed = self._coerce_pending_to_tensor(\n            resolve_observed_X(self.model, X_observed)\n        )\n\n    def _transform_reference_like_candidate(\n        self,\n        X_ref,\n        *,\n        ref: Tensor,\n    ) -> Optional[Tensor]:\n        Xr = self._coerce_pending_to_tensor(X_ref, ref=ref)\n        if Xr is None or Xr.numel() == 0:\n            return None\n        Xr_t = self._apply_input_transform(Xr)\n        Xr_t = self._ensure_q_batch(Xr_t)\n        return Xr_t.to(device=ref.device, dtype=ref.dtype)\n\n    def _transform_pending_like_candidate(\n        self,\n        X_pending,\n        *,\n        ref: Tensor,\n    ) -> Optional[Tensor]:\n        return self._transform_reference_like_candidate(X_pending, ref=ref)\n''',
        label="multi-output binary reference methods",
    )
    start = text.index("    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:\n")
    end = text.index("    def _reduce_q(self, score: Tensor) -> Tensor:\n", start)
    penalty_methods = '''    def _same_batch_duplicate_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        return hard_same_batch_duplicate_penalty_per_point(\n            self._ensure_q_batch(Xt),\n            enabled=self.exclude_same_batch_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        )\n\n    def _reference_penalty_per_point(\n        self,\n        Xt: Tensor,\n        X_ref,\n        *,\n        weight: float,\n        beta: float,\n        hard_enabled: bool,\n    ) -> Tensor:\n        Xt = self._ensure_q_batch(Xt)\n        zeros = Xt.new_zeros(Xt.shape[:-1])\n        Xr = self._transform_reference_like_candidate(X_ref, ref=Xt)\n        if Xr is None or Xr.numel() == 0:\n            return zeros\n        Xr2d = Xr.reshape(-1, Xr.shape[-1])\n        min_dist = torch.cdist(\n            Xt.reshape(-1, Xt.shape[-1]),\n            Xr2d,\n        ).min(dim=-1).values.reshape(*Xt.shape[:-1])\n        soft = weight * torch.exp(-beta * min_dist) if weight > 0.0 else zeros\n        hard = hard_reference_duplicate_penalty_per_point(\n            Xt,\n            Xr2d,\n            enabled=hard_enabled,\n            tolerance=self.hard_duplicate_tol,\n        )\n        return soft + hard\n\n    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        return self._reference_penalty_per_point(\n            Xt,\n            getattr(self, "X_pending", None),\n            weight=self.pending_penalty_weight,\n            beta=self.pending_penalty_beta,\n            hard_enabled=self.exclude_pending_duplicates,\n        )\n\n    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        return self._reference_penalty_per_point(\n            Xt,\n            getattr(self, "X_observed", None),\n            weight=self.observed_penalty_weight,\n            beta=self.observed_penalty_beta,\n            hard_enabled=self.exclude_observed_duplicates,\n        )\n\n    def _candidate_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        return (\n            self._pending_penalty_per_point(Xt)\n            + self._observed_penalty_per_point(Xt)\n            + self._same_batch_duplicate_penalty_per_point(Xt)\n        )\n\n'''
    text = text[:start] + penalty_methods + text[end:]

    for name in (
        "_MultiOutputUncertaintySamplingClassifierAcquisition",
        "_BALDMultiOutputAcquisition",
    ):
        start, end, block = class_block(text, name)
        block = replace_once(
            block,
            "        pending_penalty_beta: float = 10.0,\n",
            "        pending_penalty_beta: float = 10.0,\n"
            + duplicate_parameter_block(),
            label=f"{name} signature",
        )
        block = replace_once(
            block,
            "            pending_penalty_beta=pending_penalty_beta,\n",
            "            pending_penalty_beta=pending_penalty_beta,\n"
            + duplicate_super_block(),
            label=f"{name} super kwargs",
        )
        text = text[:start] + block + text[end:]
    write(path, text)


def patch_multioutput_binary_pof() -> None:
    path = "src/bochan/acquisition/binary/bayesian_optimization/multi_output.py"
    text = read(path)
    text = replace_once(
        text,
        "from bochan.acquisition.binary.epistemic import (\n",
        "from bochan.acquisition._duplicate_exclusion import (\n"
        "    hard_reference_duplicate_penalty_per_point,\n"
        "    hard_same_batch_duplicate_penalty_per_point,\n"
        "    resolve_observed_X,\n"
        ")\n\n"
        "from bochan.acquisition.binary.epistemic import (\n",
        label="multi-output PoF imports",
    )
    start, end, block = class_block(text, "qMultiOutputBinaryProbabilityOfFeasibility")
    block = replace_once(
        block,
        "        pending_penalty_beta: float = 10.0,\n",
        "        pending_penalty_beta: float = 10.0,\n"
        + duplicate_parameter_block(),
        label="PoF signature",
    )
    block = replace_once(
        block,
        "        self.pending_penalty_beta = float(pending_penalty_beta)\n",
        "        self.pending_penalty_beta = float(pending_penalty_beta)\n"
        "        self.observed_penalty_weight = float(observed_penalty_weight)\n"
        "        self.observed_penalty_beta = float(observed_penalty_beta)\n"
        "        self.hard_duplicate_tol = float(hard_duplicate_tol)\n"
        "        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n"
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n"
        "        if self.hard_duplicate_tol < 0.0:\n"
        "            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n",
        label="PoF fields",
    )
    block = replace_once(
        block,
        "        self.set_X_pending(None)\n",
        "        self.set_X_pending(X_pending)\n        self.set_X_observed(X_observed)\n",
        label="PoF references",
    )
    block = replace_once(
        block,
        '''    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:\n        """pending points を raw input space の値として保持する。"""\n        self.X_pending = self._coerce_pending_to_tensor(X_pending)\n''',
        '''    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:\n        """pending points を raw input space の値として保持する。"""\n        self.X_pending = self._coerce_pending_to_tensor(X_pending)\n\n    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:\n        """observed points を raw input space の値として保持する。"""\n        self.X_observed = self._coerce_pending_to_tensor(\n            resolve_observed_X(self.model, X_observed)\n        )\n''',
        label="PoF observed setter",
    )
    pending_start = block.index("    def _pending_penalty_per_point(self, expanded_X: Tensor) -> Tensor:\n")
    pending_end = block.index("    def _pointwise_pof(self, raw_X: Tensor, expanded_X: Tensor) -> Tensor:\n", pending_start)
    methods = '''    def _transform_reference_like_candidate(\n        self,\n        X_ref,\n        *,\n        ref: Tensor,\n    ) -> Optional[Tensor]:\n        Xr = self._coerce_pending_to_tensor(X_ref, ref=ref)\n        if Xr is None or Xr.numel() == 0:\n            return None\n        Xr_t = shape_X_for_model(self.model, ensure_q_batch(Xr))\n        return Xr_t.to(device=ref.device, dtype=ref.dtype)\n\n    def _same_batch_duplicate_penalty_per_point(self, expanded_X: Tensor) -> Tensor:\n        return hard_same_batch_duplicate_penalty_per_point(\n            ensure_q_batch(expanded_X),\n            enabled=self.exclude_same_batch_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        )\n\n    def _reference_penalty_per_point(\n        self,\n        expanded_X: Tensor,\n        X_ref,\n        *,\n        weight: float,\n        beta: float,\n        hard_enabled: bool,\n    ) -> Tensor:\n        expanded_X = ensure_q_batch(expanded_X)\n        zeros = expanded_X.new_zeros(expanded_X.shape[:-1])\n        Xr = self._transform_reference_like_candidate(X_ref, ref=expanded_X)\n        if Xr is None or Xr.numel() == 0:\n            return zeros\n        Xr2d = Xr.reshape(-1, Xr.shape[-1])\n        min_dist = torch.cdist(\n            expanded_X.reshape(-1, expanded_X.shape[-1]),\n            Xr2d,\n        ).min(dim=-1).values.reshape(*expanded_X.shape[:-1])\n        soft = weight * torch.exp(-beta * min_dist) if weight > 0.0 else zeros\n        hard = hard_reference_duplicate_penalty_per_point(\n            expanded_X,\n            Xr2d,\n            enabled=hard_enabled,\n            tolerance=self.hard_duplicate_tol,\n        )\n        return soft + hard\n\n    def _pending_penalty_per_point(self, expanded_X: Tensor) -> Tensor:\n        return self._reference_penalty_per_point(\n            expanded_X,\n            getattr(self, "X_pending", None),\n            weight=self.pending_penalty_weight,\n            beta=self.pending_penalty_beta,\n            hard_enabled=self.exclude_pending_duplicates,\n        )\n\n    def _observed_penalty_per_point(self, expanded_X: Tensor) -> Tensor:\n        return self._reference_penalty_per_point(\n            expanded_X,\n            getattr(self, "X_observed", None),\n            weight=self.observed_penalty_weight,\n            beta=self.observed_penalty_beta,\n            hard_enabled=self.exclude_observed_duplicates,\n        )\n\n    def _candidate_penalty_per_point(self, expanded_X: Tensor) -> Tensor:\n        return (\n            self._pending_penalty_per_point(expanded_X)\n            + self._observed_penalty_per_point(expanded_X)\n            + self._same_batch_duplicate_penalty_per_point(expanded_X)\n        )\n\n'''
    block = block[:pending_start] + methods + block[pending_end:]
    text = text[:start] + block + text[end:]
    write(path, text)


def patch_ordinal() -> None:
    paths = [
        "src/bochan/acquisition/ordinal/active_learning/single_output.py",
        "src/bochan/acquisition/ordinal/active_learning/multi_output.py",
        "src/bochan/acquisition/ordinal/active_learning/hetero_single_output.py",
        "src/bochan/acquisition/ordinal/active_learning/hetero_multi_output.py",
    ]
    for path in paths:
        text = read(path)
        text = text.replace(
            "exclude_observed_duplicates: bool = False",
            "exclude_observed_duplicates: bool = True",
        )
        write(path, text)
        add_kwargs_passthrough(path)


def patch_multiclass_single_output() -> None:
    path = "src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py"
    text = read(path)
    text = replace_once(
        text,
        "        exclude_pending_duplicates: bool = True,\n        observed_penalty_weight: float = 0.0,\n",
        "        exclude_pending_duplicates: bool = True,\n"
        "        exclude_observed_duplicates: bool = True,\n"
        "        observed_penalty_weight: float = 0.0,\n",
        label="multiclass single observed flag",
    )
    text = replace_once(
        text,
        "        X_observed: Tensor | None = None,\n",
        "        X_pending: Tensor | None = None,\n        X_observed: Tensor | None = None,\n",
        label="multiclass single references",
    )
    text = replace_once(
        text,
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n",
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n",
        label="multiclass single observed field",
    )
    text = replace_once(
        text,
        "        self.X_observed = _resolve_observed_X(model, X_observed)\n        self.set_X_pending(None)\n",
        "        self.X_observed = None\n        self.set_X_pending(X_pending)\n        self.set_X_observed(X_observed)\n",
        label="multiclass single reference init",
    )
    text = replace_once(
        text,
        "    def set_X_pending(self, X_pending: Tensor | None = None) -> None:\n        self.X_pending = None if X_pending is None else torch.as_tensor(X_pending).detach()\n",
        "    def set_X_pending(self, X_pending: Tensor | None = None) -> None:\n"
        "        self.X_pending = None if X_pending is None else torch.as_tensor(X_pending).detach()\n\n"
        "    def set_X_observed(self, X_observed: Tensor | None = None) -> None:\n"
        "        resolved = _resolve_observed_X(self.model, X_observed)\n"
        "        self.X_observed = None if resolved is None else torch.as_tensor(resolved).detach()\n",
        label="multiclass single observed setter",
    )
    old = '''    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        Xt = ensure_q_batch(Xt)\n        if self.observed_penalty_weight <= 0:\n            return Xt.new_zeros(Xt.shape[:-1])\n        Xobs = self._reference_points_transformed(self.X_observed, ref=Xt)\n        if Xobs is None:\n            return Xt.new_zeros(Xt.shape[:-1])\n        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xobs).min(dim=-1).values\n        return self.observed_penalty_weight * torch.exp(-self.observed_penalty_beta * dist.reshape(Xt.shape[:-1]))\n'''
    new = '''    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        Xt = ensure_q_batch(Xt)\n        zeros = Xt.new_zeros(Xt.shape[:-1])\n        Xobs = self._reference_points_transformed(self.X_observed, ref=Xt)\n        if Xobs is None:\n            return zeros\n        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xobs).min(dim=-1).values\n        soft = (\n            self.observed_penalty_weight\n            * torch.exp(-self.observed_penalty_beta * dist.reshape(Xt.shape[:-1]))\n            if self.observed_penalty_weight > 0.0\n            else zeros\n        )\n        hard = hard_reference_duplicate_penalty_per_point(\n            Xt,\n            Xobs,\n            enabled=self.exclude_observed_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        )\n        return soft + hard\n'''
    text = replace_once(text, old, new, label="multiclass single observed penalty")
    write(path, text)


def patch_multiclass_multi_output() -> None:
    path = "src/bochan/acquisition/multiclass/active_learning/multi_output.py"
    text = read(path)
    text = replace_once(
        text,
        "        exclude_pending_duplicates: bool = True,\n        X_observed: Tensor | None = None,\n",
        "        exclude_pending_duplicates: bool = True,\n"
        "        exclude_observed_duplicates: bool = True,\n"
        "        X_pending: Tensor | None = None,\n"
        "        X_observed: Tensor | None = None,\n",
        label="multiclass multi flags",
    )
    text = replace_once(
        text,
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n",
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)\n",
        label="multiclass multi observed field",
    )
    text = replace_once(
        text,
        "        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()\n",
        "        self.X_observed = None\n",
        label="multiclass multi observed init",
    )
    text = replace_once(
        text,
        "        self.set_X_pending(None)\n",
        "        self.set_X_pending(X_pending)\n        self.set_X_observed(X_observed)\n",
        label="multiclass multi references",
    )
    text = replace_once(
        text,
        "    hard_same_batch_duplicate_penalty_per_point,\n)",
        "    hard_same_batch_duplicate_penalty_per_point,\n    resolve_observed_X,\n)",
        label="multiclass multi helper import",
    )
    text = replace_once(
        text,
        "    def set_X_observed(self, X_observed: Tensor | None = None) -> None:\n        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()\n",
        "    def set_X_observed(self, X_observed: Tensor | None = None) -> None:\n"
        "        resolved = resolve_observed_X(self.model, X_observed)\n"
        "        self.X_observed = None if resolved is None else torch.as_tensor(resolved).detach()\n",
        label="multiclass multi observed setter",
    )
    old = '''    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        Xt = self._ensure_q_batch(Xt)\n        if self.observed_penalty_weight <= 0:\n            return Xt.new_zeros(Xt.shape[:-1])\n        Xobs = self._reference_points_transformed(self.X_observed, ref=Xt)\n        if Xobs is None:\n            return Xt.new_zeros(Xt.shape[:-1])\n        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xobs).min(dim=-1).values\n        return self.observed_penalty_weight * torch.exp(-self.observed_penalty_beta * dist.reshape(Xt.shape[:-1]))\n'''
    new = '''    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:\n        Xt = self._ensure_q_batch(Xt)\n        zeros = Xt.new_zeros(Xt.shape[:-1])\n        Xobs = self._reference_points_transformed(self.X_observed, ref=Xt)\n        if Xobs is None:\n            return zeros\n        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), Xobs).min(dim=-1).values\n        soft = (\n            self.observed_penalty_weight\n            * torch.exp(-self.observed_penalty_beta * dist.reshape(Xt.shape[:-1]))\n            if self.observed_penalty_weight > 0.0\n            else zeros\n        )\n        hard = hard_reference_duplicate_penalty_per_point(\n            Xt,\n            Xobs,\n            enabled=self.exclude_observed_duplicates,\n            tolerance=self.hard_duplicate_tol,\n        )\n        return soft + hard\n'''
    text = replace_once(text, old, new, label="multiclass multi observed penalty")
    write(path, text)


def write_tests() -> None:
    path = "tests/test_classification_duplicate_controls_e2e.py"
    content = '''from __future__ import annotations\n\nfrom types import SimpleNamespace\n\nimport torch\nfrom botorch.optim import optimize_acqf\n\nfrom bochan.acquisition._duplicate_exclusion import (\n    hard_reference_duplicate_penalty_per_point,\n    hard_same_batch_duplicate_penalty_per_point,\n)\nfrom bochan.acquisition.binary.active_learning.multi_output import (\n    qMultiOutputBinaryPredictiveEntropy,\n)\nfrom bochan.acquisition.multiclass.active_learning.single_output import (\n    qMulticlassPredictiveEntropy,\n)\nfrom bochan.acquisition.ordinal.active_learning.single_output import (\n    qOrdinalPredictiveEntropy,\n)\n\n\nclass _DummyMultiOutputBinaryModel(torch.nn.Module):\n    def __init__(self) -> None:\n        super().__init__()\n        self.train_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)\n\n    def probability_posterior(self, X: torch.Tensor):\n        x = X[..., 0]\n        p1 = torch.sigmoid(10.0 * (x - 0.45))\n        p2 = torch.sigmoid(-8.0 * (x - 0.65))\n        return SimpleNamespace(mean=torch.stack([p1, p2], dim=-1))\n\n\nclass _DummyOrdinalLikelihood(torch.nn.Module):\n    def marginal_class_probs(self, distribution):\n        x = distribution.mean\n        logits = torch.stack(\n            [\n                -10.0 * (x - 0.25).square(),\n                -10.0 * (x - 0.50).square(),\n                -10.0 * (x - 0.75).square(),\n            ],\n            dim=-1,\n        )\n        return torch.softmax(logits, dim=-1)\n\n    def class_probs_from_f(self, latent: torch.Tensor):\n        return self.marginal_class_probs(SimpleNamespace(mean=latent))\n\n\nclass _DummyOrdinalModel(torch.nn.Module):\n    def __init__(self) -> None:\n        super().__init__()\n        self.train_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)\n\n    def posterior(self, X: torch.Tensor):\n        return SimpleNamespace(distribution=SimpleNamespace(mean=X[..., 0]))\n\n\nclass _DummyMulticlassModel(torch.nn.Module):\n    def __init__(self) -> None:\n        super().__init__()\n        self.train_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)\n\n    def class_probs(self, X: torch.Tensor) -> torch.Tensor:\n        x = X[..., 0]\n        logits = torch.stack([x, 1.0 - x, -4.0 * (x - 0.5).square()], dim=-1)\n        return torch.softmax(logits, dim=-1)\n\n\ndef test_duplicate_tolerance_is_euclidean_not_squared() -> None:\n    tolerance = 1e-4\n    inside = torch.tensor([[[0.0], [0.5e-4]]], dtype=torch.double)\n    outside = torch.tensor([[[0.0], [2.0e-4]]], dtype=torch.double)\n\n    assert torch.isinf(\n        hard_same_batch_duplicate_penalty_per_point(\n            inside,\n            tolerance=tolerance,\n        )\n    ).all()\n    assert torch.equal(\n        hard_same_batch_duplicate_penalty_per_point(\n            outside,\n            tolerance=tolerance,\n        ),\n        torch.zeros(1, 2, dtype=torch.double),\n    )\n    assert torch.equal(\n        hard_reference_duplicate_penalty_per_point(\n            outside[..., 1:, :],\n            torch.tensor([[0.0]], dtype=torch.double),\n            tolerance=tolerance,\n        ),\n        torch.zeros(1, 1, dtype=torch.double),\n    )\n\n\ndef test_binary_ordinal_and_multiclass_resolve_observed_consistently() -> None:\n    binary_model = _DummyMultiOutputBinaryModel()\n    binary = qMultiOutputBinaryPredictiveEntropy(binary_model)\n    assert torch.equal(binary.X_observed, binary_model.train_X)\n    assert torch.isinf(\n        binary._observed_penalty_per_point(binary_model.train_X[:1].view(1, 1, 1))\n    ).all()\n\n    ordinal_model = _DummyOrdinalModel()\n    ordinal = qOrdinalPredictiveEntropy(\n        ordinal_model,\n        ordinal_likelihood=_DummyOrdinalLikelihood(),\n    )\n    assert torch.equal(ordinal.X_observed, ordinal_model.train_X)\n    assert torch.isinf(\n        ordinal._pointwise_reference_penalty(ordinal_model.train_X[:1].view(1, 1, 1))\n    ).all()\n\n    multiclass_model = _DummyMulticlassModel()\n    multiclass = qMulticlassPredictiveEntropy(multiclass_model)\n    assert torch.equal(multiclass.X_observed, multiclass_model.train_X)\n    assert torch.isinf(\n        multiclass._observed_penalty_per_point(\n            multiclass_model.train_X[:1].view(1, 1, 1)\n        )\n    ).all()\n\n\ndef test_public_duplicate_controls_can_be_disabled() -> None:\n    model = _DummyMultiOutputBinaryModel()\n    acquisition = qMultiOutputBinaryPredictiveEntropy(\n        model,\n        exclude_same_batch_duplicates=False,\n        exclude_pending_duplicates=False,\n        exclude_observed_duplicates=False,\n        X_pending=torch.tensor([[0.5]], dtype=torch.double),\n        X_observed=torch.tensor([[0.5]], dtype=torch.double),\n    )\n    duplicate = torch.tensor([[[0.5], [0.5]]], dtype=torch.double)\n    assert torch.equal(\n        acquisition._candidate_penalty_per_point(duplicate),\n        torch.zeros(1, 2, dtype=torch.double),\n    )\n\n\ndef test_optimize_acqf_multi_output_binary_avoids_exact_duplicates() -> None:\n    torch.manual_seed(0)\n    acquisition = qMultiOutputBinaryPredictiveEntropy(\n        _DummyMultiOutputBinaryModel(),\n        X_observed=torch.tensor([[0.45], [0.65]], dtype=torch.double),\n        hard_duplicate_tol=1e-6,\n    )\n    candidates, value = optimize_acqf(\n        acq_function=acquisition,\n        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),\n        q=2,\n        num_restarts=8,\n        raw_samples=128,\n        sequential=False,\n    )\n    assert candidates.shape == torch.Size([2, 1])\n    assert torch.isfinite(value).all()\n    assert not torch.allclose(candidates[0], candidates[1], rtol=0.0, atol=1e-6)\n    for observed in acquisition.X_observed:\n        assert not torch.allclose(candidates[0], observed, rtol=0.0, atol=1e-6)\n        assert not torch.allclose(candidates[1], observed, rtol=0.0, atol=1e-6)\n\n\ndef test_optimize_acqf_ordinal_avoids_exact_duplicates() -> None:\n    torch.manual_seed(1)\n    acquisition = qOrdinalPredictiveEntropy(\n        _DummyOrdinalModel(),\n        ordinal_likelihood=_DummyOrdinalLikelihood(),\n        X_observed=torch.tensor([[0.5]], dtype=torch.double),\n        hard_duplicate_tol=1e-6,\n    )\n    candidates, value = optimize_acqf(\n        acq_function=acquisition,\n        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),\n        q=2,\n        num_restarts=8,\n        raw_samples=128,\n        sequential=False,\n    )\n    assert candidates.shape == torch.Size([2, 1])\n    assert torch.isfinite(value).all()\n    assert not torch.allclose(candidates[0], candidates[1], rtol=0.0, atol=1e-6)\n    assert not torch.allclose(\n        candidates[0],\n        acquisition.X_observed[0],\n        rtol=0.0,\n        atol=1e-6,\n    )\n    assert not torch.allclose(\n        candidates[1],\n        acquisition.X_observed[0],\n        rtol=0.0,\n        atol=1e-6,\n    )\n'''
    write(path, content)


def main() -> None:
    patch_duplicate_helpers()
    patch_binary_base()
    patch_binary_single_output()
    patch_binary_multi_output_active_learning()
    patch_multioutput_binary_pof()
    patch_ordinal()
    patch_multiclass_single_output()
    patch_multiclass_multi_output()
    write_tests()


if __name__ == "__main__":
    main()
