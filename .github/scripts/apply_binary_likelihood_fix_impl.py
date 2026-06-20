from __future__ import annotations

import ast
import copy
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "bochan"
ACQ_BINARY = SRC / "acquisition" / "binary"
MODEL_BINARY = SRC / "models" / "classification" / "binary"

HELPER_IMPORT_LATENT = (
    "from bochan.acquisition.binary._likelihood import "
    "latent_samples_to_binary_probabilities"
)
HELPER_IMPORT_VALUES = (
    "from bochan.acquisition.binary._likelihood import "
    "values_to_binary_probabilities"
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_import(text: str, import_line: str) -> str:
    if import_line in text:
        return text

    markers = ["from torch import Tensor\n", "import torch\n"]
    for marker in markers:
        if marker in text:
            return text.replace(marker, marker + import_line + "\n", 1)
    raise RuntimeError(f"Could not find import insertion point for {import_line!r}")


def offsets(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\\n", text):
        starts.append(match.end())
    return starts


def node_span(node: ast.AST, line_offsets: list[int]) -> tuple[int, int]:
    start = line_offsets[node.lineno - 1] + node.col_offset
    end = line_offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    result: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            result[child] = parent
    return result


def enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def function_arg_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    if function.args.vararg is not None:
        args.append(function.args.vararg)
    if function.args.kwarg is not None:
        args.append(function.args.kwarg)
    return {arg.arg for arg in args}


def function_source(text: str, function: ast.AST, line_offsets: list[int]) -> str:
    start, end = node_span(function, line_offsets)
    return text[start:end]


def model_and_eps_expr(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    text: str,
    line_offsets: list[int],
) -> tuple[str, str]:
    args = function_arg_names(function)
    source = function_source(text, function, line_offsets)

    if "self" in args:
        model_expr = "self.model"
        eps_expr = "self.eps" if "self.eps" in source else "1e-6"
        return model_expr, eps_expr
    if "model" in args:
        return "model", "eps" if "eps" in args else "1e-6"

    raise RuntimeError(
        f"Cannot resolve model for likelihood conversion in function {function.name!r}."
    )


def is_torch_sigmoid(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "sigmoid"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "torch"
    )


def rewrite_plain_sigmoid_calls(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    parents = parent_map(tree)
    line_offsets = offsets(text)
    replacements: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not is_torch_sigmoid(node):
            continue
        if len(node.args) != 1 or node.keywords:
            continue
        argument = node.args[0]
        if not isinstance(argument, ast.Name):
            # Complex expressions such as smooth PI / ROI gates intentionally use sigmoid.
            continue

        function = enclosing_function(node, parents)
        if function is None:
            raise RuntimeError(f"torch.sigmoid({argument.id}) outside function in {path}")
        model_expr, eps_expr = model_and_eps_expr(function, text, line_offsets)
        start, end = node_span(node, line_offsets)
        replacement = (
            "latent_samples_to_binary_probabilities("
            f"{model_expr}, {argument.id}, eps={eps_expr}, "
            f"name=\"{argument.id} via binary likelihood\""
            ")"
        )
        replacements.append((start, end, replacement))

    if not replacements:
        return False

    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    text = ensure_import(text, HELPER_IMPORT_LATENT)
    write(path, text)
    return True


def rewrite_to_probability_calls(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    parents = parent_map(tree)
    line_offsets = offsets(text)
    replacements: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "to_probability":
            continue
        if any(keyword.arg == "model" for keyword in node.keywords):
            continue

        function = enclosing_function(node, parents)
        if function is None:
            continue
        try:
            model_expr, _ = model_and_eps_expr(function, text, line_offsets)
        except RuntimeError:
            continue

        new_node = copy.deepcopy(node)
        new_node.keywords.append(
            ast.keyword(arg="model", value=ast.parse(model_expr, mode="eval").body)
        )
        replacement = ast.unparse(new_node)
        start, end = node_span(node, line_offsets)
        replacements.append((start, end, replacement))

    if not replacements:
        return False

    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    write(path, text)
    return True


def replace_to_probability_helper() -> None:
    path = ACQ_BINARY / "bayesian_optimization" / "_utils.py"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"def to_probability\(\n.*?\n\n\ndef binary_entropy",
        flags=re.DOTALL,
    )
    replacement = '''def to_probability(
    x: Tensor,
    *,
    apply_sigmoid_if_needed: bool,
    eps: float,
    name: str,
    model: Optional[Model] = None,
) -> Tensor:
    """Convert probability values or latent values using the model likelihood.

    ``apply_sigmoid_if_needed`` is retained as a compatibility argument.  When
    conversion is required it no longer means a hard-coded sigmoid: the model's
    binary likelihood is used, so GPyTorch ``BernoulliLikelihood`` follows its
    probit link and custom likelihoods follow their own conditional link.
    """
    xmin = x.min().item()
    xmax = x.max().item()
    if 0.0 <= xmin and xmax <= 1.0:
        return x.clamp(eps, 1.0 - eps)
    if apply_sigmoid_if_needed:
        if model is None:
            raise RuntimeError(
                f"{name} requires latent-to-probability conversion, but no model "
                "was provided. Pass model=... so the binary likelihood link can be used."
            )
        return values_to_binary_probabilities(
            model,
            x,
            eps=eps,
            name=name,
        )
    raise RuntimeError(
        f"{name} is not in [0,1] (min={xmin:.4g}, max={xmax:.4g}). "
        "Return a probability posterior or enable likelihood-aware conversion."
    )


def binary_entropy'''
    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace to_probability in {path}; count={count}")
    new_text = ensure_import(new_text, HELPER_IMPORT_VALUES)
    write(path, new_text)


def canonicalize_mode_names(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = text.replace(
        'PoFMode = Literal["mc_sigmoid", "latent_cdf"]',
        'PoFMode = Literal["mc_likelihood", "mc_sigmoid", "latent_cdf"]',
    )
    text = text.replace('mode: PoFMode = "mc_sigmoid"', 'mode: PoFMode = "mc_likelihood"')
    text = text.replace('def _mc_sigmoid_prob(', 'def _mc_likelihood_prob(')
    text = text.replace(
        'if self.mode == "mc_sigmoid":',
        'if self.mode in {"mc_likelihood", "mc_sigmoid"}:',
    )
    text = text.replace('return self._mc_sigmoid_prob(', 'return self._mc_likelihood_prob(')
    text = text.replace('`mc_sigmoid` or `latent_cdf`', '`mc_likelihood` or `latent_cdf`')
    text = text.replace('例: `mc_sigmoid` または `latent_cdf`', '例: `mc_likelihood` または `latent_cdf`')
    text = text.replace('sample して sigmoid', 'sample して likelihood link で確率化')
    text = text.replace('sigmoid 変換を検討します', 'likelihood link による変換を検討します')
    text = text.replace('sigmoid で probability に変換する', 'likelihood link で probability に変換する')
    text = text.replace('sigmoid 変換するかどうか', 'likelihood link で変換するかどうか')
    if text != original:
        write(path, text)
        return True
    return False


def add_helper_module() -> None:
    path = ACQ_BINARY / "_likelihood.py"
    write(
        path,
        '''"""Likelihood-aware probability transforms for binary classification."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _resolve_binary_likelihood(model: Any) -> Any:
    """Resolve the likelihood that defines ``p(y=1 | f)`` for one model."""
    likelihood = getattr(model, "likelihood", None)
    if likelihood is not None:
        return likelihood

    inner = getattr(model, "model", None)
    likelihood = getattr(inner, "likelihood", None) if inner is not None else None
    if likelihood is not None:
        return likelihood

    raise AttributeError(
        f"Binary likelihood was not found for {type(model).__name__}. "
        "Expected model.likelihood or model.model.likelihood."
    )


def _conditional_probability(
    model: Any,
    latent_samples: Tensor,
    *,
    eps: float,
    name: str,
) -> Tensor:
    likelihood = _resolve_binary_likelihood(model)
    conditional = likelihood.forward(latent_samples)

    probs = getattr(conditional, "probs", None)
    if probs is None:
        probs = getattr(conditional, "mean", None)
    if probs is None or not torch.is_tensor(probs):
        raise TypeError(
            f"{name}: {type(likelihood).__name__}.forward(...) did not return "
            "a distribution with Tensor probs or mean."
        )
    if not torch.isfinite(probs).all():
        raise RuntimeError(f"{name}: binary likelihood returned NaN or inf probabilities.")

    pmin = probs.detach().min().item()
    pmax = probs.detach().max().item()
    tolerance = 1e-6
    if pmin < -tolerance or pmax > 1.0 + tolerance:
        raise RuntimeError(
            f"{name}: binary likelihood returned values outside [0,1] "
            f"(min={pmin:.4g}, max={pmax:.4g})."
        )
    return probs.clamp(eps, 1.0 - eps)


def latent_samples_to_binary_probabilities(
    model: Any,
    latent_samples: Tensor,
    *,
    eps: float = 1e-6,
    name: str = "latent samples",
    output_dim: int = -1,
) -> Tensor:
    """Map latent samples to probabilities with each model's own likelihood.

    For a single-output classifier this calls ``model.likelihood.forward``.
    For ModelList-like wrappers, the last output dimension is split and each
    submodel's likelihood is applied independently.  Consequently the default
    GPyTorch ``BernoulliLikelihood`` uses its probit link, while a custom
    logistic likelihood continues to use sigmoid without acquisition-side
    hard-coding.
    """
    submodels = getattr(model, "models", None)
    if submodels is None:
        return _conditional_probability(
            model,
            latent_samples,
            eps=eps,
            name=name,
        )

    submodels = list(submodels)
    if len(submodels) == 0:
        raise ValueError(f"{name}: model.models is empty.")

    dim = output_dim if output_dim >= 0 else latent_samples.ndim + output_dim
    if not 0 <= dim < latent_samples.ndim:
        raise IndexError(
            f"{name}: output_dim={output_dim} is invalid for shape "
            f"{tuple(latent_samples.shape)}."
        )
    if latent_samples.shape[dim] != len(submodels):
        if len(submodels) == 1:
            return _conditional_probability(
                submodels[0],
                latent_samples,
                eps=eps,
                name=name,
            )
        raise RuntimeError(
            f"{name}: latent output dimension {latent_samples.shape[dim]} does not "
            f"match number of submodels {len(submodels)}. "
            f"shape={tuple(latent_samples.shape)}, output_dim={output_dim}."
        )

    outputs = []
    for index, (submodel, samples_i) in enumerate(
        zip(submodels, latent_samples.unbind(dim=dim))
    ):
        outputs.append(
            _conditional_probability(
                submodel,
                samples_i,
                eps=eps,
                name=f"{name}[output={index}]",
            )
        )
    return torch.stack(outputs, dim=dim)


def values_to_binary_probabilities(
    model: Any,
    values: Tensor,
    *,
    eps: float = 1e-6,
    name: str = "binary values",
    output_dim: int = -1,
) -> Tensor:
    """Validate probability values or transform latent values via likelihood."""
    if not torch.isfinite(values).all():
        raise RuntimeError(f"{name}: values contain NaN or inf.")

    vmin = values.detach().min().item()
    vmax = values.detach().max().item()
    if 0.0 <= vmin and vmax <= 1.0:
        return values.clamp(eps, 1.0 - eps)

    return latent_samples_to_binary_probabilities(
        model,
        values,
        eps=eps,
        name=name,
        output_dim=output_dim,
    )


__all__ = [
    "latent_samples_to_binary_probabilities",
    "values_to_binary_probabilities",
]
''',
    )


def add_tests() -> None:
    path = ROOT / "tests" / "test_binary_likelihood_consistency.py"
    write(
        path,
        '''from __future__ import annotations

import ast
from pathlib import Path

import torch
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood
from torch.distributions import Bernoulli, Normal

from bochan.acquisition.binary._likelihood import (
    latent_samples_to_binary_probabilities,
    values_to_binary_probabilities,
)
from bochan.acquisition.binary.bayesian_optimization._utils import to_probability


class _Model:
    def __init__(self, likelihood) -> None:
        self.likelihood = likelihood


class _ModelList:
    def __init__(self, *models) -> None:
        self.models = list(models)


class _LogisticBernoulliLikelihood(Likelihood):
    def forward(self, function_samples: torch.Tensor, **kwargs) -> Bernoulli:
        return Bernoulli(probs=torch.sigmoid(function_samples))


def test_default_bernoulli_likelihood_uses_probit_link() -> None:
    latent = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.double)
    model = _Model(BernoulliLikelihood())

    actual = latent_samples_to_binary_probabilities(model, latent, eps=1e-12)
    expected = Normal(0.0, 1.0).cdf(latent)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(actual, torch.sigmoid(latent), atol=1e-4, rtol=1e-4)


def test_custom_likelihood_controls_the_link() -> None:
    latent = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double)
    model = _Model(_LogisticBernoulliLikelihood())

    actual = latent_samples_to_binary_probabilities(model, latent, eps=1e-12)

    assert torch.allclose(actual, torch.sigmoid(latent), atol=1e-10, rtol=1e-10)


def test_model_list_applies_each_output_likelihood() -> None:
    latent = torch.tensor(
        [[[-1.0, -1.0], [1.0, 1.0]]],
        dtype=torch.double,
    )
    model = _ModelList(
        _Model(BernoulliLikelihood()),
        _Model(_LogisticBernoulliLikelihood()),
    )

    actual = latent_samples_to_binary_probabilities(model, latent, eps=1e-12)

    assert torch.allclose(actual[..., 0], Normal(0.0, 1.0).cdf(latent[..., 0]))
    assert torch.allclose(actual[..., 1], torch.sigmoid(latent[..., 1]))


def test_probability_values_are_not_transformed_twice() -> None:
    probability = torch.tensor([0.2, 0.5, 0.8], dtype=torch.double)
    model = _Model(BernoulliLikelihood())

    actual = values_to_binary_probabilities(model, probability)
    assert torch.allclose(actual, probability)


def test_legacy_to_probability_argument_is_likelihood_aware() -> None:
    latent = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double)
    model = _Model(BernoulliLikelihood())

    actual = to_probability(
        latent,
        apply_sigmoid_if_needed=True,
        eps=1e-12,
        name="test latent",
        model=model,
    )

    assert torch.allclose(actual, Normal(0.0, 1.0).cdf(latent))


def test_binary_acquisitions_do_not_apply_sigmoid_to_plain_latent_values() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "bochan" / "acquisition" / "binary"
    violations: list[str] = []

    for path in sorted(root.rglob("*.py")):
        if path.name == "_likelihood.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_sigmoid = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "sigmoid"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            )
            if is_sigmoid and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                violations.append(
                    f"{path.relative_to(root)}:{node.lineno}: torch.sigmoid({node.args[0].id})"
                )

    assert not violations, "\n".join(violations)


def test_binary_models_do_not_hard_code_sigmoid_link() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "bochan" / "models" / "classification" / "binary"
    violations: list[str] = []

    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "sigmoid"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")

    assert not violations, "\n".join(violations)


def test_mc_sigmoid_is_only_a_compatibility_alias() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "bochan" / "acquisition" / "binary"
    bad_defaults: list[str] = []
    bad_method_names: list[str] = []

    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if 'mode: PoFMode = "mc_sigmoid"' in source:
            bad_defaults.append(str(path.relative_to(root)))
        if "def _mc_sigmoid_prob" in source:
            bad_method_names.append(str(path.relative_to(root)))

    assert not bad_defaults
    assert not bad_method_names
''',
    )


def audit() -> None:
    plain_sigmoid: list[str] = []
    for path in sorted(ACQ_BINARY.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and is_torch_sigmoid(node)
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
            ):
                plain_sigmoid.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: {node.args[0].id}"
                )

    model_sigmoid: list[str] = []
    for path in sorted(MODEL_BINARY.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and is_torch_sigmoid(node):
                model_sigmoid.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    if plain_sigmoid or model_sigmoid:
        raise RuntimeError(
            "Unresolved binary link conversions:\n"
            + "\n".join(plain_sigmoid + model_sigmoid)
        )


def main() -> None:
    add_helper_module()
    replace_to_probability_helper()

    for path in sorted(ACQ_BINARY.rglob("*.py")):
        if path.name == "_likelihood.py":
            continue
        canonicalize_mode_names(path)

    for path in sorted(ACQ_BINARY.rglob("*.py")):
        if path.name == "_likelihood.py":
            continue
        rewrite_plain_sigmoid_calls(path)

    for path in sorted((ACQ_BINARY / "bayesian_optimization").rglob("*.py")):
        rewrite_to_probability_calls(path)

    add_tests()
    audit()

    changed = [
        path.relative_to(ROOT)
        for path in sorted(ACQ_BINARY.rglob("*.py"))
        if HELPER_IMPORT_LATENT in path.read_text(encoding="utf-8")
        or HELPER_IMPORT_VALUES in path.read_text(encoding="utf-8")
        or path.name == "_likelihood.py"
    ]
    print("Likelihood-aware binary files:")
    for path in changed:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
