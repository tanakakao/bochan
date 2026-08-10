from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".json", ".txt"}


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def replace_exact(rel: str, old: str, new: str, required: bool = True) -> None:
    text = read(rel)
    if old not in text:
        if required:
            raise RuntimeError(f"{rel}: missing expected text: {old[:100]!r}")
        return
    write(rel, text.replace(old, new))


def top_function_span(text: str, name: str) -> tuple[int, int]:
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.lineno - 1, node.end_lineno
    raise RuntimeError(f"function {name} not found")


def class_method_span(text: str, class_name: str, method_name: str) -> tuple[int, int]:
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return item.lineno - 1, item.end_lineno
    raise RuntimeError(f"{class_name}.{method_name} not found")


def class_end(text: str, class_name: str) -> int:
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node.end_lineno
    raise RuntimeError(f"class {class_name} not found")


def replace_method(rel: str, class_name: str, method_name: str, source: str) -> None:
    text = read(rel)
    start, end = class_method_span(text, class_name, method_name)
    lines = text.splitlines(keepends=True)
    lines[start:end] = [source.rstrip() + "\n"]
    write(rel, "".join(lines))


def remove_function(rel: str, name: str) -> None:
    text = read(rel)
    try:
        start, end = top_function_span(text, name)
    except RuntimeError:
        return
    lines = text.splitlines(keepends=True)
    while end < len(lines) and not lines[end].strip():
        end += 1
    del lines[start:end]
    write(rel, "".join(lines))


def remove_assignment(rel: str, name: str) -> None:
    text = read(rel)
    spans = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                spans.append((node.lineno - 1, node.end_lineno))
    lines = text.splitlines(keepends=True)
    for start, end in reversed(spans):
        del lines[start:end]
    if spans:
        write(rel, "".join(lines))


def rename_method(rel: str, class_name: str, old: str, new: str) -> None:
    text = read(rel)
    start, _ = class_method_span(text, class_name, old)
    lines = text.splitlines(keepends=True)
    lines[start] = re.sub(
        rf"^(\s*def\s+){re.escape(old)}(\s*\()",
        rf"\1{new}\2",
        lines[start],
        count=1,
    )
    write(rel, "".join(lines))


def append_to_class(rel: str, class_name: str, source: str) -> None:
    text = read(rel)
    pos = class_end(text, class_name)
    lines = text.splitlines(keepends=True)
    lines.insert(pos, "\n" + source.rstrip() + "\n")
    write(rel, "".join(lines))


def function_source(rel: str, name: str) -> str:
    text = read(rel)
    start, end = top_function_span(text, name)
    return "".join(text.splitlines(keepends=True)[start:end]).rstrip()


def rename_everywhere(mapping: dict[str, str]) -> None:
    paths = []
    for root_name in ("src", "tests", "docs", "notebooks", "web"):
        root = ROOT / root_name
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES)
    for name in ("README.md", "README.ja.md", "AGENTS.md"):
        path = ROOT / name
        if path.exists():
            paths.append(path)

    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        new = text
        for old, replacement in items:
            new = re.sub(rf"\b{re.escape(old)}\b", replacement, new)
        if new != text:
            path.write_text(new, encoding="utf-8")


def formalize_projected() -> None:
    utils = "src/bochan/models/components/projected_utils.py"
    text = read(utils)
    if '"flatten_projected_one_to_many_point_axes"' not in text:
        text = text.replace(
            '    "_expand_raw_X_to_match_transformed_q",\n',
            '    "_expand_raw_X_to_match_transformed_q",\n'
            '    "flatten_projected_one_to_many_point_axes",\n',
        )
        write(utils, text)
        text = read(utils)
        _, end = top_function_span(text, "_expand_raw_X_to_match_transformed_q")
        lines = text.splitlines(keepends=True)
        helper = '''
def flatten_projected_one_to_many_point_axes(
    X: Tensor,
    transformed: Tensor,
) -> Tensor:
    if isinstance(X, tuple):
        X = X[0]
    if not torch.is_tensor(X) or not torch.is_tensor(transformed):
        return transformed
    if X.ndim < 2 or transformed.ndim < X.ndim:
        return transformed

    batch_ndim = max(0, X.ndim - 2)
    if tuple(transformed.shape[:batch_ndim]) != tuple(X.shape[:batch_ndim]):
        return transformed

    point_shape = transformed.shape[batch_ndim:-1]
    if len(point_shape) <= 1:
        return transformed

    q_like = 1
    for size in point_shape:
        q_like *= int(size)
    return transformed.reshape(
        *transformed.shape[:batch_ndim],
        q_like,
        transformed.shape[-1],
    )
'''
        lines.insert(end, "\n" + helper.strip("\n") + "\n\n")
        write(utils, "".join(lines))

    projected = "src/bochan/models/components/projected.py"
    replace_exact(
        projected,
        "    _get_cont_dims,\n",
        "    _get_cont_dims,\n    flatten_projected_one_to_many_point_axes,\n",
        required=False,
    )
    replace_method(
        projected,
        "_BaseProjectedModel",
        "transform_inputs",
        '''    def transform_inputs(self, X: Tensor) -> Tensor:
        X_raw = X[0] if isinstance(X, tuple) else X
        X_pre = self._to_preprojection_space(X_raw)
        projected = self._project_preprojected_inputs(X_pre)
        return flatten_projected_one_to_many_point_axes(X_raw, projected)
''',
    )

    write(
        "src/bochan/models/classification/multiclass/high_dim/__init__.py",
        '''from __future__ import annotations

from typing import Any

from torch import Tensor

from bochan.models.components.projected_utils import (
    flatten_projected_one_to_many_point_axes,
)
from .decomposition import (
    PCAMulticlassClassificationGPModel as _PCAMulticlassClassificationGPModel,
    PCAMulticlassClassificationMixedGPModel as _PCAMulticlassClassificationMixedGPModel,
    REMBOMulticlassClassificationGPModel as _REMBOMulticlassClassificationGPModel,
    REMBOMulticlassClassificationMixedGPModel as _REMBOMulticlassClassificationMixedGPModel,
)


class _ProjectedMulticlassModelMixin:
    def transform_inputs(self, X: Tensor) -> Tensor:
        transformed = super().transform_inputs(X)
        return flatten_projected_one_to_many_point_axes(X, transformed)

    def make_mll(self, beta: float = 1.0, **kwargs: Any):
        return self.base_model.make_mll(beta=float(beta), **kwargs)


class PCAMulticlassClassificationGPModel(
    _ProjectedMulticlassModelMixin,
    _PCAMulticlassClassificationGPModel,
):
    pass


class REMBOMulticlassClassificationGPModel(
    _ProjectedMulticlassModelMixin,
    _REMBOMulticlassClassificationGPModel,
):
    pass


class PCAMulticlassClassificationMixedGPModel(
    _ProjectedMulticlassModelMixin,
    _PCAMulticlassClassificationMixedGPModel,
):
    pass


class REMBOMulticlassClassificationMixedGPModel(
    _ProjectedMulticlassModelMixin,
    _REMBOMulticlassClassificationMixedGPModel,
):
    pass


__all__ = [
    "PCAMulticlassClassificationGPModel",
    "REMBOMulticlassClassificationGPModel",
    "PCAMulticlassClassificationMixedGPModel",
    "REMBOMulticlassClassificationMixedGPModel",
]
''',
    )

    replace_exact(
        "src/bochan/models/classification/binary/high_dim/__init__.py",
        "from .input_perturbation import configure_projected_binary_perturbation\n",
        "",
        required=False,
    )
    replace_exact(
        "src/bochan/models/classification/binary/high_dim/__init__.py",
        "\nconfigure_projected_binary_perturbation()\n",
        "\n",
        required=False,
    )
    ordinal = "src/bochan/models/ordinal/high_dim/__init__.py"
    text = read(ordinal)
    text = text.replace(
        "from bochan.models.projected_input_perturbation import configure_projected_model_classes\n\n",
        "",
    )
    text = re.sub(
        r"\nconfigure_projected_model_classes\([\s\S]*?\)\n(?=\n__all__)",
        "\n",
        text,
    )
    write(ordinal, text)

    for rel in (
        "src/bochan/models/projected_input_perturbation.py",
        "src/bochan/models/classification/binary/high_dim/input_perturbation.py",
    ):
        path = ROOT / rel
        if path.exists():
            path.unlink()


def formalize_hybrid() -> None:
    write(
        "src/bochan/models/hybrid/__init__.py",
        '''from __future__ import annotations

from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from torch import Tensor

from .class_probability_shapes import _select_class_probability_output
from .multi_output import HybridMultiOutputModel as _HybridMultiOutputModel
from .posterior import HybridPosterior
from .prediction import predict_class as _predict_class
from .prediction import predict_class_list as _predict_class_list
from .specs import OutputIndex, OutputSpec, PosteriorMode, make_output_specs
from .task_aware_sampling import _task_aware_posterior


class HybridMultiOutputModel(_HybridMultiOutputModel):
    _task_aware_original_posterior = _HybridMultiOutputModel.posterior

    def _set_transformed_inputs(self) -> None:
        return None

    def eval(self):
        self.training = False
        for model in self.models:
            model.eval()
        return self

    def _ordinal_class_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return _select_class_probability_output(
                    probs,
                    spec=spec,
                    X=X,
                    model=spec.model,
                ).clamp_min(0.0)
        return super()._ordinal_class_probs(spec, X, **kwargs)

    def _multiclass_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return _select_class_probability_output(
                    probs,
                    spec=spec,
                    X=X,
                    model=spec.model,
                ).clamp_min(0.0)
        post = self._call_accessor(
            spec.model,
            ("probability_posterior", "posterior"),
            X,
            **kwargs,
        )
        probs, _ = self._posterior_mean_variance(post, spec.name)
        return _select_class_probability_output(
            probs,
            spec=spec,
            X=X,
            model=spec.model,
        ).clamp_min(0.0)

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        *,
        output_mode: PosteriorMode = "objective",
        **kwargs: Any,
    ):
        return _task_aware_posterior(
            self,
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            output_mode=output_mode,
            **kwargs,
        )

    def predict_class_list(
        self,
        X: Tensor,
        output_indices: OutputIndex | list[OutputIndex] | Tensor | None = None,
        *,
        binary_threshold: float | Tensor = 0.5,
        **kwargs: Any,
    ) -> list[Tensor]:
        return _predict_class_list(
            self,
            X,
            output_indices=output_indices,
            binary_threshold=binary_threshold,
            **kwargs,
        )

    def predict_class(
        self,
        X: Tensor,
        output_indices: OutputIndex | list[OutputIndex] | Tensor | None = None,
        *,
        binary_threshold: float | Tensor = 0.5,
        **kwargs: Any,
    ) -> Tensor:
        return _predict_class(
            self,
            X,
            output_indices=output_indices,
            binary_threshold=binary_threshold,
            **kwargs,
        )


__all__ = [
    "HybridMultiOutputModel",
    "HybridPosterior",
    "OutputSpec",
    "PosteriorMode",
    "make_output_specs",
]
''',
    )
    remove_function(
        "src/bochan/models/hybrid/class_probability_shapes.py",
        "apply_hybrid_class_probability_shapes",
    )
    remove_assignment(
        "src/bochan/models/hybrid/class_probability_shapes.py",
        "_PATCHED",
    )
    remove_function(
        "src/bochan/models/hybrid/task_aware_sampling.py",
        "apply_task_aware_hybrid_posterior",
    )
    remove_function(
        "src/bochan/models/hybrid/prediction.py",
        "attach_prediction_methods",
    )

    for root_name in ("src", "tests"):
        root = ROOT / root_name
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            new = text.replace(
                "from bochan.models.hybrid.multi_output import HybridMultiOutputModel",
                "from bochan.models.hybrid import HybridMultiOutputModel",
            )
            if new != text:
                path.write_text(new, encoding="utf-8")


def formalize_gaussian_kronecker() -> None:
    rel = "src/bochan/models/regression/gaussian/kronecker_multitask.py"
    text = read(rel)
    start = text.index("def _install_kronecker_input_transform_support()")
    class_start = text.index("class MixedKroneckerMultiTaskGP")
    replacement = '''class GaussianKroneckerMultiTaskGP(KroneckerMultiTaskGP):
    def transform_inputs(
        self,
        X: Tensor,
        input_transform: InputTransform | None = None,
    ) -> Tensor:
        transform = (
            input_transform
            if input_transform is not None
            else getattr(self, "input_transform", None)
        )
        if transform is None:
            return X
        if bool(getattr(self, "training", True)):
            return _transform_without_one_to_many(X, transform).contiguous()
        if _is_stored_training_input(self, X):
            return X
        return super().transform_inputs(
            X,
            input_transform=transform,
        ).contiguous()

    def make_mll(self) -> ExactMarginalLogLikelihood:
        return ExactMarginalLogLikelihood(self.likelihood, self)


'''
    text = text[:start] + replacement + text[class_start:]
    text = text.replace(
        "class MixedKroneckerMultiTaskGP(KroneckerMultiTaskGP):",
        "class GaussianMixedKroneckerMultiTaskGP(GaussianKroneckerMultiTaskGP):",
    )
    text = re.sub(
        r"\n# Backward-supported alternative naming order\.\nKroneckerMultiTaskMixedGP = MixedKroneckerMultiTaskGP\n",
        "\n",
        text,
    )
    text = re.sub(
        r'__all__\s*=\s*\[[\s\S]*?\]\s*$',
        '__all__ = [\n'
        '    "GaussianKroneckerMultiTaskGP",\n'
        '    "GaussianMixedKroneckerMultiTaskGP",\n'
        ']\n',
        text,
    )
    write(rel, text)
    write(
        "src/bochan/models/regression/gaussian/__init__.py",
        '''from .kronecker_multitask import (
    GaussianKroneckerMultiTaskGP,
    GaussianMixedKroneckerMultiTaskGP,
)
from .multifidelity import (
    FidelityFeatureInputTransform,
    WideMixedMultiFidelityGP,
    WideMultiFidelityGP,
    WideMultiFidelityMixedGP,
    wide_fidelity_to_long,
)

__all__ = [
    "FidelityFeatureInputTransform",
    "GaussianKroneckerMultiTaskGP",
    "GaussianMixedKroneckerMultiTaskGP",
    "WideMixedMultiFidelityGP",
    "WideMultiFidelityGP",
    "WideMultiFidelityMixedGP",
    "wide_fidelity_to_long",
]
''',
    )


def formalize_multiclass_alignment() -> None:
    alignment = "src/bochan/models/classification/multiclass/robust/heteroscedastic_alignment.py"
    source_path = ROOT / alignment
    if not source_path.exists():
        return
    helpers = [
        function_source(alignment, "_prod"),
        function_source(alignment, "_expand_q_like_to_ref"),
        function_source(alignment, "_align_like"),
    ]
    target = "src/bochan/models/classification/multiclass/robust/heteroscedastic.py"
    text = read(target)
    start, end = top_function_span(text, "_align_like")
    lines = text.splitlines(keepends=True)
    lines[start:end] = ["\n\n".join(helpers).rstrip() + "\n"]
    write(target, "".join(lines))

    init_rel = "src/bochan/models/classification/multiclass/robust/__init__.py"
    init = read(init_rel)
    init = re.sub(r"^from \.heteroscedastic_alignment import .*\n", "", init, flags=re.M)
    init = re.sub(r"^apply_heteroscedastic_alignment\(\)\n", "", init, flags=re.M)
    write(init_rel, init)
    source_path.unlink()


def formalize_ordinal_num_classes() -> None:
    paths = (
        "src/bochan/models/ordinal/deep/deepgp.py",
        "src/bochan/models/ordinal/deep/deepkernel_configurable.py",
        "src/bochan/models/ordinal/robust/relevance_pursuit.py",
        "src/bochan/models/ordinal/robust/heteroscedastic.py",
    )
    for rel in paths:
        text = read(rel).replace("num_classes: int,", "num_classes: int | None = None,")
        prepared = "        train_Y = _prepare_ordinal_targets(train_Y, train_X)\n"
        if prepared in text:
            text = text.replace(
                prepared,
                prepared
                + "        if num_classes is None:\n"
                + "            num_classes = int(train_Y.max().item()) + 1\n",
            )
        canonical = (
            "        train_Y = self._canonicalize_train_Y(\n"
            "            train_Y,\n"
            "            raw_train_X.shape[-2],\n"
            "            raw_train_X.device,\n"
            "        )\n"
        )
        if canonical in text:
            text = text.replace(
                canonical,
                canonical
                + "        if num_classes is None:\n"
                + "            num_classes = int(train_Y.max().item()) + 1\n",
            )
        write(rel, text)

    helper = ROOT / "src/bochan/models/ordinal/robust/_num_classes.py"
    if helper.exists():
        helper.unlink()


def main() -> None:
    formalize_projected()
    formalize_hybrid()
    formalize_gaussian_kronecker()
    formalize_multiclass_alignment()

    rename_everywhere(
        {
            "list_hidden_dims": "hidden_dims",
            "inducing_points_num": "num_inducing",
            "DeepKernelDeepMixedGPModel": "DeepKernelDeepGaussianMixedGPModel",
            "DeepKernelDeepGPModel": "DeepKernelDeepGaussianGPModel",
            "DeepKernelMixedGPModel": "DeepKernelGaussianMixedGPModel",
            "DeepKernelGPModel": "DeepKernelGaussianGPModel",
            "DeepMixedGPModel": "DeepGaussianMixedGPModel",
            "DeepGPModel": "DeepGaussianGPModel",
            "SaasMixedSingleTaskGP": "SaasGaussianMixedGPModel",
            "SaasSingleTaskGP": "SaasGaussianGPModel",
            "PCAMixedSingleTaskGP": "PCAGaussianMixedGPModel",
            "PCASingleTaskGP": "PCAGaussianGPModel",
            "REMBOMixedSingleTaskGP": "REMBOGaussianMixedGPModel",
            "REMBOSingleTaskGP": "REMBOGaussianGPModel",
            "VAEMixedSingleTaskGP": "VAEGaussianMixedGPModel",
            "VAESingleTaskGP": "VAEGaussianGPModel",
            "SafeRobustRelevancePursuitMixedSingleTaskGP": "RobustRelevancePursuitGaussianMixedGPModel",
            "SafeRobustRelevancePursuitSingleTaskGP": "RobustRelevancePursuitGaussianGPModel",
            "HeteroscedasticMixedSingleTaskGP": "HeteroscedasticGaussianMixedGPModel",
            "HeteroscedasticSingleTaskGP": "HeteroscedasticGaussianGPModel",
            "PerturbationSupportedKroneckerMultiTaskGP": "GaussianKroneckerMultiTaskGP",
            "KroneckerMultiTaskMixedGP": "GaussianMixedKroneckerMultiTaskGP",
            "MixedKroneckerMultiTaskGP": "GaussianMixedKroneckerMultiTaskGP",
            "BetaDeepMixedGPModel": "DeepBetaMixedGPModel",
            "BetaMixedDeepGPModel": "DeepBetaMixedGPModel",
            "BetaDeepGPModel": "DeepBetaGPModel",
            "GammaDeepMixedGPModel": "DeepGammaMixedGPModel",
            "GammaMixedDeepGPModel": "DeepGammaMixedGPModel",
            "GammaDeepGPModel": "DeepGammaGPModel",
            "PoissonDeepMixedGPModel": "DeepPoissonMixedGPModel",
            "PoissonMixedDeepGPModel": "DeepPoissonMixedGPModel",
            "PoissonDeepGPModel": "DeepPoissonGPModel",
            "NegativeBinomialDeepMixedGPModel": "DeepNegativeBinomialMixedGPModel",
            "NegativeBinomialMixedDeepGPModel": "DeepNegativeBinomialMixedGPModel",
            "NegativeBinomialDeepGPModel": "DeepNegativeBinomialGPModel",
            "BinaryClassificationMixedDeepGPModel": "DeepBinaryClassificationMixedGPModel",
            "BinaryClassificationDeepGPModel": "DeepBinaryClassificationGPModel",
            "DeepKernelBinaryClassificationMixedDeepGPModel": "DeepKernelDeepBinaryClassificationMixedGPModel",
            "DeepKernelBinaryClassificationDeepGPModel": "DeepKernelDeepBinaryClassificationGPModel",
            "MulticlassMixedDeepGPModel": "DeepMulticlassClassificationMixedGPModel",
            "MulticlassDeepGPModel": "DeepMulticlassClassificationGPModel",
            "OrdinalMixedDeepGPModel": "DeepOrdinalMixedGPModel",
            "OrdinalDeepGPModel": "DeepOrdinalGPModel",
            "DeepKernelOrdinalMixedDeepGPModel": "DeepKernelDeepOrdinalMixedGPModel",
            "DeepKernelOrdinalDeepGPModel": "DeepKernelDeepOrdinalGPModel",
            "OutlierRelevancePursuitBinaryClassificationMixedGPModel": "RobustRelevancePursuitBinaryClassificationMixedGPModel",
            "OutlierRelevancePursuitBinaryClassificationGPModel": "RobustRelevancePursuitBinaryClassificationGPModel",
            "OutlierRelevancePursuitMulticlassClassificationMixedGPModel": "RobustRelevancePursuitMulticlassClassificationMixedGPModel",
            "OutlierRelevancePursuitMulticlassClassificationGPModel": "RobustRelevancePursuitMulticlassClassificationGPModel",
            "OutlierRelevancePursuitOrdinalMixedGPModel": "RobustRelevancePursuitOrdinalMixedGPModel",
            "OutlierRelevancePursuitOrdinalGPModel": "RobustRelevancePursuitOrdinalGPModel",
            "OutlierRelevancePursuitBetaMixedGPModel": "RobustRelevancePursuitBetaMixedGPModel",
            "OutlierRelevancePursuitBetaGPModel": "RobustRelevancePursuitBetaGPModel",
            "OutlierRelevancePursuitGammaMixedGPModel": "RobustRelevancePursuitGammaMixedGPModel",
            "OutlierRelevancePursuitGammaGPModel": "RobustRelevancePursuitGammaGPModel",
            "OutlierRelevancePursuitPoissonMixedGPModel": "RobustRelevancePursuitPoissonMixedGPModel",
            "OutlierRelevancePursuitPoissonGPModel": "RobustRelevancePursuitPoissonGPModel",
            "OutlierRelevancePursuitNegativeBinomialMixedGPModel": "RobustRelevancePursuitNegativeBinomialMixedGPModel",
            "OutlierRelevancePursuitNegativeBinomialGPModel": "RobustRelevancePursuitNegativeBinomialGPModel",
        }
    )

    formalize_ordinal_num_classes()

    write(
        "src/bochan/models/ordinal/deep/__init__.py",
        '''from .deepgp import DeepOrdinalGPModel, DeepOrdinalMixedGPModel
from .deepkernel_configurable import DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel
from .deepkerneldeepgp import DeepKernelDeepOrdinalGPModel, DeepKernelDeepOrdinalMixedGPModel

__all__ = [
    "DeepOrdinalGPModel",
    "DeepOrdinalMixedGPModel",
    "DeepKernelOrdinalGPModel",
    "DeepKernelOrdinalMixedGPModel",
    "DeepKernelDeepOrdinalGPModel",
    "DeepKernelDeepOrdinalMixedGPModel",
]
''',
    )
    write(
        "src/bochan/models/ordinal/robust/__init__.py",
        '''from .heteroscedastic import (
    HeteroscedasticOrdinalGPModel,
    HeteroscedasticOrdinalMixedGPModel,
)
from .relevance_pursuit import (
    RobustRelevancePursuitOrdinalGPModel,
    RobustRelevancePursuitOrdinalMixedGPModel,
)

__all__ = [
    "RobustRelevancePursuitOrdinalGPModel",
    "RobustRelevancePursuitOrdinalMixedGPModel",
    "HeteroscedasticOrdinalGPModel",
    "HeteroscedasticOrdinalMixedGPModel",
]
''',
    )

    forbidden = (
        "configure_projected_model_classes",
        "configure_projected_binary_perturbation",
        "apply_hybrid_class_probability_shapes",
        "apply_task_aware_hybrid_posterior",
        "attach_prediction_methods",
        "enable_num_classes_inference",
        "_install_kronecker_input_transform_support",
        "apply_heteroscedastic_alignment",
    )
    offenders = []
    for path in (ROOT / "src/bochan/models").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
    if offenders:
        raise RuntimeError("legacy model patch hooks remain:\n" + "\n".join(offenders))


if __name__ == "__main__":
    main()
