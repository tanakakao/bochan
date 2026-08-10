from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding="utf-8")


def strip_installer(rel: str, function_name: str, export_name: str) -> None:
    text = read(rel)
    marker = f"\ndef {function_name}("
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    text = text.replace("_PATCHED = False\n\n", "")
    text = re.sub(r"\n__all__\s*=\s*\[[^\]]*\]\s*$", "", text, flags=re.S)
    text = text.rstrip() + f'\n\n__all__ = ["{export_name}"]\n'
    write(rel, text)


def cleanup_hybrid() -> None:
    strip_installer(
        "src/bochan/models/hybrid/task_aware_sampling.py",
        "apply_task_aware_hybrid_posterior",
        "_task_aware_posterior",
    )
    strip_installer(
        "src/bochan/models/hybrid/class_probability_shapes.py",
        "apply_hybrid_class_probability_shapes",
        "_select_class_probability_output",
    )

    rel = "src/bochan/models/hybrid/prediction.py"
    text = read(rel)
    marker = "\ndef attach_prediction_methods("
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    text = re.sub(r"\n__all__\s*=\s*\[[^\]]*\]\s*$", "", text, flags=re.S)
    text = text.rstrip() + '\n\n__all__ = ["predict_class", "predict_class_list"]\n'
    write(rel, text)

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
    """Canonical hybrid multi-output model with task-aware posterior behavior."""

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
                    X,
                    output_index=spec.output_index,
                    name=f"{spec.name}.class_probs",
                ).clamp_min(0.0)
        return super()._ordinal_class_probs(spec, X, **kwargs)

    def _multiclass_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return _select_class_probability_output(
                    probs,
                    X,
                    output_index=spec.output_index,
                    name=f"{spec.name}.class_probs",
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
            X,
            output_index=spec.output_index,
            name=f"{spec.name}.posterior.mean",
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


def cleanup_projected_ordinal_init() -> None:
    write(
        "src/bochan/models/ordinal/high_dim/__init__.py",
        '''from .decomposition import (
    PCAOrdinalGPModel,
    PCAOrdinalMixedGPModel,
    REMBOOrdinalGPModel,
    REMBOOrdinalMixedGPModel,
)
from .saas_fixed import SaasOrdinalGPModel, SaasOrdinalMixedGPModel

__all__ = [
    "PCAOrdinalGPModel",
    "REMBOOrdinalGPModel",
    "PCAOrdinalMixedGPModel",
    "REMBOOrdinalMixedGPModel",
    "SaasOrdinalGPModel",
    "SaasOrdinalMixedGPModel",
]
''',
    )


def cleanup_multiclass_robust_init() -> None:
    rel = "src/bochan/models/classification/multiclass/robust/__init__.py"
    text = read(rel)
    text = re.sub(
        r'^from \.heteroscedastic_alignment import apply_heteroscedastic_alignment\n',
        "",
        text,
        flags=re.M,
    )
    text = re.sub(
        r'^# InputPerturbation.*\napply_heteroscedastic_alignment\(\)\n',
        "",
        text,
        flags=re.M,
    )
    text = re.sub(
        r'^\s*"apply_heteroscedastic_alignment",\n',
        "",
        text,
        flags=re.M,
    )
    write(rel, text)


def validate() -> None:
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
    offenders: list[str] = []
    for path in (ROOT / "src/bochan/models").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
    if offenders:
        raise RuntimeError("legacy model patch hooks remain:\n" + "\n".join(offenders))


def main() -> None:
    cleanup_hybrid()
    cleanup_projected_ordinal_init()
    cleanup_multiclass_robust_init()
    validate()


if __name__ == "__main__":
    main()
