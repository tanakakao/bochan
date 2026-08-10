from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding="utf-8")


def remove_export_token(rel: str, token: str) -> None:
    text = read(rel)
    text = re.sub(rf'^\s*["\']{re.escape(token)}["\'],?\s*$', "", text, flags=re.M)
    write(rel, text)


def cleanup_hybrid_exports() -> None:
    remove_export_token(
        "src/bochan/models/hybrid/prediction.py",
        "attach_prediction_methods",
    )
    remove_export_token(
        "src/bochan/models/hybrid/task_aware_sampling.py",
        "apply_task_aware_hybrid_posterior",
    )
    remove_export_token(
        "src/bochan/models/hybrid/class_probability_shapes.py",
        "apply_hybrid_class_probability_shapes",
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
    cleanup_hybrid_exports()
    cleanup_projected_ordinal_init()
    cleanup_multiclass_robust_init()
    validate()


if __name__ == "__main__":
    main()
