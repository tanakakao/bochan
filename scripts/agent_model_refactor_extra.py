from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding="utf-8")


def class_node(text: str, class_name: str) -> ast.ClassDef:
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise RuntimeError(f"class not found: {class_name}")


def method_span(text: str, class_name: str, method_name: str) -> tuple[int, int]:
    node = class_node(text, class_name)
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
            return item.lineno - 1, item.end_lineno
    raise RuntimeError(f"method not found: {class_name}.{method_name}")


def insert_before_class_end(rel: str, class_name: str, source: str) -> None:
    text = read(rel)
    node = class_node(text, class_name)
    lines = text.splitlines(keepends=True)
    lines.insert(node.end_lineno, "\n" + source.rstrip() + "\n")
    write(rel, "".join(lines))


def replace_method(rel: str, class_name: str, method_name: str, source: str) -> None:
    text = read(rel)
    start, end = method_span(text, class_name, method_name)
    lines = text.splitlines(keepends=True)
    lines[start:end] = [source.rstrip() + "\n"]
    write(rel, "".join(lines))


def formalize_probability_posterior_shapes() -> None:
    components = "src/bochan/models/components/multiclass.py"
    insert_before_class_end(
        components,
        "MulticlassProbsPosterior",
        '''    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Size:
        return torch.Size(sample_shape) + torch.Size(self.mean.shape[:-1])

    @property
    def batch_shape(self) -> torch.Size:
        mean = self.mean
        if mean.ndim <= 2:
            return torch.Size()
        return torch.Size(mean.shape[:-2])
''',
    )

    multioutput = "src/bochan/models/classification/multiclass/base/multioutput.py"
    insert_before_class_end(
        multioutput,
        "MultiOutputMulticlassProbsPosterior",
        '''    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Size:
        return torch.Size(sample_shape) + torch.Size(self.mean.shape[:-1])

    @property
    def batch_shape(self) -> torch.Size:
        mean = self.mean
        if mean.ndim <= 3:
            return torch.Size()
        return torch.Size(mean.shape[:-3])
''',
    )


def formalize_kronecker_q1_shape() -> None:
    rel = "src/bochan/models/classification/multiclass/base/kronecker_multitask.py"
    replace_method(
        rel,
        "KroneckerMultiTaskMulticlassProbsPosterior",
        "_probability_logits",
        '''    def _probability_logits(self, latent: Tensor) -> Tensor:
        if latent.shape[-3] != self.num_classes:
            raise RuntimeError(
                "Expected latent class batch at dimension -3 with size "
                f"{self.num_classes}, got shape={tuple(latent.shape)}."
            )
        logits = latent.movedim(-3, -1) / self.temperature
        if self.output_indices is not None:
            index = torch.as_tensor(
                self.output_indices,
                device=logits.device,
                dtype=torch.long,
            )
            logits = logits.index_select(dim=-2, index=index)

        if self.input_q != 1:
            return logits

        num_outputs = (
            len(self.output_indices)
            if self.output_indices is not None
            else int(logits.shape[-2])
        )
        with_q_suffix = tuple(self.input_batch_shape) + (
            1,
            num_outputs,
            self.num_classes,
        )
        if tuple(logits.shape[-len(with_q_suffix):]) == with_q_suffix:
            return logits

        without_q_suffix = tuple(self.input_batch_shape) + (
            num_outputs,
            self.num_classes,
        )
        if tuple(logits.shape[-len(without_q_suffix):]) == without_q_suffix:
            return logits.unsqueeze(-3)
        return logits
''',
    )

    text = read(rel)
    needle = "        self.temperature = float(temperature)\n"
    if needle not in text:
        raise RuntimeError("Kronecker posterior temperature assignment not found")
    text = text.replace(
        needle,
        needle
        + "        self.input_batch_shape = torch.Size()\n"
        + "        self.input_q: int | None = None\n",
        1,
    )
    write(rel, text)

    replace_method(
        rel,
        "KroneckerMultiTaskMulticlassClassificationGPModel",
        "posterior",
        '''    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> KroneckerMultiTaskMulticlassProbsPosterior:
        if observation_noise is not False:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support observation_noise."
            )
        X_tensor = torch.as_tensor(X[0] if isinstance(X, tuple) else X)
        if X_tensor.ndim == 1:
            input_batch_shape = torch.Size()
            input_q = 1
        elif X_tensor.ndim == 2:
            input_batch_shape = torch.Size()
            input_q = int(X_tensor.shape[-2])
        else:
            input_batch_shape = torch.Size(X_tensor.shape[:-2])
            input_q = int(X_tensor.shape[-2])

        indices = self._normalize_output_indices(output_indices)
        posterior = KroneckerMultiTaskMulticlassProbsPosterior(
            latent_posterior=self.latent_posterior(X),
            num_classes=self.num_classes,
            output_indices=indices,
            temperature=self.temperature,
        )
        posterior.input_batch_shape = input_batch_shape
        posterior.input_q = input_q
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior
''',
    )


def formalize_sampler_registration() -> None:
    write(
        "src/bochan/models/classification/multiclass/base/posteriors.py",
        '''from __future__ import annotations

import torch

from bochan.models.classification.multiclass.base.multioutput import (
    MultiOutputMulticlassProbsPosterior,
)
from bochan.models.components.multiclass import MulticlassProbsPosterior

try:
    from botorch.sampling.get_sampler import GetSampler
    from botorch.sampling.normal import SobolQMCNormalSampler
except Exception:  # pragma: no cover - BoTorch version guard
    GetSampler = None  # type: ignore[assignment]
    SobolQMCNormalSampler = None  # type: ignore[assignment]


def _make_sobol_sampler(
    sample_shape: torch.Size,
    seed: int | None = None,
):
    if SobolQMCNormalSampler is None:
        raise NotImplementedError("SobolQMCNormalSampler is unavailable.")
    try:
        return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)
    except TypeError:
        return SobolQMCNormalSampler(sample_shape=sample_shape)


if GetSampler is not None:

    @GetSampler.register(MultiOutputMulticlassProbsPosterior)
    def _get_multioutput_multiclass_sampler(
        posterior: MultiOutputMulticlassProbsPosterior,
        sample_shape: torch.Size,
        seed: int | None = None,
    ):
        return _make_sobol_sampler(sample_shape=sample_shape, seed=seed)

    @GetSampler.register(MulticlassProbsPosterior)
    def _get_single_multiclass_sampler(
        posterior: MulticlassProbsPosterior,
        sample_shape: torch.Size,
        seed: int | None = None,
    ):
        return _make_sobol_sampler(sample_shape=sample_shape, seed=seed)


__all__: list[str] = []
''',
    )

    rel = "src/bochan/models/classification/multiclass/base/__init__.py"
    text = read(rel)
    text = text.replace(
        "from .posteriors import apply_multiclass_posteriors\n\napply_multiclass_posteriors()\n",
        "from . import posteriors as _posterior_sampler_registration\n",
    )
    text = text.replace('    "apply_multiclass_posteriors",\n', "")
    write(rel, text)


def align_gaussian_constructor_contract() -> None:
    rel = "src/bochan/models/regression/gaussian/deep/deepgp.py"
    text = read(rel)
    text = text.replace(
        "        train_Yvar: Tensor | None = None,\n        likelihood=None,\n",
        "        train_Yvar: Tensor | None = None,\n        *,\n        likelihood=None,\n",
    )
    write(rel, text)


def validate_no_model_runtime_patch() -> None:
    forbidden = (
        "apply_multiclass_posteriors",
        "_bochan_original_posterior_before_q1_support",
        "_bochan_original_probability_logits_before_q1_support",
        "method-assign",
    )
    offenders: list[str] = []
    for path in (ROOT / "src/bochan/models").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
    if offenders:
        raise RuntimeError("runtime model patch definitions remain:\n" + "\n".join(offenders))


def main() -> None:
    formalize_probability_posterior_shapes()
    formalize_kronecker_q1_shape()
    formalize_sampler_registration()
    align_gaussian_constructor_contract()
    validate_no_model_runtime_patch()


if __name__ == "__main__":
    main()
