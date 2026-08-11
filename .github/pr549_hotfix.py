from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_FILES = (
    ROOT / "src/bochan/models/regression/beta/_components.py",
    ROOT / "src/bochan/models/regression/gamma/_components.py",
    ROOT / "src/bochan/models/regression/count/poisson/_components.py",
    ROOT / "src/bochan/models/regression/count/negative_binomial/_components.py",
)

RSAMPLE_PATTERN = re.compile(
    r"(?P<indent>^[ \t]*)f_samples = self\.latent_posterior\.rsample\(\s*"
    r"sample_shape=sample_shape,\s*base_samples=base_samples,?\s*\)",
    re.MULTILINE,
)


def _replace_latent_base_sampling(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    match = RSAMPLE_PATTERN.search(source)
    if match is None:
        expected = "self.latent_posterior.rsample_from_base_samples("
        if expected not in source:
            raise RuntimeError(f"Could not locate base-sample delegation in {path}")
        return

    indent = match.group("indent")
    child = indent + "    "
    grandchild = child + "    "
    replacement = (
        f"{indent}f_samples = (\n"
        f"{child}self.latent_posterior.rsample(sample_shape=sample_shape)\n"
        f"{child}if base_samples is None\n"
        f"{child}else self.latent_posterior.rsample_from_base_samples(\n"
        f"{grandchild}sample_shape=sample_shape,\n"
        f"{grandchild}base_samples=base_samples,\n"
        f"{child})\n"
        f"{indent})"
    )
    updated, count = RSAMPLE_PATTERN.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"Expected exactly one rsample replacement in {path}; got {count}")
    path.write_text(updated, encoding="utf-8")


def _fix_ordinal_import_order() -> None:
    path = ROOT / "tests/test_ordinal_nehvi_repeated_backward.py"
    source = path.read_text(encoding="utf-8")
    old = (
        "from bochan.models.ordinal.likelihood import OrdinalLogitLikelihood\n"
        "from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel\n"
    )
    new = (
        "from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel\n"
        "from bochan.models.ordinal.likelihood import OrdinalLogitLikelihood\n"
    )
    if old in source:
        source = source.replace(old, new, 1)
    elif new not in source:
        raise RuntimeError("Could not locate ordinal model imports")
    path.write_text(source, encoding="utf-8")


def _write_base_sample_regression_test() -> None:
    path = ROOT / "tests/test_non_gaussian_posterior_base_samples.py"
    path.write_text(
        '''from __future__ import annotations

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.models.regression.beta import BetaGPModel
from bochan.models.regression.count.negative_binomial import NegativeBinomialGPModel
from bochan.models.regression.count.poisson import PoissonGPModel
from bochan.models.regression.gamma import GammaGPModel


@pytest.mark.parametrize(
    ("model_cls", "train_y"),
    [
        (
            BetaGPModel,
            torch.linspace(0.2, 0.8, 6, dtype=torch.double).unsqueeze(-1),
        ),
        (
            GammaGPModel,
            torch.linspace(1.1, 2.0, 6, dtype=torch.double).unsqueeze(-1),
        ),
        (
            PoissonGPModel,
            torch.arange(1, 7, dtype=torch.double).unsqueeze(-1),
        ),
        (
            NegativeBinomialGPModel,
            torch.arange(1, 7, dtype=torch.double).unsqueeze(-1),
        ),
    ],
    ids=["beta", "gamma", "poisson", "negative-binomial"],
)
def test_non_gaussian_posterior_uses_latent_base_sample_protocol(
    model_cls,
    train_y: torch.Tensor,
) -> None:
    train_x = torch.linspace(0.1, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    model = model_cls(train_x, train_y)
    candidate = torch.tensor([[0.3], [0.7]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=17)

    first = sampler(posterior)
    second = sampler(posterior)

    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()
    first.sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
''',
        encoding="utf-8",
    )


def main() -> None:
    for path in COMPONENT_FILES:
        _replace_latent_base_sampling(path)
    _fix_ordinal_import_order()
    _write_base_sample_regression_test()


if __name__ == "__main__":
    main()
