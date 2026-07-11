from __future__ import annotations

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


def test_known_latent_values_inside_unit_interval_are_transformed() -> None:
    latent = torch.tensor([0.1, 0.4, 0.8], dtype=torch.double)
    model = _Model(BernoulliLikelihood())

    actual = values_to_binary_probabilities(
        model,
        latent,
        eps=1e-12,
        values_are_probabilities=False,
    )
    expected = Normal(0.0, 1.0).cdf(latent)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(actual, latent)


def test_to_probability_forces_known_latent_values_through_likelihood() -> None:
    latent = torch.tensor([0.1, 0.4, 0.8], dtype=torch.double)
    model = _Model(BernoulliLikelihood())

    actual = to_probability(
        latent,
        apply_sigmoid_if_needed=True,
        eps=1e-12,
        name="known latent values",
        model=model,
        values_are_probs=False,
    )
    expected = Normal(0.0, 1.0).cdf(latent)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(actual, latent)


def test_old_to_probability_argument_is_likelihood_aware() -> None:
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
                violations.append(f"{path.relative_to(root)}:{node.lineno}: torch.sigmoid({node.args[0].id})")

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


def test_mc_sigmoid_is_only_a_support_alias() -> None:
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
