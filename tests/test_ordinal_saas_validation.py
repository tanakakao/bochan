from __future__ import annotations

import pytest
import torch

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.ordinal.high_dim import SaasOrdinalGPModel, SaasOrdinalMixedGPModel


DTYPE = torch.double
DEVICE = torch.device("cpu")


def _train_data(n: int = 9, d: int = 3, *, cat: bool = False):
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    cols = [x]
    for j in range(1, d):
        cols.append((x + 0.17 * j).remainder(1.0))
    train_x = torch.cat(cols, dim=-1)
    if cat:
        train_x[:, -1] = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE).repeat((n + 2) // 3)[:n]
    train_y = torch.tensor([0, 1, 2] * ((n + 2) // 3), dtype=torch.long, device=DEVICE)[:n]
    return train_x, train_y


def test_saas_ordinal_cutpoint_kwargs_are_respected() -> None:
    train_x, train_y = _train_data()

    model = SaasOrdinalGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        fix_first_cutpoint=False,
        init_gap=0.5,
        num_inducing_points=4,
    )

    assert model.ordinal_likelihood.fix_first_cutpoint is False
    assert model.ordinal_likelihood.cutpoints.shape == torch.Size([2])
    assert torch.isclose(model.ordinal_likelihood.cutpoints.mean(), torch.zeros((), dtype=DTYPE))


def test_saas_ordinal_custom_likelihood_is_used() -> None:
    train_x, train_y = _train_data()
    likelihood = OrdinalLogitLikelihood(num_classes=3, fix_first_cutpoint=False, init_gap=0.5).to(train_x)

    model = SaasOrdinalGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        ordinal_likelihood=likelihood,
        num_inducing_points=4,
    )

    assert model.likelihood is likelihood
    assert model.ordinal_likelihood is likelihood
    assert model.ordinal_likelihood.fix_first_cutpoint is False


def test_saas_ordinal_mixed_custom_likelihood_is_used() -> None:
    train_x, train_y = _train_data(cat=True)
    likelihood = OrdinalLogitLikelihood(num_classes=3, fix_first_cutpoint=False, init_gap=0.5).to(train_x)

    model = SaasOrdinalMixedGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        cat_dims=[train_x.shape[-1] - 1],
        ordinal_likelihood=likelihood,
        num_inducing_points=4,
    )

    assert model.likelihood is likelihood
    assert model.ordinal_likelihood.fix_first_cutpoint is False
    assert model.train_inputs[0].shape == train_x.shape
    assert model.model.train_inputs[0].shape[-1] == model.encoded_dim


@pytest.mark.parametrize(
    "bad_y",
    [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 2, 3], dtype=torch.long),
        torch.tensor([0.0, 1.5, 2.0], dtype=DTYPE),
    ],
)
def test_saas_ordinal_rejects_invalid_inferred_labels(bad_y: torch.Tensor) -> None:
    train_x, _ = _train_data(n=bad_y.shape[0])

    with pytest.raises(ValueError):
        SaasOrdinalGPModel(
            train_X=train_x,
            train_Y=bad_y,
            num_inducing_points=3,
        )


def test_saas_ordinal_explicit_num_classes_allows_unobserved_classes() -> None:
    train_x, _ = _train_data(n=6)
    train_y = torch.tensor([0, 0, 2, 2, 0, 2], dtype=torch.long, device=DEVICE)

    model = SaasOrdinalGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        num_inducing_points=3,
    )

    assert model.num_classes == 3
    assert torch.equal(model.train_targets, train_y)


def test_saas_ordinal_rejects_labels_outside_explicit_num_classes() -> None:
    train_x, _ = _train_data(n=4)
    train_y = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=DEVICE)

    with pytest.raises(ValueError, match="num_classes"):
        SaasOrdinalGPModel(
            train_X=train_x,
            train_Y=train_y,
            num_classes=3,
            num_inducing_points=3,
        )
