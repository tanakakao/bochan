import torch
from gpytorch.mlls import VariationalELBO

from bochan.models.classification.binary.base import MultiTaskBinaryClassificationGPModel
from bochan.models.ordinal.base import MultiTaskOrdinalGPModel


def _make_multitask_X(dtype=torch.double):
    x0 = torch.linspace(0.0, 1.0, 4, dtype=dtype).unsqueeze(-1)
    x1 = torch.linspace(0.2, 1.2, 4, dtype=dtype).unsqueeze(-1)
    t0 = torch.zeros(4, 1, dtype=dtype)
    t1 = torch.ones(4, 1, dtype=dtype)
    return torch.cat([torch.cat([x0, t0], dim=-1), torch.cat([x1, t1], dim=-1)], dim=0)


def test_multitask_binary_classification_posterior_shapes():
    train_X = _make_multitask_X()
    train_Y = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1], dtype=torch.double)

    model = MultiTaskBinaryClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_tasks=2,
        task_feature=-1,
        num_inducing_points=4,
    )

    post = model.posterior(train_X[:3])
    latent = model.latent_posterior(train_X[:3])

    assert post.mean.shape == torch.Size([3, 1])
    assert post.variance.shape == torch.Size([3, 1])
    assert latent.mean.shape == torch.Size([3, 1])

    mll = VariationalELBO(model.likelihood, model.model, num_data=train_X.shape[0])
    out = model.model(model.train_inputs[0])
    loss = -mll(out, model.train_targets)
    assert torch.isfinite(loss).all()


def test_multitask_ordinal_shapes_and_class_probs():
    train_X = _make_multitask_X()
    train_Y = torch.tensor([0, 1, 1, 2, 0, 0, 1, 2], dtype=torch.long)

    model = MultiTaskOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        num_tasks=2,
        task_feature=-1,
        inducing_points_num=4,
    )

    post = model.posterior(train_X[:3])
    probs = model.class_probs(train_X[:3])

    assert post.mean.shape == torch.Size([3, 1])
    assert probs.shape == torch.Size([3, 3])
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3, dtype=train_X.dtype), atol=1e-5)

    mll = VariationalELBO(model.likelihood, model.model, num_data=train_X.shape[0])
    out = model.model(model.train_inputs[0])
    loss = -mll(out, model.train_targets)
    assert torch.isfinite(loss).all()
