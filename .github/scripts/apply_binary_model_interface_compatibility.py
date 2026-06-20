from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, got {count}.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    ROOT / "src/bochan/models/classification/binary/base/multioutput.py",
    "        joint = MultitaskMultivariateNormal.from_independent_mvns(mvns)\n",
    '''        if len(mvns) == 1:
            # from_independent_mvns は 2 出力以上を要求する。
            # 1 出力選択時も (..., q, 1) を維持する。
            joint = MultitaskMultivariateNormal.from_repeated_mvn(
                mvns[0],
                num_tasks=1,
            )
        else:
            joint = MultitaskMultivariateNormal.from_independent_mvns(mvns)
''',
)

latent_path = ROOT / "src/bochan/models/classification/binary/base/_latent_models.py"
replace_once(
    latent_path,
    '''    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class _LatentMixedBinarySVGP''',
    '''    def transform_inputs(self, X: Tensor) -> Tensor:
        """外側wrapperで変換済みの内部入力をそのまま返す。"""
        return X

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class _LatentMixedBinarySVGP''',
)
text = latent_path.read_text(encoding="utf-8")
old = '''    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)
'''
new = '''    def transform_inputs(self, X: Tensor) -> Tensor:
        """外側wrapperで変換済みの内部入力をそのまま返す。"""
        return X

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)
'''
idx = text.rfind(old)
if idx < 0:
    raise RuntimeError("Mixed latent forward block not found.")
latent_path.write_text(text[:idx] + text[idx:].replace(old, new, 1), encoding="utf-8")

rrp_path = ROOT / "src/bochan/models/classification/binary/robust/relevance_pursuit.py"
replace_once(
    rrp_path,
    '''__all__ = [
    "SparseOutlierBernoulliLikelihood",
    "OutlierRelevancePursuitBinaryClassificationGPModel",
    "OutlierRelevancePursuitBinaryClassificationMixedGPModel",
]


''',
    '''__all__ = [
    "SparseOutlierBernoulliLikelihood",
    "OutlierRelevancePursuitBinaryClassificationGPModel",
    "OutlierRelevancePursuitBinaryClassificationMixedGPModel",
]


class _RRPVariationalELBO(VariationalELBO):
    """BoTorch relevance pursuit互換のVariationalELBO。"""

    def forward(
        self,
        variational_dist_f,
        target: Tensor,
        *transformed_inputs: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        _ = transformed_inputs
        return super().forward(variational_dist_f, target, **kwargs)


''',
)
replace_once(
    rrp_path,
    '''        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.fit_train_input.shape[-2],
            beta=beta,
        )
''',
    '''        return _RRPVariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.fit_train_input.shape[-2],
            beta=beta,
        )
''',
)

replace_once(
    ROOT / "src/bochan/models/components/robust.py",
    '''    @property
    def dense_delta(self) -> Tensor:
        dense = torch.zeros(self.dim, dtype=self.raw_delta.dtype, device=self.raw_delta.device)
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense
''',
    '''    @property
    def dense_delta(self) -> Tensor:
        if not self.is_sparse:
            return self.raw_delta
        dense = torch.zeros(self.dim, dtype=self.raw_delta.dtype, device=self.raw_delta.device)
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense
''',
)

(ROOT / "tests/test_binary_model_interface_compatibility.py").write_text(
    '''from __future__ import annotations

import torch
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator

from bochan.fit import fit_rrp_binary_classifier_mll
from bochan.models.classification.binary.base._latent_models import (
    _LatentBinarySVGP,
    _LatentMixedBinarySVGP,
)
from bochan.models.classification.binary.base.multioutput import (
    MultiOutputBinaryClassificationModel,
)
from bochan.models.classification.binary.robust import (
    OutlierRelevancePursuitBinaryClassificationGPModel,
    OutlierRelevancePursuitBinaryClassificationMixedGPModel,
)
from bochan.models.components.robust import SparseOutlierBernoulliLikelihood


class _DummyBinaryModel(Model):
    num_outputs = 1
    batch_shape = torch.Size()
    cat_dims: list[int] = []

    def __init__(self) -> None:
        super().__init__()
        self.raw_train_X = torch.zeros(3, 2, dtype=torch.double)
        self.train_Y = torch.zeros(3, 1, dtype=torch.double)
        self.train_targets = self.train_Y.squeeze(-1)

    def latent_posterior(self, X: torch.Tensor) -> GPyTorchPosterior:
        mean = X[..., 0]
        return GPyTorchPosterior(
            MultivariateNormal(
                mean,
                DiagLinearOperator(torch.full_like(mean, 0.2)),
            )
        )

    def posterior(self, X: torch.Tensor, **kwargs) -> GPyTorchPosterior:
        return self.latent_posterior(X)


def _toy_data(*, mixed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [0.4, 0.3], [0.6, 0.7], [0.8, 0.9], [1.0, 1.0]],
        dtype=torch.double,
    )
    if mixed:
        X[:, 1] = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.double)
    return X, (X[:, :1] > 0.5).to(dtype=torch.double)


def test_single_selected_output_keeps_multitask_shape() -> None:
    X = torch.randn(5, 2, dtype=torch.double)
    model = MultiOutputBinaryClassificationModel(_DummyBinaryModel(), _DummyBinaryModel())
    posterior = model.latent_posterior(X, output_indices=[0])
    assert posterior.mean.shape == torch.Size([5, 1])
    assert posterior.rsample(torch.Size([4])).shape == torch.Size([4, 5, 1])


def test_inner_latent_models_use_identity_transform_inputs() -> None:
    X, Y = _toy_data()
    model = _LatentBinarySVGP(X[:3], X, Y.squeeze(-1))
    assert model.transform_inputs(X=X) is X

    X_mixed, Y_mixed = _toy_data(mixed=True)
    mixed = _LatentMixedBinarySVGP(X_mixed[:3], [1], X_mixed, Y_mixed.squeeze(-1))
    assert mixed.transform_inputs(X=X_mixed) is X_mixed


def test_dense_rrp_parameter_preserves_gradient() -> None:
    likelihood = SparseOutlierBernoulliLikelihood(dim=4)
    likelihood.to_dense()
    likelihood.sparse_parameter.requires_grad_(True)
    likelihood.dense_delta.sum().backward()
    assert likelihood.sparse_parameter.grad is not None
    assert torch.allclose(likelihood.sparse_parameter.grad, torch.ones(4))


def _fit_one_step(model) -> None:
    fit_rrp_binary_classifier_mll(
        model.make_mll(),
        method="forward",
        sparsity_levels=[0, 1],
        reset_parameters=True,
        reset_dense_parameters=False,
        record_model_trace=False,
        optimizer_kwargs={"num_epochs": 1, "lr": 0.01},
    )
    assert len(model.likelihood.support) <= 1


def test_continuous_rrp_support_expansion_runs() -> None:
    X, Y = _toy_data()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    _fit_one_step(
        OutlierRelevancePursuitBinaryClassificationGPModel(
            train_X=X,
            train_Y=Y,
            input_transform=Normalize(d=2, bounds=bounds),
        )
    )


def test_mixed_rrp_support_expansion_runs() -> None:
    X, Y = _toy_data(mixed=True)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    _fit_one_step(
        OutlierRelevancePursuitBinaryClassificationMixedGPModel(
            train_X=X,
            train_Y=Y,
            cat_dims=[1],
            input_transform=Normalize(d=2, bounds=bounds, indices=[0]),
        )
    )
''',
    encoding="utf-8",
)
