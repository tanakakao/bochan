from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:120]!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Multitask fixed-noise likelihood: allow NaN only as an explicit missing-cell
# marker while keeping the actual covariance noise finite.
# ---------------------------------------------------------------------------
Path(
    "src/bochan/models/regression/gaussian/deep/multitask_fixed_noise.py"
).write_text(
    '''"""Fixed known observation noise for correlated multitask GPs."""

from __future__ import annotations

import warnings

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.utils.warnings import GPInputWarning
from linear_operator.operators import LinearOperator, ZeroLinearOperator
from torch import Tensor


class MultitaskFixedNoiseGaussianLikelihood(FixedNoiseGaussianLikelihood):
    """Fixed per-observation, per-task variance for wide multitask targets.

    The public noise contract is ``[..., n, m]``. The likelihood converts this
    wide tensor to the covariance event order at the likelihood boundary, using
    ``MultitaskMultivariateNormal._interleaved`` rather than assuming one layout.

    ``allow_missing=True`` is reserved for partially observed correlated
    multitask training. In that mode, ``NaN`` marks an unobserved target cell.
    The underlying fixed-noise covariance stores a finite placeholder at those
    positions; the corresponding target entries are removed by the exact-GP
    missing-observation mask, so the placeholder is never used as an observed
    variance and is not an imputation.
    """

    def __init__(
        self,
        noise: Tensor,
        *,
        num_tasks: int | None = None,
        allow_missing: bool = False,
    ) -> None:
        resolved_num_tasks = self._validate_wide_noise(
            noise,
            num_tasks=num_tasks,
            argument_name="noise",
            allow_missing=allow_missing,
        )
        missing_mask = torch.isnan(noise) if allow_missing else torch.zeros_like(
            noise, dtype=torch.bool
        )
        stored_noise = torch.where(missing_mask, torch.ones_like(noise), noise)
        super().__init__(
            noise=self._flatten_event(stored_noise, interleaved=True),
            learn_additional_noise=False,
        )
        self.num_tasks = resolved_num_tasks
        self.allow_missing = bool(allow_missing)
        self.register_buffer("_missing_noise_mask", missing_mask.detach().clone())

    @staticmethod
    def _validate_wide_noise(
        noise: Tensor,
        *,
        num_tasks: int | None,
        argument_name: str,
        allow_missing: bool = False,
    ) -> int:
        if not torch.is_tensor(noise):
            raise TypeError(f"{argument_name} must be a Tensor.")
        if noise.ndim < 2:
            raise ValueError(
                f"{argument_name} must have shape [..., n, m] for multitask noise."
            )
        inferred_num_tasks = int(noise.shape[-1])
        if inferred_num_tasks < 2:
            raise ValueError(
                f"{argument_name} must contain at least two task columns."
            )
        if num_tasks is not None and inferred_num_tasks != int(num_tasks):
            raise ValueError(
                f"{argument_name} task dimension does not match num_tasks: "
                f"{inferred_num_tasks} != {int(num_tasks)}."
            )

        if allow_missing:
            if torch.isinf(noise).any():
                raise ValueError(
                    f"{argument_name} must contain finite variances or NaN missing markers."
                )
            observed = ~torch.isnan(noise)
        else:
            if not torch.isfinite(noise).all():
                raise ValueError(f"{argument_name} must contain only finite variances.")
            observed = torch.ones_like(noise, dtype=torch.bool)

        if observed.any() and (noise[observed] <= 0).any():
            raise ValueError(
                f"{argument_name} must contain strictly positive variances."
            )
        return inferred_num_tasks

    @staticmethod
    def _flatten_event(noise: Tensor, *, interleaved: bool) -> Tensor:
        ordered = noise if interleaved else noise.transpose(-1, -2)
        return ordered.reshape(*ordered.shape[:-2], -1)

    @property
    def _stored_task_noise(self) -> Tensor:
        flat_noise = self.noise_covar.noise
        if flat_noise.shape[-1] % self.num_tasks != 0:
            raise RuntimeError(
                "Stored fixed noise is incompatible with the configured task count."
            )
        num_data = flat_noise.shape[-1] // self.num_tasks
        return flat_noise.reshape(
            *flat_noise.shape[:-1],
            num_data,
            self.num_tasks,
        )

    @property
    def missing_noise_mask(self) -> Tensor:
        """Return the canonical wide mask of unobserved variance cells."""

        return self._missing_noise_mask

    @property
    def task_noise(self) -> Tensor:
        """Return fixed training variance in natural ``[..., n, m]`` shape."""

        stored = self._stored_task_noise
        if not bool(self._missing_noise_mask.any()):
            return stored
        return stored.masked_fill(self._missing_noise_mask, float("nan"))

    def _shaped_noise_covar(
        self,
        base_shape: torch.Size,
        *params,
        **kwargs,
    ) -> Tensor | LinearOperator:
        del params
        if len(base_shape) < 2 or int(base_shape[-1]) != self.num_tasks:
            raise ValueError(
                "Multitask fixed noise expects an event shape [..., n, m] with "
                f"m={self.num_tasks}; got {tuple(base_shape)}."
            )

        interleaved = bool(kwargs.pop("_interleaved", True))
        explicit_noise = kwargs.pop("noise", None)
        flat_shape = torch.Size(
            (*base_shape[:-2], int(base_shape[-2]) * self.num_tasks)
        )
        if explicit_noise is not None:
            # Test-time observation noise represents actual requested observations,
            # so it remains strict even when the stored training table is partial.
            self._validate_wide_noise(
                explicit_noise,
                num_tasks=self.num_tasks,
                argument_name="noise",
                allow_missing=False,
            )
            if tuple(explicit_noise.shape[-2:]) != tuple(base_shape[-2:]):
                raise ValueError(
                    "Explicit multitask observation noise must match the requested "
                    f"[n, m] event shape: {tuple(explicit_noise.shape[-2:])} != "
                    f"{tuple(base_shape[-2:])}."
                )
            return self.noise_covar(
                shape=flat_shape,
                noise=self._flatten_event(
                    explicit_noise,
                    interleaved=interleaved,
                ),
                **kwargs,
            )

        stored_noise = self._stored_task_noise
        if int(stored_noise.shape[-2]) == int(base_shape[-2]):
            return self.noise_covar(
                shape=flat_shape,
                noise=self._flatten_event(
                    stored_noise,
                    interleaved=interleaved,
                ),
                **kwargs,
            )

        result = self.noise_covar(shape=flat_shape, **kwargs)
        if isinstance(result, ZeroLinearOperator):
            warnings.warn(
                "The requested multitask event size does not match the stored fixed "
                "training noise and no explicit observation noise was supplied. "
                "This is treated as a no-op.",
                GPInputWarning,
                stacklevel=2,
            )
        return result

    def marginal(
        self,
        function_dist: MultitaskMultivariateNormal,
        *params,
        **kwargs,
    ) -> MultitaskMultivariateNormal:
        """Add fixed noise while preserving the multitask event ordering."""

        if not isinstance(function_dist, MultitaskMultivariateNormal):
            raise TypeError(
                "MultitaskFixedNoiseGaussianLikelihood requires a "
                "MultitaskMultivariateNormal."
            )
        mean = function_dist.mean
        covariance = function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(
            mean.shape,
            *params,
            _interleaved=bool(function_dist._interleaved),
            **kwargs,
        )
        return MultitaskMultivariateNormal(
            mean,
            covariance + noise_covar,
            interleaved=bool(function_dist._interleaved),
        )

    def get_fantasy_likelihood(self, **kwargs):
        """Append wide fixed noise for fantasy observations along the data axis."""

        if "noise" not in kwargs:
            raise RuntimeError(
                "MultitaskFixedNoiseGaussianLikelihood.fantasize requires a "
                "wide `noise` kwarg with shape [..., q, m]."
            )
        new_noise = kwargs["noise"]
        self._validate_wide_noise(
            new_noise,
            num_tasks=self.num_tasks,
            argument_name="noise",
            allow_missing=False,
        )
        old_noise = self.task_noise
        batch_shape = torch.broadcast_shapes(
            old_noise.shape[:-2],
            new_noise.shape[:-2],
        )
        old_noise = old_noise.expand(
            *batch_shape,
            old_noise.shape[-2],
            self.num_tasks,
        )
        new_noise = new_noise.expand(
            *batch_shape,
            new_noise.shape[-2],
            self.num_tasks,
        )
        fantasy = type(self)(
            torch.cat([old_noise, new_noise], dim=-2),
            num_tasks=self.num_tasks,
            allow_missing=self.allow_missing,
        )
        fantasy.train(self.training)
        return fantasy


__all__ = ["MultitaskFixedNoiseGaussianLikelihood"]
''',
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Exact multitask prediction: GPyTorch masks NaNs in the MLL and predictive
# mean, but its default predictive covariance still conditions on all event
# slots. Use the observed event subset for covariance as well.
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/models/components/layers/kernel_layers.py",
    "from gpytorch.models import ExactGP\nfrom torch import Tensor\n",
    "from gpytorch import settings\nfrom gpytorch.models import ExactGP\n"
    "from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy\n"
    "from linear_operator import to_linear_operator\n"
    "from linear_operator.operators import MaskedLinearOperator, MatmulLinearOperator\n"
    "from torch import Tensor\n",
)

replace_once(
    "src/bochan/models/components/layers/kernel_layers.py",
    "class DeepKernel(ExactGP):\n",
    '''class _PartialObservationPredictionStrategy(DefaultPredictionStrategy):
    """Exact prediction strategy that conditions only on observed task cells."""

    def _observed_event_mask(self) -> Tensor | None:
        if not bool(torch.isnan(self.train_labels).any()):
            return None
        return settings.observation_nan_policy._get_observed(
            self.train_labels,
            torch.Size((self.train_labels.shape[-1],)),
        ).reshape(-1)

    def exact_prediction(self, test_mean, test_test_covar, test_train_covar):
        if self._observed_event_mask() is None:
            return super().exact_prediction(
                test_mean,
                test_test_covar,
                test_train_covar,
            )
        with settings.observation_nan_policy("mask"):
            return super().exact_prediction(
                test_mean,
                test_test_covar,
                test_train_covar,
            )

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        observed = self._observed_event_mask()
        if observed is None:
            return super().exact_predictive_covar(
                test_test_covar,
                test_train_covar,
            )

        train_covar = MaskedLinearOperator(
            to_linear_operator(self.lik_train_train_covar),
            observed,
            observed,
        )
        test_rows = torch.ones(
            test_train_covar.shape[-2],
            dtype=torch.bool,
            device=observed.device,
        )
        test_train_observed = MaskedLinearOperator(
            to_linear_operator(test_train_covar),
            test_rows,
            observed,
        )
        correction_rhs = train_covar.solve(
            test_train_observed.transpose(-1, -2)
        )
        return to_linear_operator(test_test_covar) + MatmulLinearOperator(
            test_train_observed,
            correction_rhs.mul(-1),
        )


class _PartialObservationMultitaskKernel(MultitaskKernel):
    """Multitask kernel selecting the partial-observation prediction strategy."""

    def prediction_strategy(
        self,
        train_inputs,
        train_prior_dist,
        train_labels,
        likelihood,
    ):
        return _PartialObservationPredictionStrategy(
            train_inputs,
            train_prior_dist,
            train_labels,
            likelihood,
        )


class DeepKernel(ExactGP):
''',
)

kernel_path = Path("src/bochan/models/components/layers/kernel_layers.py")
kernel_text = kernel_path.read_text(encoding="utf-8")
count = kernel_text.count("self.covar_module = MultitaskKernel(")
if count != 2:
    raise SystemExit(f"expected 2 multitask kernel construction sites, found {count}")
kernel_path.write_text(
    kernel_text.replace(
        "self.covar_module = MultitaskKernel(",
        "self.covar_module = _PartialObservationMultitaskKernel(",
    ),
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# DeepKernel wrapper: validate the partial wide contract and make Exact MLL
# masking automatic for direct callers and the standard fit helper.
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/models/regression/gaussian/deep/deepkernel.py",
    "from gpytorch.settings import fast_pred_var\n",
    "from gpytorch.settings import fast_pred_var, observation_nan_policy\n",
)
replace_once(
    "src/bochan/models/regression/gaussian/deep/deepkernel.py",
    "class _BaseDeepKernelGPModel(DeepGP, GPyTorchModel):\n",
    '''class _ObservationMaskingExactMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """Exact MLL that automatically excludes NaN observation cells."""

    def forward(self, function_dist, target, *params, **kwargs):
        if torch.is_tensor(target) and bool(torch.isnan(target).any()):
            with observation_nan_policy("mask"):
                return super().forward(function_dist, target, *params, **kwargs)
        return super().forward(function_dist, target, *params, **kwargs)


class _BaseDeepKernelGPModel(DeepGP, GPyTorchModel):
''',
)
replace_once(
    "src/bochan/models/regression/gaussian/deep/deepkernel.py",
    "        self._model_device: torch.device | None = None\n",
    "        self._model_device: torch.device | None = None\n"
    "        self._uses_observation_nan_mask = False\n",
)
replace_once(
    "src/bochan/models/regression/gaussian/deep/deepkernel.py",
    '''    def _setup_common(
        self,
        train_X: Tensor,
''',
    '''    @classmethod
    def _validate_observation_targets(
        cls,
        train_Y: Tensor,
        train_Yvar: Tensor | None,
    ) -> bool:
        """Validate correlated partial targets and matching known variance."""

        if torch.isinf(train_Y).any():
            raise ValueError("train_Y must not contain infinite values.")
        missing = torch.isnan(train_Y)
        uses_mask = bool(missing.any())
        if uses_mask:
            if cls._get_num_outputs(train_Y) < 2:
                raise ValueError(
                    "NaN targets in a DeepKernel model require a correlated "
                    "multi-output target matrix."
                )
            observed_per_output = (~missing).sum(dim=-2)
            if bool((observed_per_output == 0).any()):
                empty = (observed_per_output == 0).nonzero(as_tuple=False).reshape(-1)
                raise ValueError(
                    "Each correlated output requires at least one observed target; "
                    f"empty output indices: {empty.tolist()}."
                )

        if train_Yvar is not None:
            if torch.isinf(train_Yvar).any():
                raise ValueError("train_Yvar must not contain infinite values.")
            variance_missing = torch.isnan(train_Yvar)
            if not torch.equal(missing, variance_missing):
                raise ValueError(
                    "train_Yvar NaN positions must exactly match missing train_Y cells "
                    "for correlated partial observations."
                )
            observed_variance = train_Yvar[~missing]
            if observed_variance.numel() and bool((observed_variance <= 0).any()):
                raise ValueError(
                    "Observed train_Yvar cells must contain strictly positive variances."
                )
        return uses_mask

    def _setup_common(
        self,
        train_X: Tensor,
''',
)
replace_once(
    "src/bochan/models/regression/gaussian/deep/deepkernel.py",
    '''        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        train_Y, train_Yvar = self._resolve_outcome_transform(
''',
    '''        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        self._uses_observation_nan_mask = self._validate_observation_targets(
            train_Y,
            train_Yvar,
        )

        train_Y, train_Yvar = self._resolve_outcome_transform(
''',
)
replace_once(
    "src/bochan/models/regression/gaussian/deep/deepkernel.py",
    "        return ExactMarginalLogLikelihood(self.likelihood, self.deepkernel)\n",
    "        return _ObservationMaskingExactMarginalLogLikelihood(\n"
    "            self.likelihood, self.deepkernel\n"
    "        )\n",
)


# ---------------------------------------------------------------------------
# Configurable Gaussian DeepKernel: material correlated models inherit this
# path, so a single opt-in feeds missing Yvar to the fixed-noise adapter.
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/models/regression/gaussian/deep/deepkernel_configurable.py",
    '''def _fixed_noise_likelihood(
    train_Yvar: Tensor | None,
    *,
    num_outputs: int,
) -> FixedNoiseGaussianLikelihood | None:
''',
    '''def _fixed_noise_likelihood(
    train_Yvar: Tensor | None,
    *,
    num_outputs: int,
    allow_missing: bool = False,
) -> FixedNoiseGaussianLikelihood | None:
''',
)
replace_once(
    "src/bochan/models/regression/gaussian/deep/deepkernel_configurable.py",
    '''    return MultitaskFixedNoiseGaussianLikelihood(
        noise=train_Yvar,
        num_tasks=num_outputs,
    )
''',
    '''    return MultitaskFixedNoiseGaussianLikelihood(
        noise=train_Yvar,
        num_tasks=num_outputs,
        allow_missing=allow_missing,
    )
''',
)
configurable_path = Path(
    "src/bochan/models/regression/gaussian/deep/deepkernel_configurable.py"
)
configurable_text = configurable_path.read_text(encoding="utf-8")
old_call = '''            likelihood = _fixed_noise_likelihood(
                self.train_Yvar,
                num_outputs=self._num_outputs,
            )
'''
new_call = '''            likelihood = _fixed_noise_likelihood(
                self.train_Yvar,
                num_outputs=self._num_outputs,
                allow_missing=self._uses_observation_nan_mask,
            )
'''
call_count = configurable_text.count(old_call)
if call_count != 2:
    raise SystemExit(f"expected 2 fixed-noise call sites, found {call_count}")
configurable_text = configurable_text.replace(old_call, new_call)
configurable_text = configurable_text.replace(
    'class DeepKernelGaussianGPModel(_BaseDeepKernelGPModel):\n',
    'class DeepKernelGaussianGPModel(_BaseDeepKernelGPModel):\n'
    '    _supports_partial_multitask_targets = True\n\n',
    1,
)
configurable_text = configurable_text.replace(
    'class DeepKernelGaussianMixedGPModel(_BaseDeepKernelGPModel):\n',
    'class DeepKernelGaussianMixedGPModel(_BaseDeepKernelGPModel):\n'
    '    _supports_partial_multitask_targets = True\n\n',
    1,
)
configurable_path.write_text(configurable_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Observation-aware builder: correlated material classes advertise wide
# partial-target support through the shared class marker.
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/api/observation/service.py",
    '''    name = _normalize_model_name(config.model_type)
    return name == "multitask" or name.endswith("widemultitask")
''',
    '''    name = _normalize_model_name(config.model_type)
    if name == "multitask" or name.endswith("widemultitask"):
        return True
    model_cls = config.model_cls
    return bool(
        model_cls is not None
        and getattr(model_cls, "_supports_partial_multitask_targets", False)
    )
''',
)


# ---------------------------------------------------------------------------
# Focused regression coverage.
# ---------------------------------------------------------------------------
Path("tests/test_material_train_yvar_phase6.py").write_text(
    '''from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from bochan.api import ModelConfig
from bochan.api.observation.service import build_objective_bundle
from bochan.fit import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep.deepkernel_configurable import (
    DeepKernelGaussianGPModel,
)
from bochan.models.regression.gaussian.deep.multitask_fixed_noise import (
    MultitaskFixedNoiseGaussianLikelihood,
)


def _partial_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.2], [0.6, 0.8], [1.0, 1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [[0.0, 1.0], [0.4, float("nan")], [float("nan"), 1.8], [1.1, 2.2]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.01, 0.02], [0.03, float("nan")], [float("nan"), 0.06], [0.07, 0.08]],
        dtype=torch.double,
    )
    return X, Y, Yvar


def test_multitask_fixed_noise_missing_mode_keeps_public_nan_contract() -> None:
    _, _, Yvar = _partial_data()
    with pytest.raises(ValueError, match="finite variances"):
        MultitaskFixedNoiseGaussianLikelihood(Yvar)

    likelihood = MultitaskFixedNoiseGaussianLikelihood(
        Yvar,
        allow_missing=True,
    )
    torch.testing.assert_close(
        likelihood.task_noise,
        Yvar,
        equal_nan=True,
    )
    torch.testing.assert_close(
        likelihood.missing_noise_mask,
        torch.isnan(Yvar),
    )
    assert torch.isfinite(likelihood.noise_covar.noise).all()


def test_partial_correlated_deepkernel_known_noise_fits_and_predicts() -> None:
    X, Y, Yvar = _partial_data()
    model = DeepKernelGaussianGPModel(
        X,
        Y,
        Yvar,
        feature_extractor=nn.Identity(),
        latent_dim=2,
        input_transform=None,
        outcome_transform=None,
    ).double()

    assert model._uses_observation_nan_mask is True
    assert isinstance(model.likelihood, MultitaskFixedNoiseGaussianLikelihood)
    torch.testing.assert_close(model.likelihood.task_noise, Yvar, equal_nan=True)

    mll = model.make_mll()
    model.train()
    output = model.deepkernel(model.transform_inputs(X))
    value = mll(output, Y)
    assert torch.isfinite(value)
    (-value).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    fit_deepkernel_mll(model.make_mll(), num_epochs=1, lr=1e-3)
    posterior = model.posterior(X[:2], observation_noise=False)
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert bool((posterior.variance >= 0).all())


def test_default_standardize_preserves_partial_variance_alignment() -> None:
    X, Y, Yvar = _partial_data()
    model = DeepKernelGaussianGPModel(
        X,
        Y,
        Yvar,
        feature_extractor=nn.Identity(),
        latent_dim=2,
        input_transform=None,
    ).double()

    assert torch.equal(torch.isnan(model.train_Y), torch.isnan(Y))
    assert model.train_Yvar is not None
    assert torch.equal(torch.isnan(model.train_Yvar), torch.isnan(Yvar))
    observed = ~torch.isnan(model.train_Y)
    assert torch.isfinite(model.train_Y[observed]).all()
    assert torch.isfinite(model.train_Yvar[observed]).all()
    assert bool((model.train_Yvar[observed] > 0).all())


def test_partial_correlated_contract_rejects_misaligned_or_empty_outputs() -> None:
    X, Y, Yvar = _partial_data()
    bad_yvar = Yvar.clone()
    bad_yvar[1, 1] = 0.04
    with pytest.raises(ValueError, match="NaN positions must exactly match"):
        DeepKernelGaussianGPModel(
            X,
            Y,
            bad_yvar,
            feature_extractor=nn.Identity(),
            latent_dim=2,
            input_transform=None,
            outcome_transform=None,
        )

    empty_output = Y.clone()
    empty_output[:, 1] = float("nan")
    empty_yvar = Yvar.clone()
    empty_yvar[:, 1] = float("nan")
    with pytest.raises(ValueError, match="at least one observed target"):
        DeepKernelGaussianGPModel(
            X,
            empty_output,
            empty_yvar,
            feature_extractor=nn.Identity(),
            latent_dim=2,
            input_transform=None,
            outcome_transform=None,
        )


def test_observation_builder_routes_marker_model_as_correlated_wide() -> None:
    X, Y, Yvar = _partial_data()
    config = ModelConfig(
        task_type="multi_objective",
        model_type="phase6_correlated_test",
        model_cls=DeepKernelGaussianGPModel,
        outcome_transform=False,
        model_kwargs={
            "feature_extractor": nn.Identity(),
            "latent_dim": 2,
            "input_transform": None,
            "outcome_transform": None,
        },
    )
    bundle = build_objective_bundle(
        train_X=X,
        train_Y=Y,
        train_Yvar=Yvar,
        config=config,
    )

    assert isinstance(bundle.model, DeepKernelGaussianGPModel)
    assert bundle.model.num_outputs == 2
    assert bundle.model._uses_observation_nan_mask is True
    torch.testing.assert_close(bundle.model.train_Yvar, Yvar, equal_nan=True)


@pytest.mark.parametrize(
    "filename",
    [
        "mace_multitask.py",
        "chgnet_multitask.py",
        "m3gnet_multitask.py",
        "alignn_multitask.py",
        "crabnet_multitask.py",
    ],
)
def test_material_correlated_families_inherit_shared_partial_contract(
    filename: str,
) -> None:
    source = (
        Path("src/bochan/models/regression/gaussian/deep") / filename
    ).read_text(encoding="utf-8")
    assert "DeepKernelGaussianGPModel" in source
    assert "train_Yvar=train_Yvar" in source
    assert DeepKernelGaussianGPModel._supports_partial_multitask_targets is True
''',
    encoding="utf-8",
)


Path("docs/material_train_yvar_phase6.md").write_text(
    '''# Material `train_Yvar` Phase 6

Phase 6 completes the core known-observation-variance path for correlated
material multitask GP/DKL models with partially observed target matrices.

## Supported contract

For a correlated wide target table, `train_Y` and `train_Yvar` both use shape
`[n, m]`.

- An observed target cell requires a finite, strictly positive variance.
- An unobserved target cell is represented by `NaN` in both `train_Y` and
  `train_Yvar`.
- The `NaN` pattern of `train_Yvar` must exactly match the missing pattern of
  `train_Y`.
- Every output task must have at least one observed target.
- Infinite targets or variances are rejected.

This contract is shared by the correlated MACE, CHGNet, M3GNet, ALIGNN, and
CrabNet multitask GP/DKL families, including their mixed-input variants.

## Exact-GP missing observation semantics

Training uses GPyTorch's exact marginal-likelihood observation mask so missing
target cells are removed from the likelihood event. Prediction uses BOCHAN's
correlated multitask prediction strategy, which applies the same observed-event
subset to both the predictive mean and predictive covariance.

The fixed-noise likelihood stores a finite internal placeholder at missing
variance positions because a covariance diagonal must be finite. That value is
never treated as an observation: the matching target event is removed before
conditioning. The public `task_noise` view retains `NaN` at missing cells.

## Outcome transforms

`Standardize` remains supported. Its NaN-aware statistics are computed per
output from observed targets, and known variances are scaled by the same output
variance. Missing target/variance cells remain aligned after transformation.

## High-level observation workflow

The observation-aware model builder recognizes correlated model classes that
advertise the shared partial-multitask contract. It therefore keeps the wide
matrix intact instead of splitting the outputs into independent models.

Kronecker and the current multi-fidelity correlated models still require a
complete rectangular target table and are not changed by Phase 6.

## Scope

Phase 6 completes the core Phase 1-6 `train_Yvar` series. Failure-classifier
cross-validation and partial-observation feature importance are separate
model-evaluation/diagnostics work rather than missing known-noise plumbing.
''',
    encoding="utf-8",
)
