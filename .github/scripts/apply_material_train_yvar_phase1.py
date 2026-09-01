from __future__ import annotations

from pathlib import Path
import re

root = Path('.')

# Shared Gaussian DeepKernel: select fixed-noise likelihood for known variance.
path = root / 'src/bochan/models/regression/gaussian/deep/deepkernel_configurable.py'
text = path.read_text(encoding='utf-8')
old_import = 'from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood\n'
new_import = '''from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)
'''
if old_import not in text:
    raise SystemExit('deepkernel_configurable likelihood import did not match expected main')
text = text.replace(old_import, new_import, 1)

marker = '\n\nclass DeepKernelGaussianGPModel(_BaseDeepKernelGPModel):\n'
helper = '''

def _fixed_noise_likelihood(
    train_Yvar: Tensor | None,
    *,
    num_outputs: int,
) -> FixedNoiseGaussianLikelihood | None:
    """Build fixed training noise for scalar-output exact GPs.

    ``train_Yvar`` is already in outcome-transform space when this helper is
    called. Correlated multitask fixed-noise likelihoods require a separate
    event/noise contract and are intentionally deferred to the next phase.
    """

    if train_Yvar is None:
        return None
    if num_outputs != 1:
        raise NotImplementedError(
            "Known observation variance is not yet supported for correlated "
            "multi-output DeepKernel Gaussian models."
        )
    noise = train_Yvar.squeeze(-1) if train_Yvar.shape[-1:] == (1,) else train_Yvar
    return FixedNoiseGaussianLikelihood(noise=noise)


class DeepKernelGaussianGPModel(_BaseDeepKernelGPModel):
'''
if marker not in text:
    raise SystemExit('deepkernel_configurable class marker missing')
text = text.replace(marker, helper, 1)

normal_old = '''        if likelihood is None:
            if self._num_outputs == 1:
                likelihood = GaussianLikelihood()
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)
'''
normal_new = '''        if likelihood is None:
            likelihood = _fixed_noise_likelihood(
                self.train_Yvar,
                num_outputs=self._num_outputs,
            )
            if likelihood is None:
                if self._num_outputs == 1:
                    likelihood = GaussianLikelihood()
                else:
                    likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)
'''
if text.count(normal_old) != 1:
    raise SystemExit(f'expected one normal likelihood block, got {text.count(normal_old)}')
text = text.replace(normal_old, normal_new, 1)

mixed_old = '''        if likelihood is None:
            if self._num_outputs == 1:
                from botorch.models.utils.gpytorch_modules import (
                    get_gaussian_likelihood_with_lognormal_prior,
                )

                likelihood = get_gaussian_likelihood_with_lognormal_prior()
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)
'''
mixed_new = '''        if likelihood is None:
            likelihood = _fixed_noise_likelihood(
                self.train_Yvar,
                num_outputs=self._num_outputs,
            )
            if likelihood is None:
                if self._num_outputs == 1:
                    from botorch.models.utils.gpytorch_modules import (
                        get_gaussian_likelihood_with_lognormal_prior,
                    )

                    likelihood = get_gaussian_likelihood_with_lognormal_prior()
                else:
                    likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)
'''
if text.count(mixed_old) != 1:
    raise SystemExit(f'expected one mixed likelihood block, got {text.count(mixed_old)}')
text = text.replace(mixed_old, mixed_new, 1)
path.write_text(text, encoding='utf-8')

# Material-family scalar / mixed model constructors forward train_Yvar.
targets = {
    'src/bochan/models/regression/gaussian/deep/mace.py': 1,
    'src/bochan/models/regression/gaussian/deep/mace_mixed.py': 1,
    'src/bochan/models/regression/gaussian/deep/chgnet.py': 2,
    'src/bochan/models/regression/gaussian/deep/m3gnet.py': 2,
    'src/bochan/models/regression/gaussian/deep/alignn.py': 1,
    'src/bochan/models/regression/gaussian/deep/alignn_mixed.py': 1,
    'src/bochan/models/regression/gaussian/deep/crabnet.py': 1,
    'src/bochan/models/regression/gaussian/deep/crabnet_mixed.py': 1,
    'src/bochan/models/regression/gaussian/deep/crabnet_mixed_dkl.py': 1,
    'src/bochan/models/regression/gaussian/deep/roost.py': 1,
}
single_line_guard = re.compile(
    r'\n        if train_Yvar is not None:\n'
    r'            raise NotImplementedError\([^\n]+\)\n'
)
multi_line_guard = re.compile(
    r'\n        if train_Yvar is not None:\n'
    r'            raise NotImplementedError\(\n'
    r'(?:                [^\n]*\n)+'
    r'            \)\n'
)
total_guards = 0
total_forwarded = 0
for relative, expected in targets.items():
    model_path = root / relative
    source = model_path.read_text(encoding='utf-8')
    source, n_multi = multi_line_guard.subn('\n', source)
    source, n_single = single_line_guard.subn('\n', source)
    removed = n_multi + n_single
    if removed != expected:
        raise SystemExit(
            f'{relative}: expected {expected} train_Yvar guards, removed {removed}'
        )
    forwarded = source.count('train_Yvar=None,')
    if forwarded != expected:
        raise SystemExit(
            f'{relative}: expected {expected} train_Yvar=None forwards, found {forwarded}'
        )
    source = source.replace('train_Yvar=None,', 'train_Yvar=train_Yvar,')
    model_path.write_text(source, encoding='utf-8')
    total_guards += removed
    total_forwarded += forwarded
if total_guards != 12 or total_forwarded != 12:
    raise SystemExit(
        f'expected 12 material guards/forwards, got guards={total_guards}, forwards={total_forwarded}'
    )

# Correct stale public docstrings that still describe train_Yvar as reserved.
doc_replacements = {
    'src/bochan/models/regression/gaussian/deep/crabnet.py': [
        (
            'train_Yvar: Reserved for future fixed-noise support.  It must currently\n            be omitted.',
            'train_Yvar: Optional known observation variances for fixed-noise GP\n            training.',
        ),
        (
            'train_Yvar: Reserved for future fixed-noise support.',
            'train_Yvar: Optional known observation variances for fixed-noise training.',
        ),
    ],
    'src/bochan/models/regression/gaussian/deep/roost.py': [
        (
            'train_Yvar: Reserved for future fixed-noise support and currently\n            unsupported.',
            'train_Yvar: Optional known observation variances for fixed-noise GP\n            training.',
        ),
    ],
}
for relative, replacements in doc_replacements.items():
    p = root / relative
    source = p.read_text(encoding='utf-8')
    for old, new in replacements:
        source = source.replace(old, new)
    p.write_text(source, encoding='utf-8')

test_content = '''from __future__ import annotations

import pytest
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood

from bochan.models.regression.gaussian.deep import (
    ALIGNNDKLModel,
    ALIGNNGPModel,
    CHGNetDKLModel,
    CHGNetGPModel,
    CrabNetDKLModel,
    CrabNetGPModel,
    DeepKernelGaussianGPModel,
    DeepKernelGaussianMixedGPModel,
    M3GNetDKLModel,
    M3GNetGPModel,
    MACEDKLModel,
    MACEGPModel,
    RoostDKLModel,
    RoostGPModel,
)


def _known_variance(train_Y: torch.Tensor) -> torch.Tensor:
    return torch.linspace(
        0.01,
        0.04,
        train_Y.shape[0],
        dtype=train_Y.dtype,
        device=train_Y.device,
    ).unsqueeze(-1)


def _assert_fixed_noise(model, expected_yvar: torch.Tensor) -> None:
    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    assert model.train_Yvar is not None
    torch.testing.assert_close(model.train_Yvar, expected_yvar)
    torch.testing.assert_close(model.likelihood.noise, expected_yvar.squeeze(-1))
    posterior = model.posterior(model.train_X[:2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_shared_deepkernel_selects_fixed_noise_only_when_yvar_is_supplied() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = train_X[:, :1] - 0.3 * train_X[:, 1:2]
    train_Yvar = _known_variance(train_Y)

    fixed = DeepKernelGaussianGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        outcome_transform=None,
    )
    learned = DeepKernelGaussianGPModel(
        train_X,
        train_Y,
        outcome_transform=None,
    )

    _assert_fixed_noise(fixed, train_Yvar)
    assert isinstance(learned.likelihood, GaussianLikelihood)
    assert not isinstance(learned.likelihood, FixedNoiseGaussianLikelihood)


def test_shared_mixed_deepkernel_selects_fixed_noise() -> None:
    continuous = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    categorical = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.double
    ).unsqueeze(-1)
    train_X = torch.cat((continuous, categorical), dim=-1)
    train_Y = continuous + 0.2 * categorical
    train_Yvar = _known_variance(train_Y)
    model = DeepKernelGaussianMixedGPModel(
        train_X,
        train_Y,
        cat_dims=[1],
        train_Yvar=train_Yvar,
        outcome_transform=None,
    )
    _assert_fixed_noise(model, train_Yvar)


def test_mace_gp_and_dkl_accept_known_observation_variance() -> None:
    pytest.importorskip('mace')
    from tests.test_mace_gp import _data, _structures, _wrapped_encoder

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    encoder, _ = _wrapped_encoder()
    gp = MACEGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
        outcome_transform=None,
    )
    encoder, _ = _wrapped_encoder()
    dkl = MACEDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_chgnet_gp_and_dkl_accept_known_observation_variance() -> None:
    pytest.importorskip('pymatgen')
    from tests.test_chgnet_gp import FakeCHGNet, _data, _structures

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = CHGNetGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = CHGNetDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_m3gnet_gp_and_dkl_accept_known_observation_variance() -> None:
    pytest.importorskip('pymatgen')
    from tests.test_m3gnet_gp import _data, _structures, _wrapped_encoder

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = M3GNetGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = M3GNetDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_alignn_gp_and_dkl_accept_known_observation_variance() -> None:
    from tests.test_alignn_gp import FakeALIGNN, _data, _graphs

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = ALIGNNGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = ALIGNNDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_crabnet_gp_and_dkl_accept_known_observation_variance() -> None:
    from tests.test_crabnet_gp import (
        FakeCrabNet,
        LayeredFakeCrabNet,
        _data,
        _element_ids,
    )

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = CrabNetGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        element_ids=_element_ids(),
        encoder=FakeCrabNet(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = CrabNetDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        element_ids=_element_ids(),
        encoder=LayeredFakeCrabNet(),
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_roost_gp_and_dkl_accept_known_observation_variance() -> None:
    from tests.test_roost_gp import (
        FakeRoostBackbone,
        LayeredFakeRoostBackbone,
        _data,
        _element_ids,
    )

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = RoostGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        element_ids=_element_ids(),
        encoder=FakeRoostBackbone(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = RoostDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        element_ids=_element_ids(),
        encoder=LayeredFakeRoostBackbone(),
        latent_dim=3,
        encoder_training='partial',
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)
'''
(root / 'tests/test_material_train_yvar_phase1.py').write_text(
    test_content, encoding='utf-8'
)

docs = '''# Material GP known observation variance — Phase 1

Phase 1 adds a shared fixed-noise contract for bochan's Gaussian material GP/DKL families.

## Meaning of `train_Yvar`

`train_Yvar` is the **known observation variance** for each training target, not a standard deviation. When it is supplied and no custom likelihood is passed, the shared Gaussian DeepKernel layer constructs `gpytorch.likelihoods.FixedNoiseGaussianLikelihood` from the outcome-transformed variance.

When `train_Yvar` is omitted, existing learned-noise behavior is unchanged.

## Phase 1 support

The scalar-output GP/DKL paths now accept `train_Yvar` for:

- MACE
- CHGNet
- M3GNet
- ALIGNN
- CrabNet
- Roost

Mixed-input variants that inherit the shared scalar Gaussian DeepKernel contract use the same fixed-noise behavior. Independent multi-output construction can reuse these scalar-output models once its caller provides one variance column per output.

## Explicit scope boundaries

Phase 1 intentionally does **not** add known-noise support to correlated multitask material models. Those models use a multitask covariance/likelihood event structure and need an explicit task-wise fixed-noise design rather than treating a wide `train_Yvar` tensor as scalar noise. That is Phase 2 work.

The high-level `TabularBayesianOptimizer` / FastAPI data contract also does not yet expose observation-variance columns. Phase 1 establishes the model-layer capability; high-level noise-column plumbing is a separate integration step.

This feature is also distinct from a learned heteroscedastic GP. `train_Yvar` means the observation variances are known inputs to the model rather than inferred as a second latent noise process.

## Custom likelihoods

An explicitly supplied likelihood remains authoritative. Automatic `FixedNoiseGaussianLikelihood` selection occurs only when `likelihood=None`.
'''
(root / 'docs/material_train_yvar_phase1.md').write_text(docs, encoding='utf-8')

workflow = '''name: Material train Yvar Phase 1 smoke

on:
  pull_request:
    paths:
      - "src/bochan/models/regression/gaussian/deep/**"
      - "tests/test_material_train_yvar_phase1.py"
      - "docs/material_train_yvar_phase1.md"
      - ".github/workflows/material-train-yvar-phase1-smoke.yml"
  workflow_dispatch:

jobs:
  fixed-noise:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install material test surface
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test,materials]"
      - name: Run Phase 1 fixed-noise tests
        run: python -m pytest tests/test_material_train_yvar_phase1.py -q
      - name: Run material GP regressions
        run: |
          python -m pytest \\
            tests/test_mace_gp.py \\
            tests/test_chgnet_gp.py \\
            tests/test_m3gnet_gp.py \\
            tests/test_alignn_gp.py \\
            tests/test_crabnet_gp.py \\
            tests/test_roost_gp.py \\
            -q
      - name: Ruff
        run: |
          ruff check \\
            src/bochan/models/regression/gaussian/deep/deepkernel_configurable.py \\
            src/bochan/models/regression/gaussian/deep/mace.py \\
            src/bochan/models/regression/gaussian/deep/mace_mixed.py \\
            src/bochan/models/regression/gaussian/deep/chgnet.py \\
            src/bochan/models/regression/gaussian/deep/m3gnet.py \\
            src/bochan/models/regression/gaussian/deep/alignn.py \\
            src/bochan/models/regression/gaussian/deep/alignn_mixed.py \\
            src/bochan/models/regression/gaussian/deep/crabnet.py \\
            src/bochan/models/regression/gaussian/deep/crabnet_mixed.py \\
            src/bochan/models/regression/gaussian/deep/crabnet_mixed_dkl.py \\
            src/bochan/models/regression/gaussian/deep/roost.py \\
            tests/test_material_train_yvar_phase1.py
'''
(root / '.github/workflows/material-train-yvar-phase1-smoke.yml').write_text(
    workflow, encoding='utf-8'
)
