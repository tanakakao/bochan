"""Posterior transfer diagnostics for single-output multi-fidelity GPs.

The diagnostics quantify how much a fitted multi-fidelity surrogate believes
that a lower-fidelity observation informs the target fidelity at the same design
point.  They also compare the learned posterior correlation with the empirical
correlation of the synthetic ground-truth functions on a shared Sobol probe set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from .experiment import SyntheticBenchmarkConfig, generate_initial_data
from .synthetic import SyntheticMultiFidelityProblem


@dataclass(frozen=True)
class FidelityTransferDiagnostic:
    """Summary of source-to-target information transfer for one fitted model."""

    problem: str
    seed: int
    source_fidelity: float
    target_fidelity: float
    n_probe: int
    mean_posterior_correlation: float
    median_posterior_correlation: float
    min_posterior_correlation: float
    max_posterior_correlation: float
    mean_squared_correlation: float
    mean_target_variance_reduction_fraction: float
    mean_target_variance: float
    true_output_correlation: float

    def row(self) -> dict[str, float | int | str]:
        """Return a CSV-friendly representation."""

        return {
            "problem": self.problem,
            "seed": int(self.seed),
            "source_fidelity": float(self.source_fidelity),
            "target_fidelity": float(self.target_fidelity),
            "n_probe": int(self.n_probe),
            "mean_posterior_correlation": float(self.mean_posterior_correlation),
            "median_posterior_correlation": float(self.median_posterior_correlation),
            "min_posterior_correlation": float(self.min_posterior_correlation),
            "max_posterior_correlation": float(self.max_posterior_correlation),
            "mean_squared_correlation": float(self.mean_squared_correlation),
            "mean_target_variance_reduction_fraction": float(
                self.mean_target_variance_reduction_fraction
            ),
            "mean_target_variance": float(self.mean_target_variance),
            "true_output_correlation": float(self.true_output_correlation),
        }


def _pearson_correlation(x: Tensor, y: Tensor) -> float:
    """Compute Pearson correlation without introducing an optional dependency."""

    x = x.reshape(-1)
    y = y.reshape(-1)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.sqrt(
        x_centered.square().sum() * y_centered.square().sum()
    )
    if float(denominator.item()) <= 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum().div(denominator).item())


def _probe_designs(
    problem: SyntheticMultiFidelityProblem,
    *,
    n_probe: int,
    seed: int,
) -> Tensor:
    """Draw deterministic design probes while leaving fidelity to be assigned later."""

    n_probe = int(n_probe)
    if n_probe < 2:
        raise ValueError("n_probe must be at least 2.")
    return draw_sobol_samples(
        bounds=problem.bounds,
        n=n_probe,
        q=1,
        seed=int(seed),
    ).squeeze(-2)


def run_fidelity_transfer_diagnostic(
    problem: SyntheticMultiFidelityProblem,
    *,
    seed: int = 0,
    n_probe: int = 128,
    config: SyntheticBenchmarkConfig | None = None,
) -> tuple[FidelityTransferDiagnostic, ...]:
    """Fit the initial MF surrogate and summarize lower-to-target transfer.

    For every non-target fidelity, the fitted posterior is queried jointly at
    ``(x, source_fidelity)`` and ``(x, target_fidelity)`` for the same Sobol
    design points.  The posterior covariance yields a pointwise latent-function
    correlation.  Its square is the fraction of target latent variance removed
    by an ideal noiseless observation at the source fidelity under the Gaussian
    conditioning identity.

    The synthetic objective is also evaluated at the paired fidelities so the
    learned posterior correlation can be compared with the true cross-fidelity
    output correlation over the same probe set.
    """

    if problem.num_objectives != 1:
        raise ValueError("Fidelity transfer diagnostics require a single-objective problem.")

    from bochan.api import BayesianOptimizer
    from bochan.api.configs import DataContext, FitConfig, ModelConfig

    config = config or SyntheticBenchmarkConfig()
    torch.manual_seed(int(seed))

    train_X = generate_initial_data(problem, n=config.n_initial, seed=int(seed))
    train_Y = problem.evaluate(train_X)

    optimizer = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            model_kwargs={
                "fidelity_features": [problem.fidelity_feature],
                "target_fidelities": {
                    problem.fidelity_feature: problem.target_fidelity,
                },
            },
        ),
        FitConfig(maxiter=config.fit_maxiter, skip_fit=config.skip_fit),
        bounds=problem.bounds,
        data_context=DataContext(bounds=problem.bounds),
    )
    optimizer.fit(train_X, train_Y)

    probes = _probe_designs(problem, n_probe=n_probe, seed=10_000 + int(seed))
    feature = problem.fidelity_feature
    target = float(problem.target_fidelity)

    results: list[FidelityTransferDiagnostic] = []
    for source in problem.fidelity_values:
        source = float(source)
        if source == target:
            continue

        source_X = probes.clone()
        source_X[:, feature] = source
        target_X = probes.clone()
        target_X[:, feature] = target
        paired_X = torch.stack([source_X, target_X], dim=1)

        posterior = optimizer.model.posterior(paired_X)
        mvn = getattr(posterior, "mvn", None)
        if mvn is None:
            raise RuntimeError("Expected a GPyTorch posterior exposing an mvn covariance.")
        covariance = mvn.covariance_matrix
        if covariance.shape[-2:] != (2, 2):
            raise RuntimeError(
                "Expected paired posterior covariance with trailing shape 2 x 2; "
                f"received {tuple(covariance.shape)}."
            )

        source_var = covariance[..., 0, 0].clamp_min(torch.finfo(covariance.dtype).eps)
        target_var = covariance[..., 1, 1].clamp_min(torch.finfo(covariance.dtype).eps)
        cross_cov = covariance[..., 0, 1]
        correlation = cross_cov / torch.sqrt(source_var * target_var)
        correlation = correlation.clamp(min=-1.0, max=1.0)
        squared_correlation = correlation.square()

        source_true = problem.evaluate(source_X)[:, 0]
        target_true = problem.evaluate(target_X)[:, 0]
        true_correlation = _pearson_correlation(source_true, target_true)

        results.append(
            FidelityTransferDiagnostic(
                problem=problem.name,
                seed=int(seed),
                source_fidelity=source,
                target_fidelity=target,
                n_probe=int(n_probe),
                mean_posterior_correlation=float(correlation.mean().item()),
                median_posterior_correlation=float(correlation.median().item()),
                min_posterior_correlation=float(correlation.min().item()),
                max_posterior_correlation=float(correlation.max().item()),
                mean_squared_correlation=float(squared_correlation.mean().item()),
                mean_target_variance_reduction_fraction=float(
                    squared_correlation.mean().item()
                ),
                mean_target_variance=float(target_var.mean().item()),
                true_output_correlation=true_correlation,
            )
        )

    return tuple(results)


__all__ = [
    "FidelityTransferDiagnostic",
    "run_fidelity_transfer_diagnostic",
]
