"""Focused tests for non-Gaussian response acquisitions."""
import torch
import pytest
from bochan.acquisition.non_gaussian._stats import observation_variance_from_mean
from bochan.acquisition.non_gaussian.active_learning import qNonGaussianResponseMeanVariance, qNonGaussianBALDProxy
from bochan.acquisition.non_gaussian.levelset_estimation import qNonGaussianStraddle, qNonGaussianLevelSetUncertainty
from bochan.api.acquisition_registry import resolve_acqf_cls
from bochan.models.regression.non_gaussian.gamma import GammaGPModel
from bochan.models.components.beta import BetaLogLikelihood
from bochan.models.components.gamma import GammaLogLikelihood
from bochan.models.components.poisson import PoissonLogLikelihood
from bochan.models.components.negative_binomial import NegativeBinomialLogLikelihood

class Holder:
    def __init__(self, likelihood): self.likelihood=likelihood

@pytest.mark.parametrize(("likelihood","mean","expected"),[
    (BetaLogLikelihood(init_concentration=9.), .4, .4*.6/10),
    (GammaLogLikelihood(init_concentration=4.), 2., 1.),
    (PoissonLogLikelihood(), 3., 3.),
    (NegativeBinomialLogLikelihood(init_total_count=2.), 2., 4.),
])
def test_family_observation_variance(likelihood, mean, expected):
    value=observation_variance_from_mean(Holder(likelihood),torch.tensor(mean))
    assert value.item()==pytest.approx(expected,rel=1e-5)

def test_fixed_sampler_determinism_gradient_and_lse():
    x=torch.linspace(.1,1,6,dtype=torch.double).unsqueeze(-1); model=GammaGPModel(x,1+x)
    candidate=torch.tensor([[[.3],[.7]]],dtype=torch.double,requires_grad=True)
    for acq in (qNonGaussianResponseMeanVariance(model,seed=4), qNonGaussianBALDProxy(model,seed=4),
                qNonGaussianStraddle(model,threshold=1.5,seed=4), qNonGaussianLevelSetUncertainty(model,threshold=1.5,seed=4)):
        first=acq(candidate); second=acq(candidate)
        torch.testing.assert_close(first,second)
        first.sum().backward(retain_graph=True)
    assert torch.isfinite(candidate.grad).all()

def test_registry_non_gaussian_canonical_routes():
    assert resolve_acqf_cls('joint_bald',task_type='nongaussian').__name__=='qNonGaussianJointBALDProxy'
    assert resolve_acqf_cls('probability_of_exceedance',task_type='nongaussian').__name__=='qNonGaussianProbabilityOfExceedanceProxy'
