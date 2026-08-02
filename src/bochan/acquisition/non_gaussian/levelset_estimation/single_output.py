"""Level-set estimation on non-Gaussian raw response scales."""
from __future__ import annotations
from typing import Any, Literal
import torch
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import get_sampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from bochan.acquisition.regression.levelset_estimation.single_output import _RegressionLevelSetBase
from .._stats import ensure_q_batch, non_gaussian_response_stats, observation_variance_from_mean, safe_logdet, safe_normal_cdf


class _NonGaussianLevelSetBase(_RegressionLevelSetBase):
    """Shared public-posterior, fixed-sampler LSE base."""
    def __init__(self, model, *, sampler: MCSampler | None = None, sample_shape: torch.Size = torch.Size([128]),
                 seed: int | None = None, num_samples: int | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs); self.sampler = sampler
        self.sample_shape = torch.Size([num_samples]) if num_samples is not None else sample_shape; self.seed = seed

    def _stats(self, X: Tensor):
        Xq = ensure_q_batch(X)
        if self.sampler is None and not getattr(self.model, "is_non_gaussian_model_list", False):
            self.sampler = get_sampler(self.model.posterior(Xq, observation_noise=False), self.sample_shape, seed=self.seed)
        return non_gaussian_response_stats(self.model, Xq, sampler=self.sampler, sample_shape=self.sample_shape, seed=self.seed, eps=self.eps)

    def _finish(self, score: Tensor, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X); Xt = self._apply_input_transform_for_distance(Xq)
        return self._finalize_pointwise_score(score, X, Xt, name=type(self).__name__)


class qNonGaussianStraddle(_NonGaussianLevelSetBase):
    """Pointwise response-mean score beta*sd(E[Y|f])-|E[Y]-threshold|."""
    def __init__(self, model, *, beta: float | Tensor = 1.96, **kwargs: Any) -> None:
        super().__init__(model, **kwargs); self.register_buffer("beta", torch.as_tensor(beta))
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate response-mean straddle without aleatoric variance."""
        s = self._stats(X); score = self.beta.to(s.response_mean) * s.response_mean_variance.sqrt() - (s.response_mean - self.threshold.to(s.response_mean)).abs()
        return self._finish(score, X)


class qNonGaussianBoundaryVariance(_NonGaussianLevelSetBase):
    """Response-mean variance weighted by Gaussian boundary proximity."""
    def __init__(self, model, *, tau: float | Tensor = 1.0, **kwargs: Any) -> None:
        super().__init__(model, **kwargs); self.register_buffer("tau", torch.as_tensor(tau))
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate boundary-weighted epistemic variance."""
        s = self._stats(X); z = (s.response_mean-self.threshold.to(s.response_mean))/self.tau.to(s.response_mean).clamp_min(self.eps)
        return self._finish(s.response_mean_variance * torch.exp(-0.5*z.square()), X)


class qNonGaussianICUProxy(_NonGaussianLevelSetBase):
    """Local contour-weighted uncertainty proxy, not fantasy-based ICU."""
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate local response-mean contour uncertainty."""
        s=self._stats(X); sd=s.response_mean_variance.sqrt().clamp_min(self.eps); z=(s.response_mean-self.threshold.to(sd))/sd
        return self._finish(torch.exp(-0.5*z.square())*sd, X)


class qNonGaussianProbabilityOfExceedanceProxy(_NonGaussianLevelSetBase):
    """Response-mean exceedance probability estimated by fixed MC or moments."""
    def __init__(self, model, *, method: Literal["mc","normal_moment","smooth_mc"]="smooth_mc",
                 temperature: float | Tensor=.05, mode: Literal["above","below","interval"]="above",
                 lower: float | Tensor | None=None, upper: float | Tensor | None=None, **kwargs: Any) -> None:
        if method not in ("mc","normal_moment","smooth_mc"): raise ValueError("Unknown method.")
        if mode == "interval" and (lower is None or upper is None or torch.any(torch.as_tensor(lower)>=torch.as_tensor(upper))):
            raise ValueError("mode='interval' requires lower < upper.")
        super().__init__(model, **kwargs); self.method=method; self.mode=mode
        self.register_buffer("temperature", torch.as_tensor(temperature)); self.register_buffer("lower", None if lower is None else torch.as_tensor(lower)); self.register_buffer("upper", None if upper is None else torch.as_tensor(upper))
    def _probability(self, s):
        mean=s.response_mean; sd=s.response_mean_variance.sqrt().clamp_min(self.eps); threshold=self.threshold.to(mean)
        lo=self.lower.to(mean) if self.lower is not None else threshold; hi=self.upper.to(mean) if self.upper is not None else threshold
        if self.method == "normal_moment":
            above=safe_normal_cdf((mean-threshold)/sd); below=1-above
            interval=safe_normal_cdf((hi-mean)/sd)-safe_normal_cdf((lo-mean)/sd)
        else:
            samples=s.response_mean_samples; temp=self.temperature.to(mean).clamp_min(self.eps)
            if self.method == "mc": above=(samples>=threshold).to(mean).mean(0); below=(samples<=threshold).to(mean).mean(0); interval=((samples>=lo)&(samples<=hi)).to(mean).mean(0)
            else: above=torch.sigmoid((samples-threshold)/temp).mean(0); below=torch.sigmoid((threshold-samples)/temp).mean(0); interval=(torch.sigmoid((samples-lo)/temp)*torch.sigmoid((hi-samples)/temp)).mean(0)
        return {"above":above,"below":below,"interval":interval}[self.mode].clamp(0,1)
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate raw-scale response-mean probability."""
        return self._finish(self._probability(self._stats(X)), X)


class qNonGaussianObservationProbabilityOfExceedance(qNonGaussianProbabilityOfExceedanceProxy):
    """Observation-space probability using family moments and differentiable CDFs.

    Gamma uses its exact CDF and Poisson uses the regularized incomplete gamma.
    Beta and Negative-Binomial use a moment-matched normal CDF where PyTorch has
    no differentiable CDF; heteroscedastic extra variance is included.
    """
    def _one(self, model, mean, variance, value):
        name=type(model.likelihood).__name__.lower()
        if "gamma" in name:
            k=model.likelihood.concentration.to(mean); return torch.special.gammainc(k, k*value.clamp_min(0)/mean.clamp_min(self.eps))
        if "poisson" in name and "negative" not in name:
            return torch.special.gammaincc(torch.floor(value).clamp_min(-1)+1, mean.clamp_min(self.eps))
        return safe_normal_cdf((value-mean)/variance.sqrt().clamp_min(self.eps))
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Integrate conditional observation probabilities over response samples."""
        s=self._stats(X); models=list(self.model.models) if getattr(self.model,"is_non_gaussian_model_list",False) else [self.model]
        vals=[]
        for i, model in enumerate(models):
            mean=s.response_mean_samples[...,i:i+1]; var=observation_variance_from_mean(model,mean)+s.extra_heteroscedastic_variance[...,i:i+1]
            threshold=self.threshold.to(mean); lo=self.lower.to(mean) if self.lower is not None else threshold; hi=self.upper.to(mean) if self.upper is not None else threshold
            if self.mode=="below": p=self._one(model,mean,var,threshold)
            elif self.mode=="interval": p=self._one(model,mean,var,hi)-self._one(model,mean,var,lo)
            else:
                cutoff=torch.ceil(threshold)-1 if any(x in type(model.likelihood).__name__.lower() for x in ("poisson","negative")) else threshold
                p=1-self._one(model,mean,var,cutoff)
            vals.append(p.mean(0).clamp(0,1))
        return self._finish(torch.cat(vals,-1),X)


class qNonGaussianLevelSetUncertainty(qNonGaussianProbabilityOfExceedanceProxy):
    """Boundary uncertainty derived from response-mean exceedance probability."""
    def __init__(self, model, *, score_mode: Literal["bernoulli_variance","binary_entropy","margin"]="bernoulli_variance", **kwargs: Any) -> None:
        super().__init__(model, **kwargs); self.score_mode=score_mode
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Score maximally near posterior boundary probability one half."""
        p=self._probability(self._stats(X)).clamp(self.eps,1-self.eps)
        score=4*p*(1-p) if self.score_mode=="bernoulli_variance" else (-(p*p.log()+(1-p)*(1-p).log()) if self.score_mode=="binary_entropy" else 1-(2*p-1).abs())
        return self._finish(score,X)


class qNonGaussianJointStraddle(qNonGaussianStraddle):
    """Joint q/output covariance straddle with a batch distance penalty."""
    def __init__(self, model, *, uncertainty_mode: Literal["trace","sqrt_trace","logdet","logdet1p"]="logdet1p", distance_reduction: Literal["mean","sum","max","rms"]="mean", **kwargs: Any) -> None:
        super().__init__(model, **kwargs); self.uncertainty_mode=uncertainty_mode; self.distance_reduction=distance_reduction
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate covariance-aware batch straddle."""
        s=self._stats(X); samples=s.response_mean_samples; flat=samples.reshape(samples.shape[0],*samples.shape[1:-2],-1); c=flat-flat.mean(0)
        cov=torch.einsum("s...i,s...j->...ij",c,c)/max(samples.shape[0]-1,1); trace=cov.diagonal(dim1=-2,dim2=-1).sum(-1)
        u={"trace":trace,"sqrt_trace":trace.sqrt(),"logdet":safe_logdet(cov),"logdet1p":safe_logdet(torch.eye(cov.shape[-1],device=X.device,dtype=X.dtype)+cov)}[self.uncertainty_mode]
        dist=(s.response_mean-self.threshold.to(s.response_mean)).abs().reshape(*s.response_mean.shape[:-2],-1)
        d={"mean":dist.mean(-1),"sum":dist.sum(-1),"max":dist.max(-1).values,"rms":dist.square().mean(-1).sqrt()}[self.distance_reduction]
        return self.beta.to(u)*u-d


qNonGaussianICU=qNonGaussianICUProxy
qNonGaussianProbabilityOfExceedance=qNonGaussianProbabilityOfExceedanceProxy
__all__=[n for n in globals() if n.startswith("qNonGaussian")]
