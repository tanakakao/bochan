from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "src/bochan/acquisition/binary/active_learning/integrated_posterior_variance.py"
text = PATH.read_text(encoding="utf-8")

anchor = "from ._utils import _apply_objective_to_pointwise_score\n"
line = "from bochan.acquisition.binary.epistemic import binary_probability_moments\n"
if line not in text:
    if anchor not in text:
        raise RuntimeError("IPV import anchor was not found")
    text = text.replace(anchor, anchor + line, 1)

old = '''        integration_beta: float = 25.0,
        local_weight: Optional[float] = None,
'''
new = '''        integration_beta: float = 25.0,
        num_epistemic_samples: int = 128,
        local_weight: Optional[float] = None,
'''
if text.count(old) != 1:
    raise RuntimeError("IPV signature block was not found")
text = text.replace(old, new, 1)

old = '''            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
'''
new = '''            reduction=reduction,
            num_samples=num_epistemic_samples,
            pending_penalty_weight=pending_penalty_weight,
'''
if text.count(old) != 1:
    raise RuntimeError("IPV parent constructor block was not found")
text = text.replace(old, new, 1)

old = '''        self.integration_beta = float(integration_beta)
        self.local_weight = (
'''
new = '''        self.integration_beta = float(integration_beta)
        self.num_epistemic_samples = int(num_epistemic_samples)
        self.local_weight = (
'''
if text.count(old) != 1:
    raise RuntimeError("IPV assignment block was not found")
text = text.replace(old, new, 1)

old = '''    def _probability(self, X: Tensor, *, name: str) -> Tensor:
        prob_fn = getattr(self.model, "probability_posterior", None)
        posterior = prob_fn(X) if callable(prob_fn) else self.model.posterior(X)
        probability = _binary_values_to_probability_for_ipv(self.model, posterior.mean, apply_sigmoid_if_needed=self.apply_sigmoid_if_needed, eps=self.eps, name=name)
        return probability.squeeze(-1) if probability.shape[-1] == 1 else probability
'''
new = '''    def _epistemic_stats(self, X: Tensor) -> tuple[Tensor, Tensor]:
        mean, epistemic_var, _, _ = binary_probability_moments(
            self.model,
            X,
            num_samples=self.num_epistemic_samples,
            eps=self.eps,
        )
        if mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        if epistemic_var.shape[-1] == 1:
            epistemic_var = epistemic_var.squeeze(-1)
        return mean, epistemic_var
'''
if text.count(old) != 1:
    raise RuntimeError("IPV probability helper was not found")
text = text.replace(old, new, 1)

old = '''        mc_probability = self._probability(
            self.mc_points,
            name="binary mc_points posterior mean",
        ).reshape(-1)
        mc_uncertainty = mc_probability * (1.0 - mc_probability)
'''
new = '''        _, mc_uncertainty = self._epistemic_stats(self.mc_points)
        mc_uncertainty = mc_uncertainty.reshape(-1)
'''
if text.count(old) != 1:
    raise RuntimeError("IPV integrated uncertainty block was not found")
text = text.replace(old, new, 1)

old = '''        probability = self._probability(
            raw_X,
            name="binary candidate posterior mean",
        )
        local_score = probability * (1.0 - probability)
'''
new = '''        _, local_score = self._epistemic_stats(raw_X)
'''
if text.count(old) != 1:
    raise RuntimeError("IPV local uncertainty block was not found")
text = text.replace(old, new, 1)

text = text.replace(
    '"""Binary IPV proxy using the same score design as multiclass IPV."""',
    '"""Binary IPV proxy based on probability epistemic variance."""',
)
PATH.write_text(text, encoding="utf-8")
