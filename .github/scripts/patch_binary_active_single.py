from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "src/bochan/acquisition/binary/active_learning/single_output.py"
text = PATH.read_text(encoding="utf-8")

anchor = "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n"
import_line = "from bochan.acquisition.binary.epistemic import binary_probability_moments\n"
if import_line not in text:
    if anchor not in text:
        raise RuntimeError("active single import anchor was not found")
    text = text.replace(anchor, anchor + import_line, 1)

old = '''        score_type: UncertaintyScoreType = "variance",
        pending_penalty_weight: float = 0.0,
'''
new = '''        score_type: UncertaintyScoreType = "variance",
        num_samples: int = 128,
        pending_penalty_weight: float = 0.0,
'''
if text.count(old) != 1:
    raise RuntimeError("uncertainty constructor signature was not found")
text = text.replace(old, new, 1)

old = '''        self.score_type = score_type
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
'''
new = '''        self.score_type = score_type
        self.num_samples = int(num_samples)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
'''
if text.count(old) != 1:
    raise RuntimeError("uncertainty constructor assignment was not found")
text = text.replace(old, new, 1)

old = '''        prob_fn = getattr(self.model, "probability_posterior", None)
        posterior = prob_fn(X) if callable(prob_fn) else self.model.posterior(X)
        p = self._normalize_prob_shape(posterior.mean, X)

        score = self._uncertainty_score(p)
        Xt = self._apply_input_transform(X)
'''
new = '''        if self.score_type == "variance":
            mean_prob, epistemic_var, _, _ = binary_probability_moments(
                self.model,
                X,
                num_samples=self.num_samples,
                eps=self.eps,
            )
            p = self._normalize_prob_shape(mean_prob, X)
            score = self._normalize_prob_shape(epistemic_var, X)
        else:
            prob_fn = getattr(self.model, "probability_posterior", None)
            posterior = prob_fn(X) if callable(prob_fn) else self.model.posterior(X)
            p = self._normalize_prob_shape(posterior.mean, X)
            score = self._uncertainty_score(p)
        Xt = self._apply_input_transform(X)
'''
if text.count(old) != 1:
    raise RuntimeError("uncertainty forward block was not found")
text = text.replace(old, new, 1)

text = text.replace(
    '``score_type="variance"`` uses Bernoulli variance ``p(1-p)``.',
    '``score_type="variance"`` uses probability epistemic variance ``Var_f[p(y=1|f)]``.',
)
text = text.replace(
    '確率 p の Bernoulli variance ``p(1-p)`` を最大化します。',
    'latent posterior が誘導する確率の epistemic variance を最大化します。',
)

old = '''        num_fantasies: int = 8,
        conditioning_steps: int = 10,
'''
new = '''        num_fantasies: int = 8,
        num_epistemic_samples: int = 128,
        conditioning_steps: int = 10,
'''
if text.count(old) != 1:
    raise RuntimeError("fantasy constructor signature was not found")
text = text.replace(old, new, 1)

old = '''        self.num_fantasies = int(num_fantasies)
        self.conditioning_steps = int(conditioning_steps)
'''
new = '''        self.num_fantasies = int(num_fantasies)
        self.num_epistemic_samples = int(num_epistemic_samples)
        self.conditioning_steps = int(conditioning_steps)
'''
if text.count(old) != 1:
    raise RuntimeError("fantasy constructor assignment was not found")
text = text.replace(old, new, 1)

old = '''    @torch.no_grad()
    def _integrated_probability_variance(self, fantasy_model: Model) -> Tensor:
        prob_fn = getattr(fantasy_model, "probability_posterior", None)
        posterior = prob_fn(self.mc_points) if callable(prob_fn) else fantasy_model.posterior(self.mc_points)
        prob = _binary_values_to_probability_for_ipv(
            fantasy_model,
            posterior.mean,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            name="fantasy posterior mean",
        )
        return (prob * (1.0 - prob)).mean()
'''
new = '''    @torch.no_grad()
    def _integrated_probability_variance(self, fantasy_model: Model) -> Tensor:
        _, epistemic_var, _, _ = binary_probability_moments(
            fantasy_model,
            self.mc_points,
            num_samples=self.num_epistemic_samples,
            eps=self.eps,
        )
        return epistemic_var.mean()
'''
if text.count(old) != 1:
    raise RuntimeError("fantasy integrated variance block was not found")
text = text.replace(old, new, 1)

PATH.write_text(text, encoding="utf-8")
