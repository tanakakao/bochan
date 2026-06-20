from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "src/bochan/acquisition/binary/bayesian_optimization/single_output.py"
text = PATH.read_text(encoding="utf-8")

import_anchor = "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n"
import_line = "from bochan.acquisition.binary.epistemic import binary_probability_samples\n"
if import_line not in text:
    if import_anchor not in text:
        raise RuntimeError("single-output BO import anchor was not found")
    text = text.replace(import_anchor, import_anchor + import_line, 1)

pattern = re.compile(
    r'''    def _posterior_samples_as_prob\(self, X: Tensor\) -> Tensor:\n.*?        return samples\n''',
    re.DOTALL,
)
replacement = '''    def _posterior_samples_as_prob(self, X: Tensor) -> Tensor:
        """Draw class-probability samples from latent GP uncertainty only.

        ``model.posterior(X).rsample`` is intentionally not used here because
        binary model posterior variance is the Bernoulli label variance
        ``p(1-p)``.  EI / PI / UCB require probability samples induced by the
        latent posterior, i.e. epistemic uncertainty.
        """
        return binary_probability_samples(
            self.model,
            X,
            sample_shape=self.sampler.sample_shape,
            eps=self.eps,
        )
'''
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError(f"_posterior_samples_as_prob replacement count={count}")

old = '''    if best_f is not None:
        return torch.as_tensor(best_f, device=ref.device, dtype=ref.dtype)
'''
new = '''    if best_f is not None:
        resolved = torch.as_tensor(best_f, device=ref.device, dtype=ref.dtype)
        if resolved.numel() != 1:
            raise ValueError("binary best_f must be a scalar probability.")
        if not torch.isfinite(resolved):
            raise ValueError("binary best_f must be finite.")
        if resolved.item() < 0.0 or resolved.item() > 1.0:
            raise ValueError(
                f"binary best_f must be in [0, 1], got {resolved.item():.6g}."
            )
        # Passing train_Y.max() gives exactly 1 for binary labels.  That makes
        # EI zero almost everywhere and PI numerically arbitrary.  Keep the
        # incumbent inside the probability domain using the same margin as the
        # automatically inferred incumbent.
        return resolved.clamp(
            min=float(eps),
            max=1.0 - float(best_f_margin),
        )
'''
if text.count(old) != 1:
    raise RuntimeError("explicit best_f block was not found")
text = text.replace(old, new, 1)

text = text.replace(
    "apply_sigmoid_if_needed: posterior mean / samples が [0, 1] にない場合に likelihood link で変換するかどうか。",
    "apply_sigmoid_if_needed: 後方互換引数。EI / PI / UCB の samples は常に latent posterior を likelihood link で確率化します。",
)

PATH.write_text(text, encoding="utf-8")
