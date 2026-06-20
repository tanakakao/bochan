import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/acquisition/binary/bayesian_optimization/single_output.py"
text = path.read_text(encoding="utf-8")
anchor = "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n"
line = "from bochan.acquisition.binary.epistemic import binary_probability_samples\n"
if line not in text:
    text = text.replace(anchor, anchor + line, 1)
pattern = re.compile(
    r"    def _posterior_samples_as_prob\(self, X: Tensor\) -> Tensor:\n.*?        return reshape_binary_samples\(probs, X\)\n",
    re.DOTALL,
[]
replacement = """    def _posterior_samples_as_prob(self, X: Tensor) -> Tensor:
        \"\"\"Draw probability samples induced only by latent GP uncertainty.\"\"\"
        X = ensure_q_batch(X)
        probs = binary_probability_samples(
            self.model,
            X,
            sample_shape=self.sampler.sample_shape,
            eps=self.eps,
        )
        if self.score_objective is not None:
            probs = self.score_objective(probs, X=X)
        probs = self._squeeze_binary_output_dim_if_present(probs, X)
        return reshape_binary_samples(probs, X)
"""
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError(f"posterior sample block count={count}")
old = "        if best_f is not None:\n            return torch.as_tensor(best_f)\n"
new = """        if best_f is not None:
            resolved = torch.as_tensor(best_f)
            if resolved.numel() != 1 or not torch.isfinite(resolved):
                raise ValueError("binary best_f must be one finite scalar probability.")
            if resolved.item() < 0.0 or resolved.item() > 1.0:
                raise ValueError("binary best_f must be in [0, 1].")
            return resolved.clamp(min=self.eps, max=1.0 - float(best_f_margin))
"""
if old not in text:
    raise RuntimeError("explicit best_f block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
