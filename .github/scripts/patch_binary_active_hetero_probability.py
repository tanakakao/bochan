from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "src/bochan/acquisition/binary/active_learning/hetero_single_output.py"
text = PATH.read_text(encoding="utf-8")

anchor = "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n"
line = "from bochan.acquisition.binary.epistemic import binary_probability_moments\n"
if line not in text:
    text = text.replace(anchor, anchor + line, 1)

start = text.index("class _HeteroProbabilityVarianceBinary")
end = text.index("# =========================================================", start)
block = text[start:end]
block = block.replace(
    '        reduction: ReductionType = "mean",\n        pending_penalty_weight: float = 0.0,\n',
    '        reduction: ReductionType = "mean",\n        num_samples: int = 128,\n        pending_penalty_weight: float = 0.0,\n',
    1,
)
block = block.replace(
    '        self._set_classification_score_objective(objective)\n',
    '        self.num_samples = int(num_samples)\n        self._set_classification_score_objective(objective)\n',
    1,
)
old = '''        post = self.model.posterior(X_in)
        Xt = self._apply_input_transform(X_in)

        p = self._squeeze_last_output_dim(post.mean).clamp(self.eps, 1.0 - self.eps)
        p = _align_pointwise_score_to_X(
            p,
            Xt,
            name="HeteroProbabilityVariance probability",
            reduce_extra="mean",
        )
        score = p * (1.0 - p)
'''
new = '''        mean_prob, epistemic_var, _, _ = binary_probability_moments(
            self.model,
            X_in,
            num_samples=self.num_samples,
            eps=self.eps,
        )
        Xt = self._apply_input_transform(X_in)
        p = self._squeeze_last_output_dim(mean_prob)
        p = _align_pointwise_score_to_X(
            p,
            Xt,
            name="HeteroProbabilityVariance probability",
            reduce_extra="mean",
        )
        score = self._squeeze_last_output_dim(epistemic_var)
        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name="HeteroProbabilityVariance epistemic variance",
            reduce_extra="mean",
        )
'''
if old not in block:
    raise RuntimeError("hetero probability variance forward block not found")
block = block.replace(old, new, 1)
text = text[:start] + block + text[end:]
PATH.write_text(text, encoding="utf-8")
