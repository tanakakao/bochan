from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "src/bochan/acquisition/binary/active_learning/hetero_single_output.py"
text = PATH.read_text(encoding="utf-8")

start = text.index("class _HeteroUncertaintySamplingBinary")
end = text.index("class qHeteroBinaryPredictiveEntropy", start)
block = text[start:end]
block = block.replace(
    '        score_type: Literal["entropy", "variance", "least_confidence"] = "variance",\n        pending_penalty_weight: float = 0.0,\n',
    '        score_type: Literal["entropy", "variance", "least_confidence"] = "variance",\n        num_samples: int = 128,\n        pending_penalty_weight: float = 0.0,\n',
    1,
)
block = block.replace(
    '        self.score_type = score_type\n        self._set_classification_score_objective(objective)\n',
    '        self.score_type = score_type\n        self.num_samples = int(num_samples)\n        self._set_classification_score_objective(objective)\n',
    1,
)
old = '''        post = self.model.posterior(X_in)
        Xt = self._apply_input_transform(X_in)

        p = self._squeeze_last_output_dim(post.mean)
        pmin = p.min().item()
        pmax = p.max().item()
        if not (0.0 <= pmin and pmax <= 1.0):
            p = latent_samples_to_binary_probabilities(self.model, p, eps=self.eps, name="p via binary likelihood")
        p = p.clamp(self.eps, 1.0 - self.eps)
        p = _align_pointwise_score_to_X(
            p,
            Xt,
            name="HeteroUncertaintySampling probability",
            reduce_extra="mean",
        )

        score = self._uncertainty_score(p)
'''
new = '''        Xt = self._apply_input_transform(X_in)
        if self.score_type == "variance":
            mean_prob, epistemic_var, _, _ = binary_probability_moments(
                self.model,
                X_in,
                num_samples=self.num_samples,
                eps=self.eps,
            )
            p = self._squeeze_last_output_dim(mean_prob)
            score = self._squeeze_last_output_dim(epistemic_var)
            score = _align_pointwise_score_to_X(
                score,
                Xt,
                name="HeteroUncertaintySampling epistemic variance",
                reduce_extra="mean",
            )
        else:
            post = self.model.posterior(X_in)
            p = self._squeeze_last_output_dim(post.mean)
            pmin = p.min().item()
            pmax = p.max().item()
            if not (0.0 <= pmin and pmax <= 1.0):
                p = latent_samples_to_binary_probabilities(
                    self.model,
                    p,
                    eps=self.eps,
                    name="p via binary likelihood",
                )
            p = p.clamp(self.eps, 1.0 - self.eps)
            score = self._uncertainty_score(p)

        p = _align_pointwise_score_to_X(
            p,
            Xt,
            name="HeteroUncertaintySampling probability",
            reduce_extra="mean",
        )
'''
if old not in block:
    raise RuntimeError("unified hetero uncertainty forward block not found")
block = block.replace(old, new, 1)
block = block.replace(
    '- "variance": probability variance p(1-p)',
    '- "variance": probability epistemic variance Var_f[p(y=1|f)]',
)
text = text[:start] + block + text[end:]
text = text.replace(
    'posterior / probability / utility の分散が大きい点を選びます。',
    'latent posterior が誘導する確率の epistemic variance が大きい点を選びます。',
)
PATH.write_text(text, encoding="utf-8")
