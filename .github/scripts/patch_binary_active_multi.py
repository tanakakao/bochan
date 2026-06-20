from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "src/bochan/acquisition/binary/active_learning/multi_output.py"
text = PATH.read_text(encoding="utf-8")

old = '''        score_type: UncertaintyScoreType = "variance",
        output_mode: MultiOutputMode = "mean",
'''
new = '''        score_type: UncertaintyScoreType = "variance",
        num_samples: int = 128,
        output_mode: MultiOutputMode = "mean",
'''
if text.count(old) != 1:
    raise RuntimeError("multi-output uncertainty signature was not found")
text = text.replace(old, new, 1)

old = '''        self.score_type = score_type
        self.output_mode = output_mode
'''
new = '''        self.score_type = score_type
        self.num_samples = int(num_samples)
        self.output_mode = output_mode
'''
if text.count(old) != 1:
    raise RuntimeError("multi-output uncertainty assignment was not found")
text = text.replace(old, new, 1)

pattern = re.compile(
    r'''    @t_batch_mode_transform\(\)\n    def forward\(self, X: Tensor\) -> Tensor:\n        X = self\._ensure_q_batch\(X\)\n        self\._set_eval_mode\(\)\n\n        raw_X = X\n        original_batch_shape = raw_X\.shape\[:-2\]\n\n        # posterior は raw_X で評価する。shape 整合と penalty には expanded_X を使う。\n        Xt = self\._apply_input_transform\(raw_X\)\n        posterior = self\._get_probability_posterior\(raw_X\)\n\n        probs = self\._normalize_mean_shape\(posterior\.mean, Xt\)\n        probs = self\._to_probability\(\n            probs,\n            apply_sigmoid_if_needed=self\.apply_sigmoid_if_needed,\n            name="probability_posterior\.mean",\n        \)\n\n        score_per_output = self\._uncertainty_score_binary_event\(probs, self\.score_type\)\n        score = self\._aggregate_outputs\(\n            score_per_output,\n            output_mode=self\.output_mode,\n            output_weights=self\.output_weights,\n            probs_for_all_positive=probs,\n            score_type_for_all_positive=self\.score_type,\n        \)  # \(\*batch, q_like\)\n''',
    re.DOTALL,
)
replacement = '''    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        self._set_eval_mode()

        raw_X = X
        original_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)

        if self.score_type == "variance":
            latent_posterior = self._get_latent_posterior(raw_X)
            latent_samples = latent_posterior.rsample(
                torch.Size([self.num_samples])
            )
            latent_samples = self._reshape_samples(
                latent_samples,
                Xt,
                self.num_samples,
            )
            probability_samples = latent_samples_to_binary_probabilities(
                self.model,
                latent_samples,
                eps=self.eps,
                name="multi-output latent posterior samples",
                output_dim=-1,
            )
            probs = probability_samples.mean(dim=0)
            if self.output_mode == "all_positive":
                event_samples = probability_samples.prod(dim=-1)
                score = event_samples.var(dim=0, unbiased=False)
            else:
                score_per_output = probability_samples.var(
                    dim=0,
                    unbiased=False,
                )
                score = self._aggregate_outputs(
                    score_per_output,
                    output_mode=self.output_mode,
                    output_weights=self.output_weights,
                )
        else:
            posterior = self._get_probability_posterior(raw_X)
            probs = self._normalize_mean_shape(posterior.mean, Xt)
            probs = self._to_probability(
                probs,
                apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
                name="probability_posterior.mean",
            )
            score_per_output = self._uncertainty_score_binary_event(
                probs,
                self.score_type,
            )
            score = self._aggregate_outputs(
                score_per_output,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
                probs_for_all_positive=probs,
                score_type_for_all_positive=self.score_type,
            )
'''
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError(f"multi-output uncertainty forward replacement count={count}")

text = text.replace(
    'return p * (1.0 - p)',
    'return p * (1.0 - p)  # predictive ambiguity; variance mode bypasses this helper',
    1,
)
text = text.replace(
    'posterior / probability / utility の分散が大きい点を選びます。',
    'latent posterior が誘導する確率の epistemic variance が大きい点を選びます。',
)
PATH.write_text(text, encoding="utf-8")
