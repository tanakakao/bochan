from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/acquisition/binary/active_learning/hetero_multi_output.py"
text = path.read_text(encoding="utf-8")
anchor = "from botorch.utils.transforms import t_batch_mode_transform\n"
line = "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n"
if line not in text:
    text = text.replace(anchor, anchor + line, 1)

# BALD must sample the latent posterior by default.
text = text.replace(
    "        samples_are_probs: bool = True,\n",
    "        samples_are_probs: bool = False,\n",
    1,
)
old = '''        posterior = self._get_probability_posterior(raw_X)

        samples = posterior.rsample(torch.Size([self.num_samples]))
        probs = self._reshape_samples(samples, Xt, self.num_samples)
        probs = self._to_probability(
            probs,
            apply_sigmoid_if_needed=not self.samples_are_probs,
            name="probability_posterior.rsample()",
        )
'''
new = '''        if self.samples_are_probs:
            posterior = self._get_probability_posterior(raw_X)
            samples = posterior.rsample(torch.Size([self.num_samples]))
            probs = self._reshape_samples(samples, Xt, self.num_samples)
            probs = self._to_probability(
                probs,
                apply_sigmoid_if_needed=False,
                name="probability_posterior.rsample()",
            )
        else:
            posterior = self._get_latent_posterior(raw_X)
            latent_samples = posterior.rsample(torch.Size([self.num_samples]))
            latent_samples = self._reshape_samples(
                latent_samples,
                Xt,
                self.num_samples,
            )
            probs = latent_samples_to_binary_probabilities(
                self.model,
                latent_samples,
                eps=self.eps,
                name="hetero multi-output latent samples",
                output_dim=-1,
            )
'''
if text.count(old) != 1:
    raise RuntimeError("hetero multi BALD sampling block not found")
text = text.replace(old, new, 1)

# Probability variance base.
start = text.index("class _HeteroMultiOutputBinaryProbabilityVarianceBase")
end = text.index("class _HeteroMultiOutputUncertaintySamplingClassifierAcquisition", start)
block = text[start:end]
block = block.replace(
    "        output_weights: Tensor | None = None,\n        pending_penalty_weight: float = 0.0,\n",
    "        output_weights: Tensor | None = None,\n        num_samples: int = 128,\n        pending_penalty_weight: float = 0.0,\n",
    1,
)
block = block.replace(
    "        self.output_weights = output_weights\n        self.mean_is_probs = bool(mean_is_probs)\n",
    "        self.output_weights = output_weights\n        self.num_samples = int(num_samples)\n        self.mean_is_probs = bool(mean_is_probs)\n",
    1,
)
old = '''        posterior = self._get_probability_posterior(raw_X)

        probs = self._normalize_mean_shape(posterior.mean, Xt)
        probs = self._to_probability(
            probs,
            apply_sigmoid_if_needed=not self.mean_is_probs,
            name="probability_posterior.mean",
        )

        if self.output_mode == "all_positive":
            log_p_all = probs.log().sum(dim=-1)
            p_all = log_p_all.exp().clamp(self.eps, 1.0 - self.eps)
            score = p_all * (1.0 - p_all)
'''
new = '''        latent_posterior = self._get_latent_posterior(raw_X)
        latent_samples = latent_posterior.rsample(torch.Size([self.num_samples]))
        latent_samples = self._reshape_samples(
            latent_samples,
            Xt,
            self.num_samples,
        )
        probability_samples = latent_samples_to_binary_probabilities(
            self.model,
            latent_samples,
            eps=self.eps,
            name="hetero multi-output latent samples",
            output_dim=-1,
        )
        probs = probability_samples.mean(dim=0)

        if self.output_mode == "all_positive":
            event_samples = probability_samples.prod(dim=-1)
            score = event_samples.var(dim=0, unbiased=False)
'''
if old not in block:
    raise RuntimeError("hetero multi probability variance block not found")
block = block.replace(old, new, 1)
block = block.replace(
    "            score_per_output = probs * (1.0 - probs)\n",
    "            score_per_output = probability_samples.var(dim=0, unbiased=False)\n",
    1,
)
text = text[:start] + block + text[end:]

# Unified variance mode.
start = text.index("class _HeteroMultiOutputUncertaintySamplingClassifierAcquisition")
end = text.index("class qHeteroMultiOutputBinaryPredictiveEntropy", start)
block = text[start:end]
block = block.replace(
    "        score_type: str = \"variance\",\n        output_mode: str = \"all_positive\",\n",
    "        score_type: str = \"variance\",\n        num_samples: int = 128,\n        output_mode: str = \"all_positive\",\n",
    1,
)
block = block.replace(
    "        self.score_type = score_type\n        self.output_mode = output_mode\n",
    "        self.score_type = score_type\n        self.num_samples = int(num_samples)\n        self.output_mode = output_mode\n",
    1,
)
old = '''        posterior = self._get_probability_posterior(raw_X)
        probs = self._normalize_mean_shape(posterior.mean, Xt)
        probs = self._to_probability(
            probs,
            apply_sigmoid_if_needed=not self.mean_is_probs,
            name="probability_posterior.mean",
        )

        if self.output_mode == "all_positive":
            p_all = probs.log().sum(dim=-1).exp().clamp(self.eps, 1.0 - self.eps)
            score = self._uncertainty_score_binary_event(p_all, self.score_type)
'''
new = '''        probability_samples = None
        if self.score_type == "variance":
            latent_posterior = self._get_latent_posterior(raw_X)
            latent_samples = latent_posterior.rsample(torch.Size([self.num_samples]))
            latent_samples = self._reshape_samples(
                latent_samples,
                Xt,
                self.num_samples,
            )
            probability_samples = latent_samples_to_binary_probabilities(
                self.model,
                latent_samples,
                eps=self.eps,
                name="hetero multi-output latent samples",
                output_dim=-1,
            )
            probs = probability_samples.mean(dim=0)
        else:
            posterior = self._get_probability_posterior(raw_X)
            probs = self._normalize_mean_shape(posterior.mean, Xt)
            probs = self._to_probability(
                probs,
                apply_sigmoid_if_needed=not self.mean_is_probs,
                name="probability_posterior.mean",
            )

        if self.output_mode == "all_positive":
            if self.score_type == "variance":
                event_samples = probability_samples.prod(dim=-1)
                score = event_samples.var(dim=0, unbiased=False)
            else:
                p_all = probs.log().sum(dim=-1).exp().clamp(self.eps, 1.0 - self.eps)
                score = self._uncertainty_score_binary_event(p_all, self.score_type)
'''
if old not in block:
    raise RuntimeError("hetero multi unified block not found")
block = block.replace(old, new, 1)
block = block.replace(
    "            score_per_output = self._uncertainty_score_binary_event(probs, self.score_type)\n",
    '''            if self.score_type == "variance":
                score_per_output = probability_samples.var(dim=0, unbiased=False)
            else:
                score_per_output = self._uncertainty_score_binary_event(
                    probs,
                    self.score_type,
                )
''',
    1,
)
text = text[:start] + block + text[end:]
text = text.replace(
    "posterior / probability / utility の分散が大きい点を選びます。",
    "latent posterior が誘導する確率の epistemic variance が大きい点を選びます。",
)
path.write_text(text, encoding="utf-8")
