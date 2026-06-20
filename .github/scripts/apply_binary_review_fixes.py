from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match in {path.relative_to(ROOT)}; found {count}.\n"
            f"Pattern:\n{old}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def update_likelihood_helper() -> None:
    path = ROOT / "src/bochan/acquisition/binary/_likelihood.py"
    replace_once(
        path,
        '''def values_to_binary_probabilities(
    model: Any,
    values: Tensor,
    *,
    eps: float = 1e-6,
    name: str = "binary values",
    output_dim: int = -1,
) -> Tensor:
    """Validate probability values or transform latent values via likelihood."""
    if not torch.isfinite(values).all():
        raise RuntimeError(f"{name}: values contain NaN or inf.")

    vmin = values.detach().min().item()
    vmax = values.detach().max().item()
    if 0.0 <= vmin and vmax <= 1.0:
        return values.clamp(eps, 1.0 - eps)

    return latent_samples_to_binary_probabilities(
        model,
        values,
        eps=eps,
        name=name,
        output_dim=output_dim,
    )
''',
        '''def values_to_binary_probabilities(
    model: Any,
    values: Tensor,
    *,
    eps: float = 1e-6,
    name: str = "binary values",
    output_dim: int = -1,
    values_are_probabilities: bool | None = None,
) -> Tensor:
    """Validate probabilities or transform latent values via likelihood.

    ``values_are_probabilities=False`` forces likelihood conversion even when
    every latent value happens to lie inside ``[0, 1]``.  ``None`` preserves the
    compatibility behavior that infers the value space from the numeric range.
    """
    if not torch.isfinite(values).all():
        raise RuntimeError(f"{name}: values contain NaN or inf.")

    vmin = values.detach().min().item()
    vmax = values.detach().max().item()
    if values_are_probabilities is not False and 0.0 <= vmin and vmax <= 1.0:
        return values.clamp(eps, 1.0 - eps)

    return latent_samples_to_binary_probabilities(
        model,
        values,
        eps=eps,
        name=name,
        output_dim=output_dim,
    )
''',
    )


def update_bo_probability_helper() -> None:
    path = ROOT / "src/bochan/acquisition/binary/bayesian_optimization/_utils.py"
    replace_once(
        path,
        '''def to_probability(
    x: Tensor,
    *,
    apply_sigmoid_if_needed: bool,
    eps: float,
    name: str,
    model: Optional[Model] = None,
) -> Tensor:
    """Convert probability values or latent values using the model likelihood.

    ``apply_sigmoid_if_needed`` is retained as a compatibility argument.  When
    conversion is required it no longer means a hard-coded sigmoid: the model's
    binary likelihood is used, so GPyTorch ``BernoulliLikelihood`` follows its
    probit link and custom likelihoods follow their own conditional link.
    """
    xmin = x.min().item()
    xmax = x.max().item()
    if 0.0 <= xmin and xmax <= 1.0:
        return x.clamp(eps, 1.0 - eps)
    if apply_sigmoid_if_needed:
        if model is None:
            raise RuntimeError(
                f"{name} requires latent-to-probability conversion, but no model "
                "was provided. Pass model=... so the binary likelihood link can be used."
            )
        return values_to_binary_probabilities(
            model,
            x,
            eps=eps,
            name=name,
        )
    raise RuntimeError(
        f"{name} is not in [0,1] (min={xmin:.4g}, max={xmax:.4g}). "
        "Return a probability posterior or enable likelihood-aware conversion."
    )
''',
        '''def to_probability(
    x: Tensor,
    *,
    apply_sigmoid_if_needed: bool,
    eps: float,
    name: str,
    model: Optional[Model] = None,
    values_are_probs: Optional[bool] = None,
) -> Tensor:
    """Convert probability values or latent values using the model likelihood.

    ``values_are_probs=False`` is the unambiguous latent path and always applies
    the model's likelihood, even when all latent values happen to be in
    ``[0, 1]``.  ``apply_sigmoid_if_needed`` remains a compatibility name for
    range-inferred inputs, but conversion is likelihood-aware rather than a
    hard-coded sigmoid.
    """
    if values_are_probs is False:
        if model is None:
            raise RuntimeError(
                f"{name} is declared latent, but no model was provided for "
                "likelihood-aware conversion."
            )
        return values_to_binary_probabilities(
            model,
            x,
            eps=eps,
            name=name,
            values_are_probabilities=False,
        )

    xmin = x.min().item()
    xmax = x.max().item()
    if 0.0 <= xmin and xmax <= 1.0:
        return x.clamp(eps, 1.0 - eps)
    if apply_sigmoid_if_needed:
        if model is None:
            raise RuntimeError(
                f"{name} requires latent-to-probability conversion, but no model "
                "was provided. Pass model=... so the binary likelihood link can be used."
            )
        return values_to_binary_probabilities(
            model,
            x,
            eps=eps,
            name=name,
            values_are_probabilities=values_are_probs,
        )
    raise RuntimeError(
        f"{name} is not in [0,1] (min={xmin:.4g}, max={xmax:.4g}). "
        "Return a probability posterior or enable likelihood-aware conversion."
    )
''',
    )


def update_multi_output_bo() -> None:
    path = ROOT / "src/bochan/acquisition/binary/bayesian_optimization/multi_output.py"
    replace_once(
        path,
        '''    MC acquisition で sigmoid 変換を使う場合、posterior(X).rsample() に
    sigmoid をかけるより、latent_posterior(X).rsample() に sigmoid をかける方が
    意味的に自然。そのため samples_are_probs=False では latent_posterior を優先する。
''',
        '''    MC acquisition で likelihood link を使う場合、probability posterior の
    samples を再変換するのではなく、latent posterior samples を各モデルの
    likelihood に通す。そのため samples_are_probs=False では latent_posterior を優先する。
''',
    )
    replace_once(
        path,
        '''            probs = to_probability(samples, apply_sigmoid_if_needed=not self.samples_are_probs or self.apply_sigmoid_if_needed, eps=self.eps, name='posterior.rsample()', model=self.model)
''',
        '''            probs = to_probability(
                samples,
                apply_sigmoid_if_needed=(
                    not self.samples_are_probs or self.apply_sigmoid_if_needed
                ),
                eps=self.eps,
                name="posterior.rsample()",
                model=self.model,
                values_are_probs=self.samples_are_probs,
            )
''',
    )
    replace_once(
        path,
        '''        values = to_probability(samples, apply_sigmoid_if_needed=not self.samples_are_probs or self.apply_sigmoid_if_needed, eps=self.eps, name='NParEGO posterior samples', model=self.model)
''',
        '''        values = to_probability(
            samples,
            apply_sigmoid_if_needed=(
                not self.samples_are_probs or self.apply_sigmoid_if_needed
            ),
            eps=self.eps,
            name="NParEGO posterior samples",
            model=self.model,
            values_are_probs=self.samples_are_probs,
        )
''',
    )


def update_hetero_single_output_bo() -> None:
    path = ROOT / "src/bochan/acquisition/binary/bayesian_optimization/hetero_single_output.py"
    replace_once(
        path,
        '''    ensure_q_batch,
    normalize_binary_mean_shape,
''',
        '''    ensure_q_batch,
    get_single_model_posterior,
    normalize_binary_mean_shape,
''',
    )
    replace_once(
        path,
        '''    mean_prob = to_probability(mean_prob, apply_sigmoid_if_needed=apply_sigmoid_if_needed, eps=eps, name='posterior.mean', model=model)
''',
        '''    mean_prob = to_probability(
        mean_prob,
        apply_sigmoid_if_needed=apply_sigmoid_if_needed,
        eps=eps,
        name="posterior.mean",
        model=model,
        values_are_probs=samples_are_probs,
    )
''',
    )
    replace_once(
        path,
        '''    samples = to_probability(samples, apply_sigmoid_if_needed=not samples_are_probs or apply_sigmoid_if_needed, eps=eps, name='posterior samples', model=model)
''',
        '''    samples = to_probability(
        samples,
        apply_sigmoid_if_needed=(
            not samples_are_probs or apply_sigmoid_if_needed
        ),
        eps=eps,
        name="posterior samples",
        model=model,
        values_are_probs=samples_are_probs,
    )
''',
    )
    replace_once(
        path,
        '''        mean_prob = to_probability(mean_prob, apply_sigmoid_if_needed=apply_sigmoid_if_needed, eps=eps, name='posterior.mean', model=model)
''',
        '''        mean_prob = to_probability(
            mean_prob,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
            name="posterior.mean",
            model=model,
            values_are_probs=True,
        )
''',
    )
    replace_once(
        path,
        '''    def _hetero_samples(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)
''',
        '''    def _hetero_samples(self, X: Tensor) -> Tensor:
        post = get_single_model_posterior(
            self.model,
            X,
            samples_are_probs=self.samples_are_probs,
        )
        samples = self.get_posterior_samples(post)
''',
    )


def update_hetero_multi_output_bo() -> None:
    path = ROOT / "src/bochan/acquisition/binary/bayesian_optimization/hetero_multi_output.py"
    replace_once(
        path,
        '''    mean_prob = to_probability(mean_prob, apply_sigmoid_if_needed=apply_sigmoid_if_needed, eps=eps, name='posterior.mean', model=model)
''',
        '''    mean_prob = to_probability(
        mean_prob,
        apply_sigmoid_if_needed=apply_sigmoid_if_needed,
        eps=eps,
        name="posterior.mean",
        model=model,
        values_are_probs=samples_are_probs,
    )
''',
    )
    replace_once(
        path,
        '''    samples = to_probability(samples, apply_sigmoid_if_needed=not samples_are_probs or apply_sigmoid_if_needed, eps=eps, name='posterior samples', model=model)
''',
        '''    samples = to_probability(
        samples,
        apply_sigmoid_if_needed=(
            not samples_are_probs or apply_sigmoid_if_needed
        ),
        eps=eps,
        name="posterior samples",
        model=model,
        values_are_probs=samples_are_probs,
    )
''',
    )
    replace_once(
        path,
        '''        mean = to_probability(mean, apply_sigmoid_if_needed=apply_sigmoid_if_needed, eps=eps, name='posterior.mean', model=model)
''',
        '''        mean = to_probability(
            mean,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
            name="posterior.mean",
            model=model,
            values_are_probs=True,
        )
''',
    )


def update_fantasy_ipv() -> None:
    path = ROOT / "src/bochan/acquisition/binary/active_learning/single_output.py"
    replace_once(
        path,
        '''        apply_sigmoid_if_needed: posterior mean が latent 値の場合に sigmoid で確率化するか。
''',
        '''        apply_sigmoid_if_needed: posterior mean が latent 値の場合に likelihood link で確率化するか。
''',
    )
    replace_once(
        path,
        '''        prob = _binary_values_to_probability_for_ipv(self.model, posterior.mean, apply_sigmoid_if_needed=self.apply_sigmoid_if_needed, eps=self.eps, name='binary posterior mean')
''',
        '''        prob = _binary_values_to_probability_for_ipv(
            self.model,
            posterior.mean,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            name="binary posterior mean",
        )
''',
    )
    replace_once(
        path,
        '''        prob = _binary_values_to_probability_for_ipv(self.model, posterior.mean, apply_sigmoid_if_needed=self.apply_sigmoid_if_needed, eps=self.eps, name='fantasy posterior mean')
''',
        '''        prob = _binary_values_to_probability_for_ipv(
            fantasy_model,
            posterior.mean,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            name="fantasy posterior mean",
        )
''',
    )


def update_tests() -> None:
    path = ROOT / "tests/test_binary_likelihood_consistency.py"
    anchor = '''def test_probability_values_are_not_transformed_twice() -> None:
    probability = torch.tensor([0.2, 0.5, 0.8], dtype=torch.double)
    model = _Model(BernoulliLikelihood())

    actual = values_to_binary_probabilities(model, probability)
    assert torch.allclose(actual, probability)


'''
    addition = '''def test_known_latent_values_inside_unit_interval_are_transformed() -> None:
    latent = torch.tensor([0.1, 0.4, 0.8], dtype=torch.double)
    model = _Model(BernoulliLikelihood())

    actual = values_to_binary_probabilities(
        model,
        latent,
        eps=1e-12,
        values_are_probabilities=False,
    )
    expected = Normal(0.0, 1.0).cdf(latent)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(actual, latent)


def test_to_probability_forces_known_latent_values_through_likelihood() -> None:
    latent = torch.tensor([0.1, 0.4, 0.8], dtype=torch.double)
    model = _Model(BernoulliLikelihood())

    actual = to_probability(
        latent,
        apply_sigmoid_if_needed=True,
        eps=1e-12,
        name="known latent values",
        model=model,
        values_are_probs=False,
    )
    expected = Normal(0.0, 1.0).cdf(latent)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(actual, latent)


'''
    replace_once(path, anchor, anchor + addition)


def main() -> None:
    update_likelihood_helper()
    update_bo_probability_helper()
    update_multi_output_bo()
    update_hetero_single_output_bo()
    update_hetero_multi_output_bo()
    update_fantasy_ipv()
    update_tests()


if __name__ == "__main__":
    main()
