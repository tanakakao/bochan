from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/acquisition/binary/bayesian_optimization/_utils.py"
text = path.read_text(encoding="utf-8")
anchor = "from bochan.acquisition.binary._likelihood import values_to_binary_probabilities\n"
line = "from bochan.acquisition.binary.epistemic import get_binary_latent_posterior\n"
if line not in text:
    if anchor not in text:
        raise RuntimeError("BO utils import anchor not found")
    text = text.replace(anchor, anchor + line, 1)
old = '''def get_single_model_posterior(model, X: Tensor, *, samples_are_probs: bool):
    if samples_are_probs:
        fn = getattr(model, "probability_posterior", None)
        if callable(fn):
            return fn(X)
        return model.posterior(X)

    for name in ("latent_posterior", "posterior_latent", "posterior_f"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn(X)
    return model.posterior(X)
'''
new = '''def get_single_model_posterior(model, X: Tensor, *, samples_are_probs: bool):
    if samples_are_probs:
        fn = getattr(model, "probability_posterior", None)
        if callable(fn):
            return fn(X)
        return model.posterior(X)
    return get_binary_latent_posterior(model, X)
'''
if text.count(old) != 1:
    raise RuntimeError("get_single_model_posterior block not found")
text = text.replace(old, new, 1)
old = '''    else:
        for name in ("latent_posterior", "posterior_latent", "posterior_f"):
            fn = getattr(model, name, None)
            if callable(fn):
                return fn(X)

    if hasattr(model, "models"):
'''
new = '''    else:
        try:
            return get_binary_latent_posterior(model, X)
        except AttributeError:
            pass

    if hasattr(model, "models"):
'''
if text.count(old) != 1:
    raise RuntimeError("get_model_posterior latent block not found")
text = text.replace(old, new, 1)
old = '''    return model.posterior(X)


def normalize_mean_shape'''
new = '''    if samples_are_probs:
        return model.posterior(X)
    raise AttributeError(
        f"{type(model).__name__} has no latent posterior for binary epistemic sampling."
    )


def normalize_mean_shape'''
if text.count(old) != 1:
    raise RuntimeError("get_model_posterior fallback block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
