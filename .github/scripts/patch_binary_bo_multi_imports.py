from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/acquisition/binary/bayesian_optimization/multi_output.py"
text = path.read_text(encoding="utf-8")
anchor = "from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform\n"
imports = """from bochan.acquisition.binary.epistemic import (
    as_epistemic_probability_model,
    get_binary_latent_posterior,
)
"""
if imports not in text:
    if anchor not in text:
        raise RuntimeError("import anchor not found")
    text = text.replace(anchor, anchor + "\n" + imports, 1)
old = """    if (not samples_are_probs) and prefer_latent:
        latent_fn = getattr(model, "latent_posterior", None)
        if callable(latent_fn):
            return latent_fn(X)

    return get_model_posterior(model, X, samples_are_probs=samples_are_probs)
"""
new = """    if (not samples_are_probs) and prefer_latent:
        return get_binary_latent_posterior(model, X)

    return get_model_posterior(model, X, samples_are_probs=samples_are_probs)
"""
if old not in text:
    raise RuntimeError("latent helper block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
