from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/acquisition/binary/bayesian_optimization/hetero_multi_output.py"
text = path.read_text(encoding="utf-8")
anchor = "from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform\n"
line = "from bochan.acquisition.binary.epistemic import as_epistemic_probability_model\n"
if line not in text:
    if anchor not in text:
        raise RuntimeError("hetero BO import anchor not found")
    text = text.replace(anchor, anchor + line, 1)

# qNEHVI delegates sampling to BoTorch; wrap the model so samples are latent-to-probability epistemic samples.
old = '''        hetero_objective = _HeteroClassificationMCMultiOutputObjective(
            base_objective=base_objective,
            model=model,
            beta=beta,
            noise_penalty=noise_penalty,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
            samples_are_probs=samples_are_probs,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
        )
        super().__init__(
            model=model,
'''
new = '''        hetero_objective = _HeteroClassificationMCMultiOutputObjective(
            base_objective=base_objective,
            model=model,
            beta=beta,
            noise_penalty=noise_penalty,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
            samples_are_probs=True,
            apply_sigmoid_if_needed=False,
            eps=eps,
        )
        super().__init__(
            model=as_epistemic_probability_model(model),
'''
if text.count(old) != 1:
    raise RuntimeError("hetero qNEHVI constructor block not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
