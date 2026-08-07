from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match in {path}, found {count}.")
    file_path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''def _normalized_acquisition_name(name: str) -> str:\n    return "".join(character for character in str(name).lower() if character.isalnum())\n\n\n''',
    '''def _normalized_acquisition_name(name: str) -> str:\n    return "".join(character for character in str(name).lower() if character.isalnum())\n\n\ndef _set_active_learning_reference_kwargs(\n    acqf_kwargs: dict[str, Any],\n    *,\n    acq_key: str,\n    train_x: Any,\n) -> None:\n    """Attach only the reference inputs supported by the selected AL acquisition.\n\n    True NIPV consumes ``mc_points`` as its integration set and has no\n    ``X_observed`` argument. Pointwise uncertainty acquisitions use\n    ``X_observed`` for optional observed-point penalties / exclusion.\n    """\n    if acq_key in {"nipv", "qnipv"}:\n        acqf_kwargs.setdefault("mc_points", train_x)\n        return\n    acqf_kwargs.setdefault("X_observed", train_x)\n\n\n''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''        acqf_kwargs.setdefault("output_reduction", "weighted_mean")\n        acqf_kwargs.setdefault("X_observed", train_x)\n        if acq_key in {"nipv", "qnipv"}:\n            acqf_kwargs.setdefault("mc_points", train_x)\n        data_context = DataContext(\n''',
    '''        acqf_kwargs.setdefault("output_reduction", "weighted_mean")\n        _set_active_learning_reference_kwargs(\n            acqf_kwargs,\n            acq_key=acq_key,\n            train_x=train_x,\n        )\n        data_context = DataContext(\n''',
)

replace_once(
    "src/bochan/models/hybrid/multi_output.py",
    '''    def condition_on_observations(\n        self,\n        X: Tensor,\n''',
    '''    def fantasize(\n        self,\n        X: Tensor,\n        sampler: Any,\n        observation_noise: bool | Tensor | None = None,\n        **kwargs: Any,\n    ) -> "HybridMultiOutputModel":\n        """Fantasize each independent submodel and rebuild the hybrid wrapper.\n\n        This enables fantasy-based acquisitions such as true regression NIPV\n        when the Web workbench represents even one target through the hybrid\n        multi-output wrapper.\n        """\n        X_tensor = self._unwrap_X(X)\n        new_specs = []\n        for i, spec in enumerate(self.specs):\n            fn = getattr(spec.model, "fantasize", None)\n            if not callable(fn):\n                raise NotImplementedError(\n                    f"Submodel {i} ({spec.name!r}) has no fantasize()."\n                )\n\n            call_kwargs = dict(kwargs)\n            if observation_noise is not None:\n                noise_i = observation_noise\n                if (\n                    torch.is_tensor(observation_noise)\n                    and observation_noise.ndim > 0\n                    and observation_noise.shape[-1] == self.num_outputs\n                ):\n                    noise_i = observation_noise[..., i : i + 1]\n                call_kwargs["observation_noise"] = noise_i\n\n            model_i = fn(\n                X=X_tensor,\n                sampler=sampler,\n                **call_kwargs,\n            )\n            new_specs.append(replace(spec, model=model_i))\n\n        return self.__class__(new_specs)\n\n    def condition_on_observations(\n        self,\n        X: Tensor,\n''',
)

Path("tests/test_webapp_single_objective_nipv.py").write_text(
    '''from __future__ import annotations\n\nimport torch\nfrom botorch.models import SingleTaskGP\nfrom botorch.optim import optimize_acqf\n\nfrom bochan.acquisition.regression.active_learning import (\n    qMultiOutputRegressionNegIntegratedPosteriorVariance,\n)\nfrom bochan.models.hybrid import HybridMultiOutputModel, OutputSpec\nfrom bochan.serving.webapp.workflows_tabular import (\n    _set_active_learning_reference_kwargs,\n)\n\n\ndef test_web_nipv_uses_mc_points_without_x_observed() -> None:\n    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)\n    kwargs: dict[str, object] = {}\n\n    _set_active_learning_reference_kwargs(\n        kwargs,\n        acq_key="nipv",\n        train_x=train_x,\n    )\n\n    assert kwargs["mc_points"] is train_x\n    assert "X_observed" not in kwargs\n\n\ndef test_web_pointwise_active_learning_keeps_x_observed() -> None:\n    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)\n    kwargs: dict[str, object] = {}\n\n    _set_active_learning_reference_kwargs(\n        kwargs,\n        acq_key="variance",\n        train_x=train_x,\n    )\n\n    assert kwargs["X_observed"] is train_x\n    assert "mc_points" not in kwargs\n\n\ndef test_single_objective_hybrid_wrapper_nipv_runs_optimize_acqf() -> None:\n    torch.manual_seed(0)\n    train_x = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.double)\n    train_y = (train_x - 0.35).square()\n    submodel = SingleTaskGP(train_x, train_y)\n    model = HybridMultiOutputModel(\n        [OutputSpec(name="y", task_type="regression", model=submodel)]\n    )\n    acquisition = qMultiOutputRegressionNegIntegratedPosteriorVariance(\n        model=model,\n        mc_points=train_x,\n        output_weights=[1.0],\n        output_reduction="weighted_mean",\n    )\n    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)\n\n    candidates, value = optimize_acqf(\n        acq_function=acquisition,\n        bounds=bounds,\n        q=1,\n        num_restarts=4,\n        raw_samples=32,\n        options={"maxiter": 50},\n    )\n\n    assert candidates.shape == (1, 1)\n    assert torch.isfinite(candidates).all()\n    assert torch.isfinite(torch.as_tensor(value)).all()\n''',
    encoding="utf-8",
    newline="\n",
)

print("Applied Web single-objective NIPV fix.")
