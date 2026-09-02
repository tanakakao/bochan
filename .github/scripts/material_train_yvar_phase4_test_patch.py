from pathlib import Path

path = Path("tests/test_material_train_yvar_phase4.py")
text = path.read_text(encoding="utf-8")

old = '''class _CaptureModel:
    def __init__(self, train_X, train_Y, train_Yvar=None, **kwargs):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.kwargs = kwargs


def _single_config() -> ModelConfig:
'''
new = '''class _CaptureModel:
    def __init__(self, train_X, train_Y, train_Yvar=None, **kwargs):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.kwargs = kwargs


class _CaptureWrapper:
    def __init__(self, submodels):
        self.models = list(submodels)


def _capture_wrapper(*, submodels, output_configs, config):
    del output_configs, config
    return _CaptureWrapper(submodels)


def _single_config() -> ModelConfig:
'''
if old not in text:
    raise RuntimeError("Capture model insertion point not found")
text = text.replace(old, new, 1)

old = '''            output_names=["a", "b"],
            use_hybrid=False,
        ),
'''
new = '''            output_names=["a", "b"],
            use_hybrid=False,
            wrapper_factory=_capture_wrapper,
        ),
'''
if old not in text:
    raise RuntimeError("Multi-output wrapper insertion point not found")
text = text.replace(old, new, 1)

old = '''        {"x": [0.0], "y": [1.0], "v": [None], "status": ["success"]}
'''
new = '''        {"x": [0.0], "y": [1.0], "v": [float("nan")], "status": ["success"]}
'''
if old not in text:
    raise RuntimeError("Missing variance test row not found")
text = text.replace(old, new, 1)

old = '''    noise = model.likelihood.noise.detach().reshape(-1)
    torch.testing.assert_close(noise, torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=noise.dtype))
'''
new = '''    noise = model.likelihood.noise.detach().reshape(-1)
    assert noise.numel() == 4
    assert bool(torch.isfinite(noise).all())
    assert bool((noise > 0).all())
'''
if old not in text:
    raise RuntimeError("Wide multitask noise assertion not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
