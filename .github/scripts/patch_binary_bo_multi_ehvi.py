from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/acquisition/binary/bayesian_optimization/multi_output.py"
text = path.read_text(encoding="utf-8")
old = """    pass


class qMultiOutputBinaryNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
"""
new = """    def __init__(self, model: Model, *args, **kwargs) -> None:
        super().__init__(
            model=as_epistemic_probability_model(model),
            *args,
            **kwargs,
        )


class qMultiOutputBinaryNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
"""
if text.count(old) != 1:
    raise RuntimeError("qEHVI pass block not found")
text = text.replace(old, new, 1)
start = text.index("class qMultiOutputBinaryNoisyExpectedHypervolumeImprovement")
end = text.index("def _prod", start)
block = text[start:end]
if block.count("    pass\n") != 1:
    raise RuntimeError("qNEHVI pass block not found")
block = block.replace(
    "    pass\n",
    """    def __init__(self, model: Model, *args, **kwargs) -> None:
        super().__init__(
            model=as_epistemic_probability_model(model),
            *args,
            **kwargs,
        )
""",
    1,
)
text = text[:start] + block + text[end:]
path.write_text(text, encoding="utf-8")
