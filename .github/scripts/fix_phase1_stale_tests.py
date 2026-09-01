from pathlib import Path

FILES = (
    Path("tests/test_crabnet_gp.py"),
    Path("tests/test_roost_gp.py"),
)

for path in FILES:
    text = path.read_text(encoding="utf-8")
    block = '''    with pytest.raises(NotImplementedError, match="train_Yvar"):\n        {model}(\n            train_X=train_X,\n            train_Y=train_Y,\n            train_Yvar=torch.full_like(train_Y, 0.01),\n            element_ids=_element_ids(),\n            encoder={encoder}(),\n        )\n'''
    if path.name == "test_crabnet_gp.py":
        expected = block.format(model="CrabNetGPModel", encoder="FakeCrabNet")
    else:
        expected = block.format(model="RoostGPModel", encoder="FakeRoostBackbone")
    if expected not in text:
        raise SystemExit(f"stale train_Yvar rejection block not found in {path}")
    path.write_text(text.replace(expected, "", 1), encoding="utf-8")
