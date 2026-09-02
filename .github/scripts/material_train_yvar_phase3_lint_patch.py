from pathlib import Path

path = Path("src/bochan/api/optimizer/core.py")
text = path.read_text(encoding="utf-8")
text = text.replace('-> "BayesianOptimizer":', '-> BayesianOptimizer:')
text = text.replace(
    "for dim, value in zip(cat_dims, row)",
    "for dim, value in zip(cat_dims, row, strict=True)",
)
path.write_text(text, encoding="utf-8")
