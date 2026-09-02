from pathlib import Path


path = Path("src/bochan/api/evaluation/cross_validation.py")
text = path.read_text(encoding="utf-8")
old = (
    "    config = cv_config or CrossValidationConfig()\n\n"
    "    X, Y = observation_data.X, observation_data.Y\n"
)
new = (
    "    config = cv_config or CrossValidationConfig()\n"
    "    if config.feature_importance_config is not None:\n"
    "        from bochan.inspection import FeatureImportanceConfig\n\n"
    "        if not isinstance(config.feature_importance_config, FeatureImportanceConfig):\n"
    "            raise ValueError(\n"
    "                \"Observation-aware feature importance requires a \"\n"
    "                \"FeatureImportanceConfig instance.\"\n"
    "            )\n\n"
    "    X, Y = observation_data.X, observation_data.Y\n"
)
count = text.count(old)
if count != 1:
    raise RuntimeError(f"Phase 8 config validation: expected one match, found {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
