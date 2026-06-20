from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "src/bochan/api/fastapi.py"
text = path.read_text(encoding="utf-8")

old_schema = '''class PredictResponse(APIBaseModel):
    posterior: Any | None = None
    mean: Any | None = None
    variance: Any | None = None
'''
new_schema = '''class PredictResponse(APIBaseModel):
    task_type: str | None = None
    prediction_space: str | None = None
    variance_kind: str | None = None
    posterior: Any | None = None
    mean: Any | None = None
    variance: Any | None = None
'''
if text.count(old_schema) != 1:
    raise RuntimeError("Legacy PredictResponse block was not found.")
text = text.replace(old_schema, new_schema, 1)

old_route = '''            posterior_payload = None if request.return_type != "posterior" else str(result.posterior)
            return PredictResponse(
                posterior=posterior_payload,
                mean=_to_python(result.mean),
                variance=_to_python(result.variance),
            )
'''
new_route = '''            mean = _to_python(result.mean)
            variance = _to_python(result.variance)
            posterior_payload = None
            if request.return_type == "posterior":
                posterior_payload = {
                    "type": type(result.posterior).__name__,
                    "mean": mean,
                    "variance": variance,
                }
            return PredictResponse(
                task_type=result.task_type,
                prediction_space=result.prediction_space,
                variance_kind=result.variance_kind,
                posterior=posterior_payload,
                mean=mean,
                variance=variance,
            )
'''
if text.count(old_route) != 1:
    raise RuntimeError("Legacy prediction route block was not found.")
path.write_text(text.replace(old_route, new_route, 1), encoding="utf-8")
