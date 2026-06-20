from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

responses = ROOT / "src/bochan/serving/fastapi/schemas/responses.py"
text = responses.read_text(encoding="utf-8")
old = '''class PredictResponse(BaseModel):
    model_id: str
    mean: Any | None = None
    variance: Any | None = None
    value: Any | None = None
'''
new = '''class PredictResponse(BaseModel):
    model_id: str
    task_type: str | None = None
    prediction_space: str | None = None
    variance_kind: str | None = None
    posterior: Any | None = None
    mean: Any | None = None
    variance: Any | None = None
    value: Any | None = None
'''
if text.count(old) != 1:
    raise RuntimeError("PredictResponse block was not found.")
responses.write_text(text.replace(old, new, 1), encoding="utf-8")

route = ROOT / "src/bochan/serving/fastapi/routers/predictions.py"
text = route.read_text(encoding="utf-8")
pattern = re.compile(
    r'''@router\.post\("/\{model_id\}/predict", response_model=PredictResponse\)\ndef predict\(\n    model_id: str,\n    request: PredictRequest,\n    store: InMemoryOptimizerStore = Depends\(get_optimizer_store\),\n\) -> PredictResponse:\n.*?    except Exception as exc:\n        raise HTTPException\(status_code=400, detail=str\(exc\)\) from exc\n''',
    re.DOTALL,
)
replacement = '''@router.post("/{model_id}/predict", response_model=PredictResponse)
def predict(
    model_id: str,
    request: PredictRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> PredictResponse:
    try:
        optimizer = store.get(model_id)
        X = to_tensor(request.X)
        result = optimizer.predict(
            X,
            return_type=request.return_type,
            return_result=True,
            posterior_kwargs=request.posterior_kwargs,
        )

        common = {
            "model_id": model_id,
            "task_type": result.task_type,
            "prediction_space": result.prediction_space,
            "variance_kind": result.variance_kind,
        }
        mean = to_serializable(result.mean)
        variance = to_serializable(result.variance)

        if request.return_type == "posterior":
            summary = {
                "type": type(result.posterior).__name__,
                "mean": mean,
                "variance": variance,
            }
            return PredictResponse(
                **common,
                posterior=summary,
                mean=mean,
                variance=variance,
                value=summary,
            )
        if request.return_type == "mean_variance":
            return PredictResponse(**common, mean=mean, variance=variance)
        if request.return_type == "mean":
            return PredictResponse(**common, mean=mean, value=mean)
        if request.return_type == "variance":
            return PredictResponse(**common, variance=variance, value=variance)
        raise ValueError(f"Unsupported return_type: {request.return_type!r}.")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
'''
updated, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError("Modular prediction route was not found.")
route.write_text(updated, encoding="utf-8")
