# FastAPI LLM planning endpoints

This document summarizes the LLM-related endpoints added to `bochan.serving.fastapi`.

API keys should stay on the server as environment variables. Do not send raw API key values in request bodies.

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

---

## 1. Plan settings only

`POST /models/plan` returns serializable bochan config dictionaries. It does not fit a model.

Use this mode when the user wants to review the model selection first.

```json
{
  "goal": "導電率を高くし、収縮率を低くしたい",
  "mode": "model_config",
  "train_X": [[800.0, 2.0, 0.0], [850.0, 3.0, 1.0]],
  "train_Y": [[10.2, 5.1], [12.4, 4.8]],
  "bounds": [[700.0, 1.0, 0.0], [950.0, 5.0, 2.0]],
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4.1-mini",
    "api_key_env": "OPENAI_API_KEY"
  },
  "llm_context": {
    "variable_names": ["temperature", "time", "atmosphere"],
    "target_names": ["conductivity", "shrinkage"],
    "variable_descriptions": {
      "temperature": "焼成温度。",
      "time": "焼成保持時間。",
      "atmosphere": "焼成雰囲気。0=air, 1=N2, 2=Ar。"
    },
    "target_descriptions": {
      "conductivity": "導電率。高いほど望ましい。",
      "shrinkage": "焼成収縮率。低いほど望ましい。"
    }
  }
}
```

Response shape:

```json
{
  "plan": {
    "model_config": {},
    "fit_config": {},
    "acquisition_config": {},
    "optimize_config": {},
    "warnings": [],
    "reasoning_summary": "..."
  }
}
```

---

## 2. Plan, fit, and generate candidates

`POST /models/auto-candidates` runs the full flow.

```text
LLM planner
  -> model_config / fit_config / acquisition_config / optimize_config
  -> BayesianOptimizer.fit(...)
  -> BayesianOptimizer.candidate(...)
```

```json
{
  "goal": "導電率を高くし、収縮率を低くしたい",
  "train_X": [[800.0, 2.0, 0.0], [850.0, 3.0, 1.0], [900.0, 4.0, 2.0]],
  "train_Y": [[10.2, 5.1], [12.4, 4.8], [11.0, 7.2]],
  "bounds": [[700.0, 1.0, 0.0], [950.0, 5.0, 2.0]],
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4.1-mini",
    "api_key_env": "OPENAI_API_KEY"
  },
  "llm_context": {
    "variable_names": ["temperature", "time", "atmosphere"],
    "target_names": ["conductivity", "shrinkage"]
  }
}
```

Response shape:

```json
{
  "model": {
    "model_id": "...",
    "task_type": "regression",
    "model_type": "base",
    "n_train": 3,
    "metadata": {}
  },
  "candidate": {
    "model_id": "...",
    "candidates": [[...]],
    "acq_value": [...]
  },
  "plan": {}
}
```

---

## 3. Explicit configs override planner output

You can pass any of the following fields to constrain or override the LLM plan:

```json
{
  "model_config": {"task_type": "regression", "model_type": "base"},
  "fit_config": {"method": "auto"},
  "acquisition_config": {"name": "NEHVI"},
  "optimize_config": {"optimizer": "llm_candidate_set", "q": 3}
}
```

The endpoint uses explicit request configs first, then fills missing pieces from the LLM plan.

---

## 4. Offline tests with planner_response

`planner_response` bypasses provider calls and is useful for tests.

```json
{
  "goal": "導電率を高くし、収縮率を低くしたい",
  "planner_response": {
    "model_config": {"task_type": "regression", "model_type": "base"},
    "fit_config": {"method": "auto"},
    "acquisition_config": {"name": "EI"},
    "optimize_config": {
      "optimizer": "llm_candidate_set",
      "q": 1,
      "optimizer_kwargs": {
        "candidate_set": [[0.2, 0.3], [0.8, 0.9]]
      }
    },
    "warnings": []
  },
  "train_X": [[0.0, 0.0], [1.0, 1.0]],
  "train_Y": [[0.0], [1.0]],
  "bounds": [[0.0, 0.0], [1.0, 1.0]]
}
```
