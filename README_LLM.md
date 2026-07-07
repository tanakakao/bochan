# LLM-assisted planning and candidate generation

This document describes the experimental LLM-assisted planning and candidate-set optimizer added to `bochan`.

There are two LLM layers:

```text
LLM planner
  -> natural-language goal + data metadata
  -> model_config / fit_config / acquisition_config / optimize_config

LLM candidate generator
  -> broad candidate set
  -> bochan validation / repair
  -> existing acquisition function reranking
  -> top-q candidates
```

The LLM is not the final optimizer. It acts as a configuration assistant and domain-aware candidate generator.

---

## Install optional LLM dependencies

```bash
pip install -e ".[llm]"
```

This installs the optional provider SDKs:

```text
openai
google-genai
```

API keys should be supplied through environment variables, not committed to code or JSON payloads.

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

Windows PowerShell:

```powershell
setx OPENAI_API_KEY "..."
setx GEMINI_API_KEY "..."
```

---

## Mode 1: model/settings planning only

Use `plan_configs()` when you want the LLM to choose settings but do not want to fit a model yet.

```python
from bochan.llm import LLMConfig, LLMContextConfig, plan_configs

plan = plan_configs(
    goal="導電率を高くし、収縮率を低くしたい",
    train_X=train_X,
    train_Y=train_Y,
    bounds=bounds,
    mode="model_config",
    llm_config=LLMConfig(provider="openai", model="gpt-4.1-mini"),
    llm_context=LLMContextConfig(
        variable_names=["temperature", "time", "atmosphere"],
        target_names=["conductivity", "shrinkage"],
        variable_descriptions={
            "temperature": "焼成温度。",
            "time": "焼成保持時間。",
            "atmosphere": "焼成雰囲気。0=air, 1=N2, 2=Ar。",
        },
        target_descriptions={
            "conductivity": "導電率。高いほど望ましい。",
            "shrinkage": "焼成収縮率。低いほど望ましい。",
        },
    ),
)

print(plan["model_config"])
print(plan["fit_config"])
```

Expected shape:

```json
{
  "model_config": {
    "task_type": "regression",
    "model_type": "base",
    "input_transform_config": {"normalize": true},
    "outcome_transform": true
  },
  "fit_config": {"method": "auto"},
  "warnings": [],
  "reasoning_summary": "..."
}
```

Explicit configs can be supplied as guardrails. The prompt tells the LLM to preserve them unless they conflict with the goal.

---

## Mode 2: plan, fit, and generate candidates

The FastAPI endpoint `/models/auto-candidates` uses the planner result to fit a model and generate candidates in one request. For Python code, the low-level pieces are still explicit: call `plan_configs()`, build `BayesianOptimizer`, then call `candidate()`.

---

## Python API: LLM candidate-set reranking

```python
from bochan.api import AcquisitionConfig, ObjectiveConfig, OptimizeConfig
from bochan.llm import LLMConfig, LLMContextConfig

acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        outputs=["conductivity", "shrinkage"],
        directions=["maximize", "minimize"],
        weights=[1.0, 0.5],
    ),
)

opt_config = OptimizeConfig(
    optimizer="llm_candidate_set",
    q=3,
    raw_samples=64,
    optimizer_kwargs={
        "n_llm_candidates": 64,
        "goal": "導電率を高くし、収縮率を低くしたい",
        "llm_config": LLMConfig(
            provider="openai",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_API_KEY",
        ),
        "llm_context": LLMContextConfig(
            variable_names=["temperature", "time", "atmosphere"],
            target_names=["conductivity", "shrinkage"],
            variable_descriptions={
                "temperature": "焼成温度。高すぎると粒成長や収縮率増加の懸念がある。",
                "time": "焼成保持時間。",
                "atmosphere": "焼成雰囲気。0=air, 1=N2, 2=Ar として符号化している。",
            },
            target_descriptions={
                "conductivity": "導電率。高いほど望ましい。",
                "shrinkage": "焼成収縮率。低いほど望ましい。",
            },
        ),
    },
)

candidates, acq_value = bo.candidate(acq_config=acq_config, opt_config=opt_config)
```

`goal` is optional when `ObjectiveConfig` already defines the optimization direction. In that case, it is only used as additional LLM prompt context.

---

## Gemini

```python
from bochan.llm import LLMConfig

llm_config = LLMConfig(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key_env="GEMINI_API_KEY",
)
```

---

## Offline candidate-set reranking

When `candidate_set` is supplied, no LLM call is made. This is useful for tests and local debugging.

```python
opt_config = OptimizeConfig(
    optimizer="llm_candidate_set",
    q=2,
    optimizer_kwargs={
        "candidate_set": [
            [840.0, 3.0, 1.0],
            [860.0, 2.5, 2.0],
            [800.0, 4.0, 1.0],
        ],
    },
)
```

`planner_response` similarly lets you test the planner path without calling a provider.

---

## FastAPI: plan only

`POST /models/plan` returns inferred configuration dictionaries without fitting a model.

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

---

## FastAPI: plan + fit + candidates

`POST /models/auto-candidates` infers configs, fits a model, stores it in the in-memory model store, and returns candidates.

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

You can still pass explicit `model_config`, `fit_config`, `acquisition_config`, or `optimize_config` fields. Explicit request configs override the LLM plan.

Do not include API key values in HTTP request bodies. Keep keys on the server side as environment variables.

---

## Current limitations

- The initial implementation reranks each LLM candidate with q=1 acquisition values.
- It is candidate-set reranking, not exact joint q-batch optimization.
- The planner returns serializable config dictionaries and warnings; applications should show these to users for review in high-risk workflows.
- Provider-backed planner and candidate generation require optional SDKs and environment variables.
