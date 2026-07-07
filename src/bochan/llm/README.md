# bochan.llm

`bochan.llm` provides experimental LLM helpers for two workflows.

```text
1. planner: natural language goal -> bochan config dictionaries
2. candidate generator: LLM candidate set -> validation / repair -> acquisition reranking
```

The LLM is not the final optimizer. It proposes settings or candidate pools; bochan still fits the model and scores candidates with the selected acquisition function.

---

## Installation

```bash
pip install -e ".[llm]"
```

Provider keys are read from environment variables.

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

## Mode 1: settings only

Use `plan_configs(..., mode="model_config")` to ask the LLM for model and fit settings without fitting a model.

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
    ),
)

print(plan["model_config"])
print(plan["fit_config"])
print(plan["warnings"])
```

`planner_response` can be supplied for offline tests. In that case no provider is called.

---

## Mode 2: plan through candidate generation

FastAPI exposes `POST /models/auto-candidates` for a full run:

```text
LLM planner -> model fit -> acquisition creation -> candidate generation
```

For Python workflows, use the same pieces explicitly:

```text
plan_configs(..., mode="full")
  -> BayesianOptimizer(...).fit(train_X, train_Y)
  -> optimizer.candidate(acq_config, opt_config)
```

---

## Candidate-set optimizer

`optimizer="llm_candidate_set"` asks an LLM for many candidates, then reranks them with the existing acquisition function.

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
        "llm_config": LLMConfig(provider="openai", model="gpt-4.1-mini"),
        "llm_context": LLMContextConfig(
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
    },
)

candidates, acq_value = bo.candidate(acq_config=acq_config, opt_config=opt_config)
```

When `ObjectiveConfig` already provides maximize / minimize directions, `goal` is only additional prompt context.

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

When `candidate_set` is supplied, no LLM call is made.

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

---

## FastAPI

Settings only:

```json
{
  "goal": "導電率を高くし、収縮率を低くしたい",
  "mode": "model_config",
  "train_X": [[800.0, 2.0, 0.0], [850.0, 3.0, 1.0]],
  "train_Y": [[10.2, 5.1], [12.4, 4.8]],
  "bounds": [[700.0, 1.0, 0.0], [950.0, 5.0, 2.0]],
  "llm_config": {"provider": "openai", "model": "gpt-4.1-mini"},
  "llm_context": {
    "variable_names": ["temperature", "time", "atmosphere"],
    "target_names": ["conductivity", "shrinkage"]
  }
}
```

Full run:

```json
{
  "goal": "導電率を高くし、収縮率を低くしたい",
  "train_X": [[800.0, 2.0, 0.0], [850.0, 3.0, 1.0], [900.0, 4.0, 2.0]],
  "train_Y": [[10.2, 5.1], [12.4, 4.8], [11.0, 7.2]],
  "bounds": [[700.0, 1.0, 0.0], [950.0, 5.0, 2.0]],
  "llm_config": {"provider": "openai", "model": "gpt-4.1-mini"},
  "llm_context": {
    "variable_names": ["temperature", "time", "atmosphere"],
    "target_names": ["conductivity", "shrinkage"]
  }
}
```

Do not put API key values in HTTP request bodies. Keep keys on the server side.

---

## Validation and reranking

```text
LLM output or explicit candidate_set
  -> JSON parse
  -> tensor conversion
  -> finite-value filtering
  -> bounds clipping
  -> duplicate removal
  -> fixed_features
  -> post_processing_func / CandidateRepairConfig
  -> linear constraint filtering
  -> acquisition reranking
  -> top-q candidates
```

Current limitations:

- candidate-set optimizer uses q=1 acquisition values for reranking
- this is candidate-set reranking, not exact joint q-batch optimization
- provider-backed calls require optional SDKs and environment variables
