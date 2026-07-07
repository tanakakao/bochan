# bochan.llm

`bochan.llm` provides experimental LLM helpers for these workflows:

```text
1. planner: natural language goal -> bochan config dictionaries
2. candidate generator: LLM candidate set -> validation / repair -> acquisition reranking
3. study suggestion: BochanStudy history -> model / acquisition / optimizer config proposal
```

The recommended Python API is `LLMSettings`. Define the shared LLM context once, then reuse it for model selection, candidate generation, and Study-level suggestions.

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

## Shared settings pattern

```python
from bochan.api import AcquisitionConfig, BayesianOptimizer, ModelConfig, ObjectiveConfig, OptimizeConfig
from bochan.llm import LLMConfig, LLMContextConfig, LLMSettings

llm_settings = LLMSettings(
    goal="導電率を高くし、収縮率を低くしたい",
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
    ),
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
    n_llm_candidates=50,
)

bo = BayesianOptimizer(
    model_config=ModelConfig(model_type="llm_selected"),
    bounds=bounds,
    llm_settings=llm_settings,
)
bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        outputs=[0, 1],
        directions=["maximize", "minimize"],
        weights=[1.0, 0.5],
    ),
)
opt_config = OptimizeConfig(
    optimizer="llm_candidate_set",
    q=3,
)

candidates, acq_value = bo.candidate(acq_config=acq_config, opt_config=opt_config)
```

In this pattern:

```text
ModelConfig(model_type="llm_selected")
  -> consumes LLMSettings.model_kwargs()

OptimizeConfig(optimizer="llm_candidate_set")
  -> consumes LLMSettings.optimizer_kwargs()

BochanStudy.suggest(mode="config")
  -> consumes LLMSettings + current trial history
```

Individual `model_kwargs` / `optimizer_kwargs` override shared settings.

---

## BochanStudy suggestions

`BochanStudy` can carry the same `LLMSettings` and use it for both `ask()` and `suggest()`.

```python
from bochan.api import BochanStudy, ModelConfig, OptimizeConfig

study = BochanStudy(
    model_config=ModelConfig(model_type="llm_selected"),
    opt_config=OptimizeConfig(optimizer="llm_candidate_set", q=3),
    bounds=bounds,
    llm_settings=llm_settings,
)

study.add_observations(train_X, train_Y)
```

Ask the LLM for a config proposal based on current study history.

```python
suggestion = study.suggest(mode="config")

print(suggestion.model_config)
print(suggestion.fit_config)
print(suggestion.acq_config)
print(suggestion.opt_config)
print(suggestion.reasoning_summary)
print(suggestion.warnings)
```

Apply it explicitly after review.

```python
study.apply_suggestion(suggestion)
batch = study.ask(return_batch=True)
```

Or apply immediately.

```python
suggestion = study.suggest(mode="config", apply=True)
```

---

## Configure later

```python
bo = BayesianOptimizer(
    model_config=ModelConfig(model_type="llm_selected"),
    bounds=bounds,
)

bo.configure_llm(
    goal="導電率を高くし、収縮率を低くしたい",
    llm_config=LLMConfig(provider="openai", model="gpt-4.1-mini"),
    llm_context=LLMContextConfig(
        variable_names=["temperature", "time", "atmosphere"],
        target_names=["conductivity", "shrinkage"],
    ),
    n_llm_candidates=50,
)

bo.fit(train_X, train_Y)
```

---

## Offline smoke test

No provider call is made when `planner_response` and `candidate_set` are supplied.

```python
from bochan.llm import LLMSettings

llm_settings = LLMSettings(
    goal="導電率を高くし、収縮率を低くしたい",
    planner_response={
        "model_config": {
            "task_type": "regression",
            "model_type": "base",
            "outcome_transform": True,
        },
        "fit_config": {"method": "auto"},
        "warnings": ["offline planner response"],
    },
    candidate_set=[
        [840.0, 3.0, 1.0],
        [860.0, 2.5, 2.0],
        [800.0, 4.0, 1.0],
    ],
    n_llm_candidates=3,
)
```

Complete examples are available at:

```bash
python examples/llm_same_pattern.py
python examples/llm_study_suggestion.py
```

---

## Gemini

Only `LLMConfig` changes.

```python
from bochan.llm import LLMConfig

llm_config = LLMConfig(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key_env="GEMINI_API_KEY",
)
```

---

## Lower-level planner API

`plan_configs()` is still available when you only want a config proposal.

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
```

---

## FastAPI

The serving API still supports request-level LLM fields:

- `POST /models/plan`: return inferred config dictionaries without fitting a model.
- `POST /models/auto-candidates`: infer configs, fit a model, and return candidates.
- `POST /models/{model_id}/candidates`: use `optimizer="llm_candidate_set"` with top-level `goal`, `llm_config`, and `llm_context`.

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

- `study.suggest(mode="config")` is implemented; `mode="next_action"` is a future phase
- candidate-set optimizer uses q=1 acquisition values for reranking
- this is candidate-set reranking, not exact joint q-batch optimization
- provider-backed calls require optional SDKs and environment variables
