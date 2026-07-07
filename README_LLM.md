# LLM-assisted planning and candidate generation

This document describes the experimental LLM-assisted planner and candidate-set optimizer.

The recommended Python API is to define LLM information once with `LLMSettings` and reuse it for model selection, candidate generation, and Study-level suggestions.

```text
LLMSettings
  -> shared goal / provider / domain context
  -> BayesianOptimizer / BochanStudy
  -> ModelConfig(model_type="llm_selected")
  -> OptimizeConfig(optimizer="llm_candidate_set")
  -> study.suggest(mode="config")
```

The LLM is not the final optimizer. It proposes configuration dictionaries or candidate pools; bochan still fits the selected model, validates / repairs candidates, and reranks them with the acquisition function.

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

## Shared LLM settings

Define the shared LLM settings once.

```python
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
            "temperature": "焼成温度。高すぎると粒成長や収縮率増加の懸念がある。",
            "time": "焼成保持時間。長いほど焼結が進む可能性がある。",
            "atmosphere": "焼成雰囲気。0=air, 1=N2, 2=Arとして符号化している。",
        },
        target_descriptions={
            "conductivity": "導電率。高いほど望ましい。",
            "shrinkage": "焼成収縮率。低いほど望ましい。",
        },
        domain_notes=[
            "高温長時間では収縮率が大きくなりやすい。",
        ],
    ),
    n_llm_candidates=50,
)
```

Individual `model_kwargs` / `optimizer_kwargs` override shared settings when needed.

---

## BayesianOptimizer usage

Use shared settings for model selection by setting `model_type="llm_selected"`.

```python
from bochan.api import BayesianOptimizer, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(model_type="llm_selected"),
    bounds=bounds,
    llm_settings=llm_settings,
)

bo.fit(train_X, train_Y)
```

Use the same shared settings for candidate generation by setting `optimizer="llm_candidate_set"`. You do not need to repeat `goal`, `llm_config`, or `llm_context`.

```python
from bochan.api import AcquisitionConfig, ObjectiveConfig, OptimizeConfig

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
    raw_samples=64,
)

candidates, acq_value = bo.candidate(acq_config=acq_config, opt_config=opt_config)
```

You can also configure LLM settings later.

```python
bo.configure_llm(
    goal="導電率を高くし、収縮率を低くしたい",
    llm_config=LLMConfig(provider="openai", model="gpt-4.1-mini"),
    llm_context=LLMContextConfig(
        variable_names=["temperature", "time", "atmosphere"],
        target_names=["conductivity", "shrinkage"],
    ),
    n_llm_candidates=50,
)
```

---

## BochanStudy usage

`BochanStudy` can now carry the same `LLMSettings` and pass it into the internal `BayesianOptimizer` used by `ask()`.

```python
from bochan.api import BochanStudy, ModelConfig, OptimizeConfig

study = BochanStudy(
    model_config=ModelConfig(model_type="llm_selected"),
    opt_config=OptimizeConfig(optimizer="llm_candidate_set", q=3),
    bounds=bounds,
    llm_settings=llm_settings,
)

study.add_observations(train_X, train_Y)
batch = study.ask(return_batch=True)
```

### Study-level config suggestion

Use `study.suggest(mode="config")` when you want the LLM to propose Study settings from the current history before generating candidates.

```python
suggestion = study.suggest(mode="config")

print(suggestion.model_config)
print(suggestion.fit_config)
print(suggestion.acq_config)
print(suggestion.opt_config)
print(suggestion.reasoning_summary)
print(suggestion.warnings)
```

The suggestion is not automatically applied unless requested.

```python
study.apply_suggestion(suggestion)
```

Or apply immediately:

```python
suggestion = study.suggest(mode="config", apply=True)
```

This is the recommended review flow for important experiments:

```text
study.suggest(mode="config")
  -> review model / acquisition / optimizer proposal
  -> study.apply_suggestion(...)
  -> study.ask(...)
```

---

## Offline smoke tests without API keys

Use `planner_response` and `candidate_set` to test the full wiring without calling OpenAI or Gemini.

```python
import torch

from bochan.api import AcquisitionConfig, BayesianOptimizer, ModelConfig, ObjectiveConfig, OptimizeConfig
from bochan.llm import LLMSettings

train_X = torch.tensor(
    [
        [800.0, 2.0, 0.0],
        [850.0, 3.0, 1.0],
        [900.0, 4.0, 2.0],
        [830.0, 2.5, 1.0],
    ],
    dtype=torch.double,
)
train_Y = torch.tensor(
    [
        [10.2, 5.1],
        [12.4, 4.8],
        [11.0, 7.2],
        [11.8, 4.9],
    ],
    dtype=torch.double,
)
bounds = torch.tensor(
    [
        [700.0, 1.0, 0.0],
        [950.0, 5.0, 2.0],
    ],
    dtype=torch.double,
)

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
        "reasoning_summary": "Use a base regression model for two continuous outputs.",
    },
    candidate_set=[
        [840.0, 3.0, 1.0],
        [860.0, 2.5, 2.0],
        [800.0, 4.0, 1.0],
        [920.0, 5.0, 2.0],
    ],
    n_llm_candidates=4,
)

bo = BayesianOptimizer(
    model_config=ModelConfig(model_type="llm_selected"),
    bounds=bounds,
    llm_settings=llm_settings,
)
bo.fit(train_X, train_Y)
```

Run the complete BayesianOptimizer smoke test:

```bash
python examples/llm_same_pattern.py
```

Run the complete BochanStudy smoke test:

```bash
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

`plan_configs()` is still available when you only want a configuration proposal and do not want to create a `BayesianOptimizer` or `BochanStudy` yet.

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

---

## FastAPI

The serving API still supports explicit request-level LLM fields.

- `POST /models/plan`: return inferred config dictionaries without fitting a model.
- `POST /models/auto-candidates`: infer configs, fit a model, and return candidates.
- `POST /models/{model_id}/candidates`: use `optimizer="llm_candidate_set"` with top-level `goal`, `llm_config`, and `llm_context`.

Do not include API key values in HTTP request bodies. Keep keys on the server side as environment variables.

---

## Current limitations

- `study.suggest(mode="config")` is implemented; `mode="next_action"` is intentionally left for a later phase.
- The candidate-set optimizer reranks each LLM candidate with q=1 acquisition values.
- It is candidate-set reranking, not exact joint q-batch optimization.
- The planner returns serializable config dictionaries and warnings; applications should show these to users for review in important workflows.
- Provider-backed planner and candidate generation require optional SDKs and environment variables.
