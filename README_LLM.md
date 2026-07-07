# LLM-assisted candidate generation

This document describes the experimental LLM-assisted candidate-set optimizer added to `bochan`.

The intended design is:

```text
LLM generates a broad candidate set
  -> bochan validates and repairs candidates
  -> existing acquisition function reranks candidates
  -> top-q candidates are returned
```

The LLM is not the final optimizer. It acts as a domain-aware candidate generator.

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

## Python API

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

---

## FastAPI

The serving API accepts top-level `goal`, `llm_config`, and `llm_context` fields in `/models/{model_id}/candidates` and `/models/{model_id}/ask` requests. These are internally moved into `OptimizeConfig.optimizer_kwargs`.

```json
{
  "acquisition_config": {
    "name": "NEHVI",
    "objective_config": {
      "mode": "multi_output",
      "outputs": ["conductivity", "shrinkage"],
      "directions": ["maximize", "minimize"],
      "weights": [1.0, 0.5]
    }
  },
  "optimize_config": {
    "optimizer": "llm_candidate_set",
    "q": 3,
    "raw_samples": 64,
    "optimizer_kwargs": {
      "n_llm_candidates": 64
    }
  },
  "goal": "導電率を高くし、収縮率を低くしたい",
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

Do not include API key values in HTTP request bodies. Keep keys on the server side as environment variables.

---

## Current limitations

- The initial implementation reranks each LLM candidate with q=1 acquisition values.
- It is candidate-set reranking, not exact joint q-batch optimization.
- Natural-language goal to full `ObjectiveConfig` / `AcquisitionConfig` planning is scaffolded via prompt helpers, but not yet automatically connected to `bo.candidate(goal=...)`.
