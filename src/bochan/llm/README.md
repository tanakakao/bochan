# bochan.llm

`bochan.llm` は、LLM を Bayesian optimization の候補生成補助として使うための実験的なモジュールです。

基本方針は次の通りです。

```text
LLM = 候補集合を広めに生成する補助器
bochan / BoTorch = surrogate model と acquisition function による最終選抜器
```

LLM が最終候補を直接決めるのではなく、LLM が出した候補集合を既存の EI / UCB / NEHVI / NIPV などで再スコアします。

---

## Installation

LLM provider SDK は optional dependency です。

```bash
pip install -e ".[llm]"
```

OpenAI だけを手動で入れる場合です。

```bash
pip install openai
```

Gemini だけを手動で入れる場合です。

```bash
pip install google-genai
```

API key はコードに直接書かず、環境変数で渡すことを推奨します。

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

Windows PowerShell では次のように設定できます。

```powershell
setx OPENAI_API_KEY "..."
setx GEMINI_API_KEY "..."
```

---

## Python API example

`optimizer="llm_candidate_set"` を使うと、LLM 候補集合を生成し、既存 acquisition function で上位 `q` 件を選びます。

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
            temperature=0.2,
        ),
        "llm_context": LLMContextConfig(
            variable_names=["temperature", "time", "atmosphere"],
            target_names=["conductivity", "shrinkage"],
            variable_descriptions={
                "temperature": "焼成温度。高すぎると粒成長や収縮率増加の懸念がある。",
                "time": "焼成保持時間。長いほど焼結が進む可能性がある。",
                "atmosphere": "焼成雰囲気。0=air, 1=N2, 2=Ar として符号化している。",
            },
            target_descriptions={
                "conductivity": "導電率。高いほど望ましい。",
                "shrinkage": "焼成収縮率。低いほど望ましい。",
            },
            domain_notes=[
                "高温かつ長時間では収縮率が大きくなる可能性がある。",
                "N2 または Ar では導電率が高い傾向がある。",
            ],
            candidate_policy="良好条件近傍と未探索領域を混ぜて、多様な候補集合を出す。",
        ),
    },
)

candidates, acq_value = bo.candidate(acq_config=acq_config, opt_config=opt_config)
```

`goal` は必須ではありません。`ObjectiveConfig` に最大化 / 最小化が明示されている場合は、目的方向はそちらが優先されます。`goal` は LLM prompt の補足文脈として使われます。

---

## Gemini example

```python
from bochan.llm import LLMConfig

llm_config = LLMConfig(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key_env="GEMINI_API_KEY",
)
```

`provider` 以外の候補生成フローは OpenAI と同じです。

---

## Offline / test mode with explicit candidate_set

`candidate_set` を渡すと LLM は呼ばれません。API key なしで reranking と repair の挙動を確認できます。

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

## FastAPI usage

`POST /models/{model_id}/candidates` と `POST /models/{model_id}/ask` で同じ仕組みを使えます。

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
    "api_key_env": "OPENAI_API_KEY",
    "temperature": 0.2
  },
  "llm_context": {
    "variable_names": ["temperature", "time", "atmosphere"],
    "target_names": ["conductivity", "shrinkage"],
    "variable_descriptions": {
      "temperature": "焼成温度。高すぎると粒成長や収縮率増加の懸念がある。",
      "time": "焼成保持時間。",
      "atmosphere": "焼成雰囲気。0=air, 1=N2, 2=Ar として符号化している。"
    },
    "target_descriptions": {
      "conductivity": "導電率。高いほど望ましい。",
      "shrinkage": "焼成収縮率。低いほど望ましい。"
    },
    "domain_notes": [
      "高温長時間では収縮率が大きくなりやすい。"
    ]
  },
  "tensor_options": {
    "dtype": "float64",
    "device": "cpu"
  }
}
```

FastAPI request に API key 本体は入れないでください。サーバー側の環境変数から読みます。

---

## Prompt shape

内部では概ね次の情報を JSON prompt として渡します。

```text
- LLM は最終候補を決めない
- 最終選抜は acquisition function が行う
- bounds と制約を守る
- 候補集合を n 個出す
- 良好条件近傍と未探索領域を混ぜる
- 変数名、bounds、変数説明、目的変数説明、domain_notes
```

出力は次の形式を想定します。

```json
{
  "candidates": [
    {
      "x": [840.0, 3.0, 1.0],
      "reason": "良好条件近傍で、温度を少し下げて収縮率抑制を狙う。"
    }
  ]
}
```

`x` は基本的に `variable_names` の順序に対応する list を推奨します。object 形式で返す場合は `LLMContextConfig.variable_names` が必要です。

---

## Validation and reranking

LLM 出力はそのまま採用しません。

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

初期実装では、各候補を `q=1` acquisition value で個別評価し、上位 `q` 件を返します。joint q-batch optimization ではなく、candidate-set reranking として扱ってください。

---

## Goal planner status

`build_goal_planner_prompt()` は用意していますが、自然言語 `goal` から `ObjectiveConfig` / `AcquisitionConfig` を自動生成する planner はまだ本格接続していません。

現時点の推奨は次の分担です。

```text
maximize / minimize / weights / constraints:
  -> ObjectiveConfig / OutcomeConstraintConfig に明示する

変数の意味、単位、実験上の注意、候補生成方針:
  -> LLMContextConfig に任意で補足する

goal:
  -> LLM 候補生成 prompt の補足文脈として使う
```
