# LLM-assisted planning and candidate generation

`bochan` では LLM を、ベイズ最適化そのものの代わりではなく、設定案や候補集合を作る補助役として利用できます。

LLM が担当できる処理は大きく3つあります。

| 役割 | 主なAPI | LLMが提案するもの | bochan側の処理 |
|---|---|---|---|
| モデル選択 | `ModelConfig(model_type="llm_selected")` | `ModelConfig`, `FitConfig` | 提案されたモデルを構築・学習 |
| Study設定提案 | `study.suggest(mode="config")` | モデル、学習設定、獲得関数、候補最適化設定 | 内容を型付きConfigへ変換し、任意で適用 |
| 候補集合生成 | `OptimizeConfig(optimizer="llm_candidate_set")` | 探索候補のプール | 制約確認、修復、獲得関数による再順位付け |

重要なのは、LLM が最終的な最適化器ではない点です。

```text
自然言語の目的・データ情報・実験履歴
  -> LLM が設定案または候補集合を生成
  -> bochan がモデル構築、獲得関数構築、制約処理、候補評価を実行
```

---

## 獲得関数をLLMに選ばせることはできますか

できます。

推奨APIは `BochanStudy.suggest(mode="config")` です。このメソッドは、現在のデータ、実験履歴、目的、既存設定をLLMへ渡し、次の4種類の設定を提案させます。

- `ModelConfig`
- `FitConfig`
- `AcquisitionConfig`
- `OptimizeConfig`

```python
suggestion = study.suggest(mode="config")

print(suggestion.model_config)
print(suggestion.fit_config)
print(suggestion.acq_config)
print(suggestion.opt_config)
print(suggestion.reasoning_summary)
print(suggestion.warnings)
```

獲得関数の提案は自動適用されません。内容を確認してから適用するのが基本です。

```python
study.apply_suggestion(suggestion)
```

確認を省略して即時適用する場合は、次のようにします。

```python
suggestion = study.suggest(mode="config", apply=True)
```

重要な実験では、次の流れを推奨します。

```text
study.suggest(mode="config")
  -> acquisition_config と warnings を確認
  -> study.apply_suggestion(...)
  -> study.ask(...)
```

---

## インストール

LLM連携用の追加依存関係をインストールします。

```bash
pip install -e ".[llm]"
```

導入される主なSDKは次のとおりです。

```text
openai
google-genai
```

開発・テストも含めて導入する場合は次のようにします。

```bash
pip install -e ".[llm,test]"
```

---

## APIキーの設定

APIキーはコード、Notebook、JSON、Gitリポジトリに直接保存せず、環境変数で渡してください。

Linux / macOS:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

Windows PowerShellの現在のセッションだけに設定する場合:

```powershell
$env:OPENAI_API_KEY = "..."
$env:GEMINI_API_KEY = "..."
```

Windowsに永続設定する場合:

```powershell
setx OPENAI_API_KEY "..."
setx GEMINI_API_KEY "..."
```

`setx` で設定した値は、新しく起動したPowerShell、コマンドプロンプト、Jupyterプロセスから有効になります。

---

## 社内プロキシ・自己署名証明書への対応

社内プロキシがHTTPS通信を中継している環境では、次のようなエラーが発生することがあります。

```text
SSL: CERTIFICATE_VERIFY_FAILED
self-signed certificate in certificate chain
```

会社のルート証明書またはCA bundleをPEM形式で用意し、`ca_bundle_path` に指定してください。

```python
from bochan.llm import LLMConfig

llm_config = LLMConfig(
    provider="openai",
    model="gpt-4.1-mini",
    api_key_env="OPENAI_API_KEY",
    ca_bundle_path=r"C:\certificates\company-ca.pem",
)
```

環境変数でも指定できます。

```powershell
$env:SSL_CERT_FILE = "C:\certificates\company-ca.pem"
```

次の環境変数を順番に参照します。

1. `SSL_CERT_FILE`
2. `REQUESTS_CA_BUNDLE`
3. `CURL_CA_BUNDLE`

一時的な疎通確認だけであれば、証明書検証を無効化できます。

```python
LLMConfig(
    provider="openai",
    model="gpt-4.1-mini",
    ssl_verify=False,
)
```

`ssl_verify=False` はHTTPS証明書の検証を無効化するため、通常運用では使用しないでください。実行時には警告が表示されます。

---

## 共通のLLM設定

モデル選択、獲得関数選択、候補生成で共通利用する情報は `LLMSettings` にまとめます。

```python
from bochan.llm import LLMConfig, LLMContextConfig, LLMSettings

llm_settings = LLMSettings(
    goal="導電率を高くし、収縮率を低くしたい",
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
        ca_bundle_path=r"C:\certificates\company-ca.pem",
    ),
    llm_context=LLMContextConfig(
        variable_names=["temperature", "time", "atmosphere"],
        target_names=["conductivity", "shrinkage"],
        variable_descriptions={
            "temperature": "焼成温度。高すぎると粒成長や収縮増加の懸念がある。",
            "time": "焼成保持時間。長いほど焼結が進む可能性がある。",
            "atmosphere": "焼成雰囲気。0=air、1=N2、2=Arとして符号化している。",
        },
        target_descriptions={
            "conductivity": "導電率。高いほど望ましい。",
            "shrinkage": "焼成収縮率。低いほど望ましい。",
        },
        domain_notes=[
            "高温かつ長時間では収縮率が大きくなりやすい。",
            "atmosphereは整数値だが連続変数ではなくカテゴリ変数である。",
        ],
        candidate_policy="既存条件から大きく離れすぎない候補を優先する。",
    ),
    n_llm_candidates=50,
)
```

### LLMに渡される主な情報

LLM plannerには、次の情報が渡されます。

- 自然言語で記述した `goal`
- `train_X` と `train_Y` のshape
- `bounds`
- 説明変数名と目的変数名
- 各変数の意味、単位、カテゴリ情報
- 実験上の注意事項
- 完了、pending、failed trial数
- 直近のtrial履歴
- 現在設定されているモデル、獲得関数、候補最適化設定

LLMはデータ配列全体をそのまま読むのではなく、主にメタデータとStudyの要約を利用して設定を提案します。

---

## 推奨例: モデルと獲得関数をまとめて提案させる

```python
import torch

from bochan.api import BochanStudy
from bochan.llm import LLMConfig, LLMContextConfig, LLMSettings

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
    goal="conductivityを最大化し、shrinkageを最小化したい。実験ノイズを考慮する。",
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
    ),
    llm_context=LLMContextConfig(
        variable_names=["temperature", "time", "atmosphere"],
        target_names=["conductivity", "shrinkage"],
        target_descriptions={
            "conductivity": "最大化する連続目的変数。",
            "shrinkage": "最小化する連続目的変数。",
        },
        domain_notes=[
            "観測値には実験ノイズが含まれる。",
            "atmosphereはカテゴリ変数である。",
        ],
    ),
)

study = BochanStudy(
    bounds=bounds,
    llm_settings=llm_settings,
)
study.add_observations(train_X, train_Y)

suggestion = study.suggest(mode="config")

print("model:", suggestion.model_config)
print("fit:", suggestion.fit_config)
print("acquisition:", suggestion.acq_config)
print("optimizer:", suggestion.opt_config)
print("reason:", suggestion.reasoning_summary)
print("warnings:", suggestion.warnings)

study.apply_suggestion(suggestion)

batch = study.ask(q=3, return_batch=True)
print(batch.candidates)
print(batch.acq_value)
```

この例では、多目的、最大化・最小化、実験ノイズという情報から、LLMが `NEHVI` などの候補を提案できます。

LLMの出力は確率的であるため、必ず同じ獲得関数が選ばれるとは限りません。`reasoning_summary` と `warnings` を確認してください。

---

## 獲得関数だけをLLMに選ばせる

モデルと学習条件を固定し、獲得関数だけをLLM提案に置き換えることもできます。

```python
from bochan.api import BochanStudy, FitConfig, ModelConfig, OptimizeConfig

study = BochanStudy(
    model_config=ModelConfig(
        task_type="regression",
        model_type="base",
        outcome_transform=True,
    ),
    fit_config=FitConfig(maxiter=128),
    opt_config=OptimizeConfig(
        optimizer="optimize_acqf",
        q=3,
        raw_samples=256,
        num_restarts=10,
    ),
    bounds=bounds,
    llm_settings=llm_settings,
)
study.add_observations(train_X, train_Y)

suggestion = study.suggest(mode="config")
print(suggestion.acq_config)
print(suggestion.reasoning_summary)
print(suggestion.warnings)

study.apply_suggestion(
    suggestion,
    model_config=False,
    fit_config=False,
    acq_config=True,
    opt_config=False,
)
```

この方法では、LLMがモデルや候補最適化設定も提案していても、Studyへ反映されるのは `AcquisitionConfig` だけです。

同様に、候補最適化設定だけを適用できます。

```python
study.apply_suggestion(
    suggestion,
    model_config=False,
    fit_config=False,
    acq_config=False,
    opt_config=True,
)
```

---

## LLMが選択対象とする獲得関数

plannerは目的とタスクに応じて、主に次の候補から選択します。

| 用途 | 主な候補 |
|---|---|
| 単目的ベイズ最適化 | `EI`, `UCB`, `PI`, `TS` |
| 多目的ベイズ最適化 | `NEHVI`, `EHVI`, `NParEGO` |
| 回帰Active Learning | `NIPV` |
| 分類Active Learning | `entropy`, `BALD`, `margin`, `variance` |
| Level-set estimation | `straddle`, `ICU`, `boundaryvariance`, `levelset` |

plannerには、次のような選択指針を与えています。

- 単目的で改善量を重視する場合は `EI`
- 探索を明示的に強める場合は `UCB`
- 改善確率そのものを重視する場合は `PI`
- posterior samplingを利用する場合は `TS`
- ノイズを含む多目的実験では `NEHVI`
- ほぼノイズのない多目的問題では `EHVI`
- スカラー化した多目的探索を意図する場合は `NParEGO`
- 分類Active Learningでは `entropy`, `BALD`, `margin`, `variance`
- 境界・しきい値探索では `straddle`, `ICU`, `boundaryvariance`, `levelset`

実際に利用可能かどうかは、選択されたモデル、task type、output構成、登録済みacquisition registryにも依存します。

### 獲得関数と候補最適化器は別物です

次の2つは分けて考えてください。

```python
AcquisitionConfig(name="NEHVI")
OptimizeConfig(optimizer="optimize_acqf")
```

- `AcquisitionConfig`: 候補点の価値を評価する関数
- `OptimizeConfig`: その獲得関数をどの方法で最大化するか

主な候補最適化backendは次のとおりです。

```text
optimize_acqf
llm_candidate_set
torch
evo
nsgaii
thompson_sampling
```

`nsgaii` は候補最適化backendであり、獲得関数名ではありません。

---

## 既存設定をLLMへ渡す意味

`BochanStudy` に既存設定がある場合、それらもLLMへ渡されます。

```python
study = BochanStudy(
    model_config=current_model_config,
    acq_config=current_acq_config,
    opt_config=current_opt_config,
    bounds=bounds,
    llm_settings=llm_settings,
)
```

plannerには、明示的な既存設定を理由なく変更しないよう指示しています。

ただし、自然言語の目的と既存設定が矛盾する場合は変更案が返ることがあります。例えば、多目的最適化が目的であるのに既存設定が単目的 `EI` の場合、`NEHVI` などへの変更が提案される可能性があります。

---

## BayesianOptimizerでのモデル自動選択

`BayesianOptimizer` で `model_type="llm_selected"` を指定すると、`fit()` の直前にモデルと学習設定をLLMに選択させます。

```python
from bochan.api import BayesianOptimizer, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(model_type="llm_selected"),
    bounds=bounds,
    llm_settings=llm_settings,
)

bo.fit(train_X, train_Y)
```

この経路で自動適用される中心的な設定は `ModelConfig` と `FitConfig` です。

獲得関数まで含めて提案・適用したい場合は、次のいずれかを使用してください。

1. 推奨: `BochanStudy.suggest(mode="config")`
2. 低レベルAPI: `plan_configs(mode="full")`

---

## LLMで候補集合を作る

獲得関数の選択と、LLMによる候補集合生成は別機能です。

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

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
)
```

この処理は次の流れです。

```text
LLMが候補集合を提案
  -> boundsと入力形式を確認
  -> 制約違反候補を除外または修復
  -> 各候補を獲得関数で評価
  -> 上位q件を返す
```

`LLMSettings` に設定した `goal`, `llm_config`, `llm_context`, `n_llm_candidates` は、自動的に `llm_candidate_set` backendへ渡されます。

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

## BochanStudyで候補生成まで行う

```python
from bochan.api import BochanStudy, ModelConfig, OptimizeConfig

study = BochanStudy(
    model_config=ModelConfig(model_type="llm_selected"),
    opt_config=OptimizeConfig(
        optimizer="llm_candidate_set",
        q=3,
    ),
    bounds=bounds,
    llm_settings=llm_settings,
)

study.add_observations(train_X, train_Y)

suggestion = study.suggest(mode="config")
study.apply_suggestion(suggestion)

batch = study.ask(return_batch=True)
print(batch.candidates)
print(batch.trial_ids)
```

次の実験結果が得られたら `tell()` で登録します。

```python
study.tell(batch.trial_ids, observed_Y)
```

その後、再度 `suggest()` を実行すると、追加された実験履歴も含めて設定案を更新できます。

---

## 低レベルplanner API

設定案だけが必要で、まだ `BayesianOptimizer` や `BochanStudy` を作成したくない場合は `plan_configs()` を利用できます。

```python
from bochan.llm import LLMConfig, LLMContextConfig, plan_configs

plan = plan_configs(
    goal="導電率を高くし、収縮率を低くしたい",
    train_X=train_X,
    train_Y=train_Y,
    bounds=bounds,
    mode="full",
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
    ),
    llm_context=LLMContextConfig(
        variable_names=["temperature", "time", "atmosphere"],
        target_names=["conductivity", "shrinkage"],
    ),
)

print(plan["model_config"])
print(plan["fit_config"])
print(plan["acquisition_config"])
print(plan["optimize_config"])
print(plan["reasoning_summary"])
print(plan["warnings"])
```

`plan_configs()` の戻り値はシリアライズ可能な辞書です。型付きConfigへ安全に変換し、そのままStudyへ適用したい場合は `BochanStudy.suggest()` を推奨します。

### model_configモード

モデル選択だけが必要な場合は次のようにします。

```python
plan = plan_configs(
    goal="予測誤差が小さい回帰モデルを選びたい",
    train_X=train_X,
    train_Y=train_Y,
    bounds=bounds,
    mode="model_config",
    llm_config=llm_config,
)
```

通常の `BayesianOptimizer(model_type="llm_selected")` も、このモデル選択中心の経路を利用します。

---

## APIを呼ばないオフラインテスト

`planner_response` と `candidate_set` を明示すれば、OpenAIやGeminiを呼ばずに一連の配線をテストできます。

```python
from bochan.api import BochanStudy
from bochan.llm import LLMSettings

study = BochanStudy(
    bounds=bounds,
    llm_settings=LLMSettings(
        goal="yを大きくしたい",
        planner_response={
            "model_config": {
                "task_type": "regression",
                "model_type": "base",
                "outcome_transform": True,
            },
            "fit_config": {
                "method": "auto",
                "skip_fit": True,
            },
            "acquisition_config": {
                "name": "UCB",
                "acqf_kwargs": {
                    "beta": 0.2,
                },
            },
            "optimize_config": {
                "optimizer": "llm_candidate_set",
                "q": 1,
                "optimizer_kwargs": {
                    "candidate_set": [
                        [0.2, 0.3],
                        [0.8, 0.9],
                    ],
                    "n_llm_candidates": 2,
                },
            },
            "warnings": ["offline study suggestion"],
            "reasoning_summary": "Use simple regression and UCB for the smoke test.",
        },
    ),
)

study.add_observations(train_X, train_Y)
suggestion = study.suggest(mode="config")
study.apply_suggestion(suggestion)
```

サンプルスクリプト:

```bash
python examples/llm_same_pattern.py
python examples/llm_study_suggestion.py
```

---

## Geminiを使用する

`LLMConfig` のproviderとmodelを変更します。

```python
from bochan.llm import LLMConfig

llm_config = LLMConfig(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key_env="GEMINI_API_KEY",
)
```

その他の `LLMSettings`, `BochanStudy`, `plan_configs()` の使い方は同じです。

---

## FastAPI

FastAPIでは、次のLLM関連endpointを利用できます。

- `POST /models/plan`: 学習せずに設定案を返す
- `POST /models/auto-candidates`: 設定提案、モデル学習、候補生成をまとめて行う
- `POST /models/{model_id}/candidates`: 学習済みモデルに対して候補を生成する

`optimizer="llm_candidate_set"` を使う場合は、request levelの `goal`, `llm_config`, `llm_context` を利用できます。

APIキーそのものはHTTP request bodyへ含めず、サーバー側の環境変数として保持してください。

---

## 注意点と制限

- LLMの提案は確率的であり、同じ入力でも設定案が変わる可能性があります。
- LLMは設定候補を提案しますが、その設定の性能を保証するものではありません。
- `warnings` が空であっても、重要な実験では人が設定を確認してください。
- 目的変数の最大化・最小化、output index、カテゴリ列は `LLMContextConfig` に明示するほど安定します。
- `study.suggest(mode="config")` は設定案を返しますが、既定では自動適用しません。
- `mode="next_action"` は未実装です。
- `llm_candidate_set` は、候補ごとに `q=1` の獲得関数値で再順位付けします。
- `llm_candidate_set` は厳密なjoint q-batch最適化ではありません。
- providerを利用するplannerと候補生成には、各SDK、APIキー、ネットワーク接続が必要です。
- 社内ネットワークでTLS inspectionが行われる場合は、会社のCA bundleを設定してください。
