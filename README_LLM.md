# LLM-assisted planning and candidate generation

`bochan` では、LLMをベイズ最適化そのものの代わりではなく、設定設計と候補集合作成の補助役として利用できます。

LLMが担当できる処理は大きく4つあります。

| 役割 | API | LLMが提案するもの | bochan側の処理 |
|---|---|---|---|
| 全設定の提案 | `BayesianOptimizer.suggest_all()` | モデル、学習、獲得関数、目的関数、候補最適化 | 型付きConfigへ変換し、任意で適用 |
| 設定カテゴリごとの提案 | `suggest_model()` / `suggest_acquisition()` / `suggest_optimizer()` | 指定したカテゴリだけ | 既存の他カテゴリを維持して任意適用 |
| Study設定の提案 | `BochanStudy.suggest(mode="config")` | Study履歴を含む全設定 | trial履歴を考慮して任意適用 |
| 候補集合の生成 | `OptimizeConfig(optimizer="llm_candidate_set")` | 探索候補のプール | 制約確認、修復、獲得関数による再順位付け |

重要なのは、LLMが最終的な候補価値を決定するわけではない点です。

```text
自然言語の目的・データ情報・既存設定
  -> LLMがConfig案または候補集合を生成
  -> bochanがConfigを検証・型変換
  -> bochanがモデル構築、獲得関数構築、制約処理、候補評価を実行
```

---

## インストール

LLM連携用の追加依存関係をインストールします。

```bash
pip install -e ".[llm]"
```

開発・テストも含める場合:

```bash
pip install -e ".[llm,test]"
```

主なprovider SDK:

```text
openai
google-genai
```

---

## APIキー

APIキーはNotebook、ソースコード、JSON、Gitリポジトリに直接保存せず、環境変数で渡してください。

Linux / macOS:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

Windows PowerShellの現在のセッション:

```powershell
$env:OPENAI_API_KEY = "..."
$env:GEMINI_API_KEY = "..."
```

Windowsへの永続設定:

```powershell
setx OPENAI_API_KEY "..."
setx GEMINI_API_KEY "..."
```

`setx` で設定した値は、新しく起動したPowerShell、コマンドプロンプト、Jupyterプロセスから有効になります。

---

## 社内プロキシ・自己署名証明書

社内プロキシがHTTPS通信を中継している環境では、次のエラーが発生する場合があります。

```text
SSL: CERTIFICATE_VERIFY_FAILED
self-signed certificate in certificate chain
```

会社のルート証明書またはCA bundleをPEM形式で用意し、`ca_bundle_path` に指定します。

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

参照される環境変数:

1. `SSL_CERT_FILE`
2. `REQUESTS_CA_BUNDLE`
3. `CURL_CA_BUNDLE`

一時的な疎通確認だけであれば次も利用できますが、通常運用では使用しないでください。

```python
LLMConfig(
    provider="openai",
    model="gpt-4.1-mini",
    ssl_verify=False,
)
```

---

## 共通のLLM設定

モデル、獲得関数、候補最適化に共通する情報は `LLMSettings` にまとめます。

```python
from bochan.llm import LLMConfig, LLMContextConfig, LLMSettings

llm_settings = LLMSettings(
    goal="conductivityを最大化し、shrinkageを最小化したい。実験ノイズを考慮する。",
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
            "temperature": "焼成温度。単位はdegC。",
            "time": "保持時間。単位はhour。",
            "atmosphere": "0=air、1=N2、2=Arとして符号化したカテゴリ変数。",
        },
        target_descriptions={
            "conductivity": "導電率。高いほど望ましい。",
            "shrinkage": "収縮率。低いほど望ましい。",
        },
        domain_notes=[
            "観測値には実験ノイズが含まれる。",
            "高温かつ長時間では収縮率が増加しやすい。",
        ],
        candidate_policy="既存条件から極端に離れすぎる候補は避ける。",
    ),
    n_llm_candidates=50,
)
```

LLM plannerへ渡される主な情報:

- 自然言語で記述した目的
- `train_X` と `train_Y` のshape
- 探索範囲 `bounds`
- 説明変数名と目的変数名
- 変数の意味、単位、カテゴリ情報
- 実験上の注意事項
- 現在のモデル、獲得関数、候補最適化設定
- `BochanStudy` の場合は完了、pending、failed trialの要約

データ配列全体をLLMへ送るのではなく、主にメタデータと設定情報を利用します。

---

# BayesianOptimizer

## 使用するデータ

以下の例では、2目的の材料条件探索を想定します。

```python
import torch

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
```

---

## 全設定をLLMに提案させる

モデル、学習設定、獲得関数関連、候補最適化関連を1回のLLM呼び出しでまとめて提案できます。

```python
from bochan.api import BayesianOptimizer, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(),
    bounds=bounds,
    llm_settings=llm_settings,
)

suggestion = bo.suggest_all(
    train_X=train_X,
    train_Y=train_Y,
)
```

返される主な属性:

```python
print(suggestion.model_config)
print(suggestion.fit_config)
print(suggestion.acq_config)
print(suggestion.opt_config)
print(suggestion.reasoning_summary)
print(suggestion.warnings)
```

提案は既定では自動適用されません。

```python
bo.apply_suggestion(suggestion)
```

確認を省略して即時適用する場合:

```python
suggestion = bo.suggest_all(
    train_X=train_X,
    train_Y=train_Y,
    apply=True,
)
```

適用後は学習し、保存済みの獲得関数・最適化設定を使って候補を生成できます。

```python
bo.fit(train_X, train_Y)

candidates, acq_value = bo.candidate()
```

`candidate()` の `acq_config` と `opt_config` は、LLM提案を適用済みなら省略できます。

---

## 全設定へ別々のプロンプトを与える

1回のLLM呼び出しの中で、モデル、獲得関数、最適化に別々の指示を与えられます。

```python
suggestion = bo.suggest_all(
    train_X=train_X,
    train_Y=train_Y,
    prompt="安全性と再現性を重視して全体を設定する。",
    model_prompt=(
        "データ数が少ないため、まず単純で安定したGPを優先する。"
        "不要な高次元モデルは避ける。"
    ),
    acquisition_prompt=(
        "2目的で観測ノイズがある。conductivityは最大化、"
        "shrinkageは最小化する。"
    ),
    optimizer_prompt=(
        "1回に3候補を提案する。atmosphereはカテゴリとして扱い、"
        "候補生成後のカテゴリ値を整数へ修復する。"
    ),
    apply=True,
)
```

役割:

- `prompt`: 全体方針
- `model_prompt`: `ModelConfig` と `FitConfig` への追加指示
- `acquisition_prompt`: `AcquisitionConfig`、`ObjectiveConfig`、制約への追加指示
- `optimizer_prompt`: `OptimizeConfig`、`q`、backend、候補修復への追加指示

辞書形式でも渡せます。

```python
suggestion = bo.suggest(
    mode="all",
    train_X=train_X,
    train_Y=train_Y,
    prompts={
        "model": "baseモデルを第一候補にする。",
        "acquisition": "多目的かつノイズありとして選択する。",
        "optimizer": "q=3でカテゴリ変数を厳密に扱う。",
    },
)
```

---

## モデル関連だけをLLMで設定する

`ModelConfig` と `FitConfig` だけを提案させます。

```python
model_suggestion = bo.suggest_model(
    prompt=(
        "説明変数は3列でデータ数も少ない。"
        "最初は単純なGPを使い、過度に複雑なモデルを避ける。"
    ),
    train_X=train_X,
    train_Y=train_Y,
)

print(model_suggestion.model_config)
print(model_suggestion.fit_config)
print(model_suggestion.reasoning_summary)
print(model_suggestion.warnings)
```

モデル関連だけを適用:

```python
bo.apply_suggestion(
    model_suggestion,
    model_config=True,
    fit_config=True,
    acq_config=False,
    opt_config=False,
)
```

即時適用:

```python
bo.suggest_model(
    prompt="base GPを優先し、学習反復数は過剰に増やさない。",
    train_X=train_X,
    train_Y=train_Y,
    apply=True,
)
```

---

## 獲得関数関連だけをLLMで設定する

`AcquisitionConfig` に加えて、必要な `ObjectiveConfig`、方向、重み、獲得関数固有パラメータを提案させます。

```python
acquisition_suggestion = bo.suggest_acquisition(
    prompt=(
        "conductivityを最大化し、shrinkageを最小化する2目的問題。"
        "実験ノイズがあるため、ノイズを考慮した獲得関数を優先する。"
    ),
    train_X=train_X,
    train_Y=train_Y,
)

print(acquisition_suggestion.acq_config)
print(acquisition_suggestion.reasoning_summary)
print(acquisition_suggestion.warnings)
```

獲得関数関連だけを適用:

```python
bo.apply_suggestion(
    acquisition_suggestion,
    model_config=False,
    fit_config=False,
    acq_config=True,
    opt_config=False,
)
```

即時適用:

```python
bo.suggest_acquisition(
    prompt="初期探索なので活用より探索を強める。UCBも候補として検討する。",
    train_X=train_X,
    train_Y=train_Y,
    apply=True,
)
```

---

## 候補最適化関連だけをLLMで設定する

`OptimizeConfig` のbackend、`q`、`raw_samples`、`num_restarts`、カテゴリ変数処理、候補修復などを提案させます。

```python
optimizer_suggestion = bo.suggest_optimizer(
    prompt=(
        "1回に3候補を提案する。atmosphereはカテゴリ変数。"
        "通常のoptimize_acqfとmixed最適化の適合性を確認する。"
    ),
    train_X=train_X,
    train_Y=train_Y,
)

print(optimizer_suggestion.opt_config)
print(optimizer_suggestion.reasoning_summary)
print(optimizer_suggestion.warnings)
```

最適化関連だけを適用:

```python
bo.apply_suggestion(
    optimizer_suggestion,
    model_config=False,
    fit_config=False,
    acq_config=False,
    opt_config=True,
)
```

即時適用:

```python
bo.suggest_optimizer(
    prompt="q=3、raw_samplesは256、再スタートは10程度を基準にする。",
    train_X=train_X,
    train_Y=train_Y,
    apply=True,
)
```

---

## 汎用suggest API

カテゴリ別メソッドは、次の汎用APIのショートカットです。

```python
bo.suggest(mode="all", ...)
bo.suggest(mode="model", ...)
bo.suggest(mode="acquisition", ...)
bo.suggest(mode="optimizer", ...)
```

利用可能な代表的alias:

```text
all / full / config
model / model_config / fit
acquisition / acq / acquisition_config
optimizer / optimization / optimize_config
```

---

## 設定を順番に試す

各カテゴリを別のLLM呼び出しとして試し、内容を比較してから適用できます。

```python
model_suggestion = bo.suggest_model(
    "base、saas、deepkernelの適合性を比較し、理由を示す。",
    train_X=train_X,
    train_Y=train_Y,
)

acq_suggestion = bo.suggest_acquisition(
    "NEHVI、EHVI、NParEGOを比較し、ノイズの影響を重視する。",
    train_X=train_X,
    train_Y=train_Y,
)

opt_suggestion = bo.suggest_optimizer(
    "optimize_acqf、nsgaii、llm_candidate_setを比較する。",
    train_X=train_X,
    train_Y=train_Y,
)
```

確認後に個別適用:

```python
bo.apply_suggestion(
    model_suggestion,
    acq_config=False,
    opt_config=False,
)

bo.apply_suggestion(
    acq_suggestion,
    model_config=False,
    fit_config=False,
    opt_config=False,
)

bo.apply_suggestion(
    opt_suggestion,
    model_config=False,
    fit_config=False,
    acq_config=False,
)
```

学習と候補生成:

```python
bo.fit(train_X, train_Y)
candidates, acq_value = bo.candidate()
```

---

## 明示設定による上書き

保存済みLLM設定があっても、`candidate()` に明示的なConfigを渡すと、その呼び出しでは明示設定が優先されます。

```python
from bochan.api import AcquisitionConfig, OptimizeConfig

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="UCB", acqf_kwargs={"beta": 2.0}),
    opt_config=OptimizeConfig(
        optimizer="optimize_acqf",
        q=2,
        raw_samples=128,
        num_restarts=5,
    ),
)
```

---

## 学習後にモデル設定を変更した場合

学習済みの `BayesianOptimizer` に対して、LLM提案でモデルまたは学習設定を変更した場合、既存モデルとConfigが不整合になります。

```python
bo.fit(train_X, train_Y)

bo.suggest_model(
    prompt="SAASを検討する。",
    train_X=train_X,
    train_Y=train_Y,
    apply=True,
)
```

この状態で `candidate()` を呼ぶと、再学習を促すエラーになります。

```python
bo.refit()
# または
bo.fit(train_X, train_Y)

candidates, acq_value = bo.candidate()
```

獲得関数または候補最適化設定だけを変更した場合は、通常はモデルの再学習は不要です。

---

## 従来のモデル自動選択API

`ModelConfig(model_type="llm_selected")` も引き続き利用できます。

```python
from bochan.api import BayesianOptimizer, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(model_type="llm_selected"),
    bounds=bounds,
    llm_settings=llm_settings,
)

bo.fit(train_X, train_Y)
```

この経路は主にモデルと学習設定を `fit()` 直前に選択します。

全設定をまとめて扱う場合は、次を推奨します。

```python
bo = BayesianOptimizer(
    model_config=ModelConfig(),
    bounds=bounds,
    llm_settings=llm_settings,
)

bo.suggest_all(
    train_X=train_X,
    train_Y=train_Y,
    apply=True,
)
bo.fit(train_X, train_Y)
bo.candidate()
```

---

## 獲得関数と候補最適化backendの違い

次の2つは別の設定です。

```python
AcquisitionConfig(name="NEHVI")
OptimizeConfig(optimizer="optimize_acqf")
```

- `AcquisitionConfig`: 候補点の価値を評価する関数
- `OptimizeConfig`: その獲得関数をどの方法で最大化するか

主な獲得関数候補:

| 用途 | 主な候補 |
|---|---|
| 単目的ベイズ最適化 | `EI`, `UCB`, `PI`, `TS` |
| 多目的ベイズ最適化 | `NEHVI`, `EHVI`, `NParEGO` |
| 回帰Active Learning | `NIPV` |
| 分類Active Learning | `entropy`, `BALD`, `margin`, `variance` |
| Level-set estimation | `straddle`, `ICU`, `boundaryvariance`, `levelset` |

主な候補最適化backend:

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
)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
)
```

処理の流れ:

```text
LLMが候補集合を提案
  -> boundsと入力形式を確認
  -> 制約違反候補を除外または修復
  -> 各候補を獲得関数で評価
  -> 上位q件を返す
```

候補集合生成は、LLM出力をそのまま採用する処理ではありません。

---

# BochanStudy

`BochanStudy` では、完了、pending、failed trialの履歴を含めて設定を提案できます。

```python
from bochan.api import BochanStudy

study = BochanStudy(
    bounds=bounds,
    llm_settings=llm_settings,
)
study.add_observations(train_X, train_Y)

suggestion = study.suggest(mode="config")

print(suggestion.model_config)
print(suggestion.fit_config)
print(suggestion.acq_config)
print(suggestion.opt_config)
print(suggestion.reasoning_summary)
print(suggestion.warnings)
```

確認後に適用:

```python
study.apply_suggestion(suggestion)
batch = study.ask(q=3, return_batch=True)
```

即時適用:

```python
suggestion = study.suggest(mode="config", apply=True)
batch = study.ask(q=3, return_batch=True)
```

---

# APIを呼ばないオフラインテスト

`planner_response` を使うと、OpenAIやGeminiを呼ばずにConfig変換と適用処理をテストできます。

```python
bo = BayesianOptimizer(
    model_config=ModelConfig(),
    bounds=bounds,
)

suggestion = bo.suggest_all(
    train_X=train_X,
    train_Y=train_Y,
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
            "acqf_kwargs": {"beta": 0.2},
        },
        "optimize_config": {
            "optimizer": "llm_candidate_set",
            "q": 1,
            "optimizer_kwargs": {
                "candidate_set": [
                    [840.0, 3.0, 1.0],
                    [860.0, 2.5, 2.0],
                ],
                "n_llm_candidates": 2,
            },
        },
        "warnings": ["offline planner response"],
        "reasoning_summary": "Use a base GP and exploratory UCB.",
    },
    apply=True,
)

bo.fit(train_X, train_Y)
candidates, acq_value = bo.candidate()
```

---

# 低レベルplanner API

`BayesianOptimizer` を作成せず、設定辞書だけを取得する場合は `plan_configs()` を利用します。

## 全設定

```python
from bochan.llm import plan_configs

plan = plan_configs(
    goal="conductivityを最大化し、shrinkageを最小化する。",
    llm_config=llm_settings.llm_config,
    llm_context=llm_settings.llm_context,
    train_X=train_X,
    train_Y=train_Y,
    bounds=bounds,
    mode="full",
)
```

## 獲得関数だけ

```python
plan = plan_configs(
    goal="conductivityを最大化し、shrinkageを最小化する。",
    llm_config=llm_settings.llm_config,
    llm_context=llm_settings.llm_context,
    train_X=train_X,
    train_Y=train_Y,
    bounds=bounds,
    mode="acquisition",
    requested_sections=["acquisition"],
    section_prompts={
        "acquisition": "観測ノイズを重視し、NEHVIとEHVIを比較する。",
    },
)
```

## カテゴリ別プロンプト

```python
plan = plan_configs(
    goal="材料条件を最適化する。",
    llm_config=llm_settings.llm_config,
    llm_context=llm_settings.llm_context,
    train_X=train_X,
    train_Y=train_Y,
    bounds=bounds,
    mode="full",
    requested_sections=["model", "acquisition", "optimizer"],
    section_prompts={
        "model": "単純なGPを優先する。",
        "acquisition": "多目的かつノイズありとして選ぶ。",
        "optimizer": "q=3でカテゴリ変数を扱う。",
    },
)
```

---

# Gemini

provider設定だけを変更します。

```python
from bochan.llm import LLMConfig

llm_config = LLMConfig(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key_env="GEMINI_API_KEY",
)
```

---

# FastAPI

HTTP APIでは、サーバー側の環境変数にAPIキーを保存してください。

- `POST /models/plan`: 設定辞書を提案
- `POST /models/auto-candidates`: 設定提案、学習、候補生成
- `POST /models/{model_id}/candidates`: 学習済みモデルから候補生成

APIキー値をrequest bodyへ含めないでください。

---

# 注意事項

- LLM出力は確率的であり、同じデータとプロンプトでも提案が変わる場合があります。
- `reasoning_summary` と `warnings` を確認してから重要な実験へ適用してください。
- LLMはConfig候補を返しますが、選択したモデル・task type・獲得関数の組合せが実装上有効かはbochan側でも検証されます。
- 目的変数の方向、出力index、カテゴリ列、制約条件が曖昧な場合は、プロンプトまたは `LLMContextConfig` で明示してください。
- `llm_candidate_set` は候補集合の再順位付けであり、厳密なjoint q-batch最適化ではありません。
- 学習後にモデル設定を変更した場合は、候補生成前に `fit()` または `refit()` が必要です。
