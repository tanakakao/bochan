# LLMによる最終候補の物理・化学・製造・開発視点の説明

`BayesianOptimizer` が返した候補条件について、単に「獲得関数値が高い」と説明するだけでなく、次の視点から構造化して説明できます。

- モデルが直接示している根拠
- 物理的な解釈
- 化学的な解釈
- 製造現場での実行性、制御性、品質、安全、スケールアップ
- 開発現場での仮説検証価値、学習価値、次の実験への寄与
- リスクとトレードオフ
- 実施前後に確認すべき測定・実験

説明は候補生成とは分離されています。候補を作り直さず、同じ候補に対して説明条件や代表点数を変更して再評価できます。

---

## 1. 基本的な使用方法

まず、通常どおり候補を生成し、`CandidateResult` を取得します。

```python
from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    ModelConfig,
    ObjectiveConfig,
    OptimizeConfig,
)
from bochan.llm import LLMConfig, LLMContextConfig, LLMSettings

llm_settings = LLMSettings(
    goal=(
        "導電率を高くし、収縮率を低くしたい。"
        "量産時に安定して制御できる条件を優先したい。"
    ),
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    llm_context=LLMContextConfig(
        variable_names=[
            "raw_material_1",
            "raw_material_2",
            "temperature",
            "holding_time",
        ],
        target_names=["conductivity", "shrinkage"],
        variable_descriptions={
            "raw_material_1": "導電相を形成する主原料。単位はmass fraction。",
            "raw_material_2": "焼結助剤。過剰添加では異相形成の懸念がある。",
            "temperature": "焼成温度。単位はdegC。",
            "holding_time": "焼成保持時間。単位はhour。",
        },
        target_descriptions={
            "conductivity": "導電率。高いほど望ましい。",
            "shrinkage": "焼成収縮率。低いほど望ましい。",
        },
        domain_notes=[
            "高温長時間では粒成長が進み、収縮率が増える可能性がある。",
            "raw_material_2は少量では焼結を促進するが、過剰添加は避けたい。",
            "量産炉では温度の面内分布が±5 degC程度ある。",
        ],
        candidate_policy=(
            "極端な外挿条件は避ける。量産時の温度変動を考慮し、"
            "狭いプロセスウィンドウの候補には警告を付ける。"
        ),
    ),
)

bo = BayesianOptimizer(
    model_config=ModelConfig(model_type="llm_selected"),
    bounds=bounds,
    llm_settings=llm_settings,
)
bo.fit(train_X, train_Y)

result = bo.candidate(
    acq_config=AcquisitionConfig(
        name="llm_selected",
        objective_config=ObjectiveConfig(
            mode="multi_output",
            outputs=[0, 1],
            directions=["maximize", "minimize"],
            weights=[1.0, 0.5],
        ),
    ),
    opt_config=OptimizeConfig(
        optimizer="llm_candidate_set",
        q=8,
    ),
    return_result=True,
)
```

候補生成後に説明を作成します。

```python
explanation = bo.explain_candidates(result)
```

主な結果を確認します。

```python
print(explanation.summary)
print(explanation.common_patterns)
print(explanation.representative_indices)
print(explanation.warnings)
```

候補ごとの説明を確認します。

```python
for item in explanation.candidate_explanations:
    print(item.candidate_index)
    print(item.headline)
    print(item.model_evidence)
    print(item.physical_interpretation)
    print(item.chemical_interpretation)
    print(item.manufacturing_interpretation)
    print(item.development_interpretation)
    print(item.risks_and_tradeoffs)
    print(item.recommended_checks)
    print(item.confidence)
```

説明は元の結果にも保存されます。

```python
assert result.explanation is explanation
assert bo.last_candidate_explanation is explanation
```

LLMへ送った最終プロンプトは、監査やデバッグ用に確認できます。

```python
print(bo.last_candidate_explanation_prompt)
```

---

## 2. 候補数が多い場合の代表点選択

既定では最大5点を説明します。

```python
explanation = bo.explain_candidates(
    result,
    max_representatives=5,
)
```

候補数が5点以下の場合はすべて説明します。候補数が多い場合は、次の順序で代表点を選びます。

1. 候補ごとの獲得関数値が得られる場合、その最大点
2. 候補条件の中心に近い点
3. すでに選択した点から離れた多様な点

候補ごとの獲得関数値がなく、joint batch全体に1つの獲得関数値だけがある場合は、optimizerが返した先頭候補を起点にします。

選択された役割は候補ごとの説明に保存されます。

```python
for item in explanation.candidate_explanations:
    print(item.representative_role)
```

代表的な値:

```text
highest_acquisition
optimizer_first
central_candidate
diverse_candidate
all_candidates
```

候補数と省略数も保持されます。

```python
print(explanation.total_candidates)
print(explanation.omitted_count)
```

代表点数を増やす場合:

```python
explanation = bo.explain_candidates(
    result,
    max_representatives=10,
)
```

説明コストを抑える場合:

```python
explanation = bo.explain_candidates(
    result,
    max_representatives=3,
)
```

---

## 3. LLMへ渡される情報

説明時には、生データ全体ではなく主に次の情報を渡します。

- 自然言語の開発目標
- 変数名と目的変数名
- 変数・目的変数の説明
- 物理化学・工程上の既知事項
- 候補選択方針
- モデル設定
- 獲得関数設定
- 候補最適化設定
- 学習データ各列の最小値、最大値、平均値、有限値数
- 代表候補の条件値
- 代表候補が学習範囲内か、範囲外か
- 代表候補の予測平均
- 代表候補の予測分散
- 代表点として選ばれた理由

これにより、LLMは候補条件だけを見て一般論を述べるのではなく、モデル結果と与えられた技術文脈を区別して説明します。

---

## 4. モデル根拠と物理化学仮説の区別

出力では、次のフィールドを分離しています。

```python
item.model_evidence
item.physical_interpretation
item.chemical_interpretation
```

`model_evidence` には、次のようなモデル・設定から直接確認できる事項を記述させます。

- 予測平均が他の代表候補より高い
- 予測分散が大きく、探索的価値が高い
- 学習データ範囲の外側にある
- 制約境界に近い
- 獲得関数上で高く評価された

`physical_interpretation` と `chemical_interpretation` は仮説です。プロンプトでは次を禁止しています。

- 相関を因果関係として断定する
- モデル予測を実験事実として表現する
- 与えられていない反応機構や相を発明する
- 変数説明が不足しているのに詳細な機構を断定する

情報が不足している場合は、`confidence="low"`、`warnings`、`assumptions` に明示されます。

---

## 5. 製造現場の視点

`manufacturing_interpretation` では、文脈に応じて次を確認します。

- 装置で設定可能な条件か
- 量産時に再現できる条件幅があるか
- 温度、圧力、流量、組成などの制御精度に対して頑健か
- 工程能力や設備ばらつきに対して余裕があるか
- 原料切替、段取り、清掃、設備負荷への影響
- 安全性、腐食、詰まり、析出、発熱、ガス発生などの懸念
- スケールアップ時の熱・物質移動差
- 検査可能性と品質保証方法
- コスト、タクト、歩留まりへの影響

これらは `domain_notes` と `candidate_policy` に具体的な現場情報を記載するほど有用になります。

```python
LLMContextConfig(
    domain_notes=[
        "量産機では昇温速度を2 degC/min未満にできない。",
        "粘度が高い配合ではポンプ圧力上限に注意する。",
        "原料Bは吸湿しやすく、開封後8時間以内に使用する。",
    ],
    candidate_policy=(
        "設備上限から5%以上の余裕を確保する。"
        "特殊な段取りが必要な条件には警告を付ける。"
    ),
)
```

---

## 6. 開発現場の視点

`development_interpretation` は、候補の性能だけでなく学習価値を説明します。

- 現在の最良条件を更新する活用候補か
- 不確かな領域を確認する探索候補か
- 競合する仮説を識別できるか
- 変数間の交互作用を確認できるか
- 制約境界やプロセスウィンドウを明確にできるか
- 外挿性能を確認する候補か
- 次サイクルのモデル更新にどのような情報を与えるか
- 再現確認、ロバスト性確認、メカニズム確認のどれに該当するか

例えば、性能予測が最高ではなくても、条件空間の離れた候補は「モデルの仮説を識別する点」として説明される場合があります。

---

## 7. 視点を限定する

必要な視点だけを指定できます。

```python
explanation = bo.explain_candidates(
    result,
    perspectives=["chemistry", "manufacturing"],
)
```

物理・開発視点を重視する場合:

```python
explanation = bo.explain_candidates(
    result,
    perspectives=["physics", "development"],
)
```

出力schema自体は共通ですが、LLMへの重点指示が変更されます。

---

## 8. 追加の説明指示

通常のドメイン情報に加えて、今回だけの指示を渡せます。

```python
explanation = bo.explain_candidates(
    result,
    prompt=(
        "今回は量産移管前の確認である。"
        "特に設備ばらつき、原料ロット差、測定再現性を重視して説明する。"
    ),
)
```

安全・環境面を重視する場合:

```python
explanation = bo.explain_candidates(
    result,
    prompt=(
        "発熱、ガス発生、腐食、廃液、作業者曝露の可能性を優先して確認する。"
        "根拠が不足する項目は断定せず、確認項目として示す。"
    ),
)
```

---

## 9. 最新候補を説明する

`CandidateResult` を明示しない場合、直近の候補生成結果を使います。

```python
explanation = bo.explain_last_candidates()
```

これは次と同じです。

```python
explanation = bo.explain_candidates(bo.history[-1])
```

候補履歴がない場合はエラーになります。

---

## 10. 候補行列を直接説明する

`CandidateResult` がない場合でも、候補行列を直接渡せます。

```python
explanation = bo.explain_candidates(
    candidates=candidates,
    acq_value=acq_values,
    max_representatives=4,
)
```

この場合、`CandidateResult` に保存されている獲得関数設定やoptimizer設定は利用できません。可能であれば `return_result=True` を使う方法を推奨します。

---

## 11. オフラインテスト

APIを呼ばずに、説明結果の型変換と画面表示をテストできます。

```python
response = {
    "summary": "高性能候補と探索候補を組み合わせた提案である。",
    "selection_note": "獲得関数最大点、中心点、多様点を選択した。",
    "common_patterns": [
        "温度は既存データ範囲の上側に集中している。",
    ],
    "candidate_explanations": [
        {
            "candidate_index": 0,
            "representative_role": "highest_acquisition",
            "headline": "高性能が期待される活用候補",
            "model_evidence": [
                "予測導電率が代表候補中で最も高い。",
            ],
            "physical_interpretation": [
                "高温化による緻密化が寄与する可能性があるが、機構は未確認である。",
            ],
            "chemical_interpretation": [
                "焼結助剤による反応促進の可能性がある。相分析で確認する必要がある。",
            ],
            "manufacturing_interpretation": [
                "量産炉の温度分布を考慮すると上限余裕が小さい。",
            ],
            "development_interpretation": [
                "現在の最良条件更新を狙う活用候補である。",
            ],
            "risks_and_tradeoffs": [
                "粒成長と収縮率増加のリスクがある。",
            ],
            "recommended_checks": [
                "XRD、密度、粒径分布、収縮率を同時に測定する。",
            ],
            "confidence": "medium",
        },
    ],
    "assumptions": [
        "温度と保持時間の説明が実工程と一致している。",
    ],
    "warnings": [],
}

explanation = bo.explain_candidates(
    result,
    max_representatives=1,
    explanation_response=response,
)
```

`explanation_response` を指定した場合、OpenAIやGeminiは呼び出されません。

---

## 12. JSON化

保存やAPI返却には `to_dict()` を利用できます。

```python
payload = explanation.to_dict()
```

返される主な構造:

```text
CandidateExplanation
├── total_candidates
├── representative_indices
├── omitted_count
├── summary
├── selection_note
├── common_patterns
├── candidate_explanations
│   ├── candidate_index
│   ├── representative_role
│   ├── headline
│   ├── model_evidence
│   ├── physical_interpretation
│   ├── chemical_interpretation
│   ├── manufacturing_interpretation
│   ├── development_interpretation
│   ├── risks_and_tradeoffs
│   ├── recommended_checks
│   └── confidence
├── assumptions
└── warnings
```

---

## 13. 注意事項

この説明は実験結果の代わりではありません。

- LLMが述べる物理・化学機構は、与えられた説明に基づく仮説です。
- モデル予測が高くても、実験再現性や量産再現性が保証されるわけではありません。
- 予測分散が小さくても、モデルバイアスや未入力因子は残ります。
- 学習範囲外の候補では、予測値そのものより検証計画を重視してください。
- 安全、法規、設備保護に関わる判断をLLMへ委ねないでください。
- 目的変数、単位、カテゴリの意味、設備制約を `LLMContextConfig` に明示してください。
- 因果関係を確認するには、追加実験、分析、再現試験が必要です。

推奨する運用は次です。

```text
BayesianOptimizerの候補
  -> モデル根拠を確認
  -> LLMの物理・化学・製造・開発仮説を確認
  -> warningsとassumptionsを確認
  -> 専門家が実施可否を判断
  -> 必要な分析・再現試験を付けて実験
  -> tell()で結果をモデルへ戻す
```
