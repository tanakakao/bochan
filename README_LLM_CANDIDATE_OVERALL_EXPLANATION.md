# 候補ごとの総合説明

`BayesianOptimizer.explain_candidates()` は、物理・化学・製造・開発の各視点に加えて、各代表候補の総合説明を返します。

```python
result = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
    return_result=True,
)

explanation = bo.explain_candidates(
    result,
    max_representatives=5,
    prompt="量産性、設備ばらつき、反応機構を重視して説明する。",
)
```

## 候補群全体の総合説明

```python
print(explanation.summary)
```

`summary` は、提案された候補群全体について、技術的な狙い、探索と活用の構成、実務上の価値、主要なリスクをまとめます。

## 候補ごとの総合説明

```python
for item in explanation.candidate_explanations:
    print(item.headline)
    print(item.overall_interpretation)
```

`overall_interpretation` は、次の情報を統合した意思決定向けの説明です。

- モデル予測と不確かさ
- 物理的な解釈
- 化学的な解釈
- 製造性、制御性、スケールアップ
- 開発上の学習価値
- リスクとトレードオフ
- 次に行うべき実験・測定

各視点は個別フィールドとしても保持されます。

```python
item.model_evidence
item.physical_interpretation
item.chemical_interpretation
item.manufacturing_interpretation
item.development_interpretation
item.risks_and_tradeoffs
item.recommended_checks
item.confidence
```

総合説明で概要を把握し、必要に応じて各視点へ掘り下げる使い方を想定しています。

## JSON保存

```python
payload = explanation.to_dict()
```

各候補の辞書には `overall_interpretation` が含まれます。

## オフライン応答例

```python
explanation = bo.explain_candidates(
    result,
    explanation_response={
        "summary": "候補群全体では性能向上と工程成立性を同時に確認する。",
        "candidate_explanations": [
            {
                "candidate_index": 0,
                "headline": "性能有望だが工程確認が必要な候補",
                "overall_interpretation": (
                    "モデル上は有望で物理化学的にも妥当な可能性がある一方、"
                    "量産時の制御余裕を確認してから採用判断すべき候補である。"
                ),
                "model_evidence": ["予測値が高い。"],
                "physical_interpretation": ["構造変化が寄与する可能性がある。"],
                "chemical_interpretation": ["反応進行度が変化する可能性がある。"],
                "manufacturing_interpretation": ["設備ばらつきの確認が必要である。"],
                "development_interpretation": ["仮説識別価値が高い。"],
                "risks_and_tradeoffs": ["制御幅が狭い可能性がある。"],
                "recommended_checks": ["条件を振った再現実験を行う。"],
                "confidence": "medium",
            }
        ],
        "warnings": [],
    },
)
```

旧形式の応答に `overall_interpretation` がない場合は空文字として扱うため、後方互換性があります。

## 注意

総合説明は候補を自動承認するものではありません。設備能力、安全性、法規、原料互換性、測定系、量産再現性は専門家が最終確認してください。
