# Web / FastAPIで単一組成式を組成比として扱う

bochan Web Appでは、1列の組成式を合計1の組成比として学習・候補生成に使用できます。

## Web App

1. Select画面で組成式列を説明変数として選択します。
2. 同じカードの「入力表記」で「組成式」を選択します。
3. Suggest画面の「単一組成式の比率探索」で変換方法と制約を設定します。

対応範囲:

- 組成式列は1列
- Fraction / CLR / ALR / ILR
- 原子比・mol比 / 重量比
- 元素ごとの上下限、刻み、必須指定
- 最小・最大使用元素数
- 元素間の線形等式・不等式

候補は変換座標ではなく、合計1に正規化された組成式と元素分率で表示されます。

## 組成式の事前検証

```http
POST /api/v1/composition/validate
Content-Type: application/json
```

```json
{
  "formulas": ["Fe2Co5Ni4", "Fe4Co10Ni8"],
  "settings": {
    "column": "formula",
    "representation": "ilr",
    "normalization": "atomic_fraction"
  }
}
```

倍率だけ異なる組成式は同じ分率になります。

## 型付き候補生成エンドポイント

```http
POST /api/v1/composition/regression/run
Content-Type: application/json
```

```json
{
  "run": {
    "dataset_id": "dataset-id",
    "feature_columns": ["formula", "temperature"],
    "target_column": "property",
    "search_space": [
      {
        "name": "formula",
        "type": "auto"
      },
      {
        "name": "temperature",
        "type": "numeric",
        "lower": 800.0,
        "upper": 1200.0
      }
    ],
    "acquisition": {
      "name": "EI"
    },
    "optimizer": {
      "q": 3,
      "num_restarts": 10,
      "raw_samples": 256
    }
  },
  "composition": {
    "column": "formula",
    "elements": ["Fe", "Co", "Ni"],
    "normalization": "atomic_fraction",
    "representation": "ilr",
    "bounds": {
      "Fe": [0.1, 0.8],
      "Co": [0.0, 0.8],
      "Ni": [0.0, 0.8]
    },
    "steps": {
      "Fe": 0.01,
      "Co": 0.01,
      "Ni": 0.01
    },
    "element_constraints": [
      {
        "terms": [
          {
            "element": "Co",
            "coefficient": 1.0
          },
          {
            "element": "Fe",
            "coefficient": -0.5
          }
        ],
        "operator": "=",
        "rhs": 0.0,
        "basis": "atomic_amount"
      }
    ]
  }
}
```

上記の元素制約は次を表します。

```text
Co = 0.5 × Fe
```

通常のWeb回帰エンドポイントへ `model_kwargs.web_composition` を渡す互換経路もありますが、新規APIクライアントでは型付きの `/composition/regression/run` を推奨します。

## 対象外

- A/Bサイトなどの複数サイト
- 複数の組成式列
- 組成記述子を探索変数とした候補生成
- 結晶構造からのサイト自動判定
