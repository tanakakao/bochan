import numpy as np
import pandas as pd
import torch
import pytest
import sys
import os

path = os.getcwd()

sys.path.append(path)

from bayes_optimization.models import BOCandidateModel


@pytest.fixture
def base_df():
    """BOCandidateModel の fit テストで使うベースデータ"""
    np.random.seed(0)

    n = 20
    base = np.concatenate([np.zeros(1), np.random.rand(n - 2), np.ones(1)])

    df = pd.DataFrame({
        "feature1": base,
        "feature2": base * 2.0,
        "cat_feature": np.random.choice(["a", "b", "c"], n),
        "feature3": 0.2 + base,
        "target1": np.random.rand(n),
        "target2": np.random.rand(n),
        "cat_target": np.random.choice(["x", "y"], n)
    })
    return df
    
@pytest.mark.parametrize(
    [
        "feature_cols",
        "target_cols",
        "numeric_cols",
        "categorical_cols",
        "cat_targets_cols",
        "cat_target_items",
        "bounds_norm",
        "multi_model_type",
        "impute",
        "robust",
        "perturbation",
        "heteroscedastic",
        "deep_gp",
        "deep_kernel",
        "lr",
        "epoch"
    ],
    [
        # 数値だけ
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="single", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed single", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            torch.tensor([[0.,0.,0.],[2.,2.,2.]]).to(torch.double), # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="single with bounds", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="multi", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed multi", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            "multi_task_multi_output", # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="multi with multi task", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            "model_list", # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="multi with model list", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["cat_target"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            ["cat_target"], # カテゴリ値の目的変数
            {'cat_target': ['x']}, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="categorical target", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["cat_target", "target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            ["cat_target"], # カテゴリ値の目的変数
            {'cat_target': ['x']}, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed target", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            True, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="single robust", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            True, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed single robust", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            True, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="single perturbation", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            True, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed single perturbation", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="single heteroscedastic", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed single heteroscedastic", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            True, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="single hetero and pert", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            True, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed single hetero and pert", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            True, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="single hetero and robust", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            True, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed single hetero and robust", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            "model_list", # 多目的モデルの種類
            None, # 欠損値の補完
            True, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="multi robust", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            "model_list", # 多目的モデルの種類
            None, # 欠損値の補完
            True, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed multi robust", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            True, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="multi perturbation", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            True, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed multi perturbation", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="multi heteroscedastic", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed multi heteroscedastic", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            True, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="multi hetero and pert", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            True, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed multi hetero and pert", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            "model_list", # 多目的モデルの種類
            None, # 欠損値の補完
            True, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="single hetero and robust", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            "model_list", # 多目的モデルの種類
            None, # 欠損値の補完
            True, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            True, # 分散予測
            None, # 深層ガウス過程
            None, # 深層カーネル学習
            None, # 学習率(深層系)
            None, # エポック数(深層系)
            id="mixed multi hetero and robust", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            True, # 深層ガウス過程
            None, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="single deep gp", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            True, # 深層ガウス過程
            None, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="mixed single deep gp", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            True, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="single deep kernel", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            True, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="mixed single deep kernel", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            True, # 深層ガウス過程
            True, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="single deep gp and deep kernel", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            True, # 深層ガウス過程
            True, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="mixed single deep gp and deep kernel", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            True, # 深層ガウス過程
            None, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="multi deep gp", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            True, # 深層ガウス過程
            None, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="mixed multi deep gp", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            True, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="multi deep kernel", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            None, # 深層ガウス過程
            True, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="mixed multi deep kernel", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            None, # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            True, # 深層ガウス過程
            True, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="multi deep gp and deep kernel", # テスト名
        ),
        pytest.param(
            ["feature1", "feature2", "feature3", "cat_feature"], # 全説明変数
            ["target1", "target2"], # 目的変数
            ["feature1", "feature2", "feature3"], # 連続値の説明変数
            ["cat_feature"], # カテゴリ値の説明変数
            None, # カテゴリ値の目的変数
            None, # カテゴリ値の目的変数ごとのカテゴリ値のリスト
            None, # 正規化範囲
            None, # 多目的モデルの種類
            None, # 欠損値の補完
            None, # ロバストモデル(外れ値)
            None, # ロバスト(入力摂動)
            None, # 分散予測
            True, # 深層ガウス過程
            True, # 深層カーネル学習
            1e-2, # 学習率(深層系)
            50, # エポック数(深層系)
            id="mixed multi deep gp and deep kernel", # テスト名
        ),
    ],
    # ids=lambda v: v if isinstance(v, str) else None,  # case_name をテスト名に反映
)

def test_fit_various_patterns(
    base_df,
    feature_cols,
    target_cols,
    numeric_cols,
    categorical_cols,
    cat_targets_cols,
    cat_target_items,
    bounds_norm,
    multi_model_type,
    impute,
    robust,
    perturbation,
    heteroscedastic,
    deep_gp,
    deep_kernel,
    lr,
    epoch
):
    df = base_df.copy()

    model = BOCandidateModel()

    fit_kwargs = dict(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        categorical_cols=categorical_cols,
        cat_targets_cols=cat_targets_cols,
        cat_target_items=cat_target_items,
        bounds_norm=bounds_norm,
        multi_model_type=multi_model_type,
        impute=impute,
        robust=robust,
        perturbation=perturbation,
        heteroscedastic=heteroscedastic,
        deep_gp=deep_gp,
        deep_kernel=deep_kernel,
        lr=lr,
        epoch=epoch
    )

    # ---- 実行 ----
    model.fit(**fit_kwargs)

    # ---- 共通のアサーション ----
    assert model.model is not None
    assert model.feature_cols == feature_cols
    assert model.target_cols == target_cols
    assert model.cat_targets_cols == (cat_targets_cols or [])
    if len(target_cols)==1:
        assert model.cat_targets_idx == ([0] if cat_targets_cols is not None else [])
    else:
        assert model.cat_targets_idx == ([1] if cat_targets_cols is not None else [])
    
    # 実装仕様に合わせて:
    assert model.numeric_cols == numeric_cols
    assert model.numeric_idx == [0,1,2]
    assert model.categorical_cols == (categorical_cols or [])
    assert model.categorical_idx == ([3] if categorical_cols is not None else [])

    assert model.cat_features_list == ([{3: 0}, {3: 1}, {3: 2}] if categorical_cols is not None else None)
    assert model.labels == ({'cat_feature': {'a': 0, 'b': 1, 'c': 2}} if categorical_cols is not None else {})
    assert model.categorical_features == ({3: [0, 1, 2]} if categorical_cols is not None else None)
    if cat_targets_cols is None:
        assert model.multi_model_type == ("single_task_multi_output" if len(target_cols)>1 and multi_model_type is None else multi_model_type)
    else:
        assert model.multi_model_type == ("model_list" if len(target_cols)>1 else None)
    assert model.multi_task_type is None

    assert model.robust == (robust or False)
    assert model.perturbation == (perturbation or False)
    assert model.heteroscedastic == (heteroscedastic or False)

    assert model.deep_gp == (deep_gp or False)
    assert model.deep_kernel == (deep_kernel or False)
    
    assert model.lr == (lr or 1e-2)
    assert model.epoch == (epoch or 300)

    if categorical_cols is None:
        bounds_norm_expected = torch.tensor(
            [
                [0.0000, 0.0000, 0.2000],
                [1.0000, 2.0000, 1.2000],
            ],
            dtype=model.dtype,
        ) if bounds_norm is None else bounds_norm
    else:
        bounds_norm_expected = torch.tensor(
            [
                [0.0000, 0.0000, 0.2000, 0.0000],
                [1.0000, 2.0000, 1.2000, 2.0000],
            ],
            dtype=model.dtype,
        ) if bounds_norm is None else bounds_norm

    assert torch.allclose(
        model.bounds_norm,
        bounds_norm_expected,
        atol=1e-9,
    )

    if hasattr(model.model, "models"):
        assert model.model.models[0].input_transform is not None
    else:
        assert model.model.input_transform is not None

    assert model.predict(model.train_X)[0].shape[0]==model.train_Y.shape[0]
    assert model.predict(model.train_X)[0].shape[-1]==model.train_Y.shape[-1]