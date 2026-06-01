"""High-level configuration objects for the bochan API.

このモジュールは、モデル生成・学習・獲得関数生成・候補点最適化を
疎結合に扱うための dataclass 群を定義します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Callable, Literal, Sequence


TaskType = Literal[
    "regression",
    "multi_objective",
    "binary",
    "multiclass",
    "ordinal",
    "hybrid",
]
InputType = Literal["normal", "mixed"]
ModelType = str
OptimizerName = Literal["optimize_acqf", "optimize_acqf_mixed"]


@dataclass
class ModelConfig:
    """モデル生成に必要な設定。

    `ModelConfig` には4つの使い方があります。

    1. 直接指定モード:
        `model_cls=SingleTaskGP` のようにモデルクラスを直接渡します。
        この場合、`task_type` / `model_type` は履歴・ログ用のメタ情報です。

    2. registry 解決モード:
        `model_cls=None` とし、`task_type` / `model_type` / `cat_dims` から
        外部の registry を使ってモデルクラスを解決します。

    3. factory 指定モード:
        `model_factory` を指定し、任意の関数でモデルを生成します。

    4. multi-output 指定モード:
        `multi_output_config` を指定すると、出力列ごとに single-output submodel を
        作成し、`fit_model()` 時に submodel ごとに学習します。

    Args:
        model_cls: 直接生成したいモデルクラス。None の場合は registry から解決する。
        model_factory: 任意のモデル生成関数。指定時は `model_cls` / registry より優先される。
        task_type: regression / multi_objective / binary / multiclass / ordinal / hybrid などのタスク種別。
        model_type: base / deepgp / deepkernel / saas / pca / rembo / rrp / hetero など。
        input_type: normal / mixed。None の場合は cat_dims の有無から自動推定する。
        cat_dims: カテゴリ変数の列番号。空なら通常モデルとして扱う。
        input_transform: BoTorch 互換の input_transform。
        outcome_transform: BoTorch 互換の outcome_transform。
        model_kwargs: モデルクラスまたは model_factory へ追加で渡す kwargs。
        multi_output_config: multi-output / hybrid 自動構築用の設定。
        train_x_name: モデルコンストラクタで使う train_X 引数名。
        train_y_name: モデルコンストラクタで使う train_Y 引数名。
        pass_train_data: train_X / train_Y をモデルコンストラクタへ渡すか。
        pass_cat_dims: cat_dims をモデルに渡すか。None なら cat_dims が非空の場合だけ渡す。
        pass_input_transform: input_transform を渡すか。
        pass_outcome_transform: outcome_transform を渡すか。
    """

    model_cls: type | Callable[..., Any] | None = None
    model_factory: Callable[..., Any] | None = None

    task_type: TaskType | str = "regression"
    model_type: ModelType = "base"
    input_type: InputType | None = None

    cat_dims: Sequence[int] | None = None
    input_transform: Any = None
    outcome_transform: Any = None

    model_kwargs: dict[str, Any] = field(default_factory=dict)
    multi_output_config: MultiOutputConfig | None = None

    train_x_name: str = "train_X"
    train_y_name: str = "train_Y"

    pass_train_data: bool = True
    pass_cat_dims: bool | None = None
    pass_input_transform: bool = True
    pass_outcome_transform: bool = True


@dataclass
class FitConfig:
    """モデル学習に必要な設定。"""

    fit_func: Callable[..., Any] | None = None
    mll_factory: Callable[..., Any] | None = None
    mll_cls: type | Callable[..., Any] | None = None

    mll_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_kwargs: dict[str, Any] = field(default_factory=dict)

    use_model_make_mll: bool = True
    skip_fit: bool = False


@dataclass
class OutputConfig:
    """multi-output / hybrid の各出力を文字列中心で指定するための設定。

    Args:
        task_type: 出力のタスク種別。例: regression / binary / ordinal / multiclass。
        model_type: registry からモデルを引くための model_type。例: base / deepkernel / saas。
        name: 出力名。hybrid の OutputSpec.name に使う。None なら y0, y1, ...。
        input_type: normal / mixed。None の場合は親 ModelConfig または cat_dims から推定する。
        cat_dims: 出力ごとの cat_dims。None の場合は親 ModelConfig.cat_dims を使う。
        model_kwargs: 出力ごとのモデル生成 kwargs。
        fit_config: 出力ごとの fit 設定。None の場合は親または MultiOutputConfig.output_fit_configs を使う。
        output_spec_kwargs: hybrid OutputSpec に渡す kwargs。
    """

    task_type: str
    model_type: str = "base"
    name: str | None = None
    input_type: InputType | None = None
    cat_dims: Sequence[int] | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_config: FitConfig | None = None
    output_spec_kwargs: dict[str, Any] = field(default_factory=dict)


OutputConfigLike = ModelConfig | OutputConfig | Mapping[str, Any] | str


@dataclass
class MultiOutputConfig:
    """出力ごとに submodel を作り、multi-output / hybrid wrapper に束ねる設定。

    `output_configs` は文字列・辞書・OutputConfig・ModelConfig のいずれでも指定できます。

    Examples:
        文字列だけで指定する場合:

        ```python
        MultiOutputConfig(output_configs=["regression", "binary", "ordinal"])
        ```

        辞書で model_type や hybrid 用 kwargs まで指定する場合:

        ```python
        MultiOutputConfig(
            output_configs=[
                {"name": "strength", "task_type": "regression", "model_type": "base"},
                {
                    "name": "defect",
                    "task_type": "binary",
                    "model_type": "base",
                    "output_spec_kwargs": {"sign": -1.0, "positive_class": 1},
                },
            ]
        )
        ```
    """

    output_configs: Sequence[OutputConfigLike] | None = None
    output_fit_configs: Sequence[FitConfig | None] | FitConfig | None = None
    output_task_types: Sequence[str] | None = None
    output_names: Sequence[str] | None = None

    wrapper_cls: type | Callable[..., Any] | None = None
    wrapper_factory: Callable[..., Any] | None = None
    wrapper_kwargs: dict[str, Any] = field(default_factory=dict)

    use_hybrid: bool | None = None
    fit_submodels: bool = True
    fit_wrapper: bool = False

    output_spec_kwargs: Sequence[dict[str, Any]] | None = None
    train_y_slice_dim: int = -1


@dataclass
class MultiObjectiveConfig:
    """多目的獲得関数用の設定。"""

    ref_point: Any | None = None
    Y_baseline: Any | None = None
    partitioning: Any | None = None
    objective_thresholds: Any | None = None
    constraints: Any | None = None
    objective: Any | None = None

    scalarization_weights: Any | None = None
    scalarization_alpha: float = 0.05

    auto_partitioning: bool = True
    auto_scalarization: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionConfig:
    """獲得関数生成に必要な設定。"""

    name: str
    acqf_cls: type | Callable[..., Any] | None = None
    acqf_factory: Callable[..., Any] | None = None

    objective: Any = None
    sampler: Any = None
    acqf_kwargs: dict[str, Any] = field(default_factory=dict)

    context_fields: tuple[str, ...] = (
        "X_baseline",
        "X_pending",
        "Y_baseline",
        "best_f",
        "ref_point",
        "partitioning",
        "objective_thresholds",
        "mc_points",
        "constraints",
    )
    filter_kwargs_by_signature: bool = True


@dataclass
class DataContext:
    """獲得関数・最適化で共有するデータ文脈。"""

    bounds: Any | None = None
    X_baseline: Any | None = None
    X_pending: Any | None = None
    Y_baseline: Any | None = None

    best_f: Any | None = None
    ref_point: Any | None = None
    partitioning: Any | None = None
    objective_thresholds: Any | None = None
    mc_points: Any | None = None
    constraints: Any | None = None

    multi_objective: MultiObjectiveConfig | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizeConfig:
    """候補点最適化に必要な設定。"""

    q: int = 1
    num_restarts: int = 10
    raw_samples: int = 256
    sequential: bool = False

    optimizer: OptimizerName | Callable[..., Any] = "optimize_acqf"
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    post_processing_func: Callable[..., Any] | None = None
    fixed_features: dict[int, float] | None = None
    fixed_features_list: list[dict[int, float]] | None = None

    inequality_constraints: Any | None = None
    equality_constraints: Any | None = None

    return_best_only: bool = True


@dataclass
class ModelBundle:
    """生成・学習済みモデルと、その周辺情報をまとめた入れ物。"""

    model: Any
    train_X: Any
    train_Y: Any

    model_config: ModelConfig
    fit_config: FitConfig | None = None

    input_type: InputType | str = "normal"
    task_type: str = "regression"
    model_type: str = "base"
    cat_dims: list[int] = field(default_factory=list)

    mll: Any | None = None
    fit_result: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult:
    """候補点生成結果。"""

    candidates: Any
    acq_value: Any
    acqf: Any
    acq_config: AcquisitionConfig
    opt_config: OptimizeConfig
    data_context: DataContext


@dataclass
class PredictionResult:
    """予測結果。"""

    posterior: Any
    mean: Any | None = None
    variance: Any | None = None
