"""High-level configuration objects for the bochan API.

このモジュールは、モデル生成・学習・獲得関数生成・候補点最適化を
疎結合に扱うための dataclass 群を定義します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence


TaskType = Literal[
    "regression",
    "multi_objective",
    "binary",
    "multiclass",
    "ordinal",
]
InputType = Literal["normal", "mixed"]
ModelType = str
OptimizerName = Literal["optimize_acqf", "optimize_acqf_mixed"]


@dataclass
class ModelConfig:
    """モデル生成に必要な設定。

    `ModelConfig` には2つの使い方があります。

    1. 直接指定モード:
        `model_cls=SingleTaskGP` のようにモデルクラスを直接渡します。
        この場合、`task_type` / `model_type` は履歴・ログ用のメタ情報です。
        そのため、単に SingleTaskGP を使うだけなら `model_type="base"` を
        明示する必要はありません。

    2. registry 解決モード:
        `model_cls=None` とし、`task_type` / `model_type` / `cat_dims` から
        外部の registry を使ってモデルクラスを解決します。
        この場合は `model_type="base"` のような指定が意味を持ちます。

    Args:
        model_cls: 直接生成したいモデルクラス。None の場合は registry から解決する。
        task_type: regression / multi_objective / binary / multiclass / ordinal などのタスク種別。
        model_type: base / deepgp / deepkernel / saas / pca / rembo / rrp / hetero など。
            `model_cls` を直接指定する場合は主にメタ情報として扱う。
        input_type: normal / mixed。None の場合は cat_dims の有無から自動推定する。
        cat_dims: カテゴリ変数の列番号。空なら通常モデルとして扱う。
        input_transform: BoTorch 互換の input_transform。
        outcome_transform: BoTorch 互換の outcome_transform。
        model_kwargs: モデルクラスへ追加で渡す kwargs。
        train_x_name: モデルコンストラクタで使う train_X 引数名。
        train_y_name: モデルコンストラクタで使う train_Y 引数名。
        pass_cat_dims: cat_dims をモデルに渡すか。None なら cat_dims が非空の場合だけ渡す。
        pass_input_transform: input_transform を渡すか。
        pass_outcome_transform: outcome_transform を渡すか。
    """

    model_cls: type | Callable[..., Any] | None = None
    task_type: TaskType | str = "regression"
    model_type: ModelType = "base"
    input_type: InputType | None = None

    cat_dims: Sequence[int] | None = None
    input_transform: Any = None
    outcome_transform: Any = None

    model_kwargs: dict[str, Any] = field(default_factory=dict)

    train_x_name: str = "train_X"
    train_y_name: str = "train_Y"

    pass_cat_dims: bool | None = None
    pass_input_transform: bool = True
    pass_outcome_transform: bool = True


@dataclass
class FitConfig:
    """モデル学習に必要な設定。

    Args:
        fit_func: 学習関数。通常は `fit_gpytorch_mll` や独自の `fit_classifier_mll`。
        mll_factory: `model -> mll` を作る関数。
        mll_cls: MLL クラス。指定時は基本的に `mll_cls(model.likelihood, model, **mll_kwargs)` で作る。
        mll_kwargs: MLL 生成時の追加 kwargs。
        fit_kwargs: fit_func に渡す追加 kwargs。
        use_model_make_mll: model.make_mll() があれば優先して使う。
        skip_fit: True の場合、モデル生成だけ行い学習はしない。
    """

    fit_func: Callable[..., Any] | None = None
    mll_factory: Callable[..., Any] | None = None
    mll_cls: type | Callable[..., Any] | None = None

    mll_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_kwargs: dict[str, Any] = field(default_factory=dict)

    use_model_make_mll: bool = True
    skip_fit: bool = False


@dataclass
class MultiObjectiveConfig:
    """多目的獲得関数用の設定。

    EHVI / NEHVI / NParEGO などで共通して必要になりやすい情報をまとめます。

    Args:
        ref_point: Hypervolume 系獲得関数の参照点。基本的には目的空間で悪い側の点を指定する。
        Y_baseline: Pareto 分割や scalarization の基準に使う目的値。None なら train_Y を使う。
        partitioning: qEHVI などに渡す box decomposition。None かつ auto_partitioning=True なら自動生成を試みる。
        objective_thresholds: qNEHVI などで使う objective thresholds。
        constraints: BoTorch の constraints callable 群。
        objective: MCMultiOutputObjective など。AcquisitionConfig.objective が優先される。
        scalarization_weights: NParEGO 風の Chebyshev scalarization に使う重み。
        scalarization_alpha: Chebyshev scalarization の alpha。
        auto_partitioning: qEHVI 用 partitioning を自動生成するか。
        auto_scalarization: objective が未指定で scalarization_weights がある場合、自動で GenericMCObjective を作るか。
    """

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
    """獲得関数生成に必要な設定。

    Args:
        name: 獲得関数名。ログ・履歴用。
        acqf_cls: 獲得関数クラス。None の場合は `acqf_factory` を使う。
        acqf_factory: 獲得関数生成関数。`(bundle, config, data_context) -> acqf` を推奨。
        objective: MCObjective / MCMultiOutputObjective など。None の場合は渡さない。
        sampler: MC sampler。None の場合は渡さない。
        acqf_kwargs: 獲得関数に追加で渡す kwargs。
        context_fields: DataContext から獲得関数へ自動転送するフィールド。
        filter_kwargs_by_signature: True の場合、獲得関数の signature にない kwargs を落とす。
    """

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
