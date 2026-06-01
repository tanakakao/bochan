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
FinalPriority = Literal["grid", "constraints"]
SparseScore = Literal["abs", "value"]
SupportSelection = Literal["topk", "sample"]
InequalitySense = Literal["le", "ge"]


@dataclass
class InputTransformConfig:
    """input_transform を文字列API側から簡易構築するための設定。

    `bochan.models.transforms.input.build_input_transform` を内部で呼びます。

    Args:
        perturbation: 入力摂動を使うか。
        n_w: 摂動サンプル数。
        std: 摂動の標準偏差。Normalize 後の空間での標準偏差。
        bounds: 明示的な bounds。None の場合は train_X の min/max から自動生成する。
        categorical_idx: Normalize / perturbation から除外するカテゴリ列。
            None の場合は ModelConfig.cat_dims を使う。
    """

    perturbation: bool = False
    n_w: int = 16
    std: float = 0.1
    bounds: Any | None = None
    categorical_idx: Sequence[int] | None = None


@dataclass
class ModelConfig:
    """モデル生成に必要な設定。

    基本的には `task_type` と `model_type` を文字列で指定し、モデル固有の引数は
    `model_kwargs` に渡します。`model_cls` や `model_factory` は高度な利用・テスト用です。
    """

    model_cls: type | Callable[..., Any] | None = None
    model_factory: Callable[..., Any] | None = None

    task_type: TaskType | str = "regression"
    model_type: ModelType = "base"
    input_type: InputType | None = None

    cat_dims: Sequence[int] | None = None
    input_transform: Any = None
    input_transform_config: InputTransformConfig | None = None
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
    """学習設定。

    通常利用では `fit_func`, `mll_factory`, `mll_cls` を指定する必要はありません。
    それらは `task_type` / `model_type` / model の持つ `make_mll()` から内部で自動選択します。

    Advanced:
        `fit_func`, `mll_factory`, `mll_cls` を指定すると自動解決を上書きできます。
    """

    method: str = "auto"
    num_epochs: int | None = None
    lr: float | None = None
    batch_size: int | None = None
    shuffle: bool = True
    verbose: bool = False
    clip_grad_norm: float | None = None
    maxiter: int | None = None
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    fit_kwargs: dict[str, Any] = field(default_factory=dict)
    mll_kwargs: dict[str, Any] = field(default_factory=dict)

    skip_fit: bool = False

    fit_func: Callable[..., Any] | None = None
    mll_factory: Callable[..., Any] | None = None
    mll_cls: type | Callable[..., Any] | None = None
    use_model_make_mll: bool = True


@dataclass
class OutputConfig:
    """multi-output / hybrid の各出力を文字列中心で指定するための設定。"""

    task_type: str
    model_type: str = "base"
    name: str | None = None
    input_type: InputType | None = None
    cat_dims: Sequence[int] | None = None
    input_transform_config: InputTransformConfig | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_config: FitConfig | None = None
    output_spec_kwargs: dict[str, Any] = field(default_factory=dict)


OutputConfigLike = ModelConfig | OutputConfig | Mapping[str, Any] | str


@dataclass
class MultiOutputConfig:
    """出力ごとに submodel を作り、multi-output / hybrid wrapper に束ねる設定。"""

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
class CandidateRepairConfig:
    """候補点の丸め・k-sparse・制約補修を自動生成する設定。

    `bochan.constraints.postprocess.make_grid_k_sparse_post_processing_func` を
    `OptimizeConfig` 内で組み立てるための薄い設定です。

    Args:
        bounds: repair 用 bounds。None の場合は `optimize_candidates(..., bounds=...)` を使う。
        numeric_indices: grid rounding 対象の数値列。None なら全列。
        steps: grid step。None なら丸めなし。
        comp_idx: k-sparse 対象列。None / [] なら k-sparse は実質無効。
        k: comp_idx 内で許容する active component 数。
        equality_constraints: repair 用の等式制約。None なら OptimizeConfig.equality_constraints を使う。
        inequality_constraints: repair 用の不等式制約。None なら OptimizeConfig.inequality_constraints を使う。
        inequality_sense: k-sparse repair 側の不等式向き。
        fixed_features: repair 用 fixed_features。None なら OptimizeConfig.fixed_features を使う。
        final_sum_constraint: active support 上の最終和制約。
        diversify: q-batch 内の重複を軽くずらして再補修するか。
        final_priority: 最後に grid / constraints のどちらを優先するか。
    """

    bounds: Any | None = None
    numeric_indices: Sequence[int] | None = None
    steps: Any | None = None
    comp_idx: Sequence[int] | None = None
    k: int = 0

    equality_constraints: Any | None = None
    inequality_constraints: Any | None = None
    inequality_sense: InequalitySense = "le"
    fixed_features: dict[int, float] | None = None
    final_sum_constraint: tuple[Sequence[int], float] | None = None

    diversify: bool = False
    diversify_kwargs: dict[str, Any] | None = None
    score: SparseScore = "abs"
    support_selection: SupportSelection = "topk"
    sample_tau: float = 0.2
    sample_eps: float = 0.05
    generator: Any | None = None
    max_iters: int = 12
    num_alternations: int = 2
    final_priority: FinalPriority = "grid"
    support_eps: float = 0.0


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
    repair_config: CandidateRepairConfig | None = None
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
