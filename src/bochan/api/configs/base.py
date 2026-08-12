"""High-level configuration objects for the bochan API.

このモジュールは、モデル生成・学習・獲得関数生成・候補点最適化を
疎結合に扱うための dataclass 群を定義します。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from torch import nn

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
OptimizerName = Literal[
    "optimize_acqf",
    "optimize_acqf_mixed",
    "evo",
    "optimize_acqf_evo",
    "evo_mixed",
    "optimize_acqf_evo_mixed",
    "torch",
    "optimize_acqf_torch",
    "torch_mixed",
    "optimize_acqf_torch_mixed",
    "nsgaii",
    "optimize_acqf_nsgaii",
]
FinalPriority = Literal["grid", "constraints"]
SparseScore = Literal["abs", "value"]
SupportSelection = Literal["topk", "sample"]
InequalitySense = Literal["le", "ge"]
ObjectiveMode = Literal["auto", "none", "scalar", "multi_output"]
Direction = Literal["maximize", "minimize"]
REGRESSION_OUTCOME_TASK_TYPES: set[str] = {"regression", "multi_objective"}


class AutoStandardizeOutcomeTransform(nn.Module):
    """train_Y の出力次元から ``Standardize`` を遅延生成する transform。

    ``ModelConfig.outcome_transform=True`` のときに使う内部用 helper です。
    ``train_Y`` を受け取るまで出力次元 ``m`` が確定しないため、モデル生成時の
    初回 ``forward`` で ``botorch.models.transforms.outcome.Standardize`` を構築します。
    """

    _is_linear = True

    def __init__(self) -> None:
        super().__init__()
        self.standardize: Any | None = None

    def _ensure_transform(self, Y: Any | None = None) -> Any:
        if self.standardize is None:
            if Y is None or not hasattr(Y, "shape"):
                raise RuntimeError(
                    "AutoStandardizeOutcomeTransform must be initialized by calling it with train_Y first."
                )
            from botorch.models.transforms.outcome import Standardize

            m = int(Y.shape[-1]) if len(Y.shape) >= 2 else 1
            self.standardize = Standardize(m=m)
            if hasattr(Y, "device") and hasattr(Y, "dtype"):
                self.standardize = self.standardize.to(device=Y.device, dtype=Y.dtype)
            self.standardize.train(self.training)
        return self.standardize

    def forward(self, Y: Any, Yvar: Any | None = None, X: Any | None = None) -> tuple[Any, Any | None]:
        return self._ensure_transform(Y)(Y, Yvar, X=X)

    def untransform(self, Y: Any, Yvar: Any | None = None, X: Any | None = None) -> tuple[Any, Any | None]:
        return self._ensure_transform().untransform(Y, Yvar, X=X)

    def untransform_posterior(self, posterior: Any, X: Any | None = None) -> Any:
        return self._ensure_transform().untransform_posterior(posterior, X=X)

    def subset_output(self, idcs: Any) -> AutoStandardizeOutcomeTransform:
        new = type(self)()
        if self.standardize is not None:
            new.standardize = self.standardize.subset_output(idcs)
        return new


def is_regression_outcome_task(task_type: str) -> bool:
    """Return whether ``task_type`` supports BoTorch outcome transforms."""

    return str(task_type) in REGRESSION_OUTCOME_TASK_TYPES


def build_outcome_transform_for_task(task_type: str, outcome_transform: bool | Any | None) -> Any | None:
    """Build an outcome transform only for regression-like tasks.

    ``True`` means ``Standardize`` should be used.  Classification, multiclass,
    ordinal, and hybrid wrapper tasks always return ``None``; hybrid submodels
    are handled after each output is resolved to its own task type.
    """

    if not is_regression_outcome_task(str(task_type)):
        return None
    if isinstance(outcome_transform, bool):
        return AutoStandardizeOutcomeTransform() if outcome_transform else None
    if outcome_transform is None:
        return None
    if isinstance(outcome_transform, AutoStandardizeOutcomeTransform):
        return AutoStandardizeOutcomeTransform()
    return outcome_transform


@dataclass
class InputTransformConfig:
    """input_transform を文字列API側から簡易構築するための設定。

    `bochan.models.transforms.input.build_input_transform` を内部で呼びます。

    Args:
        normalize: Normalize を使うか。既存挙動との互換のため、既定値は True。
        perturbation: 入力摂動を使うか。
        n_w: 摂動サンプル数。
        std: 摂動の標準偏差。normalize=True の場合は Normalize 後の空間での標準偏差。
            normalize=False の場合は raw 空間での標準偏差。
        bounds: 明示的な bounds。None の場合は train_X の min/max から自動生成する。
        categorical_idx: Normalize / perturbation から除外するカテゴリ列。
            None の場合は ModelConfig.cat_dims を使う。

    Notes:
        factory 側との互換性を保つため、内部的には bounds に lightweight な
        metadata dict を格納します。`build_input_transform(...)` がこの metadata を解釈し、
        normalize / perturbation を独立に切り替えます。
    """

    normalize: bool = True
    perturbation: bool = False
    n_w: int = 16
    std: float = 0.1
    bounds: Any | None = None
    categorical_idx: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, dict) or not self.bounds.get("__bochan_input_transform_config__", False):
            self.bounds = {
                "__bochan_input_transform_config__": True,
                "normalize": bool(self.normalize),
                "bounds": self.bounds,
            }


@dataclass
class ModelConfig:
    """モデル生成に必要な設定。

    基本的には `task_type` と `model_type` を文字列で指定し、モデル固有の引数は
    `model_kwargs` に渡します。`model_cls` や `model_factory` は高度な利用・テスト用です。

    Args:
        outcome_transform: regression 系モデルに Standardize を適用するか。
            True の場合は ``AutoStandardizeOutcomeTransform`` を使って train_Y の出力次元から
            ``Standardize(m=...)`` を自動構築します。False の場合は outcome_transform を渡しません。
            BoTorch 互換の transform オブジェクトを直接渡すこともできます。
            binary / multiclass / ordinal では自動的に無効化されます。
            hybrid では親設定として保持し、submodel が regression / multi_objective に解決されたときだけ
            適用されます。
    """

    model_cls: type | Callable[..., Any] | None = None
    model_factory: Callable[..., Any] | None = None

    task_type: TaskType | str = "regression"
    model_type: ModelType = "base"
    input_type: InputType | None = None

    cat_dims: Sequence[int] | None = None
    input_transform: Any = None
    input_transform_config: InputTransformConfig | None = None
    outcome_transform: bool | Any = True

    model_kwargs: dict[str, Any] = field(default_factory=dict)
    multi_output_config: MultiOutputConfig | None = None

    train_x_name: str = "train_X"
    train_y_name: str = "train_Y"

    pass_train_data: bool = True
    pass_cat_dims: bool | None = None
    pass_input_transform: bool = True
    pass_outcome_transform: bool = True

    def __post_init__(self) -> None:
        task_type = str(self.task_type)

        if task_type == "hybrid":
            # hybrid wrapper 自体には outcome_transform を適用しない。
            # 値は submodel 解決時に dataclasses.replace(...) で継承され、
            # 各 submodel の task_type に応じて再評価される。
            self.pass_outcome_transform = False
            return

        if not is_regression_outcome_task(task_type):
            self.outcome_transform = None
            self.pass_outcome_transform = False
            return

        if str(self.model_type).startswith("gamma_") and self.outcome_transform is True:
            from bochan.models.transforms.outcome import PositiveScaleOutcomeTransform

            self.outcome_transform = PositiveScaleOutcomeTransform(validate_positive=True)

        if str(self.model_type).startswith("beta_"):
            if self.outcome_transform not in (None, False, True):
                raise ValueError(
                    "Beta regression does not support generic outcome transforms; "
                    "targets must remain on the open unit interval."
                )
            self.outcome_transform = None

        if str(self.model_type).startswith("poisson_"):
            if self.outcome_transform not in (None, False, True):
                raise ValueError(
                    "Poisson models require raw non-negative integer counts and do not "
                    "support outcome_transform."
                )
            self.outcome_transform = None

        if str(self.model_type).startswith("negative_binomial_"):
            if self.outcome_transform not in (None, False, True):
                raise ValueError(
                    "Negative Binomial models require raw non-negative integer counts "
                    "and do not support outcome_transform."
                )
            self.outcome_transform = None

        self.outcome_transform = build_outcome_transform_for_task(task_type, self.outcome_transform)
        self.pass_outcome_transform = self.outcome_transform is not None


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
    """出力ごとに submodel を作り、multi-output / hybrid wrapper に束ねる設定。

    `output_configs` を省略した場合、`train_Y` の出力数だけ親 `ModelConfig.task_type`
    と `ModelConfig.model_type` を複製します。例えば親が `task_type="binary"` なら、
    すべての出力が binary submodel として構築されます。

    出力ごとに task / model / kwargs を変えたい場合だけ、`output_configs` を指定します。
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
class ObjectiveConfig:
    """API 側で objective を自動構築するための設定。

    ユーザーは objective のクラスを直接選ばず、出力・方向・risk 集約などを指定します。
    API は model / task_type / hybrid 出力情報から、適切な objective 実装を選びます。
    """

    mode: ObjectiveMode = "auto"
    output: Any | None = None
    outputs: Sequence[Any] | None = None
    specs: Sequence[Any] | None = None
    directions: Sequence[Direction | bool | float | int] | None = None
    weights: Sequence[float] | None = None
    eq_targets: Sequence[float | None] | None = None
    direction: Direction | bool | float | int = "maximize"
    weight: float = 1.0
    eq_target: float | None = None
    n_w: int | None = None
    risk_type: str | None = None
    alpha: float = 0.5
    maximize: bool = True
    aggregate_mean_when_no_risk: bool = True
    allow_unexpanded: bool = True
    utility_values: Sequence[float] | Any | None = None
    ordinal_likelihood: Any | None = None
    objective_kwargs: dict[str, Any] = field(default_factory=dict)


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
    """獲得関数生成に必要な設定。

    Args:
        objective: 生成済み objective。指定された場合はそのまま獲得関数に渡します。
        objective_config: objective を API 側で自動生成するための設定。
            直接 objective が渡されていない場合に使います。
        objective_factory: 高度な上書き用の objective factory。
            通常は ``objective_config`` を推奨します。
        objective_kwargs: ``objective_factory`` に渡す追加引数。
    """

    name: str
    acqf_cls: type | Callable[..., Any] | None = None
    acqf_factory: Callable[..., Any] | None = None

    objective: Any = None
    objective_config: ObjectiveConfig | None = None
    objective_factory: Callable[..., Any] | None = None
    objective_kwargs: dict[str, Any] = field(default_factory=dict)
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
    """候補点の丸め・k-sparse・制約補修を自動生成する設定。"""

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
    """予測結果。

    binary の ``mean`` はクラス1確率です。``variance_kind`` が
    ``bernoulli_observation`` の場合、variance は ``p * (1 - p)`` であり、
    確率推定値そのものの epistemic variance ではありません。
    """

    posterior: Any
    mean: Any | None = None
    variance: Any | None = None
    task_type: str | None = None
    prediction_space: str | None = None
    variance_kind: str | None = None
