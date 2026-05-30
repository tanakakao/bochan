from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

from torch import Tensor


TaskType = Literal[
    "regression",
    "binary",
    "ordinal",
    "multiclass",
]

PosteriorMode = Literal[
    "objective",
    "mean",
    "latent",
    "probability",
    "expected_utility",
]


@dataclass(frozen=True)
class OutputSpec:
    """HybridMultiOutputModel に束ねる 1 出力分の定義。

    Args:
        name:
            出力名。`subset_output` / `output_indices` で文字列指定するためにも使う。
        task_type:
            出力のタスク種別。`regression`, `binary`, `ordinal`, `multiclass` を指定する。
        model:
            対応する single-output model または wrapper。
        output_index:
            submodel が multi-output の場合に、どの列をこの出力として使うか。
            通常は 0 のままでよい。
        sign:
            最大化方向へそろえる符号。最小化したい指標は -1 を指定する。
        weight:
            objective scale での重み。
        eq_target:
            目標値に近いほどよい出力に変換する場合の目標値。
            指定時は `-abs(y - eq_target) * weight` を使う。
        utility_values:
            binary / ordinal / multiclass の expected utility 変換に使う utility。
            binary は `[u0, u1]`、ordinal / multiclass は `[u0, ..., u_{K-1}]`。
        positive_class:
            binary / multiclass で probability mode の対象にするクラス。
            binary では 0 または 1、multiclass ではクラス index。
        transform:
            objective 化後の平均値に適用する任意の callable。
            分散は線形近似として `weight ** 2` のみ反映する。
    """

    name: str
    task_type: TaskType
    model: Any
    output_index: int = 0
    sign: float = 1.0
    weight: float = 1.0
    eq_target: Optional[float] = None
    utility_values: Optional[Sequence[float] | Tensor] = None
    positive_class: Optional[int] = None
    transform: Optional[Callable[[Tensor], Tensor]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ValueError("OutputSpec.name must be a non-empty string.")

        if self.task_type not in {"regression", "binary", "ordinal", "multiclass"}:
            raise ValueError(
                "OutputSpec.task_type must be one of "
                "'regression', 'binary', 'ordinal', or 'multiclass'. "
                f"Got {self.task_type!r}."
            )

        if int(self.output_index) < 0:
            raise ValueError("OutputSpec.output_index must be non-negative.")

        object.__setattr__(self, "output_index", int(self.output_index))
        object.__setattr__(self, "sign", float(self.sign))
        object.__setattr__(self, "weight", float(self.weight))

        if self.positive_class is not None:
            object.__setattr__(self, "positive_class", int(self.positive_class))

        if self.task_type == "binary" and self.positive_class not in (None, 0, 1):
            raise ValueError("For binary task, positive_class must be None, 0, or 1.")


__all__ = [
    "OutputSpec",
    "PosteriorMode",
    "TaskType",
]
