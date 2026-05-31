from __future__ import annotations

from typing import Any, Sequence, Union

import torch
from torch import Tensor

from .specs import OutputSpec

OutputIndex = Union[int, str]


def _coerce_binary_threshold(
    threshold: float | Tensor,
    *,
    ref: Tensor,
) -> Tensor:
    """binary threshold を ref と同じ device / dtype の Tensor に変換する。"""

    return torch.as_tensor(threshold, device=ref.device, dtype=ref.dtype)


def _predict_one_output(
    self,
    spec: OutputSpec,
    X: Tensor,
    *,
    binary_threshold: float | Tensor = 0.5,
    **kwargs: Any,
) -> Tensor:
    """1つの OutputSpec に対する予測値 / 予測ラベルを返す。

    - regression: posterior mean をそのまま返す。
    - binary: P(y=1) >= threshold で 0/1 ラベルを返す。
    - ordinal: class probability の argmax を返す。
    - multiclass: class probability の argmax を返す。
    """

    if spec.task_type == "regression":
        mean, _ = self._regression_stats(
            spec,
            X,
            output_mode="mean",
            **kwargs,
        )
        return mean

    if spec.task_type == "binary":
        _, _, p1 = self._binary_probability_stats(
            spec,
            X,
            **kwargs,
        )
        threshold = _coerce_binary_threshold(binary_threshold, ref=p1)
        return (p1 >= threshold).to(torch.long)

    if spec.task_type == "ordinal":
        probs = self._ordinal_class_probs(
            spec,
            X,
            **kwargs,
        )
        return probs.argmax(dim=-1).to(torch.long)

    if spec.task_type == "multiclass":
        probs = self._multiclass_probs(
            spec,
            X,
            **kwargs,
        )
        return probs.argmax(dim=-1).to(torch.long)

    raise RuntimeError(f"Unsupported task_type={spec.task_type!r}.")


def predict_class_list(
    self,
    X: Tensor,
    output_indices: OutputIndex | Sequence[OutputIndex] | Tensor | None = None,
    *,
    binary_threshold: float | Tensor = 0.5,
    **kwargs: Any,
) -> list[Tensor]:
    """各出力の予測値 / 予測ラベルを list で返す。

    Returns:
        list[Tensor]:
            各要素の shape は ``X.shape[:-1]``。
            regression は予測平均、binary / ordinal / multiclass は予測クラス。

    Notes:
        list 形式では regression の float dtype と分類ラベルの long dtype を
        そのまま保持できる。
    """

    X = self._unwrap_X(X)
    outputs = []
    for i in self._normalize_output_indices(output_indices):
        outputs.append(
            _predict_one_output(
                self,
                self.specs[i],
                X,
                binary_threshold=binary_threshold,
                **kwargs,
            )
        )
    return outputs


def predict_class(
    self,
    X: Tensor,
    output_indices: OutputIndex | Sequence[OutputIndex] | Tensor | None = None,
    *,
    binary_threshold: float | Tensor = 0.5,
    **kwargs: Any,
) -> Tensor:
    """Hybrid model の各出力を 1 つの Tensor として予測する。

    - regression 出力は posterior mean をそのまま返す。
    - binary 出力は ``P(y=1) >= binary_threshold`` による 0/1 ラベルを返す。
    - ordinal 出力は class probability の argmax ラベルを返す。
    - multiclass 出力は class probability の argmax ラベルを返す。

    Returns:
        Tensor:
            shape = ``X.shape[:-1] + (m_selected,)``。

    Notes:
        Tensor は単一 dtype しか持てないため、regression を含む場合は
        分類ラベルも regression と同じ浮動小数 dtype に変換される。
        dtype を出力ごとに保持したい場合は ``predict_class_list`` を使う。
    """

    values = predict_class_list(
        self,
        X,
        output_indices=output_indices,
        binary_threshold=binary_threshold,
        **kwargs,
    )
    if len(values) == 0:
        raise RuntimeError("No outputs were selected.")

    ref_shape = values[0].shape
    has_floating = any(torch.is_floating_point(v) for v in values)
    if has_floating:
        dtype = next(v.dtype for v in values if torch.is_floating_point(v))
    else:
        dtype = torch.long

    device = values[0].device
    out = []
    for i, value in enumerate(values):
        if value.shape != ref_shape:
            try:
                value = value.expand(ref_shape)
            except RuntimeError as e:
                raise RuntimeError(
                    "All predicted tensors must have the same shape before stacking. "
                    f"0={tuple(ref_shape)}, {i}={tuple(value.shape)}."
                ) from e
        out.append(value.to(device=device, dtype=dtype).unsqueeze(-1))

    return torch.cat(out, dim=-1)


def attach_prediction_methods(cls) -> None:
    """HybridMultiOutputModel に prediction helper を追加する。"""

    cls.predict_class_list = predict_class_list
    cls.predict_class = predict_class


__all__ = [
    "attach_prediction_methods",
    "predict_class",
    "predict_class_list",
]
