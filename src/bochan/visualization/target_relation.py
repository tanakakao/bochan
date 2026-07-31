"""Target-to-target Plotly visualization for mixed output types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from .utils import cycle_color_map, cycle_series


def _normalize_task_type(task_type: Any, series: pd.Series) -> str:
    """Normalize a task name into regression, ordinal, or categorical."""

    normalized = str(task_type or "").strip().lower()
    if normalized in {"binary", "multiclass", "classification", "categorical"}:
        return "categorical"
    if normalized == "ordinal":
        return "ordinal"
    if normalized in {"regression", "continuous", "numeric"}:
        return "regression"
    return "regression" if pd.api.types.is_numeric_dtype(series) else "categorical"


def show_target_relation_plot(
    y: pd.DataFrame,
    target1: str,
    target2: str,
    *,
    task_types: Mapping[str, str] | None = None,
    category_orders: Mapping[str, Sequence[Any]] | None = None,
    df_cand: pd.DataFrame | None = None,
    directions: Mapping[str, str] | None = None,
    show_pareto_front: bool = False,
    cycle: str | Sequence[Any] | pd.Series | None = None,
    aggregate_categorical: bool = True,
) -> Figure:
    """目的変数同士の関係を回帰・分類・順序変数に対応して描画する。

    Parameters
    ----------
    y:
        目的変数を含むデータフレーム。
    target1, target2:
        横軸・縦軸に使用する目的変数名。
    task_types:
        目的変数ごとのタスク種別。``regression``、``classification``、
        ``binary``、``multiclass``、``ordinal`` を受け付ける。省略時はdtypeから推定する。
    category_orders:
        分類・順序変数の表示順。順序回帰では低位から高位の順を指定する。
    df_cand:
        候補点の予測平均・標準偏差を含むデータフレーム。両軸が回帰の場合に
        ``show_pareto_plot`` へ渡す。
    directions:
        目的変数ごとの ``maximize`` または ``minimize``。両軸が回帰の場合の
        パレートフロント判定に使用する。
    show_pareto_front:
        両軸が回帰の場合に、現データの非支配点を結ぶフロントを表示する。
    cycle:
        サイクル列名またはサイクル系列。指定時はサイクル別にトレースを分ける。
    aggregate_categorical:
        両軸が非回帰の場合に、同じカテゴリ組み合わせを件数集約するかどうか。

    Returns
    -------
    Figure
        回帰、カテゴリ、順序カテゴリを混在して表示できるPlotly Figure。
    """

    for column in (target1, target2):
        if column not in y.columns:
            raise ValueError(f"y に列 {column!r} が存在しません。")
    if target1 == target2:
        raise ValueError("target1 and target2 must be different target variables.")

    normalized_tasks = dict(task_types or {})
    normalized_orders = dict(category_orders or {})
    task1 = _normalize_task_type(normalized_tasks.get(target1), y[target1])
    task2 = _normalize_task_type(normalized_tasks.get(target2), y[target2])
    if task1 == "regression" and task2 == "regression":
        from .plots import show_pareto_plot

        return show_pareto_plot(
            y,
            target1,
            target2,
            df_cand=df_cand,
            directions=directions,
            show_pareto_front=show_pareto_front,
            cycle=cycle,
        )

    categorical_pair = task1 != "regression" and task2 != "regression"

    frame = y[[target1, target2]].copy()
    cycles = cycle_series(cycle, y=y, length=len(y)) if cycle is not None else None
    if cycles is not None:
        frame["__cycle__"] = cycles.to_numpy()
    frame = frame.dropna(subset=[target1, target2])
    if frame.empty:
        raise ValueError("選択した目的変数に有効な組み合わせがありません。")

    figure = go.Figure()
    color_map = cycle_color_map(cycles)

    def add_trace(values: pd.DataFrame, name: str, color: Any | None = None) -> None:
        marker: dict[str, Any] = {"opacity": 0.76}
        if color is not None:
            marker["color"] = color

        if aggregate_categorical and categorical_pair:
            counts = (
                values.groupby([target1, target2], sort=False, dropna=False)
                .size()
                .reset_index(name="count")
            )
            marker["size"] = [min(46, 14 + 4 * int(value)) for value in counts["count"]]
            figure.add_trace(
                go.Scatter(
                    x=counts[target1],
                    y=counts[target2],
                    mode="markers+text",
                    text=[str(value) for value in counts["count"]],
                    textposition="middle center",
                    customdata=counts["count"].to_numpy(),
                    marker=marker,
                    name=name,
                    hovertemplate=(
                        f"{target1}: %{{x}}<br>{target2}: %{{y}}"
                        "<br>件数: %{customdata}<extra></extra>"
                    ),
                )
            )
            return

        marker["size"] = 9
        figure.add_trace(
            go.Scatter(
                x=values[target1],
                y=values[target2],
                mode="markers",
                marker=marker,
                name=name,
                hovertemplate=f"{target1}: %{{x}}<br>{target2}: %{{y}}<extra></extra>",
            )
        )

    if cycles is None:
        add_trace(frame, "入力データ")
    else:
        for cycle_value, color in color_map.items():
            subset = frame.loc[frame["__cycle__"] == cycle_value]
            if not subset.empty:
                add_trace(subset, f"cycle {cycle_value}", color)

    def axis_options(target: str, task: str) -> dict[str, Any]:
        options: dict[str, Any] = {"title": target}
        if task != "regression":
            options["type"] = "category"
            order = list(normalized_orders.get(target) or [])
            if order:
                options["categoryorder"] = "array"
                options["categoryarray"] = order
        return options

    figure.update_xaxes(**axis_options(target1, task1))
    figure.update_yaxes(**axis_options(target2, task2))
    figure.update_layout(
        height=600,
        width=700,
        legend_title_text="系列",
        font_size=16,
    )
    return figure


__all__ = ["show_target_relation_plot"]