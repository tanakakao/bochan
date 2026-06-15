"""Plotly based visualization functions for bochan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from .data import (
    candidates_dataframe,
    grid_1d_plot,
    grid_2d,
    prediction_dataframe,
    study_target_dataframe,
    training_dataframe,
    tri_grid,
)
from .utils import cycle_color_map, cycle_series


def _finite_range(values: Any) -> list[float] | None:
    """有限な数値から Plotly axis range を作成する。"""

    try:
        arr = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if lo == hi:
        pad = 0.5 if lo == 0.0 else abs(lo) * 0.05
        lo -= pad
        hi += pad
    return [lo, hi]


def show_yyplot(
    y: pd.DataFrame,
    target: str,
    preds: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    df_cand: pd.DataFrame | None = None,
    *,
    cv: bool = False,
    cv_result: dict[str, pd.DataFrame] | None = None,
    cycle: str | Sequence[Any] | pd.Series | None = None,
) -> Figure:
    """実測値と予測値の YY plot を作成する。"""

    if target not in y.columns:
        raise ValueError(f"y に列 {target!r} が存在しません。")
    y_true = pd.to_numeric(y[target], errors="coerce")
    cyc = cycle_series(cycle, y=y, length=len(y)) if cycle is not None else None
    cmap = cycle_color_map(cyc)
    fig = go.Figure()

    if df_cand is not None and f"{target}_mean" in df_cand and f"{target}_std" in df_cand:
        mu = pd.to_numeric(df_cand[f"{target}_mean"], errors="coerce")
        sd = pd.to_numeric(df_cand[f"{target}_std"], errors="coerce").abs()
        fig.add_trace(go.Scatter(x=mu, y=mu, mode="markers", name="候補点", marker=dict(color="green", size=10), error_y=dict(type="data", array=sd, visible=True)))

    cv_ok = cv and isinstance(cv_result, dict) and all(k in cv_result for k in ("mean_train_cv", "std_train_cv", "mean_test_cv", "std_test_cv"))
    if cv_ok:
        for label, marker_symbol in (("train", "circle"), ("test", "x")):
            mean = pd.to_numeric(cv_result[f"mean_{label}_cv"][target], errors="coerce")
            std = pd.to_numeric(cv_result[f"std_{label}_cv"][target], errors="coerce").abs()
            if cyc is None:
                fig.add_trace(go.Scatter(x=y_true, y=mean, mode="markers", name=f"入力データ({label})", marker=dict(symbol=marker_symbol, size=10), error_y=dict(type="data", array=std, visible=True)))
            else:
                for c, color in cmap.items():
                    mask = cyc == c
                    fig.add_trace(go.Scatter(x=y_true[mask], y=mean[mask], mode="markers", name=f"{label} (cycle {c})", marker=dict(symbol=marker_symbol, color=color, size=9, line=dict(width=0.5, color="black")), error_y=dict(type="data", array=std[mask], visible=True)))
    elif preds is not None:
        mean_df, std_df = preds
        mean = pd.to_numeric(mean_df[target], errors="coerce")
        std = pd.to_numeric(std_df[target], errors="coerce").abs()
        if cyc is None:
            fig.add_trace(go.Scatter(x=y_true, y=mean, mode="markers", name="入力データ", marker=dict(color="blue", size=10), error_y=dict(type="data", array=std, visible=True)))
        else:
            for c, color in cmap.items():
                mask = cyc == c
                fig.add_trace(go.Scatter(x=y_true[mask], y=mean[mask], mode="markers", name=f"cycle {c}", marker=dict(color=color, size=10, line=dict(width=0.5, color="black")), error_y=dict(type="data", array=std[mask], visible=True)))

    values = [y_true]
    if preds is not None:
        values.append(pd.to_numeric(preds[0][target], errors="coerce"))
    if df_cand is not None and f"{target}_mean" in df_cand:
        values.append(pd.to_numeric(df_cand[f"{target}_mean"], errors="coerce"))
    all_v = pd.concat(values).dropna()
    lo, hi = (0.0, 1.0) if all_v.empty else (float(all_v.min()), float(all_v.max()))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, line=dict(color="red")))
    fig.update_layout(height=600, width=600, xaxis_title="実測値", yaxis_title="予測値", legend_title_text="系列", font_size=16)
    return fig


def show_pareto_plot(y: pd.DataFrame, target1: str, target2: str, df_cand: pd.DataFrame | None = None, *, cycle: str | Sequence[Any] | pd.Series | None = None) -> Figure:
    """2目的の実測値と候補点を散布する。"""

    for col in (target1, target2):
        if col not in y.columns:
            raise ValueError(f"y に列 {col!r} が存在しません。")
    x = pd.to_numeric(y[target1], errors="coerce")
    z = pd.to_numeric(y[target2], errors="coerce")
    cyc = cycle_series(cycle, y=y, length=len(y)) if cycle is not None else None
    cmap = cycle_color_map(cyc)
    fig = go.Figure()
    if df_cand is not None and all(c in df_cand for c in (f"{target1}_mean", f"{target1}_std", f"{target2}_mean", f"{target2}_std")):
        fig.add_trace(go.Scatter(
            x=pd.to_numeric(df_cand[f"{target1}_mean"], errors="coerce"),
            y=pd.to_numeric(df_cand[f"{target2}_mean"], errors="coerce"),
            mode="markers", name="候補点", marker=dict(color="green", size=10),
            error_x=dict(type="data", array=pd.to_numeric(df_cand[f"{target1}_std"], errors="coerce").abs(), visible=True),
            error_y=dict(type="data", array=pd.to_numeric(df_cand[f"{target2}_std"], errors="coerce").abs(), visible=True),
        ))
    if cyc is None:
        fig.add_trace(go.Scatter(x=x, y=z, mode="markers", name="入力データ", marker=dict(color="blue", size=10)))
    else:
        for c, color in cmap.items():
            mask = cyc == c
            fig.add_trace(go.Scatter(x=x[mask], y=z[mask], mode="markers", name=f"cycle {c}", marker=dict(color=color, size=9, line=dict(width=0.5, color="black"))))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=600, width=600, xaxis_title=target1, yaxis_title=target2, legend_title_text="系列", font_size=16)
    return fig


def show_target_over_cycle_study(study: Any, target: str, *, agg: str | None = None, target_cols: Sequence[str] | None = None, cycle_col: str = "cycle") -> Figure:
    """BochanStudy の trial metadata に保存された cycle ごとの目的変数推移を描く。"""

    df = study_target_dataframe(study, target=target, target_cols=target_cols, cycle_col=cycle_col)
    valid_aggs = {"mean", "median", "min", "max", "count"}
    if agg is not None and agg not in valid_aggs:
        raise ValueError(f"agg must be one of {valid_aggs} or None.")
    fig = go.Figure()
    if agg is None:
        fig.add_trace(go.Scatter(x=df[cycle_col], y=df[target], mode="markers", name=f"{target} (raw)"))
    elif agg == "mean":
        g = df.groupby(cycle_col, dropna=False)[target].agg(["mean", "std"]).reset_index()
        fig.add_trace(go.Scatter(x=g[cycle_col], y=g["mean"], mode="lines+markers", name=f"{target} (mean)", error_y=dict(type="data", array=g["std"].fillna(0), visible=True)))
    else:
        g = df.groupby(cycle_col, dropna=False)[target].agg(agg).reset_index(name="val")
        fig.add_trace(go.Scatter(x=g[cycle_col], y=g["val"], mode="lines+markers", name=f"{target} ({agg})"))
    fig.update_layout(height=450, width=700, xaxis_title=cycle_col, yaxis_title=target, font_size=14)
    return fig


def show_1dplot_with_pred(feature: str, target: str, data_1d_plot: tuple[pd.DataFrame, pd.DataFrame, np.ndarray], X: pd.DataFrame, y: pd.DataFrame, df_cand: pd.DataFrame | None = None, *, cycle: str | Sequence[Any] | pd.Series | None = None) -> Figure:
    """1D 予測平均±標準偏差と学習点・候補点を描く。"""

    mean_df, std_df, x_grid = data_1d_plot
    mu = pd.to_numeric(mean_df[target], errors="coerce").to_numpy()
    sd = pd.to_numeric(std_df[target], errors="coerce").abs().to_numpy()
    xg = np.asarray(x_grid).ravel()
    try:
        order = np.argsort(xg.astype(float))
        xg, mu, sd = xg[order], mu[order], sd[order]
    except Exception:
        pass
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xg, y=mu + sd, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xg, y=mu - sd, mode="lines", fill="tonexty", line=dict(width=0), name=f"{target} ±1σ", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xg, y=mu, mode="lines", name=f"{target} 予測平均", line=dict(color="red", width=2)))

    cyc = cycle_series(cycle, X=X, y=y, length=len(X)) if cycle is not None else None
    cmap = cycle_color_map(cyc)
    if cyc is None:
        fig.add_trace(go.Scatter(x=X[feature], y=y[target], mode="markers", name="入力データ", marker=dict(color="blue", size=9)))
    else:
        for c, color in cmap.items():
            mask = cyc == c
            fig.add_trace(go.Scatter(x=X.loc[mask, feature], y=y.loc[mask, target], mode="markers", name=f"cycle {c}", marker=dict(color=color, size=9, line=dict(width=0.5, color="black"))))
    if df_cand is not None and all(c in df_cand for c in (feature, f"{target}_mean", f"{target}_std")):
        fig.add_trace(go.Scatter(x=df_cand[feature], y=df_cand[f"{target}_mean"], mode="markers", name="候補点", marker=dict(color="green", size=10), error_y=dict(type="data", array=df_cand[f"{target}_std"].abs(), visible=True)))
    fig.update_layout(height=600, width=800, xaxis_title=feature, yaxis_title=target, legend_title_text="系列", font_size=16)
    return fig


def show_scatter_with_acqf(feature_col1: str, feature_col2: str, target_col: str, data_2d_plot: Sequence[Any], X: pd.DataFrame, y: pd.DataFrame, df_cand: pd.DataFrame | None = None, *, show_type: str = "acqf", cycle: str | Sequence[Any] | pd.Series | None = None) -> Figure:
    """2D 散布図に獲得関数または予測値の等高線を重ねる。"""

    Z_raw, grid1, grid2 = data_2d_plot
    grid1_arr = np.asarray(grid1).ravel()
    grid2_arr = np.asarray(grid2).ravel()
    Z = np.asarray(Z_raw)[0] if np.asarray(Z_raw).ndim == 3 else np.asarray(Z_raw)
    fig = go.Figure()
    fig.add_trace(go.Contour(z=Z, x=grid1_arr, y=grid2_arr, ncontours=25, contours_coloring="heatmap", colorscale="RdBu_r", colorbar=dict(title="獲得関数" if show_type == "acqf" else "予測値", lenmode="pixels", len=200), hoverinfo="none"))
    if df_cand is not None and feature_col1 in df_cand and feature_col2 in df_cand:
        fig.add_trace(go.Scatter(x=df_cand[feature_col1], y=df_cand[feature_col2], mode="markers", name="候補点", marker=dict(color="green", size=12, symbol="diamond", line=dict(width=0.8, color="black"))))
    cyc = cycle_series(cycle, X=X, y=y, length=len(X)) if cycle is not None else None
    cmap = cycle_color_map(cyc)
    if cyc is None:
        fig.add_trace(go.Scatter(x=X[feature_col1], y=X[feature_col2], mode="markers", name="入力データ", marker=dict(size=10, color=y[target_col], colorscale="RdBu_r", showscale=False)))
    else:
        for c, color in cmap.items():
            mask = cyc == c
            fig.add_trace(go.Scatter(x=X.loc[mask, feature_col1], y=X.loc[mask, feature_col2], mode="markers", name=f"入力データ (cycle {c})", marker=dict(size=10, color=color, line=dict(width=0.6, color="black"))))
    fig.update_layout(height=600, width=800, xaxis_title=feature_col1, yaxis_title=feature_col2, legend_title_text="cycle" if cyc is not None else "系列", font_size=16)
    x_range = _finite_range(grid1_arr)
    y_range = _finite_range(grid2_arr)
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def show_triscatter_with_acqf(feature_col1: str, feature_col2: str, feature_col3: str, target_col: str, data_tri_plot: tuple[np.ndarray, Any], X: pd.DataFrame, y: pd.DataFrame, df_cand: pd.DataFrame | None = None, *, show_type: str = "acqf", cycle: str | Sequence[Any] | pd.Series | None = None) -> Figure:
    """三角散布図に獲得関数または予測値の等高線を重ねる。"""

    values, grid = data_tri_plot
    grid = np.asarray(grid, dtype=float)
    values_flat = np.ravel(values).astype(float)

    if grid.ndim != 2 or grid.shape[0] != 3:
        raise ValueError("grid must have shape (3, n).")
    if grid.shape[1] != values_flat.size:
        raise ValueError("grid と values の点数が一致していません。")

    finite_mask = np.isfinite(values_flat) & np.all(np.isfinite(grid), axis=0)
    if not np.any(finite_mask):
        raise ValueError("テルナリー等高線を生成できる有限な grid / values がありません。")

    grid_valid = grid[:, finite_mask]
    values_valid = values_flat[finite_mask]

    fig = go.Figure()

    try:
        contour_fig = ff.create_ternary_contour(
            grid_valid,
            values_valid,
            pole_labels=[feature_col1, feature_col2, feature_col3],
            ncontours=12,
            coloring=None,
            colorscale="RdBu",
            showscale=False,
            showmarkers=False,
        )
        contour_fig.update_traces(showlegend=False)
        fig.add_traces(contour_fig.data)
    except Exception as e:
        raise ValueError(
            "テルナリー等高線の生成に失敗しました。"
            "`grid` は shape=(3, n) で、各点の和が 1 になるように正規化されている必要があります。"
            f" 元のエラー: {e}"
        ) from e

    value_range = _finite_range(values_valid) or [0.0, 1.0]
    fig.add_trace(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="markers",
            marker=dict(
                color=value_range,
                cmin=value_range[0],
                cmax=value_range[1],
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(
                    title="獲得関数" if show_type == "acqf" else "予測値",
                    lenmode="pixels",
                    len=200,
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if df_cand is not None and all(c in df_cand for c in (feature_col1, feature_col2, feature_col3)):
        fig.add_trace(go.Scatterternary(a=df_cand[feature_col1], b=df_cand[feature_col2], c=df_cand[feature_col3], mode="markers", name="候補点", marker=dict(color="green", size=12, symbol="diamond", line=dict(width=0.8, color="black"))))
    cyc = cycle_series(cycle, X=X, y=y, length=len(X)) if cycle is not None else None
    cmap = cycle_color_map(cyc)
    if cyc is None:
        fig.add_trace(go.Scatterternary(a=X[feature_col1], b=X[feature_col2], c=X[feature_col3], mode="markers", name="入力データ", marker=dict(color=y[target_col], colorscale="RdBu_r", showscale=False, size=10)))
    else:
        for c, color in cmap.items():
            mask = cyc == c
            fig.add_trace(go.Scatterternary(a=X.loc[mask, feature_col1], b=X.loc[mask, feature_col2], c=X.loc[mask, feature_col3], mode="markers", name=f"入力データ (cycle {c})", marker=dict(color=color, size=10, line=dict(width=0.6, color="black"))))
    fig.update_layout(
        height=600,
        width=800,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        ternary=dict(aaxis=dict(title=feature_col1), baxis=dict(title=feature_col2), caxis=dict(title=feature_col3), sum=1),
        legend_title_text="cycle" if cyc is not None else "系列",
        font_size=12,
    )
    return fig


def show_yyplot_from_optimizer(obj: Any, target: str, *, feature_cols: Sequence[str] | None = None, target_cols: Sequence[str] | None = None, candidate_result: Any | None = None, cycle: str | Sequence[Any] | pd.Series | None = None) -> Figure:
    X_df, y_df = training_dataframe(obj, feature_cols=feature_cols, target_cols=target_cols)
    pred = prediction_dataframe(obj, X_df.to_numpy(), target_cols=list(y_df.columns))
    df_cand = candidates_dataframe(obj, candidate_result=candidate_result, feature_cols=feature_cols, target_cols=list(y_df.columns))
    return show_yyplot(y_df, target, pred, df_cand, cycle=cycle)


def show_1dplot_from_optimizer(obj: Any, feature: str, target: str, *, feature_cols: Sequence[str] | None = None, target_cols: Sequence[str] | None = None, value_dict: dict[str, Any] | None = None, candidate_result: Any | None = None, n: int = 50, cycle: str | Sequence[Any] | pd.Series | None = None) -> Figure:
    X_df, y_df = training_dataframe(obj, feature_cols=feature_cols, target_cols=target_cols)
    data = grid_1d_plot(obj, feature, value_dict, feature_cols=list(X_df.columns), target_cols=list(y_df.columns), n=n)
    df_cand = candidates_dataframe(obj, candidate_result=candidate_result, feature_cols=list(X_df.columns), target_cols=list(y_df.columns))
    return show_1dplot_with_pred(feature, target, data, X_df, y_df, df_cand, cycle=cycle)


def show_scatter_with_acqf_from_optimizer(obj: Any, feature_col1: str, feature_col2: str, target_col: str, *, feature_cols: Sequence[str] | None = None, target_cols: Sequence[str] | None = None, value_dict: dict[str, Any] | None = None, candidate_result: Any | None = None, n: int = 25, show_type: str = "acqf", cycle: str | Sequence[Any] | pd.Series | None = None) -> Figure:
    X_df, y_df = training_dataframe(obj, feature_cols=feature_cols, target_cols=target_cols)
    data = grid_2d(obj, [feature_col1, feature_col2], target_col, value_dict, feature_cols=list(X_df.columns), target_cols=list(y_df.columns), candidate_result=candidate_result, n=n, show_type=show_type)  # type: ignore[arg-type]
    df_cand = candidates_dataframe(obj, candidate_result=candidate_result, feature_cols=list(X_df.columns), target_cols=list(y_df.columns))
    return show_scatter_with_acqf(feature_col1, feature_col2, target_col, data, X_df, y_df, df_cand, show_type=show_type, cycle=cycle)


def show_triscatter_with_acqf_from_optimizer(obj: Any, feature_col1: str, feature_col2: str, feature_col3: str, target_col: str, *, feature_cols: Sequence[str] | None = None, target_cols: Sequence[str] | None = None, value_dict: dict[str, Any] | None = None, candidate_result: Any | None = None, sum_value: float | None = None, n: int = 35, show_type: str = "acqf", cycle: str | Sequence[Any] | pd.Series | None = None) -> Figure:
    X_df, y_df = training_dataframe(obj, feature_cols=feature_cols, target_cols=target_cols)
    data = tri_grid(obj, [feature_col1, feature_col2, feature_col3], target_col, value_dict, feature_cols=list(X_df.columns), target_cols=list(y_df.columns), candidate_result=candidate_result, sum_value=sum_value, n=n, show_type=show_type)  # type: ignore[arg-type]
    df_cand = candidates_dataframe(obj, candidate_result=candidate_result, feature_cols=list(X_df.columns), target_cols=list(y_df.columns))
    return show_triscatter_with_acqf(feature_col1, feature_col2, feature_col3, target_col, data, X_df, y_df, df_cand, show_type=show_type, cycle=cycle)
