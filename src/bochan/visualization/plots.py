"""Plotly based visualization functions for bochan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def _pareto_front_dataframe(
    y: pd.DataFrame,
    target1: str,
    target2: str,
    directions: Mapping[str, str] | None,
) -> pd.DataFrame:
    """Return unique non-dominated observed points in the original value scale."""

    frame = pd.DataFrame(
        {
            target1: pd.to_numeric(y[target1], errors="coerce"),
            target2: pd.to_numeric(y[target2], errors="coerce"),
        }
    )
    finite_mask = np.isfinite(frame[[target1, target2]].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[finite_mask].drop_duplicates().reset_index(drop=True)
    if frame.empty:
        return frame

    resolved_directions: dict[str, str] = {}
    for target in (target1, target2):
        direction = str((directions or {}).get(target, "maximize")).strip().lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"directions[{target!r}] must be maximize or minimize, got {direction!r}."
            )
        resolved_directions[target] = direction

    aligned = frame[[target1, target2]].to_numpy(dtype=float, copy=True)
    for index, target in enumerate((target1, target2)):
        if resolved_directions[target] == "minimize":
            aligned[:, index] *= -1.0

    dominated = np.zeros(len(frame), dtype=bool)
    for index, point in enumerate(aligned):
        no_worse = np.all(aligned >= point, axis=1)
        strictly_better = np.any(aligned > point, axis=1)
        dominated[index] = bool(np.any(no_worse & strictly_better))

    return frame.loc[~dominated].sort_values(target1, kind="stable").reset_index(drop=True)


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


def show_pareto_plot(
    y: pd.DataFrame,
    target1: str,
    target2: str,
    df_cand: pd.DataFrame | None = None,
    *,
    directions: Mapping[str, str] | None = None,
    show_pareto_front: bool = False,
    cycle: str | Sequence[Any] | pd.Series | None = None,
    range_padding: float = 0.05,
) -> Figure:
    """2目的の実測値・候補点と、任意で現データのパレートフロントを描画する。

    初期表示範囲は実測点と候補点の予測平均から算出する。候補点はフロント判定に
    含めない。方向を省略した目的変数は最大化として扱う。
    """

    for col in (target1, target2):
        if col not in y.columns:
            raise ValueError(f"y に列 {col!r} が存在しません。")
    if target1 == target2:
        raise ValueError("target1 and target2 must be different target variables.")
    if range_padding < 0:
        raise ValueError("range_padding must be greater than or equal to zero.")

    x = pd.to_numeric(y[target1], errors="coerce")
    z = pd.to_numeric(y[target2], errors="coerce")
    x_values = [x]
    y_values = [z]
    cyc = cycle_series(cycle, y=y, length=len(y)) if cycle is not None else None
    cmap = cycle_color_map(cyc)
    fig = go.Figure()

    candidate_columns = (
        f"{target1}_mean",
        f"{target1}_std",
        f"{target2}_mean",
        f"{target2}_std",
    )
    if df_cand is not None and all(column in df_cand for column in candidate_columns):
        candidate_x = pd.to_numeric(df_cand[f"{target1}_mean"], errors="coerce")
        candidate_y = pd.to_numeric(df_cand[f"{target2}_mean"], errors="coerce")
        candidate_x_std = pd.to_numeric(
            df_cand[f"{target1}_std"], errors="coerce"
        ).abs()
        candidate_y_std = pd.to_numeric(
            df_cand[f"{target2}_std"], errors="coerce"
        ).abs()
        x_values.append(candidate_x)
        y_values.append(candidate_y)
        fig.add_trace(
            go.Scatter(
                x=candidate_x,
                y=candidate_y,
                mode="markers",
                name="候補点",
                marker=dict(color="green", size=10, symbol="diamond"),
                error_x=dict(type="data", array=candidate_x_std, visible=True),
                error_y=dict(type="data", array=candidate_y_std, visible=True),
            )
        )

    if cyc is None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=z,
                mode="markers",
                name="入力データ",
                marker=dict(color="blue", size=10),
            )
        )
    else:
        for c, color in cmap.items():
            mask = cyc == c
            fig.add_trace(
                go.Scatter(
                    x=x[mask],
                    y=z[mask],
                    mode="markers",
                    name=f"cycle {c}",
                    marker=dict(
                        color=color,
                        size=9,
                        line=dict(width=0.5, color="black"),
                    ),
                )
            )

    if show_pareto_front:
        pareto_front = _pareto_front_dataframe(y, target1, target2, directions)
        if not pareto_front.empty:
            fig.add_trace(
                go.Scatter(
                    x=pareto_front[target1],
                    y=pareto_front[target2],
                    mode="lines+markers",
                    name="パレートフロント",
                    line=dict(color="crimson", width=3),
                    marker=dict(
                        color="crimson",
                        size=9,
                        symbol="circle-open",
                        line=dict(width=2, color="crimson"),
                    ),
                    hovertemplate=(
                        f"{target1}: %{{x}}<br>{target2}: %{{y}}"
                        "<extra>パレートフロント</extra>"
                    ),
                )
            )

    def displayed_range(values: list[pd.Series]) -> list[float] | None:
        combined = pd.concat(values, ignore_index=True)
        finite = pd.to_numeric(combined, errors="coerce").to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        span = upper - lower
        if span == 0.0:
            padding = 0.5 if lower == 0.0 else max(abs(lower) * 0.05, 1e-12)
        else:
            padding = span * float(range_padding)
        return [lower - padding, upper + padding]

    x_range = displayed_range(x_values)
    y_range = displayed_range(y_values)
    if x_range is not None:
        fig.update_xaxes(range=x_range, autorange=False, fixedrange=False)
    if y_range is not None:
        fig.update_yaxes(range=y_range, autorange=False, fixedrange=False)
    fig.update_layout(
        height=600,
        width=600,
        xaxis_title=target1,
        yaxis_title=target2,
        legend_title_text="系列",
        font_size=16,
        dragmode="zoom",
    )
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
    fig.add_trace(go.Contour(z=Z, x=grid1_arr, y=grid2_arr, ncontours=25, contours_coloring="heatmap", colorscale="RdBu", reversescale=True, line=dict(color="rgba(128,128,128,0.65)", width=0.6), colorbar=dict(title="獲得関数" if show_type == "acqf" else "予測値", lenmode="pixels", len=200), hoverinfo="none"))
    if df_cand is not None and feature_col1 in df_cand and feature_col2 in df_cand:
        fig.add_trace(go.Scatter(x=df_cand[feature_col1], y=df_cand[feature_col2], mode="markers", name="候補点", marker=dict(color="green", size=12, symbol="diamond", line=dict(width=0.8, color="black"))))
    cyc = cycle_series(cycle, X=X, y=y, length=len(X)) if cycle is not None else None
    cmap = cycle_color_map(cyc)
    if cyc is None:
        fig.add_trace(go.Scatter(x=X[feature_col1], y=X[feature_col2], mode="markers", name="入力データ", marker=dict(size=10, color=y[target_col], colorscale="RdBu", reversescale=True, showscale=False, line=dict(width=0.6, color="black"))))
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


def show_triscatter_with_acqf(
    feature_col1: str,
    feature_col2: str,
    feature_col3: str,
    target_col: str,
    data_tri_plot: tuple[np.ndarray, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    df_cand: pd.DataFrame | None = None,
    *,
    show_type: str = "acqf",
    cycle: str | Sequence[Any] | pd.Series | None = None,
    ncontours: int = 25,
) -> Figure:
    """三角散布図に獲得関数または予測値の等高線を重ねる。"""

    if not (isinstance(data_tri_plot, (tuple, list)) and len(data_tri_plot) == 2):
        raise ValueError("`data_tri_plot` は (ac, grid) のタプルで指定してください。")
    ac, grid = data_tri_plot
    ac_flat = np.ravel(ac)

    if isinstance(grid, pd.DataFrame) and grid.shape[1] != 3:
        raise ValueError("`grid` は 3 列の DataFrame を想定しています。")
    if isinstance(grid, np.ndarray) and (grid.ndim != 2 or grid.shape[0] != 3):
        raise ValueError("`grid` は形状 (3, N) の ndarray を想定しています。")

    fig = go.Figure()

    contour_fig = ff.create_ternary_contour(
        grid,
        -ac_flat,
        pole_labels=[feature_col1, feature_col2, feature_col3],
        ncontours=int(ncontours),
        coloring=None,
        colorscale="RdBu",
        showscale=False,
        interp_mode="cartesian",
        showmarkers=False,
    )
    contour_fig.update_traces(showlegend=False, hoverinfo="skip")
    fig.add_traces(contour_fig.data)

    ac_min = float(np.nanmin(ac_flat)) if np.isfinite(ac_flat).any() else 0.0
    ac_max = float(np.nanmax(ac_flat)) if np.isfinite(ac_flat).any() else 1.0
    if ac_min == ac_max:
        ac_min, ac_max = ac_min - 0.5, ac_max + 0.5

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale="RdBu",
                reversescale=True,
                cmin=ac_min,
                cmax=ac_max,
                colorbar=dict(
                    title="獲得関数" if show_type == "acqf" else "予測値",
                    lenmode="pixels",
                    len=200,
                ),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    use_cycle = cycle is not None
    if use_cycle:
        cyc_series = cycle_series(cycle, X=X, y=y, length=len(X)).astype("Int64")
        unique_cycles = sorted(pd.unique(cyc_series.dropna().astype(int)))
        color_map = cycle_color_map(cyc_series)

    if df_cand is not None and all(c in df_cand for c in (feature_col1, feature_col2, feature_col3)):
        cand_a = pd.to_numeric(df_cand[feature_col1], errors="coerce")
        cand_b = pd.to_numeric(df_cand[feature_col2], errors="coerce")
        cand_c = pd.to_numeric(df_cand[feature_col3], errors="coerce")
        cand_mask = np.isfinite(cand_a) & np.isfinite(cand_b) & np.isfinite(cand_c)
        if cand_mask.any():
            fig.add_trace(
                go.Scatterternary(
                    a=cand_a[cand_mask],
                    b=cand_b[cand_mask],
                    c=cand_c[cand_mask],
                    mode="markers",
                    name="候補点",
                    marker=dict(color="green", size=12, symbol="diamond", line=dict(width=0.8, color="black"), showscale=False),
                    customdata=np.stack([cand_a[cand_mask], cand_b[cand_mask], cand_c[cand_mask]], axis=1),
                    hovertemplate=(
                        f"{feature_col1}: %{{customdata[0]}}<br>"
                        f"{feature_col2}: %{{customdata[1]}}<br>"
                        f"{feature_col3}: %{{customdata[2]}}<extra></extra>"
                    ),
                )
            )

    a = pd.to_numeric(X[feature_col1], errors="coerce")
    b = pd.to_numeric(X[feature_col2], errors="coerce")
    c = pd.to_numeric(X[feature_col3], errors="coerce")
    col = pd.to_numeric(y[target_col], errors="coerce")
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(col)

    if m.any():
        if use_cycle:
            for cyc in unique_cycles:
                mask = (cyc_series == cyc) & m
                if mask.any():
                    fig.add_trace(
                        go.Scatterternary(
                            a=a[mask],
                            b=b[mask],
                            c=c[mask],
                            mode="markers",
                            name=f"入力データ (cycle {cyc})",
                            marker=dict(color=color_map.get(cyc, "#000000"), size=10, line=dict(width=0.6, color="black"), showscale=False),
                            customdata=np.stack([a[mask], b[mask], c[mask], col[mask]], axis=1),
                            hovertemplate=(
                                f"{feature_col1}: %{{customdata[0]}}<br>"
                                f"{feature_col2}: %{{customdata[1]}}<br>"
                                f"{feature_col3}: %{{customdata[2]}}<br>"
                                f"{target_col}: %{{customdata[3]}}<extra></extra>"
                            ),
                        )
                    )
        else:
            fig.add_trace(
                go.Scatterternary(
                    a=a[m],
                    b=b[m],
                    c=c[m],
                    mode="markers",
                    name="入力データ",
                    marker=dict(color=col[m], colorscale="RdBu", reversescale=True, showscale=False, size=10, line=dict(width=0.6, color="black")),
                    customdata=np.stack([a[m], b[m], c[m], col[m]], axis=1),
                    hovertemplate=(
                        f"{feature_col1}: %{{customdata[0]}}<br>"
                        f"{feature_col2}: %{{customdata[1]}}<br>"
                        f"{feature_col3}: %{{customdata[2]}}<br>"
                        f"{target_col}: %{{customdata[3]}}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=600,
        width=800,
        showlegend=True,
        font_size=16,
        legend_title_text="cycle" if use_cycle else "系列",
        ternary=dict(
            aaxis=dict(title=feature_col1),
            baxis=dict(title=feature_col2),
            caxis=dict(title=feature_col3),
            sum=1,
        ),
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


def show_triscatter_with_acqf_from_optimizer(
    obj: Any,
    feature_col1: str,
    feature_col2: str,
    feature_col3: str,
    target_col: str,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    value_dict: dict[str, Any] | None = None,
    candidate_result: Any | None = None,
    sum_value: float | None = None,
    n: int = 50,
    show_type: str = "acqf",
    cycle: str | Sequence[Any] | pd.Series | None = None,
    ncontours: int = 25,
) -> Figure:
    X_df, y_df = training_dataframe(obj, feature_cols=feature_cols, target_cols=target_cols)
    data = tri_grid(obj, [feature_col1, feature_col2, feature_col3], target_col, value_dict, feature_cols=list(X_df.columns), target_cols=list(y_df.columns), candidate_result=candidate_result, sum_value=sum_value, n=n, show_type=show_type)  # type: ignore[arg-type]
    df_cand = candidates_dataframe(obj, candidate_result=candidate_result, feature_cols=list(X_df.columns), target_cols=list(y_df.columns), include_prediction=False)
    return show_triscatter_with_acqf(feature_col1, feature_col2, feature_col3, target_col, data, X_df, y_df, df_cand, show_type=show_type, cycle=cycle, ncontours=ncontours)