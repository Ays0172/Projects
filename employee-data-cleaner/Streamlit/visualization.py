"""
visualization.py — Chart generation for both static (Matplotlib/Seaborn) and interactive (Plotly) modes.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Lazy imports handled in functions to avoid top-level errors when libs absent.

ENGINE = Literal["static", "interactive"]

# ── Dark palette shared across both engines ───────────────────────────────────
_GOLD = "#C9A84C"
_BG = "#1C1C1E"
_SURFACE = "#242428"
_BORDER = "#3A3A42"
_TEXT = "#F0EDE8"
_MUTED = "#8A8A9A"
_PALETTE = ["#C9A84C", "#5C9BE8", "#4CAF7D", "#E85C5C", "#A85CE8", "#E8A838", "#5CE8D8", "#E85CA8"]


# ══════════════════════════════════════════════════════════════════════════════
# STATIC (Matplotlib / Seaborn)
# ══════════════════════════════════════════════════════════════════════════════

def _mpl_style() -> None:
    """Apply dark theme defaults to Matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": _SURFACE,
        "axes.facecolor": _BG,
        "axes.edgecolor": _BORDER,
        "axes.labelcolor": _TEXT,
        "axes.titlecolor": _TEXT,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "text.color": _TEXT,
        "grid.color": _BORDER,
        "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
        "figure.dpi": 120,
    })


def static_distribution(df: pd.DataFrame, column: str) -> "plt.Figure":  # type: ignore[name-defined]
    """Histogram + KDE for a numeric column (Seaborn)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    _mpl_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df[column].dropna(), kde=True, color=_GOLD, ax=ax, edgecolor=_BORDER, bins=40)
    ax.set_title(f"Distribution — {column}", fontsize=13, pad=12)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def static_boxplot(df: pd.DataFrame, column: str, group_col: Optional[str] = None) -> "plt.Figure":  # type: ignore[name-defined]
    """Box plot, optionally grouped."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    _mpl_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    if group_col and group_col in df.columns:
        order = df.groupby(group_col)[column].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=group_col, y=column, order=order,
                    palette=_PALETTE[:len(order)], ax=ax, linewidth=0.8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        sns.boxplot(data=df[[column]].dropna(), color=_GOLD, ax=ax, linewidth=0.8)
    ax.set_title(f"Box Plot — {column}" + (f" by {group_col}" if group_col else ""), fontsize=13, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def static_correlation_heatmap(corr_matrix: pd.DataFrame) -> "plt.Figure":  # type: ignore[name-defined]
    """Seaborn correlation heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    _mpl_style()
    n = len(corr_matrix)
    size = max(6, n * 0.75)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    cmap = sns.diverging_palette(230, 45, as_cmap=True)
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap=cmap,
        center=0, linewidths=0.4, linecolor=_BORDER,
        cbar_kws={"shrink": 0.75}, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Correlation Matrix", fontsize=13, pad=12)
    fig.tight_layout()
    return fig


def static_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str = "") -> "plt.Figure":  # type: ignore[name-defined]
    """Horizontal bar chart for grouped statistics."""
    import matplotlib.pyplot as plt
    _mpl_style()
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.45)))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(df))]
    ax.barh(df[x_col].astype(str), df[y_col], color=colors, edgecolor=_BORDER, linewidth=0.5)
    ax.set_xlabel(y_col)
    ax.set_title(title or f"{y_col} by {x_col}", fontsize=13, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def static_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str] = None) -> "plt.Figure":  # type: ignore[name-defined]
    """Scatter plot with optional color grouping."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    _mpl_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(df[hue_col].unique())} if hue_col else None
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=palette,
                    alpha=0.65, s=35, ax=ax, linewidth=0)
    ax.set_title(f"{y_col} vs {x_col}", fontsize=13, pad=12)
    ax.grid(linestyle="--", alpha=0.4)
    if hue_col:
        ax.legend(loc="best", framealpha=0.3, edgecolor=_BORDER)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE (Plotly)
# ══════════════════════════════════════════════════════════════════════════════

def _plotly_layout(title: str = "") -> dict:
    """Shared Plotly layout dictionary."""
    return dict(
        title=dict(text=title, font=dict(color=_TEXT, size=14, family="Syne, sans-serif")),
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, family="DM Sans, sans-serif", size=11),
        xaxis=dict(gridcolor=_BORDER, zerolinecolor=_BORDER),
        yaxis=dict(gridcolor=_BORDER, zerolinecolor=_BORDER),
        margin=dict(l=50, r=30, t=55, b=50),
        colorway=_PALETTE,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=_BORDER),
    )


def interactive_distribution(df: pd.DataFrame, column: str) -> "go.Figure":  # type: ignore[name-defined]
    """Interactive histogram + violin."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = df[column].dropna()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3],
                        subplot_titles=["Histogram + KDE", "Violin"])
    fig.add_trace(go.Histogram(x=data, nbinsx=50, name="Count",
                               marker_color=_GOLD, opacity=0.8), row=1, col=1)
    fig.add_trace(go.Violin(y=data, name="Distribution", line_color=_GOLD,
                            fillcolor=f"rgba(201,168,76,0.2)", meanline_visible=True), row=1, col=2)
    fig.update_layout(**_plotly_layout(f"Distribution — {column}"))
    return fig


def interactive_boxplot(df: pd.DataFrame, column: str, group_col: Optional[str] = None) -> "go.Figure":  # type: ignore[name-defined]
    """Interactive box plot."""
    import plotly.express as px

    if group_col and group_col in df.columns:
        fig = px.box(df, x=group_col, y=column, color=group_col,
                     color_discrete_sequence=_PALETTE)
    else:
        fig = px.box(df, y=column, color_discrete_sequence=[_GOLD])
    fig.update_layout(**_plotly_layout(f"Box Plot — {column}" + (f" by {group_col}" if group_col else "")))
    return fig


def interactive_correlation_heatmap(corr_matrix: pd.DataFrame) -> "go.Figure":  # type: ignore[name-defined]
    """Interactive Plotly correlation heatmap."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(thickness=14, len=0.8),
    ))
    fig.update_layout(**_plotly_layout("Correlation Matrix"))
    return fig


def interactive_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str = "") -> "go.Figure":  # type: ignore[name-defined]
    """Interactive horizontal bar chart."""
    import plotly.express as px

    fig = px.bar(df.sort_values(y_col), x=y_col, y=x_col, orientation="h",
                 color=y_col, color_continuous_scale=[[0, "#3A3A42"], [1, _GOLD]])
    fig.update_layout(**_plotly_layout(title or f"{y_col} by {x_col}"))
    fig.update_coloraxes(showscale=False)
    return fig


def interactive_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str] = None) -> "go.Figure":  # type: ignore[name-defined]
    """Interactive scatter plot."""
    import plotly.express as px

    fig = px.scatter(df, x=x_col, y=y_col, color=hue_col,
                     color_discrete_sequence=_PALETTE, opacity=0.7,
                     trendline="ols" if hue_col is None else None,
                     trendline_color_override=_GOLD)
    fig.update_layout(**_plotly_layout(f"{y_col} vs {x_col}"))
    return fig


def interactive_pie(df: pd.DataFrame, column: str, top_n: int = 8) -> "go.Figure":  # type: ignore[name-defined]
    """Donut chart for categorical distribution."""
    import plotly.graph_objects as go

    vc = df[column].value_counts().head(top_n)
    fig = go.Figure(go.Pie(
        labels=vc.index.tolist(), values=vc.values.tolist(),
        hole=0.55, marker_colors=_PALETTE,
        textfont=dict(size=11),
    ))
    fig.update_layout(**_plotly_layout(f"Distribution — {column}"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED RENDER DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def render_distribution(df: pd.DataFrame, column: str, engine: ENGINE) -> None:
    """Render a distribution chart using the selected engine."""
    if engine == "static":
        st.pyplot(static_distribution(df, column), use_container_width=True)
    else:
        st.plotly_chart(interactive_distribution(df, column), use_container_width=True)


def render_boxplot(df: pd.DataFrame, column: str, group_col: Optional[str], engine: ENGINE) -> None:
    """Render a box plot using the selected engine."""
    if engine == "static":
        st.pyplot(static_boxplot(df, column, group_col), use_container_width=True)
    else:
        st.plotly_chart(interactive_boxplot(df, column, group_col), use_container_width=True)


def render_correlation_heatmap(corr_matrix: pd.DataFrame, engine: ENGINE) -> None:
    """Render a correlation heatmap using the selected engine."""
    if engine == "static":
        st.pyplot(static_correlation_heatmap(corr_matrix), use_container_width=True)
    else:
        st.plotly_chart(interactive_correlation_heatmap(corr_matrix), use_container_width=True)


def render_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, engine: ENGINE) -> None:
    """Render a bar chart using the selected engine."""
    if engine == "static":
        st.pyplot(static_bar(df, x_col, y_col, title), use_container_width=True)
    else:
        st.plotly_chart(interactive_bar(df, x_col, y_col, title), use_container_width=True)


def render_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str], engine: ENGINE) -> None:
    """Render a scatter plot using the selected engine."""
    if engine == "static":
        st.pyplot(static_scatter(df, x_col, y_col, hue_col), use_container_width=True)
    else:
        st.plotly_chart(interactive_scatter(df, x_col, y_col, hue_col), use_container_width=True)


def fig_to_bytes(fig: "plt.Figure") -> bytes:  # type: ignore[name-defined]
    """Convert a Matplotlib figure to PNG bytes for download."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    return buf.getvalue()