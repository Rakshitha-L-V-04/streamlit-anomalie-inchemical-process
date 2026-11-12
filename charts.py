import pandas as pd
import plotly.graph_objects as go

# =====================================================================
# Shared styling (used by all charts)
# =====================================================================
NORMAL_COLOR = "#1f77b4"
NORMAL2_COLOR = "#6baed6"
ANOMALOUS_COLOR = "#ff7f0e"
ANOMALY_TEXT = "#b91c1c"
GRID_COLOR = "#ececec"
TITLE_COLOR = "#1f2a44"
AXIS_COLOR = "#6b7280"
FONT_FAMILY = "Arial, Helvetica, sans-serif"

CHART_HEIGHT = 400
TOP_MARGIN = 72          # toolbar + title + legend
TITLE_SIZE = 13
FONT_SIZE = 11

CATEGORY_ORDER = ["transient", "recurrent", "permanent"]
CATEGORY_COLORS = {
    "transient": "#0b6bcb",
    "recurrent": "#9467bd",
    "permanent": "#d62728",
}
OTHER_CATEGORY_COLOR = "#9ca3af"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def experiment_color(label: str) -> str:
    return ANOMALOUS_COLOR if label.startswith("Anomalous") else NORMAL_COLOR


def run_style(label: str) -> tuple[str, str]:
    if label.startswith("Anomalous"):
        return ANOMALOUS_COLOR, "solid"
    if label == "Normal":
        return NORMAL_COLOR, "solid"
    return NORMAL2_COLOR, "dash"


def _category_sort_key(category: str) -> int:
    """Known categories first, in a fixed order; anything else last."""
    return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)


def _apply_base_layout(fig: go.Figure, x_title: str, y_title: str,
                       title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=55, r=10, t=TOP_MARGIN, b=45),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=AXIS_COLOR),
    )

    if title:
        fig.update_layout(
            title=dict(
                text=f'<span style="font-weight:normal">{title}</span>',
                x=0, xref="paper", xanchor="left",
                y=1, yref="container", yanchor="top",
                pad=dict(t=28),
                font=dict(family=FONT_FAMILY, size=TITLE_SIZE, color=TITLE_COLOR),
            )
        )

    fig.update_xaxes(title_text=x_title, title_font_size=FONT_SIZE,
                     automargin=True, showgrid=True, gridcolor=GRID_COLOR,
                     zeroline=False, showline=False)
    fig.update_yaxes(title_text=y_title, title_font_size=FONT_SIZE,
                     automargin=True, showgrid=True, gridcolor=GRID_COLOR,
                     zeroline=False, showline=False)
    return fig


def _top_legend(fig: go.Figure) -> None:
    """Horizontal legend above the plot; each entry is as wide as its own text."""
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0,
            xanchor="left", x=0, title_text="",
            font=dict(size=FONT_SIZE - 1),
            itemwidth=30,            # short colour swatch -> more room for text
            tracegroupgap=0,
        ),
    )

def _bottom_legend(fig: go.Figure) -> None:
    """Horizontal legend under the x-axis; extra rows grow downward, never into the title."""
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.12,          # just below the tick labels
            xanchor="left", x=0,
            title_text="",
            font=dict(size=FONT_SIZE - 1),
            itemwidth=30,
            traceorder="normal",             # transient → recurrent → permanent → unknown
        ),
    )


def _add_anomaly_windows(fig: go.Figure, annotations: pd.DataFrame,
                         opacity: float = 0.22, font_size: int = 10) -> None:
    valid = annotations.dropna(subset=["start_min", "end_min"])
    for _, a in valid.iterrows():
        fig.add_vrect(
            x0=a["start_min"],
            x1=a["end_min"],
            fillcolor=f"rgba(239, 68, 68, {opacity})",
            line_width=0,
            layer="below",
            annotation_text="Anomaly",
            annotation_position="top left",
            annotation_font=dict(size=font_size, color=ANOMALY_TEXT),
        )


# =====================================================================
# Chart 1: Sensor Behavior Over Time
# =====================================================================
def sensor_over_time_fig(
    df: pd.DataFrame,
    sensor: str,
    annotations: pd.DataFrame,
    color: str,
    title: str | None = "Sensor Behavior Over Time",
) -> go.Figure:
    fig = go.Figure()
    _add_anomaly_windows(fig, annotations)

    fig.add_trace(
        go.Scatter(
            x=df["ElapsedMin"],
            y=df[sensor],
            mode="lines",
            name=sensor,
            line=dict(color=color, width=1.6),
            customdata=df["Time"],
            hovertemplate=(
                "Elapsed: %{x:.1f} min<br>"
                "Clock: %{customdata}<br>"
                f"{sensor}: " + "%{y:.2f}<extra></extra>"
            ),
        )
    )
    return _apply_base_layout(fig, "Elapsed Time (minutes)", sensor, title)


# =====================================================================
# Chart 2: Normal vs Anomalous Behavior
# =====================================================================
def normal_vs_anomalous_fig(
    runs: dict,
    sensor: str,
    highlight: str | None = None,
    anomaly_annotations: pd.DataFrame | None = None,
    title: str | None = "Normal vs Anomalous Behavior",
) -> go.Figure:
    fig = go.Figure()

    if anomaly_annotations is not None:
        _add_anomaly_windows(fig, anomaly_annotations, opacity=0.12, font_size=9)

    ordered = sorted(runs.items(), key=lambda kv: kv[0].startswith("Anomalous"))
    for label, run_df in ordered:
        if sensor not in run_df.columns:
            continue
        color, dash = run_style(label)
        fig.add_trace(
            go.Scatter(
                x=run_df["ElapsedMin"],
                y=run_df[sensor],
                mode="lines",
                name=label,
                line=dict(color=color, dash=dash,
                          width=2.4 if label == highlight else 1.4),
                customdata=run_df["Time"],
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Elapsed: %{x:.1f} min<br>"
                    "Clock: %{customdata}<br>"
                    f"{sensor}: " + "%{y:.2f}<extra></extra>"
                ),
            )
        )

    _apply_base_layout(fig, "Elapsed Time (minutes)", sensor, title)
    _top_legend(fig)
    return fig


# =====================================================================
# Chart 3a: Recorded Anomaly Durations (one operating point, per anomaly)
# =====================================================================
def anomaly_durations_fig(
    annotations: pd.DataFrame,
    title: str | None = "Recorded Anomaly Durations",
) -> go.Figure:
    data = annotations.dropna(subset=["duration_min"]).copy()

    text = data["perturbation"].astype(str)
    data["perturbation_short"] = text.where(text.str.len() <= 80,
                                            text.str.slice(0, 77) + "…")

    fig = go.Figure()
    categories = sorted(data["category"].unique(), key=_category_sort_key)

    for category in categories:
        group = data[data["category"] == category]
        fig.add_trace(
            go.Bar(
                x=group["id"],
                y=group["duration_min"],
                name=category,
                marker_color=CATEGORY_COLORS.get(category, OTHER_CATEGORY_COLOR),
                text=group["duration_min"].map(lambda d: f"{d:.1f} min"),
                textposition="outside",
                textfont=dict(size=FONT_SIZE),
                cliponaxis=False,
                customdata=group[["start_min", "end_min", "perturbation_short"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Category: {category}<br>"
                    "Duration: %{y:.1f} min<br>"
                    "Window: %{customdata[0]:.1f}–%{customdata[1]:.1f} min<br>"
                    "%{customdata[2]}<extra></extra>"
                ),
            )
        )

    _apply_base_layout(fig, "", "Duration (minutes)", title)
    _bottom_legend(fig)
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(rangemode="tozero")
    # "stack": each ID has one category, so every bar gets the FULL width
    fig.update_layout(barmode="stack", bargap=0.3)
    return fig


# =====================================================================
# Chart 3b: Recorded Anomaly Durations (all operating points, per OP)
# =====================================================================
def anomaly_durations_by_op_fig(
    all_annotations: pd.DataFrame,
    title: str | None = "Recorded Anomaly Durations · all OPs",
) -> go.Figure:
    data = all_annotations.dropna(subset=["duration_min"])

    # Total minutes + number of anomalies per OP and category
    summary = (
        data.groupby(["op", "category"])
        .agg(total=("duration_min", "sum"), count=("id", "size"))
        .reset_index()
    )
    ops = sorted(summary["op"].unique())
    categories = sorted(summary["category"].unique(), key=_category_sort_key)

    fig = go.Figure()

    # One stacked segment per category
    for category in categories:
        g = summary[summary["category"] == category]
        fig.add_trace(
            go.Bar(
                x=g["op"],
                y=g["total"],
                name=category,
                marker_color=CATEGORY_COLORS.get(category, OTHER_CATEGORY_COLOR),
                customdata=g[["count"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Category: {category}<br>"
                    "Anomalies: %{customdata[0]}<br>"
                    "Total duration: %{y:.1f} min<extra></extra>"
                ),
            )
        )

    # Label on top of each OP: "n · total min"
    per_op = summary.groupby("op").agg(total=("total", "sum"), count=("count", "sum"))
    per_op = per_op.reindex(ops)
    fig.add_trace(
        go.Scatter(
            x=ops,
            y=per_op["total"],
            mode="text",
            text=[f"{c} · {t:.0f} min" for c, t in zip(per_op["count"], per_op["total"])],
            textposition="top center",
            textfont=dict(size=FONT_SIZE - 1, color=TITLE_COLOR),
            showlegend=False,
            hoverinfo="skip",
            cliponaxis=False,
        )
    )

    _apply_base_layout(fig, "", "Total duration (minutes)", title)
    _bottom_legend(fig)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ops,
                     showgrid=False)
    fig.update_yaxes(range=[0, per_op["total"].max() * 1.15])
    fig.update_layout(barmode="stack", bargap=0.35)
    return fig