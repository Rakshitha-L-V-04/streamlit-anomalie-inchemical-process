import json
from pathlib import Path
import pandas as pd
import streamlit as st

from data_loader import (
    DATA_DIR, SENSORS,
    list_operating_points, list_experiments, load_experiment,
    load_annotations, load_all_annotations,
)

from charts import (
    PLOTLY_CONFIG, experiment_color,
    sensor_over_time_fig, normal_vs_anomalous_fig,
    anomaly_durations_fig, anomaly_durations_by_op_fig,
)

APP_TITLE = "Understanding Anomalies in Chemical Processes"
APP_ICON = "⚗️"      

# =====================================================================
# Page config (must be the first Streamlit call)
# =====================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Custom CSS (loaded from assets/style.css)
# =====================================================================
def load_css(file_name: str) -> None:
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_path}")
load_css("assets/style.css")


# =====================================================================
# Small helpers
# =====================================================================
def fmt(value, digits: int = 2) -> str:
    """Format a number, or show '—' if it is missing."""
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


# ---------- Sanity check: can we see the dataset? ----------
if not DATA_DIR.exists():
    st.error(f"Dataset folder not found at: {DATA_DIR}")
    st.stop()


# =====================================================================
# SIDEBAR: Dashboard Controls
# =====================================================================
st.sidebar.header("Dashboard Controls")

# 1) Operating point
operating_points = list_operating_points()
selected_op = st.sidebar.selectbox(
    "Select Operating Point",
    options=operating_points,
    index=0,
    key="op",
)

# 2) Experiment
experiments = list_experiments(selected_op)
selected_exp = st.sidebar.radio(
    "Select Experiment",
    options=list(experiments.keys()),
    index=0,
    key="experiment",
)

# --- Load the data for the current selection ---
csv_path = str(experiments[selected_exp])
df = load_experiment(csv_path)
annotations = load_annotations(csv_path, df["Seconds"].iloc[0])
anomaly_windows = annotations.dropna(subset=["start_min", "end_min"])

# 3) Sensor
available_sensors = [s for s in SENSORS if s in df.columns]
default_sensor = "T712" if "T712" in available_sensors else available_sensors[0]

selected_sensor = st.sidebar.selectbox(
    "Select Sensor",
    options=available_sensors,
    index=available_sensors.index(default_sensor),
    format_func=lambda s: f"{s} · {SENSORS[s]}",
    key="sensor",
)
st.sidebar.caption(f"**{selected_sensor}**: {SENSORS[selected_sensor]}")

# 4) Chart options
st.sidebar.divider()
st.sidebar.subheader("Chart Options")

show_window = st.sidebar.checkbox(
    "Show anomaly window in comparison chart",
    value=False,
    key="show_window_cmp",
)
scope = st.sidebar.radio(
    "Anomaly durations",
    options=["This operating point", "All operating points"],
    key="duration_scope",
)

# 5) Help
st.sidebar.divider()
with st.sidebar.expander("ℹ️ How to read this dashboard"):
    st.markdown(
        """
        **Sensor Behavior Over Time**: the selected sensor in the selected run.
        The 🟥 red band is the annotated anomaly interval from the YAML file.

        **Normal vs Anomalous Behavior**: all runs of the same operating point
        on one time axis (minutes since start). Where the orange line leaves the
        blue one, the process behaves differently from normal.

        **Recorded Anomaly Durations**: how long each annotated anomaly lasted,
        coloured by category (transient, recurrent, permanent).

        **Tip:** keep one anomalous run selected and switch through the sensors
        to find out which ones react to the disturbance.
        """
    )


# --- Load ALL runs of this OP (for comparison) ---
all_runs = {label: load_experiment(str(path)) for label, path in experiments.items()}

anomalous_label = next((l for l in all_runs if l.startswith("Anomalous")), None)
if anomalous_label:
    anomalous_annotations = load_annotations(
        str(experiments[anomalous_label]),
        all_runs[anomalous_label]["Seconds"].iloc[0],
    )
else:
    anomalous_annotations = None


# =====================================================================
# MAIN AREA: Title
# =====================================================================
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(
    "Interactive visual analysis of normal and anomalous operating "
    "conditions in a batch distillation process."
)


# =====================================================================
# OVERVIEW: one compact line + 4 metric cards
# =====================================================================
n_measurements = len(df)
n_sensors = len(available_sensors)
n_anomalies = len(annotations)
run_minutes = df["ElapsedMin"].iloc[-1]

st.markdown(
    f'<div class="section-title">Overview'
    f'<span class="section-sub">{selected_op} · {selected_exp} experiment · '
    f'{n_measurements:,} measurements · {run_minutes:.0f} minutes recorded</span></div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Operating Point", selected_op,
          help="A specific combination of process settings (pressure, heat input, …).")
m2.metric("Measurements", f"{n_measurements:,}",
          help="Number of time steps in this run (one reading per second).")
m3.metric("Sensors", n_sensors,
          help="Project sensors used in this dashboard (see README).")
m4.metric("Recorded Anomalies", n_anomalies,
          help="Annotated anomaly events from the YAML file. Normal runs have none.")


# =====================================================================
# CHART ROW (titles are drawn inside the Plotly charts)
# =====================================================================
chart_col1, chart_col2, chart_col3 = st.columns(3, gap="medium")

# ---------- Chart 1: Sensor Behavior Over Time ----------
with chart_col1:
    fig1 = sensor_over_time_fig(
        df=df,
        sensor=selected_sensor,
        annotations=annotations,
        color=experiment_color(selected_exp),
    )
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    if len(anomaly_windows):
        windows = ", ".join(
            f"{w['start_min']:.1f}–{w['end_min']:.1f} min"
            for _, w in anomaly_windows.iterrows()
        )
        st.caption(f"🟥 Annotated anomaly interval: {windows}")
    else:
        st.caption("Normal run: no anomalies annotated.")

# ---------- Chart 2: Normal vs Anomalous Behavior ----------
with chart_col2:
    fig2 = normal_vs_anomalous_fig(
        runs=all_runs,
        sensor=selected_sensor,
        highlight=selected_exp,
        anomaly_annotations=anomalous_annotations if show_window else None,
    )
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    lengths = " · ".join(
        f"{label}: {run['ElapsedMin'].iloc[-1]:.0f} min"
        for label, run in all_runs.items()
    )
    st.caption(f"Run lengths: {lengths}")

# ---------- Chart 3: Recorded Anomaly Durations ----------
with chart_col3:
    if scope == "All operating points":
        duration_data = load_all_annotations()
        scope_note = "All OPs"
    elif len(annotations):
        duration_data = annotations
        scope_note = f"{selected_op} · {selected_exp}"
    elif anomalous_annotations is not None and len(anomalous_annotations):
        duration_data = anomalous_annotations
        scope_note = f"Reference: {selected_op} · {anomalous_label}"
    else:
        duration_data = pd.DataFrame()
        scope_note = ""

    valid_durations = (duration_data.dropna(subset=["duration_min"])
                       if len(duration_data) else duration_data)

    if len(valid_durations) == 0:
        st.info("No anomalies recorded for this selection.")
    else:
        if scope == "All operating points":
            fig3 = anomaly_durations_by_op_fig(valid_durations)
        else:
            fig3 = anomaly_durations_fig(valid_durations)
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

        n = len(valid_durations)
        st.caption(
            f"{scope_note} · {n} anomal{'y' if n == 1 else 'ies'} · "
            f"total {valid_durations['duration_min'].sum():.0f} min · "
            f"mean {valid_durations['duration_min'].mean():.1f} min"
        )


# =====================================================================
# EXPANDER: View Recorded Anomaly Details
# =====================================================================
with st.expander("🚩 View Recorded Anomaly Details", expanded=False):

    if len(annotations):
        details = annotations
        details_run = df
        details_label = selected_exp
    elif anomalous_annotations is not None and len(anomalous_annotations):
        details = anomalous_annotations
        details_run = all_runs[anomalous_label]
        details_label = anomalous_label
        st.info(f"The **{selected_exp}** run has no anomalies. "
                f"Showing the **{anomalous_label}** run of {selected_op} instead.")
    else:
        details = pd.DataFrame()

    if len(details) == 0:
        st.write("No anomalies recorded for this operating point.")
    else:
        st.markdown("##### Summary")
        st.dataframe(
            details[["id", "category", "class", "start_clock", "end_clock",
                     "start_min", "end_min", "duration_min"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": "Anomaly ID",
                "category": "Category",
                "class": "Status",
                "start_clock": "Begins (clock)",
                "end_clock": "Ends (clock)",
                "start_min": st.column_config.NumberColumn("Start (min)", format="%.1f"),
                "end_min": st.column_config.NumberColumn("End (min)", format="%.1f"),
                "duration_min": st.column_config.NumberColumn("Duration (min)", format="%.1f"),
            },
        )

        normal_run = all_runs.get("Normal")

        for _, a in details.iterrows():
            st.divider()
            st.markdown(f"##### {a['id']} · {a['category']}")
            st.markdown(f"**Perturbation:** {a['perturbation']}")

            if pd.notna(a["start_min"]) and pd.notna(a["end_min"]):
                t = details_run["ElapsedMin"]
                values = details_run[selected_sensor]

                before = values[(t >= a["start_min"] - 10) & (t < a["start_min"])].mean()
                during = values[t.between(a["start_min"], a["end_min"])].mean()

                if normal_run is not None and selected_sensor in normal_run.columns:
                    nt = normal_run["ElapsedMin"]
                    normal_same = normal_run[selected_sensor][
                        nt.between(a["start_min"], a["end_min"])
                    ].mean()
                else:
                    normal_same = None

                st.markdown(f"**Effect on {selected_sensor}** ({details_label} run)")
                k1, k2, k3 = st.columns(3)
                k1.metric("Mean 10 min before", fmt(before))
                k2.metric(
                    "Mean during anomaly", fmt(during),
                    delta=(None if pd.isna(before) or pd.isna(during)
                           else f"{during - before:+.2f} vs before"),
                    delta_color="off",
                )
                k3.metric(
                    "Normal run, same minutes", fmt(normal_same),
                    delta=(None if normal_same is None or pd.isna(normal_same) or pd.isna(during)
                           else f"{during - normal_same:+.2f} anomalous − normal"),
                    delta_color="off",
                )

            pm = a["perturbation_mode"]
            if pm:
                st.markdown("**Perturbation mode** (how and when the disturbance was applied)")
                st.caption("Times in here describe when the disturbance was applied "
                           "and can differ from the annotated interval above.")
                st.json(json.loads(json.dumps(pm, default=str)), expanded=False)


# =====================================================================
# FOOTER
# =====================================================================
st.markdown(
    '<div class="footer">Data: Batch Distillation Dataset (butan-1-ol, propan-2-ol, water) · '
    '<a href="https://doi.org/10.5281/zenodo.17395544" target="_blank">Zenodo</a> · '
    'Built with Streamlit and Plotly</div>',
    unsafe_allow_html=True,
)