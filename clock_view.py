"""
Decoding the Disturbance: Understanding Anomalies in Batch Distillation: an interactive drill-down component.

    All runs  ->  one run (clock + graphs)  ->  one event  ->  one sensor

The drawing and the animations live in assets/process_clocks.html (D3.js).
This module builds the data the page needs from the dataset and embeds it
into the page, so the component works without any extra server.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from data_loader import (
    SENSORS,
    list_operating_points,
    list_experiments,
    load_experiment,
    load_raw_yaml,
)

TEMPLATE = Path(__file__).parent / "assets" / "process_clocks.html"

STEP = 20            # keep one reading every 20 s (enough for the overview)
MEDIAN_WINDOW = 601  # 10-minute rolling median (1 reading per second)
PEAK_WINDOW = 31     # max over 30 s
SCORE_CAP = 5


# =====================================================================
# Deviation score
# =====================================================================
def _residual(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """How far each sensor is from its own 10-minute rolling median."""
    values = df[sensors]
    median = values.rolling(MEDIAN_WINDOW, center=True, min_periods=1).median()
    return (values - median).abs()


def _peak(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rolling(PEAK_WINDOW, center=True, min_periods=1).max()


# =====================================================================
# Annotation helpers (YAML structure varies a bit between files)
# =====================================================================
def _first(value):
    return value[0] if isinstance(value, list) and value else value


def _clock_seconds(value):
    value = _first(value)
    if value is None:
        return None
    try:
        return pd.to_timedelta(str(value).strip()).total_seconds()
    except (ValueError, TypeError):
        return None


def _sensors_of(anomaly: dict) -> list[str]:
    """Sensor IDs named under PerturbationMode > hasLocalEffect > madeBySensor."""
    found = []
    effects = (anomaly.get("PerturbationMode") or {}).get("hasLocalEffect")
    for effect in effects if isinstance(effects, list) else [effects]:
        if not isinstance(effect, dict):
            continue
        effect = effect.get("LocalEffect", effect)
        made_by = effect.get("madeBySensor") if isinstance(effect, dict) else None
        if isinstance(made_by, dict):
            made_by = made_by.get("Sensor", made_by)
        for s in made_by if isinstance(made_by, list) else [made_by]:
            if isinstance(s, dict):
                s = s.get("Sensor", s)
            if isinstance(s, dict) and "id" in s:
                found.append(str(s["id"]))
    return sorted(set(found))


def _events(csv_path: str, run_start_sec: float) -> list[dict]:
    data = load_raw_yaml(csv_path) or {}
    anomalies = data.get("anomalies") or []
    if isinstance(anomalies, dict):
        anomalies = list(anomalies.values())

    def to_min(value):
        sec = _clock_seconds(value)
        if sec is None:
            return None
        if sec < run_start_sec - 12 * 3600:      # run crossed midnight
            sec += 24 * 3600
        return round((sec - run_start_sec) / 60, 2)

    events, seen = [], {}
    for a in anomalies:
        if not isinstance(a, dict):
            continue
        key = (a.get("id"), str(_first(a.get("hasBeginning"))), str(_first(a.get("hasEnd"))))
        if key in seen:                           # same event listed twice: merge sensors
            seen[key]["sens"] = sorted(set(seen[key]["sens"]) | set(_sensors_of(a)))
            continue
        category = a.get("hasCategory")
        category = str(category) if category else (
            "unconfirmed" if a.get("class") == "Anomaly" else "unknown")
        mode = a.get("PerturbationMode") or {}
        event = {
            "id": str(a.get("id")),
            "cls": str(a.get("class")),
            "cat": category,
            "s": to_min(a.get("hasBeginning")),
            "e": to_min(a.get("hasEnd")),
            "p": " ".join(str(a.get("Perturbation") or "No description").split()),
            "mode": str(mode.get("label", "")) if isinstance(mode, dict) else "",
            "sens": _sensors_of(a),
            "rec": str(a.get("hasRecoveryAction") or ""),
        }
        seen[key] = event
        events.append(event)
    return events


# =====================================================================
# Build the data for all operating points
# =====================================================================
def _round(values) -> list:
    return [None if pd.isna(v) else float(f"{v:.4g}") for v in values]


@st.cache_data(show_spinner="Decoding the disturbance …")
def build_clock_data() -> dict:
    sensors = list(SENSORS)
    ops = {}
    for op in list_operating_points():
        experiments = list_experiments(op)
        anomalous = [p for label, p in experiments.items() if label.startswith("Anomalous")]
        normals = [p for label, p in experiments.items() if label.startswith("Normal")]
        if not anomalous or not normals:
            continue

        a_path = str(anomalous[0])
        run = load_experiment(a_path)
        normal_runs = [load_experiment(str(p)) for p in normals]
        present = [s for s in sensors if s in run.columns and all(s in n.columns for n in normal_runs)]

        # scale: 99.5th percentile of the normal run(s) -> "1 = worst normal moment"
        normal_peaks = pd.concat([_peak(_residual(n, present)) for n in normal_runs])
        scale = normal_peaks.quantile(0.995).clip(lower=1e-6)
        score = (_peak(_residual(run, present)) / scale).clip(0, SCORE_CAP)
        score_n = (_peak(_residual(normal_runs[0], present)) / scale).clip(0, SCORE_CAP)

        idx = np.arange(0, len(run), STEP)
        normal = normal_runs[0]
        normal_all = pd.concat(normal_runs)
        jdx = np.arange(0, len(normal), STEP)

        ops[op] = {
            "t": run["ElapsedMin"].values[idx].round(2).tolist(),
            "tn": normal["ElapsedMin"].values[jdx].round(2).tolist(),
            "raw": {s: _round(run[s].values[idx]) for s in present},
            "rawn": {s: _round(normal[s].values[jdx]) for s in present},
            "dev": {s: score[s].values[idx].round(2).tolist() for s in present},
            "devn": {s: score_n[s].values[jdx].round(2).tolist() for s in present},
            # ideal range = middle 98 % of the normal run(s) of this operating point
            "band": {s: [float(f"{v:.4g}") for v in normal_all[s].quantile([0.01, 0.99]).values]
                     for s in present},
            "ev": _events(a_path, run["Seconds"].iloc[0]),
        }
    return {"sensors": sensors, "ops": ops}


# =====================================================================
# Render
# =====================================================================
def render_process_clocks(height: int = 1010) -> None:
    if not TEMPLATE.exists():
        st.error(f"Missing {TEMPLATE}")
        return
    data = build_clock_data()
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":"))
    )
    st.iframe(html, height=height)
