import re
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# =====================================================================
# Constants
# =====================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "batch_distillation_dataset"

SENSORS = {
    "FT703":  "Flow in the top product stream",
    "FT704":  "Flow in the reflux stream",
    "FYI702": "Cooling-water flow signal",
    "PDI701": "Pressure difference along the column",
    "PDI702": "Pressure difference at buffer vessel V002",
    "PY23":   "Pressure in top vessel G001",
    "T703":   "Liquid temperature in reboiler V001",
    "T705":   "Temperature at top of section C003",
    "T709":   "Temperature below section C001",
    "T711":   "Temperature between C001 and C002",
    "T712":   "Temperature between C002 and C003",
}


# =====================================================================
# Step 2: Discovering and loading measurements
# =====================================================================
def list_operating_points() -> list[str]:
    return sorted(p.name for p in DATA_DIR.iterdir()
                  if p.is_dir() and p.name.startswith("OP"))


def list_experiments(op: str) -> dict[str, Path]:
    meas_dir = DATA_DIR / op / "measurements"
    experiments = {}

    anomalous = sorted(meas_dir.glob("test_anormal_experiment_*.csv"))
    normal = sorted(meas_dir.glob("train_normal_experiment_*.csv"))

    for i, path in enumerate(anomalous, start=1):
        experiments["Anomalous" if i == 1 else f"Anomalous {i}"] = path
    for i, path in enumerate(normal, start=1):
        experiments["Normal" if i == 1 else f"Normal {i}"] = path

    return experiments


def _time_to_seconds(time_col: pd.Series) -> pd.Series:
    seconds = pd.to_timedelta(time_col.astype(str).str.strip(),
                              errors="coerce").dt.total_seconds()
    crossed_midnight = (seconds.diff() < -12 * 3600).cumsum()
    return seconds + crossed_midnight * 24 * 3600


@st.cache_data
def load_experiment(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df["Seconds"] = _time_to_seconds(df["Time"])
    df = df.dropna(subset=["Seconds"]).reset_index(drop=True)
    df["ElapsedMin"] = (df["Seconds"] - df["Seconds"].iloc[0]) / 60

    sensors_present = [s for s in SENSORS if s in df.columns]
    for s in sensors_present:
        df[s] = pd.to_numeric(df[s], errors="coerce")

    return df[["Time", "Seconds", "ElapsedMin"] + sensors_present]


# =====================================================================
# Step 3: Anomaly annotations (YAML)
# =====================================================================
ANNOTATION_COLUMNS = [
    "id", "class", "category", "perturbation",
    "start_clock", "end_clock",
    "start_min", "end_min", "duration_min",
    "perturbation_mode",
]


def annotation_path(csv_path: str) -> Path:
    csv_path = Path(csv_path)
    ann_dir = csv_path.parent.parent / "annotations"
    for ext in (".yaml", ".yml"):
        candidate = ann_dir / (csv_path.stem + ext)
        if candidate.exists():
            return candidate
    return ann_dir / (csv_path.stem + ".yaml")


def _get(d: dict, *keys, default=None):
    if not isinstance(d, dict):
        return default
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return default


def _clock_to_seconds(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        inner = _get(value, "value", "time", "timestamp", "@value")
        if inner is None and value:
            inner = next(iter(value.values()))
        return _clock_to_seconds(inner)
    if isinstance(value, list):
        return _clock_to_seconds(value[0]) if value else None
    if isinstance(value, (datetime, time)):
        return value.hour * 3600 + value.minute * 60 + value.second
    match = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(value))
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + int(s or 0)
    return None


def _clean_label(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, list):
        value = value[0] if value else "unknown"
    if isinstance(value, dict):
        value = _get(value, "value", "type", "name") or next(iter(value.values()), "unknown")
    text = str(value).rstrip("/").split("#")[-1].split("/")[-1]
    return text.strip().lower()


def _to_text(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_to_text(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "; ".join(_to_text(v) for v in value)
    return str(value)


def _iter_anomalies(data: dict):
    anomalies = _get(data, "anomalies", default=[]) if isinstance(data, dict) else []
    if isinstance(anomalies, list):
        for i, a in enumerate(anomalies, start=1):
            if isinstance(a, dict):
                yield _get(a, "id", default=f"A{i:03d}"), a
    elif isinstance(anomalies, dict):
        if _get(anomalies, "hasBeginning") is not None:
            yield _get(anomalies, "id", default="A001"), anomalies
        else:
            for key, a in anomalies.items():
                if isinstance(a, dict):
                    yield _get(a, "id", default=key), a


@st.cache_data
def load_raw_yaml(csv_path: str) -> dict | None:
    path = annotation_path(csv_path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_annotations(csv_path: str, run_start_seconds: float) -> pd.DataFrame:
    data = load_raw_yaml(csv_path)
    if not data:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    def to_elapsed_min(clock_sec):
        if clock_sec is None:
            return None
        if clock_sec < run_start_seconds - 12 * 3600:
            clock_sec += 24 * 3600
        return (clock_sec - run_start_seconds) / 60

    rows = []
    for anomaly_id, a in _iter_anomalies(data):
        begin_raw = _get(a, "hasBeginning")
        end_raw = _get(a, "hasEnd")
        start_min = to_elapsed_min(_clock_to_seconds(begin_raw))
        end_min = to_elapsed_min(_clock_to_seconds(end_raw))

        rows.append({
            "id": str(anomaly_id),
            "class": _clean_label(_get(a, "class")),
            "category": _clean_label(_get(a, "hasCategory")),
            "perturbation": _to_text(_get(a, "Perturbation")),
            "start_clock": _to_text(begin_raw),
            "end_clock": _to_text(end_raw),
            "start_min": start_min,
            "end_min": end_min,
            "duration_min": (end_min - start_min)
                            if start_min is not None and end_min is not None else None,
            "perturbation_mode": _get(a, "PerturbationMode"),
        })

    return pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)


# =====================================================================
# Step 8: All annotations across every operating point
# =====================================================================
@st.cache_data
def load_all_annotations() -> pd.DataFrame:
    frames = []
    for op in list_operating_points():
        for label, path in list_experiments(op).items():
            if not label.startswith("Anomalous"):
                continue
            run = load_experiment(str(path))
            ann = load_annotations(str(path), run["Seconds"].iloc[0])
            if len(ann):
                frames.append(ann.assign(op=op, experiment=label))

    if not frames:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS + ["op", "experiment"])
    return pd.concat(frames, ignore_index=True)