# ⚗️ Understanding Anomalies in Chemical Processes

An interactive Streamlit dashboard for exploring sensor data from a **batch distillation** process and understanding when, where and for how long anomalies occur.

## Features
- **Sensor over time:** plot any sensor signal for a chosen experiment, with annotated anomaly periods
- **Normal vs. anomalous:** compare sensor behaviour across normal and anomalous runs
- **Anomaly durations:** see how long anomalies last, overall and per operating point
- Sidebar controls to pick the operating point, experiment and sensor

## Tech stack
Python · Streamlit · Plotly · pandas

## Project structure
```
app.py              # Streamlit app: layout, sidebar controls, pages
charts.py           # Plotly figure builders
data_loader.py      # Loading experiments, sensors and anomaly annotations
assets/style.css    # Custom styling
batch_distillation_dataset/   # Dataset
```

## Run locally
```bash
git clone https://github.com/Rakshitha-L-V-04/streamlit-anomalie-inchemical-process.git
cd streamlit-anomalie-inchemical-process
pip install -r requirements.txt
streamlit run app.py
```
