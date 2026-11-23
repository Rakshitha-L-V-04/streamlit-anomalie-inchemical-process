from pathlib import Path
import streamlit as st

from data_loader import DATA_DIR
from clock_view import render_process_clocks

APP_TITLE = "Spotting the Outliers: Anomalies Through Time"
APP_ICON = "🧭"      

# =====================================================================
# Page config (must be the first Streamlit call)
# =====================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
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
# Fit the dashboard to the window: trim Streamlit's own top/bottom spacing
st.markdown(
    "<style>.block-container{padding-top:0.6rem !important;padding-bottom:0 !important}"
    "footer{display:none}</style>",
    unsafe_allow_html=True,
)


# ---------- Sanity check: can we see the dataset? ----------
if not DATA_DIR.exists():
    st.error(f"Dataset folder not found at: {DATA_DIR}")
    st.stop()


# =====================================================================
# 🧭: interactive drill-down overview (all operating points)
# =====================================================================
render_process_clocks()
