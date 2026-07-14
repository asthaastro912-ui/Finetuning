"""Real-time monitoring dashboard over the FastAPI service's request log.

Run: streamlit run src/monitoring/dashboard.py
"""
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common import load_config, resolve_path

st.set_page_config(page_title="Financial QA LoRA — Monitoring", layout="wide")

cfg = load_config()
db_path = resolve_path(cfg["monitoring"]["db_path"])

st.title("Financial QA LoRA — Serving Monitor")
st.caption(f"Reading live requests from `{db_path}`. Refresh with the button below.")

if st.button("Refresh"):
    st.rerun()


@st.cache_data(ttl=5)
def load_requests(path: str, _cache_bust: float):
    if not Path(path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT * FROM requests ORDER BY ts DESC", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
    return df


df = load_requests(str(db_path), time.time() // 5)

if df.empty:
    st.info("No requests logged yet. Hit the /generate endpoint on the FastAPI service to see data here.")
else:
    n_total = len(df)
    n_errors = (df["status"] == "error").sum()
    avg_latency = df.loc[df["status"] == "ok", "latency_ms"].mean()
    avg_hallucination = df.loc[df["status"] == "ok", "numeric_hallucination_rate"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total requests", n_total)
    c2.metric("Error rate", f"{100 * n_errors / n_total:.1f}%")
    c3.metric("Avg latency", f"{avg_latency:.0f} ms" if pd.notna(avg_latency) else "n/a")
    c4.metric("Avg numeric hallucination rate", f"{avg_hallucination:.2f}" if pd.notna(avg_hallucination) else "n/a")

    st.subheader("Latency over time")
    st.plotly_chart(px.scatter(df.sort_values("timestamp"), x="timestamp", y="latency_ms",
                                color="status", title=None), use_container_width=True)

    st.subheader("Numeric hallucination rate over time")
    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        st.plotly_chart(px.scatter(ok_df.sort_values("timestamp"), x="timestamp",
                                    y="numeric_hallucination_rate", title=None),
                         use_container_width=True)

    st.subheader("Recent requests")
    st.dataframe(
        df[["timestamp", "question", "prediction", "latency_ms",
            "numeric_hallucination_rate", "status", "error"]].head(50),
        use_container_width=True,
    )

st.divider()
st.subheader("Offline evaluation report (base vs fine-tuned)")
report_path = resolve_path(cfg["evaluation"]["report_path"])
if report_path.exists():
    import json
    report = json.loads(report_path.read_text())
    base = report["base_model"]
    ft = report["fine_tuned_model"]
    comparison = pd.DataFrame({
        "base (pre-fine-tune)": base,
        "fine-tuned (LoRA)": ft,
    })
    st.dataframe(comparison, use_container_width=True)
    if report.get("hallucination_reduction_pct") is not None:
        st.metric("Hallucination rate reduction vs base model",
                   f"{report['hallucination_reduction_pct']}%")
else:
    st.info(f"No eval report yet at {report_path}. Run `python -m src.evaluation.evaluate` first.")
