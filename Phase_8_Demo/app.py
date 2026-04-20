"""
Phase 8 — Interactive Streamlit Demo

Run from project root:
    streamlit run Phase_8_Demo/app.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    os.path.join(_ROOT, "Phase_2_Ingestion_Pipeline"),
    os.path.join(_ROOT, "Phase_3_Signal_Processing"),
    os.path.join(_ROOT, "Phase_4_Feature_Engineering"),
    os.path.join(_ROOT, "Phase_6_Pipeline"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loader    import load_file, BASE_DIR
from processor import condition_signal
from extractor import extract_features
from pipeline  import EMIPipeline

INDEX_CSV    = os.path.join(_ROOT, "Phase_1_Dataset_Audit",    "dataset_index.csv")
FEATURES_CSV = os.path.join(_ROOT, "Phase_4_Feature_Engineering", "features.csv")
FIG_DIR      = os.path.join(_ROOT, "Phase_7_Results_Analysis")

DEVICE_COLORS = {
    "motor_1":   "#3b82f6",
    "motor_2":   "#10b981",
    "motor_3":   "#8b5cf6",
    "charger_1": "#ef4444",
    "charger_2": "#f97316",
}
FAMILY_COLORS = {"motor": "#3b82f6", "charger": "#ef4444"}

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EMI Device Fingerprinting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  — modern card-based UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── global typography ── */
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }

/* ── sidebar brand ── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.93rem; padding: 0.3rem 0; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700;
    color: #1e293b;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── dividers ── */
hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── buttons ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}

/* ── expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}

/* ── info / warning boxes ── */
[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def hero(title: str, subtitle: str, accent: str = "#2563eb"):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {accent} 0%, #1e40af 100%);
        color: white;
        padding: 2.2rem 2rem 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    ">
        <div style="font-size:1.8rem; font-weight:700; margin-bottom:0.4rem;">{title}</div>
        <div style="font-size:1rem; opacity:0.88; line-height:1.5;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def card(content_html: str, bg: str = "#f8fafc", border: str = "#e2e8f0"):
    st.markdown(f"""
    <div style="
        background:{bg};
        border:1px solid {border};
        border-radius:12px;
        padding:1.4rem 1.6rem;
        margin:0.6rem 0;
        box-shadow:0 1px 4px rgba(0,0,0,0.05);
    ">{content_html}</div>
    """, unsafe_allow_html=True)


def stage_header(n: int, title: str, subtitle: str):
    st.markdown(f"""
    <div style="
        display:flex; align-items:flex-start; gap:1rem;
        background:white;
        border:1px solid #e2e8f0;
        border-left:4px solid #2563eb;
        border-radius:0 10px 10px 0;
        padding:0.8rem 1.2rem;
        margin: 1.8rem 0 0.6rem;
        box-shadow:0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="
            background:#2563eb; color:white;
            border-radius:6px; padding:0.15em 0.6em;
            font-size:0.8rem; font-weight:700;
            white-space:nowrap; margin-top:2px;
        ">Stage {n}</div>
        <div>
            <div style="font-size:1rem; font-weight:600; color:#1e293b;">{title}</div>
            <div style="font-size:0.83rem; color:#64748b; margin-top:0.15rem;">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def phase_card(number: int, title: str, color: str,
               challenge: str, built: str, output: str):
    st.markdown(f"""
    <div style="
        background:white;
        border:1px solid #e2e8f0;
        border-radius:14px;
        overflow:hidden;
        margin:0.75rem 0;
        box-shadow:0 2px 6px rgba(0,0,0,0.06);
    ">
        <div style="
            background:{color};
            padding:0.9rem 1.4rem;
            display:flex; align-items:center; gap:0.9rem;
        ">
            <div style="
                background:rgba(255,255,255,0.25);
                color:white;
                font-size:1rem; font-weight:800;
                border-radius:8px;
                width:2.2rem; height:2.2rem;
                display:flex; align-items:center; justify-content:center;
            ">{number}</div>
            <div style="color:white; font-size:1.05rem; font-weight:600;">{title}</div>
        </div>
        <div style="padding:1.2rem 1.4rem; display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.2rem;">
            <div>
                <div style="font-size:0.7rem; font-weight:700; color:#64748b;
                            text-transform:uppercase; letter-spacing:0.07em;
                            margin-bottom:0.35rem;">The challenge</div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.5;">{challenge}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; font-weight:700; color:#64748b;
                            text-transform:uppercase; letter-spacing:0.07em;
                            margin-bottom:0.35rem;">What we built</div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.5;">{built}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; font-weight:700; color:#64748b;
                            text-transform:uppercase; letter-spacing:0.07em;
                            margin-bottom:0.35rem;">Key output</div>
                <div style="font-size:0.88rem; color:#334155; line-height:1.5;">{output}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def prediction_badge(label: str, value: str, color: str = "#2563eb"):
    st.markdown(f"""
    <div style="
        background:white; border:2px solid {color};
        border-radius:12px; padding:1rem 1.4rem;
        text-align:center;
    ">
        <div style="font-size:0.72rem; font-weight:700; color:{color};
                    text-transform:uppercase; letter-spacing:0.07em;">{label}</div>
        <div style="font-size:1.45rem; font-weight:800; color:#1e293b;
                    margin-top:0.3rem;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading trained models ...")
def get_pipeline() -> EMIPipeline:
    return EMIPipeline()


@st.cache_data(show_spinner=False)
def get_index() -> pd.DataFrame:
    df = pd.read_csv(INDEX_CSV)
    return df[(df["domain"] == "time") & (~df["is_duplicate"].astype(bool))].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV)


@st.cache_data(show_spinner=False)
def get_pca_projection():
    df = get_features()
    feat_cols = [c for c in df.columns if c.startswith("td_pri_") or c.startswith("fd_pri_")]
    df = df.dropna(subset=feat_cols).drop_duplicates(subset=feat_cols).reset_index(drop=True)
    X = df[feat_cols].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=2, random_state=42).fit(Xs)
    coords = pca.transform(Xs)
    return {
        "feat_cols": feat_cols,
        "scaler":    scaler,
        "pca":       pca,
        "coords":    coords,
        "device_id": df["device_id"].values,
        "family":    df["device_family"].values,
        "explained": pca.explained_variance_ratio_,
    }


@st.cache_data(show_spinner=False)
def get_class_stats():
    df = get_features()
    feat_cols = [c for c in df.columns if c.startswith("td_pri_") or c.startswith("fd_pri_")]
    grouped = df.groupby("device_id")[feat_cols].median()
    return grouped, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEV_LABEL = {
    "motor_1":   "Motor 1",
    "motor_2":   "Motor 2",
    "motor_3":   "Motor 3",
    "charger_1": "Charger 1",
    "charger_2": "Charger 2",
}


def _display_name(row: pd.Series) -> str:
    """Human-readable name for a CSV capture row."""
    m = re.search(r"(\d+)", row["filename"])
    num = int(m.group(1)) if m else 0
    dev = _DEV_LABEL.get(row["device_id"], row["device_id"])
    state = row["state"].upper()
    dist = str(row.get("distance_label", "")).lower()
    dist_suffix = ""
    if dist not in ("unknown", "ok", "", "nan", "none"):
        dist_suffix = f" [{dist}]"
    return f"{dev}  \u00b7  {state}{dist_suffix}  \u00b7  Capture #{num}"


def _primary(sig: dict):
    mode = sig.get("channel_mode", "")
    if mode == "single_ch2" or (sig.get("ch1") is None and sig.get("ch2") is not None):
        return sig["ch2"], "ch2"
    return sig["ch1"], "ch1"


def _fft(x, fs):
    if len(x) < 4 or not fs:
        return np.array([]), np.array([])
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mags  = np.abs(np.fft.rfft(x - np.mean(x))) / len(x)
    return freqs, mags


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-size:1.1rem; font-weight:800; color:#f1f5f9;
                    letter-spacing:0.01em;">EMI Fingerprinting</div>
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.2rem;">
            ADSC  ×  IEEE  —  Interactive Demo
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "nav",
        ["Overview", "Live Demo", "Project Phases", "Dataset Explorer", "Results & Methodology"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
    <div style="font-size:0.73rem; color:#64748b; line-height:1.7;">
        <b style="color:#94a3b8;">Start here:</b><br>
        Overview &#8594; Project Phases &#8594; Live Demo<br><br>
        <b style="color:#94a3b8;">Data & results:</b><br>
        Dataset Explorer &#8594; Results & Methodology
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────

def render_overview():
    hero(
        "Can we identify a device just from its electrical noise?",
        "Every motor and laptop charger leaks a unique electromagnetic interference (EMI) "
        "signature. This project captures those signatures with an oscilloscope, extracts "
        "signal features, and uses machine learning to identify the exact device — model, "
        "unit, and operating state.",
        accent="#1d4ed8",
    )

    df_idx  = get_index()
    df_feat = get_features()
    n_feats = sum(c.startswith(("td_pri_", "fd_pri_")) for c in df_feat.columns)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time-domain captures", f"{len(df_idx):,}")
    c2.metric("Distinct devices", df_idx["device_id"].nunique())
    c3.metric("Features per signal", n_feats)
    c4.metric("Best test accuracy", "100%")

    st.divider()

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("### The problem")
        card("""
        <ul style="margin:0; padding-left:1.2rem; color:#334155; line-height:1.7; font-size:0.92rem;">
            <li>Two laptop chargers of the same make look <b>identical</b> from the outside.</li>
            <li>Three commodity motors of the same model are <b>interchangeable</b> to the eye.</li>
            <li>But each one emits a <b>different pattern of electromagnetic interference</b>
                because of microscopic variation in its switching electronics and mechanical parts.</li>
            <li>Question: <b>can a classifier trained on EMI captures tell them apart?</b></li>
        </ul>
        """)

        st.markdown("### How we solved it")
        steps = [
            ("Capture", "Raw oscilloscope voltage traces from each device in different states."),
            ("Condition", "Low-pass filter and z-score each signal, individually per device rate."),
            ("Extract", "Compute 23 time- and frequency-domain features per trace."),
            ("Classify", "Hierarchical ML: family first (motor vs charger), then identity within."),
            ("Validate", "Proper 80/20 splits, deduplication, rate-agnostic controls."),
        ]
        for step, desc in steps:
            st.markdown(f"""
            <div style="display:flex; gap:0.75rem; align-items:flex-start;
                        margin:0.5rem 0; font-size:0.9rem;">
                <div style="background:#dbeafe; color:#1d4ed8; border-radius:6px;
                            padding:0.1em 0.6em; font-weight:700; white-space:nowrap;
                            margin-top:1px;">{step}</div>
                <div style="color:#334155; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("### Classification hierarchy")
        st.graphviz_chart("""
        digraph G {
            rankdir=TB; splines=ortho;
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
            edge [fontsize=10];

            RAW [label="Raw oscilloscope CSV",
                 fillcolor="#f1f5f9", color="#94a3b8"];
            FAM [label="Family classifier\l(SVM, rate-agnostic features)\l",
                 fillcolor="#dbeafe", color="#3b82f6"];
            MID [label="Motor ID classifier\l(SVM, 3-class)\l",
                 fillcolor="#d1fae5", color="#10b981"];
            CST [label="Charger state\l(RF, ON / OFF)\l",
                 fillcolor="#fff7ed", color="#f97316"];
            CID [label="Charger ID\l(SVM, 2-class)\l",
                 fillcolor="#fff7ed", color="#f97316"];

            RAW -> FAM;
            FAM -> MID  [label=" motor"];
            FAM -> CST  [label=" charger"];
            FAM -> CID  [label=" charger"];
        }
        """)

        st.markdown("### Headline results")
        results_md = pd.DataFrame([
            ["Motor vs Charger (rate-agnostic)", "SVM", "100.0%"],
            ["Charger ON vs OFF",                "RF",  "100.0%"],
            ["Motor ID — 3 units",               "LR",  "100.0%"],
            ["Charger ID — 2 units",             "SVM", " 94.6%"],
        ], columns=["Task", "Model", "Test acc"])
        st.dataframe(results_md, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#64748b; font-size:0.88rem; padding:0.5rem 0 1rem;">
        Navigate to <b>Live Demo</b> to run the full pipeline on a real captured signal.
        Visit <b>Project Phases</b to understand how each stage was built.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Live Demo
# ─────────────────────────────────────────────────────────────────────────────

def render_live_demo():
    hero(
        "Live Pipeline Demo",
        "Pick a captured signal and watch every processing stage run in real time — "
        "from raw oscilloscope voltage to device identity.",
        accent="#1d4ed8",
    )

    with st.expander("First time here? Read this first.", expanded=False):
        st.markdown("""
        **What this demo does:**
        1. You pick a captured signal from our dataset (or upload your own oscilloscope CSV).
        2. The same pipeline that produced our benchmark results runs live.
        3. Each of the 6 stages renders an interactive chart with a plain-English explanation.
        4. At the end you see where this signal lands in the full feature space relative to all training data.

        **Tips for demoing:**
        - Try signals from different devices and compare how the spectra and feature fingerprints differ.
        - Upload mode accepts any oscilloscope CSV in the same 2-row-header format.
        - The PCA panel (Stage 6) is the best single visual to show to an audience.
        """)

    idx  = get_index()
    pipe = get_pipeline()

    # ── File selector ────────────────────────────────────────────────────────
    st.markdown("#### Select a signal")
    mode = st.radio(
        "Source", ["Choose from dataset", "Upload CSV"],
        horizontal=True, label_visibility="collapsed",
    )

    selected_path = None
    truth         = None

    if mode == "Choose from dataset":
        # Build display-name → row mapping per (family, device, state) grouping
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fam_label = {"motor": "Motor (induction motor)",
                         "charger": "Charger (laptop power supply)"}
            fams = {fam_label.get(f, f): f for f in sorted(idx["device_family"].unique())}
            fam_disp = st.selectbox("Device family", list(fams.keys()))
            fam = fams[fam_disp]

        sub = idx[idx["device_family"] == fam]
        dev_labels = {_DEV_LABEL.get(d, d): d for d in sorted(sub["device_id"].unique())}
        with col_b:
            dev_disp = st.selectbox("Device", list(dev_labels.keys()))
            dev = dev_labels[dev_disp]

        sub = sub[sub["device_id"] == dev]
        state_labels = {s.upper(): s for s in sorted(sub["state"].unique())}
        with col_c:
            state_disp = st.selectbox("Operating state", list(state_labels.keys()))
            state = state_labels[state_disp]

        sub = sub[sub["state"] == state].copy()
        sub["_display"] = sub.apply(_display_name, axis=1)
        disp_to_row = {row["_display"]: row for _, row in sub.iterrows()}

        chosen = st.selectbox(
            f"Capture ({len(sub)} available in this group)",
            list(disp_to_row.keys()),
        )
        row_sel = disp_to_row[chosen]
        selected_path = os.path.join(BASE_DIR, row_sel["file_path"])
        truth = {
            "family": row_sel["device_family"],
            "device": row_sel["device_id"],
            "state":  row_sel["state"],
        }
    else:
        up = st.file_uploader(
            "Upload an oscilloscope CSV (2-row-header format)", type=["csv"]
        )
        if up is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(up.read())
            tmp.close()
            selected_path = tmp.name

    run = st.button(
        "Run pipeline on this signal",
        type="primary",
        disabled=(selected_path is None),
    )
    if not run:
        st.info(
            "Pick a signal above and press **Run pipeline** to see the full processing "
            "chain rendered step by step."
        )
        return

    # ── Load + condition + extract ───────────────────────────────────────────
    with st.spinner("Processing signal through all pipeline stages ..."):
        try:
            sig = load_file(selected_path)
        except Exception as e:
            st.error(f"Could not load file: {e}")
            return
        if sig is None:
            st.error(
                "File could not be loaded. Make sure it is a time-domain oscilloscope CSV, "
                "not a frequency-domain export."
            )
            return

        cond      = condition_signal(sig)
        raw_arr, ch_name = _primary(sig)
        cond_arr  = cond[ch_name]
        t         = sig["time"]
        fs        = sig["sample_rate_hz"]
        row_feats = extract_features(sig, cond)
        result    = pipe.predict_file(selected_path)

    # ── Prediction banner ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Prediction result")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        prediction_badge("Device family", result.get("device_family", "—"), "#2563eb")
    with p2:
        if result.get("device_family") == "motor":
            val = _DEV_LABEL.get(result.get("motor_id", ""), result.get("motor_id", "—"))
            prediction_badge("Motor identity", val, "#10b981")
        else:
            val = _DEV_LABEL.get(result.get("charger_id", ""), result.get("charger_id", "—"))
            prediction_badge("Charger identity", val, "#ef4444")
    with p3:
        if result.get("device_family") == "charger":
            prediction_badge("Operating state", result.get("charger_state", "—").upper(), "#f97316")
        else:
            prediction_badge("Sample rate", f"{int(fs/1000)} kHz", "#6366f1")
    with p4:
        if truth:
            dev_pred = result.get("motor_id") or result.get("charger_id")
            ok_fam   = truth["family"] == result.get("device_family")
            ok_dev   = truth["device"] == dev_pred
            if ok_fam and ok_dev:
                prediction_badge("Verdict", "Correct", "#059669")
            elif ok_fam:
                prediction_badge("Verdict", "Family OK / ID wrong", "#d97706")
            else:
                prediction_badge("Verdict", "Wrong family", "#dc2626")
        else:
            prediction_badge("Source", "Uploaded file", "#64748b")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    stage_header(1, "Raw signal",
        f"Captured at {fs/1000:.1f} kHz · {len(raw_arr)} samples · "
        f"{len(raw_arr)/fs*1000:.1f} ms window · primary channel: {ch_name.upper()}")

    st.markdown("""
    <p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem;">
    This is the raw voltage the oscilloscope probe picked up.
    You are seeing <b>electromagnetic interference leaking out of the device</b> — not a signal
    the device intentionally transmits.  The amplitude, shape, and texture already encode
    information about what is running.
    </p>""", unsafe_allow_html=True)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=t * 1000, y=raw_arr,
        mode="lines", line=dict(color="#475569", width=1), name="voltage",
    ))
    fig1.update_layout(
        height=240, margin=dict(l=50, r=20, t=10, b=45),
        xaxis_title="time (ms)", yaxis_title="voltage (V)",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    stage_header(2, "Signal conditioning",
        "Zero-phase Butterworth low-pass at 80% Nyquist, then z-score normalisation")

    st.markdown("""
    <p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem;">
    The filter removes noise above the band of interest (8 kHz for motors, 2 kHz for chargers)
    without distorting peak timing.  Z-score sets every signal to zero mean / unit variance so
    that motors and chargers can be compared on <b>shape</b>, not just voltage scale.
    Amplitude features are extracted <i>before</i> this step from the raw signal.
    </p>""", unsafe_allow_html=True)

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Raw (DC removed)", "Conditioned (filtered + z-scored)"),
    )
    raw_c = raw_arr - np.mean(raw_arr)
    fig2.add_trace(go.Scatter(x=t*1000, y=raw_c,    mode="lines",
                              line=dict(color="#94a3b8", width=1)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=t*1000, y=cond_arr, mode="lines",
                              line=dict(color="#2563eb", width=1)), row=1, col=2)
    fig2.update_layout(
        height=270, margin=dict(l=50, r=20, t=40, b=45), showlegend=False,
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
    )
    fig2.update_xaxes(title_text="time (ms)", showgrid=True, gridcolor="#e2e8f0")
    fig2.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    stage_header(3, "Frequency spectrum (FFT)",
        "The EMI fingerprint lives here — each device has a characteristic spectral signature")

    st.markdown("""
    <p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem;">
    The FFT converts the time-domain waveform into frequency components.
    Motors show sharp harmonics from their commutation frequency;
    chargers show broader switching noise bands.  The dominant frequency,
    spectral entropy, and band-energy ratios are extracted as features from this spectrum.
    The log scale prevents a single dominant peak from masking the rest of the spectrum.
    </p>""", unsafe_allow_html=True)

    freqs, mags = _fft(cond_arr, fs)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=freqs, y=mags,
        mode="lines", line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
    ))
    if len(mags):
        dom_idx = int(np.argmax(mags))
        fig3.add_vline(
            x=freqs[dom_idx], line_dash="dash", line_color="#1e293b", line_width=1.5,
            annotation_text=f"dominant  {freqs[dom_idx]:.0f} Hz",
            annotation_font_size=11,
        )
    fig3.update_layout(
        height=270, margin=dict(l=50, r=20, t=10, b=45),
        xaxis_title="frequency (Hz)", yaxis_title="|FFT| (log)",
        yaxis_type="log",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    stage_header(4, "Feature fingerprint",
        "How this signal's features compare to every device's median profile")

    st.markdown("""
    <p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem;">
    Each line is the median feature profile for one device class.
    The <b>black diamond trace</b> is this signal.
    The closer it tracks a coloured line across all features, the more likely it is
    to belong to that class.  No single feature is decisive — the classifier uses
    all 23 simultaneously.
    </p>""", unsafe_allow_html=True)

    class_medians, feat_cols = get_class_stats()
    highlight = [
        "td_pri_peak_to_peak", "td_pri_kurtosis", "td_pri_crest_factor",
        "td_pri_zero_crossing_rate", "fd_pri_band_energy_rel_high",
        "fd_pri_band_energy_rel_low", "fd_pri_spectral_entropy",
        "fd_pri_dominant_freq_norm",
    ]
    highlight = [f for f in highlight if f in class_medians.columns]

    fig4 = go.Figure()
    xs = list(range(len(highlight)))
    for did in class_medians.index:
        vals = class_medians.loc[did, highlight].values
        fig4.add_trace(go.Scatter(
            x=xs, y=vals, mode="lines+markers", name=_DEV_LABEL.get(did, did),
            line=dict(color=DEVICE_COLORS.get(did, "#999"), width=2),
            marker=dict(size=7), opacity=0.6,
        ))
    this_vals = [row_feats.get(f, np.nan) for f in highlight]
    fig4.add_trace(go.Scatter(
        x=xs, y=this_vals, mode="lines+markers", name="This signal",
        line=dict(color="#0f172a", width=3),
        marker=dict(size=13, symbol="diamond", color="#0f172a"),
    ))
    labels = [
        f.replace("td_pri_","TD: ").replace("fd_pri_","FD: ")
         .replace("_"," ") for f in highlight
    ]
    fig4.update_layout(
        height=380, margin=dict(l=50, r=20, t=10, b=120),
        xaxis=dict(tickmode="array", tickvals=xs, ticktext=labels, tickangle=-35),
        yaxis_title="feature value",
        legend=dict(orientation="h", y=-0.52, font_size=11),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis_gridcolor="#e2e8f0", yaxis_gridcolor="#e2e8f0",
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Stage 5 ──────────────────────────────────────────────────────────────
    fam_pred = result.get("device_family", "unknown")
    stage_header(5, "Hierarchical prediction",
        f"Family classifier ran first (rate-agnostic SVM), "
        f"then the {fam_pred}-specific classifier resolved identity")

    st.markdown("""
    <p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem;">
    The family classifier deliberately <b>excludes</b> any feature denominated in Hz
    (dominant frequency, spectral centroid) to prevent it from separating devices purely
    by sampling rate.  Passing rate-agnostic validation at 100% means the separation
    is driven by real EMI characteristics, not an instrumentation shortcut.
    </p>""", unsafe_allow_html=True)

    flow = f"""
    digraph G {{
        rankdir=LR; splines=ortho;
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
        IN  [label="17 rate-agnostic\\nfeatures", fillcolor="#f1f5f9", color="#94a3b8"];
        FAM [label="Family classifier\\n→  {fam_pred}", fillcolor="#dbeafe", color="#3b82f6"];
    """
    if fam_pred == "motor":
        mid = _DEV_LABEL.get(result.get("motor_id",""), result.get("motor_id","?"))
        flow += f"""
        MID [label="Motor ID\\n→  {mid}", fillcolor="#d1fae5", color="#10b981"];
        IN -> FAM -> MID;
        """
    else:
        cid = _DEV_LABEL.get(result.get("charger_id",""), result.get("charger_id","?"))
        cst = (result.get("charger_state","?") or "").upper()
        flow += f"""
        CST [label="Charger state\\n→  {cst}", fillcolor="#fff7ed", color="#f97316"];
        CID [label="Charger ID\\n→  {cid}", fillcolor="#fff7ed", color="#f97316"];
        IN -> FAM; FAM -> CST; FAM -> CID;
        """
    flow += "}"
    st.graphviz_chart(flow)

    # ── Stage 6 ──────────────────────────────────────────────────────────────
    stage_header(6, "Signal location in feature space",
        "PCA of the full 23-feature training set — where does this signal land?")

    st.markdown("""
    <p style="font-size:0.85rem; color:#475569; margin:0 0 0.5rem;">
    Principal component analysis compresses 23 features into two axes that explain
    most of the variance.  Well-separated clusters confirm the classifier has
    genuinely distinct distributions to learn from.
    The <b>black diamond</b> is this signal — it should land inside or near its predicted cluster.
    </p>""", unsafe_allow_html=True)

    proj = get_pca_projection()
    x_new = np.array([row_feats.get(c, np.nan) for c in proj["feat_cols"]]).reshape(1, -1)
    if np.any(np.isnan(x_new)):
        st.warning("Some features could not be computed for this signal — PCA projection skipped.")
    else:
        xy = proj["pca"].transform(proj["scaler"].transform(x_new))[0]
        fig6 = go.Figure()
        for did in np.unique(proj["device_id"]):
            m = proj["device_id"] == did
            fig6.add_trace(go.Scatter(
                x=proj["coords"][m, 0], y=proj["coords"][m, 1],
                mode="markers",
                name=_DEV_LABEL.get(did, did),
                marker=dict(size=9, color=DEVICE_COLORS.get(did, "#999"),
                            opacity=0.55, line=dict(width=0.5, color="white")),
            ))
        fig6.add_trace(go.Scatter(
            x=[xy[0]], y=[xy[1]], mode="markers", name="This signal",
            marker=dict(size=24, symbol="diamond", color="#0f172a",
                        line=dict(width=2.5, color="white")),
        ))
        pct1, pct2 = proj["explained"][0]*100, proj["explained"][1]*100
        fig6.update_layout(
            height=520, margin=dict(l=50, r=20, t=10, b=50),
            xaxis_title=f"PC 1  ({pct1:.1f}% of variance)",
            yaxis_title=f"PC 2  ({pct2:.1f}% of variance)",
            legend=dict(font_size=11),
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ── All features table ───────────────────────────────────────────────────
    with st.expander("Full feature table for this signal (all 23 primary-channel features)"):
        pri = [c for c in row_feats if c.startswith(("td_pri_", "fd_pri_"))]
        fdf = pd.DataFrame({
            "Feature":  [c.replace("td_pri_","").replace("fd_pri_","").replace("_"," ") for c in pri],
            "Group":    ["Time-domain" if c.startswith("td") else "Frequency-domain" for c in pri],
            "Value":    [round(row_feats[c], 6) if row_feats[c] is not None else None for c in pri],
        })
        st.dataframe(fdf, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Project Phases
# ─────────────────────────────────────────────────────────────────────────────

_PHASES = [
    dict(
        number=1, title="Dataset Audit",
        color="#4f46e5",
        challenge=(
            "783 raw oscilloscope CSV files were spread across folders with no central "
            "inventory. Files mixed time-domain and frequency-domain exports, contained "
            "duplicates (e.g. NewFile1(1).csv), and spanned two distinct sampling rates "
            "with no documentation."
        ),
        built=(
            "<b>dataset_index.csv</b> — a machine-readable manifest tagging every file "
            "with: device family, device ID, operating state, domain (time/freq), "
            "channel mode (single/dual), sampling increment, duplicate flag, and quality notes."
        ),
        output=(
            "A 784-row CSV that all downstream phases consume as ground truth. "
            "Without it, labels would have to be inferred from folder names — fragile "
            "and error-prone."
        ),
    ),
    dict(
        number=2, title="Ingestion Pipeline",
        color="#0891b2",
        challenge=(
            "Oscilloscopes export a non-standard 2-row-header CSV format. Row 0 holds "
            "column names and metadata keys; Row 1 holds units and metadata values. "
            "Files may have one or two channels (CH1, CH2) and the time axis must be "
            "reconstructed from Start and Increment fields."
        ),
        built=(
            "<b>loader.py</b> — parses the 2-row header, extracts both channels, "
            "reconstructs the time axis as <code>start + i × increment</code>, attaches "
            "Phase 1 labels, and returns a standardised Signal dict used by every phase."
        ),
        output=(
            "A consistent <code>Signal</code> dict: <code>{ ch1, ch2, time, "
            "sample_rate_hz, device_family, device_id, state, ... }</code>.  "
            "The contract between all downstream stages."
        ),
    ),
    dict(
        number=3, title="Signal Conditioning",
        color="#059669",
        challenge=(
            "Motors are captured at 20 kHz; chargers at 5 kHz. Applying a fixed-frequency "
            "filter would treat them inconsistently. Additionally, z-score normalisation "
            "must be done <i>after</i> amplitude features are extracted — normalising first "
            "would collapse RMS to ~1.0 for every signal."
        ),
        built=(
            "<b>processor.py</b> — zero-phase Butterworth low-pass at 80% of each "
            "signal's own Nyquist (8 kHz for motors, 2 kHz for chargers), followed by "
            "z-score. <i>Per-signal</i>, not global, so conditioning is proportionally "
            "equivalent across device types."
        ),
        output=(
            "A conditioned Signal dict used for shape and spectral feature extraction. "
            "The raw Signal is preserved separately for amplitude features."
        ),
    ),
    dict(
        number=4, title="Feature Engineering",
        color="#d97706",
        challenge=(
            "A 1200-sample waveform is too high-dimensional for a small dataset. "
            "We need compact, meaningful representations that capture both the "
            "voltage amplitude and the spectral texture of each device's EMI without "
            "leaking sample-rate information into the features."
        ),
        built=(
            "<b>extractor.py</b> — 23 primary-channel features: "
            "<b>amplitude</b> (mean, RMS, peak-to-peak, energy) from <i>raw</i> signals; "
            "<b>shape</b> (kurtosis, crest factor, ZCR, IQR) from <i>conditioned</i>; "
            "<b>spectral</b> (dominant freq, spectral entropy, band energies) from FFT "
            "of conditioned."
        ),
        output=(
            "<b>features.csv</b> — 371 rows × 51 columns (23 primary + cross-channel). "
            "This file trains all classifiers and powers the Live Demo's feature fingerprint plot."
        ),
    ),
    dict(
        number=5, title="Modelling & Evaluation",
        color="#dc2626",
        challenge=(
            "Small datasets invite overfitting and data leakage. Key risks: "
            "27 exact duplicate feature vectors that could straddle train/test splits; "
            "Hz-denominated features that let a classifier cheat by reading sampling rate; "
            "confusion matrices evaluated on training data (a common mistake)."
        ),
        built=(
            "5 tasks with <b>stratified 80/20 splits after deduplication</b>. "
            "Task 1b uses rate-agnostic features only, proving separation isn't a "
            "sample-rate shortcut. All confusion matrices are on held-out test data. "
            "5-fold CV on training split only."
        ),
        output=(
            "4 saved classifiers (family SVM, charger-state RF, motor-ID SVM, charger-ID SVM) "
            "with test accuracies of 100% / 100% / 100% / 94.6%. Methodology report in "
            "Phase 7 figures."
        ),
    ),
    dict(
        number=6, title="Pipeline Integration",
        color="#7c3aed",
        challenge=(
            "All 5 phases exist as separate scripts. A user running an experiment "
            "should not need to manually chain loader → processor → extractor → classifier "
            "and manage intermediate files. The pipeline must also handle any raw CSV, "
            "not just files in the training set."
        ),
        built=(
            "<b>EMIPipeline</b> class: one call <code>pipe.predict_file(path)</code> "
            "runs all stages — Load → Condition → Extract → Hierarchical predict — "
            "and returns a clean dict. Models load once and are cached. "
            "Batch prediction via <code>predict_batch(paths)</code>."
        ),
        output=(
            "Integration test: 21 held-out captures, 20 pass (95.2%). The one failure "
            "is a charger-1 / charger-2 swap, consistent with the 94.6% test accuracy "
            "on that task."
        ),
    ),
    dict(
        number=7, title="Results Analysis",
        color="#0e7490",
        challenge=(
            "High accuracy numbers alone do not explain <i>why</i> the pipeline works. "
            "An audience needs to see which physical EMI characteristics differ between "
            "devices, whether the feature clusters are genuinely separated, and which "
            "features are responsible."
        ),
        built=(
            "5 publication-quality figures: signal traces for all 5 devices, "
            "log-scale spectral comparison, feature-narrative bar chart (8 discriminative "
            "features), PCA separation scatter, and a headline results panel."
        ),
        output=(
            "fig_p7_01 through fig_p7_05. All displayed in the "
            "<b>Results & Methodology</b> page. The PCA figure is the clearest single "
            "visual showing that the feature space is genuinely separable."
        ),
    ),
]


def render_phases():
    hero(
        "Project Phases 1 – 7",
        "A walkthrough of every step — the engineering challenge, what was built, "
        "and why each stage is necessary for the pipeline to work.",
        accent="#4f46e5",
    )
    st.markdown("""
    <p style="color:#64748b; font-size:0.9rem; margin-bottom:0.5rem;">
    Each card shows the challenge we faced, what code or artifact was built to solve it,
    and the output that the next phase depends on.  The <b>Live Demo</b> page runs all
    seven phases sequentially in real time on a signal you pick.
    </p>
    """, unsafe_allow_html=True)

    for p in _PHASES:
        phase_card(**p)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────

def render_dataset_explorer():
    hero(
        "Dataset Explorer",
        "Browse all 783 captured signals — filter by device, state, or channel mode.",
        accent="#0891b2",
    )

    idx = get_index()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total signals", f"{len(idx):,}")
    c2.metric("Motor signals",   int((idx["device_family"] == "motor").sum()))
    c3.metric("Charger signals", int((idx["device_family"] == "charger").sum()))
    c4.metric("Dual-channel",    int((idx["channel_mode"] == "dual").sum()))

    st.divider()

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Signals per device and state")
        by_dev = idx.groupby(["device_id", "state"]).size().reset_index(name="count")
        fig = go.Figure()
        state_colors = {"on": "#2563eb", "off": "#94a3b8"}
        for s in sorted(by_dev["state"].unique()):
            sub = by_dev[by_dev["state"] == s]
            fig.add_trace(go.Bar(
                x=[_DEV_LABEL.get(d, d) for d in sub["device_id"]],
                y=sub["count"], name=s.upper(),
                marker_color=state_colors.get(s, "#888"),
            ))
        fig.update_layout(
            barmode="stack", height=350,
            margin=dict(l=40, r=10, t=10, b=40),
            yaxis_title="capture count",
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Sampling rate distribution")
        valid_inc = idx["increment_sec"].replace(0, float("nan")).dropna()
        rates = (1.0 / valid_inc).round(0).astype(int)
        fig = go.Figure(go.Histogram(
            x=rates, nbinsx=20,
            marker_color="#2563eb", marker_line_color="#1d4ed8", marker_line_width=0.5,
        ))
        fig.update_layout(
            height=350, margin=dict(l=40, r=10, t=10, b=40),
            xaxis_title="sample rate (Hz)",
            yaxis_title="file count",
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <p style="font-size:0.83rem; color:#64748b;">
        Two distinct clusters: motors at <b>20 kHz</b> and chargers at <b>5 kHz</b>.
        This is exactly why the family classifier is validated on <i>rate-agnostic</i>
        features — we want to rule out the classifier simply reading the sample rate.
        </p>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Filterable capture table")
    f1, f2, f3 = st.columns(3)
    with f1:
        fam_filter   = st.multiselect("Family",  sorted(idx["device_family"].unique()))
    with f2:
        dev_filter   = st.multiselect("Device",  [_DEV_LABEL.get(d, d) for d in sorted(idx["device_id"].unique())])
    with f3:
        state_filter = st.multiselect("State",   sorted(idx["state"].unique()))

    rev_label = {v: k for k, v in _DEV_LABEL.items()}
    view = idx.copy()
    if fam_filter:
        view = view[view["device_family"].isin(fam_filter)]
    if dev_filter:
        view = view[view["device_id"].isin([rev_label.get(d, d) for d in dev_filter])]
    if state_filter:
        view = view[view["state"].isin(state_filter)]

    view = view.copy()
    view["display_name"] = view.apply(_display_name, axis=1)
    st.dataframe(
        view[["display_name", "device_family", "state", "channel_mode",
              "sample_count", "increment_sec", "quality_flags"]].rename(columns={
            "display_name":  "Signal",
            "device_family": "Family",
            "state":         "State",
            "channel_mode":  "Channels",
            "sample_count":  "Samples",
            "increment_sec": "Increment (s)",
            "quality_flags": "Quality",
        }),
        hide_index=True, use_container_width=True, height=400,
    )
    st.caption(f"Showing {len(view):,} of {len(idx):,} signals.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Results & Methodology
# ─────────────────────────────────────────────────────────────────────────────

def _modeling_flow_chart():
    """Plotly-based modeling pipeline flow diagram."""
    fig = go.Figure()

    fig.update_layout(
        height=380,
        xaxis=dict(range=[0, 10], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[0, 7], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
    )

    # ── Node definitions ──────────────────────────────────────────────────────
    # (cx, cy, w, h, label_html, bg, border, text_color)
    nodes = [
        # Input
        (1.1,  3.5, 1.5, 0.75,
         "23 Signal<br>Features",
         "#f1f5f9", "#94a3b8", "#475569"),
        # Family classifier
        (3.3,  3.5, 1.7, 0.75,
         "Family Classifier<br><i>(SVM · rate-agnostic)</i>",
         "#dbeafe", "#3b82f6", "#1e40af"),
        # Motor ID
        (6.0,  5.5, 1.7, 0.75,
         "Motor ID<br><i>(SVM · 3-class)</i>",
         "#d1fae5", "#10b981", "#065f46"),
        # Charger state
        (6.0,  3.5, 1.7, 0.75,
         "Charger State<br><i>(RF · ON / OFF)</i>",
         "#fff7ed", "#f97316", "#7c2d12"),
        # Charger ID
        (6.0,  1.5, 1.7, 0.75,
         "Charger ID<br><i>(SVM · 2-class)</i>",
         "#fff7ed", "#f97316", "#7c2d12"),
        # Outputs
        (8.8,  5.5, 1.5, 0.75,
         "Motor 1 / Motor 2<br>/ Motor 3",
         "#ecfdf5", "#10b981", "#065f46"),
        (8.8,  3.5, 1.5, 0.75,
         "ON  /  OFF",
         "#fef3c7", "#f59e0b", "#78350f"),
        (8.8,  1.5, 1.5, 0.75,
         "Charger 1<br>/ Charger 2",
         "#fef3c7", "#f59e0b", "#78350f"),
    ]

    for cx, cy, w, h, label, bg, border, tc in nodes:
        x0, x1 = cx - w / 2, cx + w / 2
        y0, y1 = cy - h / 2, cy + h / 2
        # Rectangle
        fig.add_shape(type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=bg,
            line=dict(color=border, width=1.8),
            layer="below",
        )
        # Label
        fig.add_annotation(
            x=cx, y=cy, text=label,
            showarrow=False,
            font=dict(size=10.5, color=tc, family="Inter, Segoe UI, sans-serif"),
            align="center",
        )

    # ── Arrows ────────────────────────────────────────────────────────────────
    def arrow(x0, y0, x1, y1, color="#94a3b8", label=""):
        # Line
        fig.add_shape(type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=color, width=1.8),
        )
        # Arrowhead as a scatter point
        fig.add_trace(go.Scatter(
            x=[x1], y=[y1], mode="markers",
            marker=dict(symbol="arrow", size=10, color=color,
                        angle=_arrow_angle(x0, y0, x1, y1),
                        line=dict(width=0)),
            hoverinfo="none",
        ))
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_annotation(x=mx, y=my + 0.18, text=label,
                showarrow=False,
                font=dict(size=9, color="#64748b", family="Inter, sans-serif"),
                bgcolor="white",
            )

    def _arrow_angle(x0, y0, x1, y1):
        import math
        dx, dy = x1 - x0, y1 - y0
        return -math.degrees(math.atan2(dy, dx)) + 90

    # Input → Family
    arrow(1.85, 3.5, 2.44, 3.5, "#94a3b8")

    # Family → Motor ID (up)
    arrow(4.15, 3.88, 5.14, 5.18, "#10b981", "motor")
    # Family → Charger State (straight)
    arrow(4.15, 3.5,  5.14, 3.5,  "#f97316", "charger")
    # Family → Charger ID (down)
    arrow(4.15, 3.12, 5.14, 1.82, "#f97316")

    # Motor ID → output
    arrow(6.85, 5.5, 8.04, 5.5, "#10b981")
    # Charger State → output
    arrow(6.85, 3.5, 8.04, 3.5, "#f59e0b")
    # Charger ID → output
    arrow(6.85, 1.5, 8.04, 1.5, "#f59e0b")

    # Column labels (top of chart)
    for x, lbl in [(1.1, "Input"), (3.3, "Stage 1"), (6.0, "Stage 2"), (8.8, "Output")]:
        fig.add_annotation(x=x, y=6.55, text=f"<b>{lbl}</b>",
            showarrow=False,
            font=dict(size=10, color="#64748b", family="Inter, sans-serif"),
            align="center",
        )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_results():
    hero(
        "Results & Methodology",
        "Benchmark numbers, trust controls, and publication-quality analysis figures.",
        accent="#7c3aed",
    )

    st.markdown("### Modeling architecture")
    _modeling_flow_chart()
    st.divider()
    st.markdown("### Classification benchmarks")
    results = pd.DataFrame([
        ["Motor vs Charger — all features",      "SVM", "100.0%", "100.0%",
         "Trivially separable when Hz-denominated features are included."],
        ["Motor vs Charger — rate-agnostic",     "SVM", "100.0%", "100.0%",
         "Proves separation is real, not a sample-rate shortcut."],
        ["Charger ON vs OFF",                    "RF",  "100.0%", "100.0%",
         "Charger state is crisply identifiable from EMI alone."],
        ["Motor ID — Motor 1 / 2 / 3",           "LR",  "100.0%", "100.0%",
         "Three same-model units are distinguishable by EMI fingerprint."],
        ["Charger ID — Charger 1 / 2",           "SVM", " 94.6%", " 97.4%",
         "Two same-make chargers: minor feature overlap causes occasional swaps."],
    ], columns=["Task", "Best model", "Test acc", "5-fold CV", "Interpretation"])
    st.dataframe(results, hide_index=True, use_container_width=True)

    st.markdown("""
    <p style="font-size:0.8rem; color:#64748b; margin-top:0.3rem;">
    Stratified 80/20 split after deduplication (27 exact feature-vector duplicates removed).
    Cross-validation is on training split only; test set is never seen during training.
    </p>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Why these results are trustworthy")

    trust_items = [
        ("Deduplication",
         "27 feature vectors were exact duplicates across files. Without removal, some "
         "duplicates straddled train/test splits — inflating accuracy. Removed before any "
         "split is created."),
        ("Rate-agnostic validation",
         "Motors sample 4× faster than chargers. A naive classifier could cheat by reading "
         "sample rate off Hz-denominated features. Task 1b excludes those features entirely "
         "and still achieves 100% — the separation is driven by real EMI content."),
        ("Amplitude vs shape split",
         "Amplitude features (RMS, peak-to-peak) are extracted from the raw signal before "
         "z-score normalisation. Extracting them post-normalisation would collapse RMS to "
         "~1.0 for every signal, destroying the most informative feature group."),
        ("Held-out integration test",
         "21 files not seen during training were run through the full pipeline end-to-end. "
         "20 pass. The one failure (Charger 1 misidentified as Charger 2) is consistent "
         "with the 94.6% test accuracy on that task."),
    ]
    for title, body in trust_items:
        card(f"""
        <div style="font-size:0.92rem; font-weight:600; color:#1e293b; margin-bottom:0.3rem;">
            {title}
        </div>
        <div style="font-size:0.87rem; color:#475569; line-height:1.55;">{body}</div>
        """)

    st.divider()
    st.markdown("### Phase 7 analysis figures")

    fig_meta = [
        ("Signal traces — time-domain view of all 5 devices",
         "fig_p7_01_signal_traces.png"),
        ("Spectral comparison — EMI fingerprints in the frequency domain",
         "fig_p7_02_spectral_comparison.png"),
        ("Feature narrative — 8 most discriminative features",
         "fig_p7_03_feature_narrative.png"),
        ("Separation in feature space — PCA of all 23 features",
         "fig_p7_04_separation_space.png"),
        ("Headline results at a glance",
         "fig_p7_05_results_summary.png"),
    ]
    for caption, fname in fig_meta:
        path = os.path.join(FIG_DIR, fname)
        if os.path.isfile(path):
            st.markdown(f"**{caption}**")
            st.image(path, use_container_width=True)
            st.write("")


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

if page == "Overview":
    render_overview()
elif page == "Live Demo":
    render_live_demo()
elif page == "Project Phases":
    render_phases()
elif page == "Dataset Explorer":
    render_dataset_explorer()
else:
    render_results()
