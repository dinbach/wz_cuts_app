"""
app.py  —  WZ → 3ℓ ν  Cut Optimisation Tool
============================================
Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader  import load_data
from utils.cuts         import CUT_DEFS, CUT_MAP, VARIABLES, apply_cuts, default_cut_values
from utils.significance import (
    compute_yields, scan_cut, compute_roc, SIGNIFICANCE_FORMULAE
)
from utils.plotting     import (
    plot_variable_panel, plot_roc, plot_mZ_summary
)

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WZ → 3ℓν  Cut Optimisation",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Compact sidebar labels */
    .stSlider label, .stNumberInput label { font-size: 0.82rem !important; }
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 8px 14px;
    }
    /* Tab font */
    button[data-baseweb="tab"] { font-size: 0.80rem; }
    /* Slightly tighter main padding */
    .main .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state: persistent cut values across reruns ────────────────────────

if "cut_values" not in st.session_state:
    st.session_state.cut_values = default_cut_values()

if "n_bins" not in st.session_state:
    st.session_state.n_bins = 50

# ── Load data ─────────────────────────────────────────────────────────────────

sig_raw, bkg_raw = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚛️ WZ → 3ℓ ν")
    st.caption("Cut Optimisation Tool — CERN Open Data")
    st.divider()

    # ── Significance formula ─────────────────────────────────────────────
    sig_formula_label = st.selectbox(
        "Significance metric",
        options=list(SIGNIFICANCE_FORMULAE.keys()),
        index=0,
        help="Formula used to compute S/B significance in the scan plots.",
    )
    significance_fn = SIGNIFICANCE_FORMULAE[sig_formula_label]

    st.divider()
    st.subheader("Cuts")

    cv = st.session_state.cut_values   # shorthand

    # ── Lepton pT cuts ───────────────────────────────────────────────────
    with st.expander("Lepton $p_T$ cuts", expanded=True):
        for key in ("pT1", "pT2", "pT3"):
            c = CUT_MAP[key]
            cv[key] = st.slider(
                f"{c.label} > [GeV]",
                min_value=float(c.range[0]),
                max_value=float(c.range[1]),
                value=float(cv[key]),
                step=c.step,
                help=c.description,
            )

    # ── MET & mT_W ───────────────────────────────────────────────────────
    with st.expander("MET & $m_T^W$", expanded=True):
        for key in ("MET", "mT_W"):
            c = CUT_MAP[key]
            cv[key] = st.slider(
                f"{c.label} > [GeV]",
                min_value=float(c.range[0]),
                max_value=float(c.range[1]),
                value=float(cv[key]),
                step=c.step,
                help=c.description,
            )

    # ── Z mass window ────────────────────────────────────────────────────
    with st.expander("Z mass window", expanded=True):
        c = CUT_MAP["m_Z"]
        lo, hi = cv["m_Z"]
        new_lo, new_hi = st.slider(
            r"$m_Z$ window [GeV]",
            min_value=float(c.range[0]),
            max_value=float(c.range[1]),
            value=(float(lo), float(hi)),
            step=c.step,
            help=c.description,
        )
        cv["m_Z"] = (new_lo, new_hi)

    # ── Lepton quality ───────────────────────────────────────────────────
    with st.expander("Lepton quality", expanded=False):
        for key in ("isolation", "d0_sig"):
            c = CUT_MAP[key]
            label = f"{c.label} <"
            cv[key] = st.slider(
                label,
                min_value=float(c.range[0]),
                max_value=float(c.range[1]),
                value=float(cv[key]),
                step=c.step,
                help=c.description,
            )

    st.divider()

    # ── Display options ──────────────────────────────────────────────────
    st.subheader("Display")
    log_y     = st.checkbox("Log Y axis",  value=False)
    show_roc  = st.checkbox("Show ROC curves", value=True)
    n_bins    = st.slider("Histogram bins", 20, 120, st.session_state.n_bins, step=5)
    st.session_state.n_bins = n_bins

    st.divider()

    # ── Reset button ─────────────────────────────────────────────────────
    if st.button("↺  Reset to defaults", use_container_width=True):
        st.session_state.cut_values = default_cut_values()
        st.rerun()

# ── Apply all cuts ────────────────────────────────────────────────────────────

sig_cut = apply_cuts(sig_raw, cv)
bkg_cut = apply_cuts(bkg_raw, cv)

S_total, B_total = compute_yields(sig_cut, bkg_cut)
S_raw,   B_raw   = compute_yields(sig_raw, bkg_raw)

sig_eff  = S_total / (S_raw + 1e-9)
bkg_pass = B_total / (B_raw + 1e-9)   # fraction of bkg that survives
bkg_rej  = 1.0 - bkg_pass

significance = significance_fn(S_total, B_total)

# ── Header & summary metrics ──────────────────────────────────────────────────

st.title("WZ → 3ℓ ν  —  Signal vs Background Cut Optimisation")
st.caption("Adjust the cuts in the sidebar. Plots and metrics update in real time.")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Signal events",       f"{S_total:,.0f}",
            delta=f"{sig_eff*100:.1f}% of total",
            delta_color="normal")
col2.metric("Background events",   f"{B_total:,.0f}",
            delta=f"{bkg_pass*100:.1f}% of total",
            delta_color="inverse")
col3.metric("Signal efficiency",   f"{sig_eff*100:.1f} %")
col4.metric("Background rejection", f"{bkg_rej*100:.1f} %")
col5.metric(sig_formula_label,     f"{significance:.2f}")

st.divider()

# ── Main panels ───────────────────────────────────────────────────────────────

# Top row: variable tabs | m_Z summary
left_col, right_col = st.columns([3, 1], gap="large")

with left_col:
    # ── Variable explorer tabs ────────────────────────────────────────────
    tab_labels = [CUT_MAP[k].label for k in VARIABLES]
    tabs = st.tabs(tab_labels)

    for tab, key in zip(tabs, VARIABLES):
        with tab:
            c = CUT_MAP[key]

            # Compute significance scan for this variable (other cuts fixed)
            scan_x, scan_y = scan_cut(
                sig_raw, bkg_raw, c, cv, significance_fn, n_points=80
            )

            # After-cut data for histograms
            sig_col = sig_cut[key].values
            bkg_col = bkg_cut[key].values
            sig_w   = sig_cut["weight"].values
            bkg_w   = bkg_cut["weight"].values

            fig = plot_variable_panel(
                sig_col, bkg_col, sig_w, bkg_w,
                c, scan_x, scan_y,
                current_cut=cv[key],
                significance_label=sig_formula_label,
                n_bins=n_bins,
                log_y=log_y,
            )
            st.pyplot(fig, use_container_width=True)

            # Best cut hint
            if len(scan_y) > 0 and c.direction in (">", "<"):
                best_x = scan_x[np.argmax(scan_y)]
                best_z = scan_y.max()
                st.info(
                    f"🎯 Best cut for **{c.label}**: "
                    f"**{best_x:.2f} {c.unit}**  →  "
                    f"{sig_formula_label} = **{best_z:.2f}**",
                    icon=None,
                )

with right_col:
    # ── m_Z summary plot (always visible) ────────────────────────────────
    st.subheader("Z mass window")
    mZ_fig = plot_mZ_summary(
        sig_cut["m_Z"].values,
        bkg_cut["m_Z"].values,
        sig_cut["weight"].values,
        bkg_cut["weight"].values,
        window=cv["m_Z"],
        n_bins=50,
    )
    st.pyplot(mZ_fig, use_container_width=True)

    st.caption(
        "Orange shaded region shows the current Z mass window. "
        "The dashed grey line marks $m_Z = 91.2$ GeV."
    )

    # ── Per-cut summary table ─────────────────────────────────────────────
    st.subheader("Cut summary")
    rows = []
    for c in CUT_DEFS:
        val = cv[c.key]
        if c.direction == "window":
            lo, hi = val
            cut_str = f"{lo:.1f} – {hi:.1f} {c.unit}"
        else:
            cut_str = f"{c.direction} {val:.2f} {c.unit}"
        rows.append({"Variable": c.label, "Cut": cut_str})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── ROC curves ────────────────────────────────────────────────────────────────

if show_roc:
    st.divider()
    st.subheader("ROC curves — per variable")
    st.caption(
        "Each curve shows signal efficiency vs. background rejection "
        "as a single variable's cut is varied (all other cuts held fixed)."
    )

    roc_vars  = [k for k in VARIABLES if CUT_MAP[k].direction in (">", "<")]
    effs, rejs, lbls = [], [], []
    for key in roc_vars:
        se, br = compute_roc(sig_raw, bkg_raw, CUT_MAP[key], cv, n_points=100)
        effs.append(se); rejs.append(br); lbls.append(CUT_MAP[key].label)

    roc_fig = plot_roc(effs, rejs, lbls)
    # Centre the ROC plot (don't stretch it to full width)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.pyplot(roc_fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: CERN Open Data (synthetic in demo mode) · "
    "WZ → W(ℓν) Z(ℓℓ) signal · "
    "Built with [Streamlit](https://streamlit.io) · "
    "Physics: ATLAS Open Data framework"
)
