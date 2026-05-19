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
    compute_yields, scan_cut, compute_roc,
    SIGNIFICANCE_FORMULAE, SIGNIFICANCE_PLOT_LABELS,
)
from utils.plotting     import plot_variable_panel, plot_roc

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
    .stNumberInput label { font-size: 0.82rem !important; }
    div[data-testid="stMetric"] {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 8px 14px;
    }
    button[data-baseweb="tab"] { font-size: 0.80rem; }
    .main .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Initialise session-state keys for every cut ───────────────────────────────
#
# Each number_input is given key="cut_<varname>" (or "cut_<varname>_lo/hi" for
# windows). Streamlit owns the widget state under these keys, which means:
#   • The value is kept correctly across reruns.
#   • The +/− buttons work on a single click.
#   • We never pass value= from session_state back into the widget (that is what
#     causes the "double-click" problem).
#
_DEFAULTS = default_cut_values()

def _init_cut_state():
    for k, v in _DEFAULTS.items():
        if isinstance(v, tuple):
            lo, hi = v
            st.session_state.setdefault(f"cut_{k}_lo", lo)
            st.session_state.setdefault(f"cut_{k}_hi", hi)
        else:
            st.session_state.setdefault(f"cut_{k}", v)
    st.session_state.setdefault("n_bins", 50)

_init_cut_state()

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
        help="Formula used to compute significance in the scan plots.",
    )
    significance_fn       = SIGNIFICANCE_FORMULAE[sig_formula_label]
    significance_plot_lbl = SIGNIFICANCE_PLOT_LABELS[sig_formula_label]

    st.divider()
    st.subheader("Cuts")

    # ── Lepton pT cuts ───────────────────────────────────────────────────
    with st.expander("Lepton pT cuts", expanded=True):
        for k in ("pT1", "pT2", "pT3"):
            c = CUT_MAP[k]
            st.number_input(
                f"{c.plain_label} > (GeV)",
                min_value=float(c.range[0]),
                max_value=float(c.range[1]),
                step=1.0,
                format="%.0f",
                key=f"cut_{k}",
                help=c.description,
            )

    # ── MET & mT(W) ──────────────────────────────────────────────────────
    with st.expander("MET & mT(W)", expanded=True):
        for k in ("MET", "mT_W"):
            c = CUT_MAP[k]
            st.number_input(
                f"{c.plain_label} > (GeV)",
                min_value=float(c.range[0]),
                max_value=float(c.range[1]),
                step=1.0,
                format="%.0f",
                key=f"cut_{k}",
                help=c.description,
            )

    # ── Z mass window ────────────────────────────────────────────────────
    with st.expander("Z mass window", expanded=True):
        c = CUT_MAP["m_Z"]
        col_lo, col_hi = st.columns(2)
        with col_lo:
            st.number_input(
                "m(Z) low (GeV)",
                min_value=0.0,
                max_value=199.0,
                step=1.0,
                format="%.0f",
                key="cut_m_Z_lo",
                help="Lower edge of the Z mass window (0–200 GeV).",
            )
        with col_hi:
            st.number_input(
                "m(Z) high (GeV)",
                min_value=1.0,
                max_value=200.0,
                step=1.0,
                format="%.0f",
                key="cut_m_Z_hi",
                help="Upper edge of the Z mass window (0–200 GeV).",
            )

    # ── Lepton quality ───────────────────────────────────────────────────
    with st.expander("Lepton quality", expanded=False):
        c = CUT_MAP["isolation"]
        st.number_input(
            f"{c.plain_label} <",
            min_value=float(c.range[0]),
            max_value=float(c.range[1]),
            step=0.05,
            format="%.2f",
            key="cut_isolation",
            help=c.description,
        )
        c = CUT_MAP["d0_sig"]
        st.number_input(
            f"{c.plain_label} <",
            min_value=float(c.range[0]),
            max_value=float(c.range[1]),
            step=0.2,
            format="%.1f",
            key="cut_d0_sig",
            help=c.description,
        )

    st.divider()

    # ── Display options ──────────────────────────────────────────────────
    st.subheader("Display")
    log_y      = st.checkbox("Log Y axis", value=False)
    show_roc   = st.checkbox("Show ROC curves", value=False)
    show_table = st.checkbox("Show data table", value=False)
    st.number_input(
        "Histogram bins",
        min_value=20, max_value=100,
        step=5,
        format="%d",
        key="n_bins",
    )

    st.divider()

    # ── Reset button ─────────────────────────────────────────────────────
    if st.button("↺  Reset to defaults", use_container_width=True):
        # Delete all managed keys so _init_cut_state() repopulates from defaults.
        # We must DELETE rather than SET keys that are bound to active widgets —
        # Streamlit raises StreamlitAPIException if you write to a widget key
        # after the widget has already been instantiated in this run.
        for sk in list(st.session_state.keys()):
            if sk.startswith("cut_") or sk == "n_bins":
                del st.session_state[sk]
        st.rerun()

# ── Read current cut values from session state ────────────────────────────────

cv = {}
for k, v in _DEFAULTS.items():
    if isinstance(v, tuple):
        cv[k] = (
            float(st.session_state[f"cut_{k}_lo"]),
            float(st.session_state[f"cut_{k}_hi"]),
        )
    else:
        cv[k] = float(st.session_state[f"cut_{k}"])

n_bins = int(st.session_state["n_bins"])

# ── Apply all cuts ────────────────────────────────────────────────────────────

sig_cut = apply_cuts(sig_raw, cv)
bkg_cut = apply_cuts(bkg_raw, cv)

S_total, B_total = compute_yields(sig_cut, bkg_cut)
S_raw,   B_raw   = compute_yields(sig_raw, bkg_raw)

sig_eff  = S_total / (S_raw + 1e-9)
bkg_pass = B_total / (B_raw + 1e-9)
bkg_rej  = 1.0 - bkg_pass

significance = significance_fn(S_total, B_total)

# ── Header & summary metrics ──────────────────────────────────────────────────

st.title("WZ → 3ℓ ν  —  Signal vs Background Cut Optimisation")
st.caption("Adjust the cuts in the sidebar. Plots and metrics update in real time.")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Signal events",        f"{S_total:,.0f} / {S_raw:,.0f}",
            delta=f"{sig_eff*100:.1f}% efficiency", delta_color="normal")
col2.metric("Background events",    f"{B_total:,.0f} / {B_raw:,.0f}",
            delta=f"{bkg_pass*100:.1f}% pass rate", delta_color="inverse")
col3.metric("Signal efficiency",    f"{sig_eff*100:.1f} %")
col4.metric("Background rejection", f"{bkg_rej*100:.1f} %")
col5.metric(sig_formula_label,      f"{significance:.2f}")

st.divider()

# ── Variable explorer (full width) ───────────────────────────────────────────

tab_labels = [CUT_MAP[k].plain_label for k in VARIABLES]
tabs = st.tabs(tab_labels)

for tab, key in zip(tabs, VARIABLES):
    with tab:
        c = CUT_MAP[key]

        scan_x, scan_y = scan_cut(
            sig_raw, bkg_raw, c, cv, significance_fn, n_points=80
        )

        sig_col = sig_cut[key].values
        bkg_col = bkg_cut[key].values
        sig_w   = sig_cut["weight"].values
        bkg_w   = bkg_cut["weight"].values

        fig = plot_variable_panel(
            sig_col, bkg_col, sig_w, bkg_w,
            c, scan_x, scan_y,
            current_cut=cv[key],
            significance_label=significance_plot_lbl,
            n_bins=n_bins,
            log_y=log_y,
        )
        st.pyplot(fig, use_container_width=True)

# ── Cut summary table ─────────────────────────────────────────────────────────

st.divider()
st.subheader("Current cut summary")
rows = []
for c in CUT_DEFS:
    val = cv[c.key]
    if c.direction == "window":
        lo, hi = val
        cut_str = f"{lo:.0f} – {hi:.0f} GeV"
    elif c.unit == "GeV":
        cut_str = f"{c.direction} {val:.0f} GeV"
    else:
        cut_str = f"{c.direction} {val:.2f}"
    rows.append({"Variable": c.plain_label, "Cut": cut_str})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Data table ────────────────────────────────────────────────────────────────

if show_table:
    st.divider()
    st.subheader("Data table")

    display_cols = ["pT1", "pT2", "pT3", "MET", "mT_W", "m_Z", "isolation", "d0_sig"]
    col_rename   = {c.key: c.plain_label for c in CUT_DEFS}

    MAX_ROWS = 500

    tab_sig, tab_bkg = st.tabs(
        [f"Signal  ({len(sig_cut)} events)", f"Background  ({len(bkg_cut)} events)"]
    )

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df[display_cols].rename(columns=col_rename)
        return out.reset_index(drop=True)

    with tab_sig:
        shown = min(len(sig_cut), MAX_ROWS)
        if len(sig_cut) > MAX_ROWS:
            st.caption(f"Showing first {MAX_ROWS} of {len(sig_cut)} signal events.")
        st.dataframe(
            _fmt(sig_cut.head(MAX_ROWS)).style.format("{:.2f}"),
            use_container_width=True,
        )

    with tab_bkg:
        shown = min(len(bkg_cut), MAX_ROWS)
        if len(bkg_cut) > MAX_ROWS:
            st.caption(f"Showing first {MAX_ROWS} of {len(bkg_cut)} background events.")
        st.dataframe(
            _fmt(bkg_cut.head(MAX_ROWS)).style.format("{:.2f}"),
            use_container_width=True,
        )

# ── ROC curves ────────────────────────────────────────────────────────────────

if show_roc:
    st.divider()
    st.subheader("ROC curves — per variable")
    st.caption(
        "Each curve shows signal efficiency vs. background rejection "
        "as a single variable's cut is varied (all other cuts held fixed)."
    )

    roc_vars = [k for k in VARIABLES if CUT_MAP[k].direction in (">", "<")]
    effs, rejs, lbls = [], [], []
    for key in roc_vars:
        se, br = compute_roc(sig_raw, bkg_raw, CUT_MAP[key], cv, n_points=100)
        effs.append(se); rejs.append(br); lbls.append(CUT_MAP[key].plain_label)

    roc_fig = plot_roc(effs, rejs, lbls)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.pyplot(roc_fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: CERN Open Data · "
    "WZ → W(ℓν) Z(ℓℓ) signal · "
    "Background: Z+jets, Diboson, tt̄ · "
    "Built with [Streamlit](https://streamlit.io) · "
    "Physics: ATLAS Open Data framework"
)
