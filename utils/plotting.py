"""
plotting.py
-----------
All matplotlib / mplhep plot functions used by the Streamlit app.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    import mplhep as hep
    hep.style.use("ATLAS")
except ImportError:
    pass   # graceful degradation without mplhep

from utils.cuts import CutDef

# ── Colour palette ─────────────────────────────────────────────────────────

SIG_COLOR  = "#1f77b4"    # blue
BKG_COLOR  = "#d62728"    # red
SCAN_COLOR = "#2ca02c"    # green
DATA_COLOR = "black"
GRID_ALPHA = 0.25


# ── Helper ──────────────────────────────────────────────────────────────────

def _stats_title(label: str, data: np.ndarray, weights: np.ndarray | None = None) -> str:
    if weights is None:
        weights = np.ones_like(data)
    n     = weights.sum()
    mean  = np.average(data, weights=weights)
    rms   = np.sqrt(np.average((data - mean) ** 2, weights=weights))
    return f"{label}   Entries: {n:.0f}   Mean: {mean:.2f}   RMS: {rms:.2f}"


def _make_bins(cut_def: CutDef, n_bins: int) -> np.ndarray:
    return np.linspace(cut_def.range[0], cut_def.range[1], n_bins + 1)


# ── Main variable panel: signal + background + significance scan ─────────────

def plot_variable_panel(
    sig_col: np.ndarray,
    bkg_col: np.ndarray,
    sig_w: np.ndarray,
    bkg_w: np.ndarray,
    cut_def: CutDef,
    scan_x: np.ndarray,
    scan_y: np.ndarray,
    current_cut,
    significance_label: str,
    n_bins: int = 50,
    log_y: bool = False,
) -> plt.Figure:
    """
    Three-panel figure for one variable:
      top    – signal histogram
      middle – background histogram
      bottom – significance scan vs cut value
    """
    bins = _make_bins(cut_def, n_bins)

    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)
    ax_sig  = fig.add_subplot(gs[0])
    ax_bkg  = fig.add_subplot(gs[1])
    ax_scan = fig.add_subplot(gs[2])

    # ── Signal histogram ─────────────────────────────────────────────────
    ax_sig.hist(sig_col, bins=bins, weights=sig_w, color=SIG_COLOR, alpha=0.75)
    ax_sig.set_title(_stats_title(f"{cut_def.label} Signal", sig_col, sig_w),
                     fontsize=9, pad=4)
    ax_sig.set_xlabel(f"{cut_def.label} {f'[{cut_def.unit}]' if cut_def.unit else ''}",
                      fontsize=9)
    ax_sig.set_ylabel("Events", fontsize=9)
    ax_sig.set_xlim(cut_def.range)
    if log_y:
        ax_sig.set_yscale("log")
    ax_sig.grid(True, alpha=GRID_ALPHA)
    _draw_cut_line(ax_sig, cut_def, current_cut)

    # ── Background histogram ─────────────────────────────────────────────
    ax_bkg.hist(bkg_col, bins=bins, weights=bkg_w, color=BKG_COLOR, alpha=0.75)
    ax_bkg.set_title(_stats_title(f"{cut_def.label} Background", bkg_col, bkg_w),
                     fontsize=9, pad=4)
    ax_bkg.set_xlabel(f"{cut_def.label} {f'[{cut_def.unit}]' if cut_def.unit else ''}",
                      fontsize=9)
    ax_bkg.set_ylabel("Events", fontsize=9)
    ax_bkg.set_xlim(cut_def.range)
    if log_y:
        ax_bkg.set_yscale("log")
    ax_bkg.grid(True, alpha=GRID_ALPHA)
    _draw_cut_line(ax_bkg, cut_def, current_cut)

    # ── Significance scan ────────────────────────────────────────────────
    ax_scan.plot(scan_x, scan_y, color=SCAN_COLOR, lw=2)
    ax_scan.set_title(f"{cut_def.label} — {significance_label}", fontsize=9, pad=4)
    ax_scan.set_xlabel(f"Cut value on {cut_def.label}"
                       f"{f' [{cut_def.unit}]' if cut_def.unit else ''}",
                       fontsize=9)
    ax_scan.set_ylabel(significance_label, fontsize=9)
    ax_scan.set_xlim(cut_def.range)
    ax_scan.grid(True, alpha=GRID_ALPHA)

    # Mark current cut on scan
    if cut_def.direction in (">", "<") and current_cut is not None:
        ax_scan.axvline(current_cut, color="orange", lw=1.5,
                        linestyle="--", label=f"current = {current_cut:.2f}")
        ax_scan.legend(fontsize=8)

    # Highlight optimal cut
    if len(scan_y) > 0:
        best_idx = np.argmax(scan_y)
        ax_scan.axvline(scan_x[best_idx], color="magenta", lw=1.2,
                        linestyle=":", label=f"best = {scan_x[best_idx]:.2f}")
        ax_scan.legend(fontsize=8)

    return fig


def _draw_cut_line(ax, cut_def: CutDef, current_cut):
    """Draw a vertical line at the current cut value on a histogram."""
    if cut_def.direction == ">" and current_cut is not None:
        ax.axvline(current_cut, color="orange", lw=1.5, linestyle="--", alpha=0.85)
    elif cut_def.direction == "<" and current_cut is not None:
        ax.axvline(current_cut, color="orange", lw=1.5, linestyle="--", alpha=0.85)
    elif cut_def.direction == "window" and current_cut is not None:
        lo, hi = current_cut
        ax.axvline(lo, color="orange", lw=1.5, linestyle="--", alpha=0.85)
        ax.axvline(hi, color="orange", lw=1.5, linestyle="--", alpha=0.85)
        ax.axvspan(lo, hi, color="orange", alpha=0.08)


# ── ROC curve ──────────────────────────────────────────────────────────────

def plot_roc(
    sig_effs_list: list[np.ndarray],
    bkg_rejs_list: list[np.ndarray],
    labels: list[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.tab10.colors
    for i, (se, br, lbl) in enumerate(zip(sig_effs_list, bkg_rejs_list, labels)):
        ax.plot(se, br, lw=1.8, color=colors[i % 10], label=lbl)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Signal Efficiency", fontsize=10)
    ax.set_ylabel("Background Rejection", fontsize=10)
    ax.set_title("ROC Curves (per variable)", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig


# ── m_Z invariant mass summary plot ────────────────────────────────────────

def plot_mZ_summary(
    sig_mZ: np.ndarray,
    bkg_mZ: np.ndarray,
    sig_w: np.ndarray,
    bkg_w: np.ndarray,
    window: tuple[float, float],
    n_bins: int = 60,
) -> plt.Figure:
    bins = np.linspace(60, 120, n_bins + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bkg_mZ, bins=bins, weights=bkg_w,
            color=BKG_COLOR, alpha=0.55, label="Background")
    ax.hist(sig_mZ, bins=bins, weights=sig_w,
            color=SIG_COLOR, alpha=0.65, label="WZ Signal")
    ax.axvspan(window[0], window[1], color="orange", alpha=0.12, label="Z window")
    ax.axvline(window[0], color="orange", lw=1.5, linestyle="--")
    ax.axvline(window[1], color="orange", lw=1.5, linestyle="--")
    ax.axvline(91.2, color="gray", lw=1.0, linestyle=":", alpha=0.7)
    ax.set_xlabel(r"$m_Z$ [GeV]", fontsize=11)
    ax.set_ylabel("Events", fontsize=11)
    ax.set_title(r"$m_Z$ — signal vs. background (all other cuts applied)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    return fig
