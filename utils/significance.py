"""
significance.py
---------------
Signal-over-background significance metrics and per-variable scan.
"""

import numpy as np
import pandas as pd
from utils.cuts import CutDef, apply_cuts, CUT_DEFS

EPSILON = 1e-9   # avoid division by zero


def s_over_sqrtb(S: float, B: float) -> float:
    return S / np.sqrt(B + EPSILON)


def s_over_sqrt_splusb(S: float, B: float) -> float:
    return S / np.sqrt(S + B + EPSILON)


def asimov_z(S: float, B: float) -> float:
    """
    Approximate Asimov significance (Cowan et al. 2011, Eq. 97 simplified).
    Z_A ≈ sqrt(2 * ((S+B)*ln(1 + S/B) - S))
    """
    if B < EPSILON:
        return 0.0
    ratio = S / (B + EPSILON)
    return np.sqrt(max(0.0, 2.0 * ((S + B) * np.log1p(ratio) - S)))


SIGNIFICANCE_FORMULAE = {
    r"$S / \sqrt{B}$":             s_over_sqrtb,
    r"$S / \sqrt{S+B}$":           s_over_sqrt_splusb,
    r"Asimov $Z_A$":               asimov_z,
}


def compute_yields(
    sig: pd.DataFrame,
    bkg: pd.DataFrame,
) -> tuple[float, float]:
    """Weighted event yields."""
    S = float(sig["weight"].sum())
    B = float(bkg["weight"].sum())
    return S, B


def scan_cut(
    sig: pd.DataFrame,
    bkg: pd.DataFrame,
    cut_def: CutDef,
    cut_values: dict,
    significance_fn,
    n_points: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan the cut value for `cut_def`, all other cuts fixed.
    Returns (scan_x, significance_y).

    For ">" and "<" cuts: scan the cut threshold.
    For "window" cuts: scan the half-width symmetrically around m_Z = 91.2 GeV.
    """
    # Apply all OTHER cuts first
    sig_base = apply_cuts(sig, cut_values, exclude_key=cut_def.key)
    bkg_base = apply_cuts(bkg, cut_values, exclude_key=cut_def.key)

    lo, hi = cut_def.range
    xs = np.linspace(lo, hi, n_points)

    significances = []
    for x in xs:
        if cut_def.direction == ">":
            s_cut = sig_base[sig_base[cut_def.key] >= x]
            b_cut = bkg_base[bkg_base[cut_def.key] >= x]
        elif cut_def.direction == "<":
            s_cut = sig_base[sig_base[cut_def.key] <= x]
            b_cut = bkg_base[bkg_base[cut_def.key] <= x]
        elif cut_def.direction == "window":
            # x = half-width around 91.2 GeV
            center = 91.2
            half = max(x - lo, 0.5)  # half-width grows as x moves from lo→hi
            hw = (hi - lo) / 2.0 * (x - lo) / (hi - lo + EPSILON)
            hw = max(hw, 0.5)
            s_cut = sig_base[
                (sig_base[cut_def.key] >= center - hw) &
                (sig_base[cut_def.key] <= center + hw)
            ]
            b_cut = bkg_base[
                (bkg_base[cut_def.key] >= center - hw) &
                (bkg_base[cut_def.key] <= center + hw)
            ]

        S, B = compute_yields(s_cut, b_cut)
        significances.append(significance_fn(S, B))

    return xs, np.array(significances)


def compute_roc(
    sig: pd.DataFrame,
    bkg: pd.DataFrame,
    cut_def: CutDef,
    cut_values: dict,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ROC curve for a single variable:
    signal efficiency vs. background rejection as the cut is varied.
    Returns (sig_eff, bkg_rejection).
    """
    sig_base = apply_cuts(sig, cut_values, exclude_key=cut_def.key)
    bkg_base = apply_cuts(bkg, cut_values, exclude_key=cut_def.key)

    S_total = sig_base["weight"].sum() + EPSILON
    B_total = bkg_base["weight"].sum() + EPSILON

    lo, hi = cut_def.range
    xs = np.linspace(lo, hi, n_points)
    sig_effs, bkg_rejs = [], []

    for x in xs:
        if cut_def.direction == ">":
            s = sig_base[sig_base[cut_def.key] >= x]["weight"].sum()
            b = bkg_base[bkg_base[cut_def.key] >= x]["weight"].sum()
        elif cut_def.direction == "<":
            s = sig_base[sig_base[cut_def.key] <= x]["weight"].sum()
            b = bkg_base[bkg_base[cut_def.key] <= x]["weight"].sum()
        else:
            continue

        sig_effs.append(s / S_total)
        bkg_rejs.append(1.0 - b / B_total)

    return np.array(sig_effs), np.array(bkg_rejs)
