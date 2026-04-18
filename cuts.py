"""
cuts.py
-------
Defines the cut variables, their metadata, and functions to apply them.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

CutDirection = Literal[">", "<", "window"]


@dataclass
class CutDef:
    """Metadata for a single cut variable."""
    key: str                    # DataFrame column name
    label: str                  # Display label (supports LaTeX via mplhep)
    unit: str                   # Unit string, e.g. "GeV" or ""
    direction: CutDirection     # ">" min-cut, "<" max-cut, "window" for both
    default: float | tuple      # Default cut value(s)
    range: tuple                # (min, max) slider range
    step: float                 # Slider step
    description: str = ""       # Tooltip / explanation for students
    # For "window" cuts only:
    default_lo: float = 0.0
    default_hi: float = 0.0


# ── Cut definitions for WZ → 3ℓ ν ──────────────────────────────────────────

CUT_DEFS: list[CutDef] = [
    CutDef(
        key="pT1", label=r"$p_{T1}$", unit="GeV", direction=">",
        default=25.0, range=(0.0, 100.0), step=1.0,
        description="Minimum pT of the leading lepton (from Z or W decay).",
    ),
    CutDef(
        key="pT2", label=r"$p_{T2}$", unit="GeV", direction=">",
        default=20.0, range=(0.0, 80.0), step=1.0,
        description="Minimum pT of the sub-leading lepton.",
    ),
    CutDef(
        key="pT3", label=r"$p_{T3}$", unit="GeV", direction=">",
        default=15.0, range=(0.0, 60.0), step=1.0,
        description="Minimum pT of the third (softest) lepton.",
    ),
    CutDef(
        key="MET", label=r"$E_T^{miss}$", unit="GeV", direction=">",
        default=20.0, range=(0.0, 120.0), step=1.0,
        description="Missing transverse energy — proxy for the neutrino from W decay.",
    ),
    CutDef(
        key="mT_W", label=r"$m_T^W$", unit="GeV", direction=">",
        default=30.0, range=(0.0, 150.0), step=1.0,
        description="Transverse mass of the W candidate (lepton + MET). "
                    "Jacobian peak near m_W ≈ 80 GeV for real W bosons.",
    ),
    CutDef(
        key="m_Z", label=r"$m_Z$", unit="GeV", direction="window",
        default=0.0, range=(60.0, 120.0), step=0.5,
        default_lo=76.0, default_hi=106.0,
        description="Invariant mass of the SFOS lepton pair (Z candidate). "
                    "Select events inside the Z mass window.",
    ),
    CutDef(
        key="isolation", label=r"Isolation", unit="", direction="<",
        default=0.5, range=(0.0, 2.0), step=0.02,
        description="Track isolation of the leptons. "
                    "Small values → well-isolated, prompt leptons.",
    ),
    CutDef(
        key="d0_sig", label=r"$|d_0|$ significance", unit="", direction="<",
        default=5.0, range=(0.0, 10.0), step=0.1,
        description="Transverse impact parameter significance. "
                    "Small values → lepton originates from the primary vertex.",
    ),
]

# Keyed lookup for convenience
CUT_MAP: dict[str, CutDef] = {c.key: c for c in CUT_DEFS}

# Variables displayed as histogram tabs (same order as CUT_DEFS)
VARIABLES = [c.key for c in CUT_DEFS]


# ── Cut application ───────────────────────────────────────────────────────────

def apply_cuts(
    df: pd.DataFrame,
    cut_values: dict,          # {key: value} or {key: (lo, hi)} for windows
    exclude_key: str | None = None,  # omit one variable (for significance scan)
) -> pd.DataFrame:
    """
    Apply all cuts to df, optionally excluding one variable.
    Returns the filtered DataFrame (preserves index).
    """
    mask = pd.Series(True, index=df.index)

    for c in CUT_DEFS:
        if c.key == exclude_key:
            continue

        val = cut_values.get(c.key)
        if val is None:
            continue

        if c.direction == ">":
            mask &= df[c.key] >= val
        elif c.direction == "<":
            mask &= df[c.key] <= val
        elif c.direction == "window":
            lo, hi = val
            mask &= (df[c.key] >= lo) & (df[c.key] <= hi)

    return df[mask]


def default_cut_values() -> dict:
    """Return the default cut value dict (used for reset)."""
    vals = {}
    for c in CUT_DEFS:
        if c.direction == "window":
            vals[c.key] = (c.default_lo, c.default_hi)
        else:
            vals[c.key] = c.default
    return vals
