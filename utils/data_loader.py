"""
data_loader.py
--------------
Loads real CERN Open Data CSV files for the WZ → 3ℓ ν analysis.

Files expected in data/:
  Signal.csv   — WZ → W(ℓν) Z(ℓℓ) signal
  Zjets.csv    — Z+jets background
  ttbar.csv    — tt̄ background
  Diboson.csv  — other diboson background (ZZ, WW, …)

Each CSV has per-lepton kinematics. This module derives the physics
variables needed by the app:

  pT1, pT2, pT3     — leading/sub/third lepton pT [GeV]
  eta1, eta2, eta3  — pseudorapidities
  m_Z               — invariant mass of SFOS pair closest to 91.2 GeV [GeV]
  mT_W              — transverse mass proxy for W candidate [GeV]
  MET               — missing transverse energy [GeV]
  isolation         — max relative track isolation across 3 leptons
  d0_sig            — max |d0 significance| across 3 leptons
  weight            — event weight (1.0 for all events in this exercise)
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Isolation is clipped at this value to keep slider ranges sensible
ISO_CLIP = 5.0

# ── Physics helpers ───────────────────────────────────────────────────────────

def _invariant_mass(E1, px1, py1, pz1, E2, px2, py2, pz2) -> np.ndarray:
    """4-vector invariant mass of a pair. Returns 0 if m² < 0 (rounding)."""
    m2 = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
    return np.sqrt(np.maximum(0.0, m2))


def _process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a raw ATLAS-style CSV DataFrame into the variable set
    expected by the app.
    """
    n = len(df)

    # ── Per-lepton 4-vectors ──────────────────────────────────────────────
    lep = {}
    for i in [1, 2, 3]:
        pt  = df[f"lep{i}_pt"].values.astype(float)
        eta = df[f"lep{i}_eta"].values.astype(float)
        phi = df[f"lep{i}_phi"].values.astype(float)
        lep[i] = dict(
            pt      = pt,
            eta     = eta,
            phi     = phi,
            e       = df[f"lep{i}_e"].values.astype(float),
            px      = pt * np.cos(phi),
            py      = pt * np.sin(phi),
            pz      = pt * np.sinh(eta),
            charge  = df[f"lep{i}_charge"].values.astype(int),
            d0sig   = df[f"lep{i}_d0sig"].values.astype(float),
            ptvarcone30 = df[f"lep{i}_ptvarcone30"].values.astype(float),
        )

    # ── Z candidate: SFOS pair with invariant mass closest to 91.2 GeV ───
    #
    # With 3 leptons there are 3 possible pairs. We evaluate all pairs,
    # set the mass of same-sign pairs to 9999 (unphysical sentinel) and
    # pick the one whose mass is closest to m_Z = 91.2 GeV.
    #
    # pair index → which lepton is the W candidate
    pairs       = [(1, 2), (1, 3), (2, 3)]
    W_for_pair  = {0: 3, 1: 2, 2: 1}   # (1,2)→W=3, (1,3)→W=2, (2,3)→W=1

    pair_masses = []
    for (i, j) in pairs:
        is_os = (lep[i]["charge"] * lep[j]["charge"] < 0)
        m = _invariant_mass(
            lep[i]["e"],  lep[i]["px"], lep[i]["py"], lep[i]["pz"],
            lep[j]["e"],  lep[j]["px"], lep[j]["py"], lep[j]["pz"],
        )
        pair_masses.append(np.where(is_os, m, 9999.0))

    mass_arr  = np.stack(pair_masses, axis=1)            # (n, 3)
    best_pair = np.argmin(np.abs(mass_arr - 91.2), axis=1)  # index 0/1/2
    m_Z       = mass_arr[np.arange(n), best_pair]

    # ── W lepton ──────────────────────────────────────────────────────────
    W_idx  = np.array([W_for_pair[b] for b in best_pair])  # lepton number 1/2/3
    pt_arr  = np.stack([lep[i]["pt"]  for i in [1, 2, 3]], axis=1)  # (n, 3)
    phi_arr = np.stack([lep[i]["phi"] for i in [1, 2, 3]], axis=1)  # (n, 3)
    pT_W   = pt_arr [np.arange(n), W_idx - 1]
    phi_W  = phi_arr[np.arange(n), W_idx - 1]

    # ── mT_W using full formula with MET φ ───────────────────────────────
    # mT = sqrt(2 · pT_lep · MET · (1 − cos Δφ))
    # where Δφ = φ(W lepton) − φ(MET), wrapped to [−π, π].
    MET     = df["met"].values.astype(float)
    met_phi = df["met_phi"].values.astype(float)
    dphi    = phi_W - met_phi
    dphi    = (dphi + np.pi) % (2.0 * np.pi) - np.pi          # wrap to [−π, π]
    mT_W    = np.sqrt(np.maximum(0.0, 2.0 * pT_W * MET * (1.0 - np.cos(dphi))))

    # ── Relative track isolation  max over 3 leptons ─────────────────────
    # iso_i = |ptvarcone30_i| / pT_i  (0 = perfectly isolated)
    iso_vals = []
    for i in [1, 2, 3]:
        cone = np.abs(lep[i]["ptvarcone30"])
        iso_vals.append(cone / (lep[i]["pt"] + 1e-9))
    isolation = np.maximum(np.maximum(iso_vals[0], iso_vals[1]), iso_vals[2])
    isolation = np.clip(isolation, 0.0, ISO_CLIP)

    # ── |d0| significance  max over 3 leptons ────────────────────────────
    d0_sig = np.maximum(
        np.maximum(np.abs(lep[1]["d0sig"]), np.abs(lep[2]["d0sig"])),
        np.abs(lep[3]["d0sig"]),
    )

    # ── Filter events with no valid SFOS pair (m_Z sentinel = 9999) ──────
    valid = m_Z < 9000.0

    result = pd.DataFrame({
        "pT1":       lep[1]["pt"],
        "pT2":       lep[2]["pt"],
        "pT3":       lep[3]["pt"],
        "eta1":      lep[1]["eta"],
        "eta2":      lep[2]["eta"],
        "eta3":      lep[3]["eta"],
        "m_Z":       m_Z,
        "mT_W":      mT_W,
        "MET":       MET,
        "isolation": isolation,
        "d0_sig":    d0_sig,
        "weight":    np.ones(n),
    })

    return result[valid].reset_index(drop=True)


# ── Public loader ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading CERN Open Data…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (signal_df, background_df).

    Signal  = Signal.csv   (WZ → 3ℓν)
    Background = Zjets.csv + ttbar.csv + Diboson.csv  (concatenated, weight=1)
    """
    sig_path  = DATA_DIR / "Signal.csv"
    bkg_files = ["Zjets.csv", "Diboson.csv", "ttbar.csv"]

    if not sig_path.exists():
        raise FileNotFoundError(
            f"Signal.csv not found in {DATA_DIR}. "
            "Please place the CERN Open Data CSV files in the data/ directory."
        )

    sig_raw = pd.read_csv(sig_path)
    sig = _process(sig_raw)

    bkg_parts = []
    for fname in bkg_files:
        path = DATA_DIR / fname
        if path.exists():
            bkg_parts.append(_process(pd.read_csv(path)))
        else:
            st.warning(f"Background file not found: {path.name}")

    if not bkg_parts:
        raise FileNotFoundError("No background CSV files found in data/.")

    bkg = pd.concat(bkg_parts, ignore_index=True)

    return sig, bkg
