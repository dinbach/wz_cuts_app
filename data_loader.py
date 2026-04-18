"""
data_loader.py
--------------
Loads signal (WZ→3ℓν) and background datasets.

Currently generates physically motivated synthetic data.
To use real CERN Open Data ROOT files, replace `generate_synthetic_data()`
with the uproot-based loader at the bottom of this file.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

RNG_SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data"


# ── Synthetic data generation ─────────────────────────────────────────────────

def _generate_signal(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    WZ → 3ℓ ν signal.
    Two SFOS leptons from Z (m_Z ≈ 91.2 GeV),
    one lepton + neutrino from W (MET, mT_W peak near m_W ≈ 80 GeV).
    Prompt, well-isolated leptons.
    """
    # Lepton pT: Breit-Wigner-ish smeared by detector, use gamma distribution
    pT1 = rng.gamma(shape=6.0, scale=8.0, size=n) + 28.0   # leading,  mean ~76
    pT2 = rng.gamma(shape=5.0, scale=6.0, size=n) + 20.0   # sub-lead, mean ~50
    pT3 = rng.gamma(shape=3.5, scale=5.0, size=n) + 15.0   # third,    mean ~32

    # Pseudorapidity: roughly uniform up to |η| < 2.5
    eta1 = rng.uniform(-2.4, 2.4, size=n)
    eta2 = rng.uniform(-2.4, 2.4, size=n)
    eta3 = rng.uniform(-2.4, 2.4, size=n)

    # Z candidate invariant mass: Breit-Wigner around 91.2 GeV
    m_Z = rng.standard_cauchy(size=n) * 2.5 + 91.2
    m_Z = np.clip(m_Z, 60.0, 120.0)

    # W transverse mass: Jacobian peak near 80 GeV
    # Simulate as a truncated distribution rising then falling sharply
    u = rng.uniform(0, 1, size=n)
    mT_W = 80.0 * np.sqrt(1 - (1 - u) ** 0.5)
    mT_W = np.clip(mT_W, 0.0, 150.0)

    # MET: neutrino pT, roughly folded Gaussian
    MET = np.abs(rng.normal(loc=38.0, scale=18.0, size=n))

    # Isolation: prompt leptons → small values, exponential tail
    isolation = rng.exponential(scale=0.08, size=n)
    isolation = np.clip(isolation, 0.0, 2.0)

    # Impact parameter significance: prompt → Gaussian near 0
    d0_sig = np.abs(rng.normal(loc=0.0, scale=1.2, size=n))

    return pd.DataFrame({
        "pT1": pT1, "pT2": pT2, "pT3": pT3,
        "eta1": eta1, "eta2": eta2, "eta3": eta3,
        "m_Z": m_Z, "mT_W": mT_W, "MET": MET,
        "isolation": isolation, "d0_sig": d0_sig,
        "weight": np.ones(n),
    })


def _generate_background(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Mixed background: Z+jets (dominant), ttbar, ZZ, fake leptons.
    No sharp m_Z constraint for the 3-lepton system, broader MET,
    worse isolation, larger d0.
    """
    # Softer pT spectra
    pT1 = rng.gamma(shape=4.0, scale=7.0, size=n) + 22.0
    pT2 = rng.gamma(shape=3.0, scale=5.5, size=n) + 15.0
    pT3 = rng.gamma(shape=2.0, scale=5.0, size=n) + 10.0

    eta1 = rng.uniform(-2.5, 2.5, size=n)
    eta2 = rng.uniform(-2.5, 2.5, size=n)
    eta3 = rng.uniform(-2.5, 2.5, size=n)

    # m_Z: some Z+jets events have a real Z, but third lepton is fake/mis-ID
    # Blend: 60% have a Z peak, 40% are off-peak
    has_z = rng.random(size=n) < 0.55
    m_Z = np.where(
        has_z,
        np.clip(rng.standard_cauchy(size=n) * 3.5 + 91.2, 60.0, 120.0),
        rng.uniform(40.0, 120.0, size=n),
    )

    # mT_W: no Jacobian peak → roughly uniform / falling
    mT_W = rng.exponential(scale=35.0, size=n)
    mT_W = np.clip(mT_W, 0.0, 150.0)

    # MET: lower on average (Z+jets has fake MET from jet mis-measurement)
    MET = np.abs(rng.normal(loc=18.0, scale=20.0, size=n))

    # Isolation: broader — Z+jets leptons less isolated
    isolation = rng.exponential(scale=0.35, size=n)
    isolation = np.clip(isolation, 0.0, 2.0)

    # d0_sig: ttbar component gives wider tails
    d0_sig = np.abs(rng.normal(loc=0.0, scale=2.5, size=n))
    d0_sig = np.clip(d0_sig, 0.0, 10.0)

    return pd.DataFrame({
        "pT1": pT1, "pT2": pT2, "pT3": pT3,
        "eta1": eta1, "eta2": eta2, "eta3": eta3,
        "m_Z": m_Z, "mT_W": mT_W, "MET": MET,
        "isolation": isolation, "d0_sig": d0_sig,
        "weight": np.ones(n),
    })


@st.cache_data(show_spinner="Loading datasets…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (signal_df, background_df).

    Priority:
      1. Parquet files in data/ (fast, pre-processed from ROOT)
      2. Synthetic data (fallback for development / demo)
    """
    sig_path = DATA_DIR / "signal.parquet"
    bkg_path = DATA_DIR / "background.parquet"

    if sig_path.exists() and bkg_path.exists():
        sig = pd.read_parquet(sig_path)
        bkg = pd.read_parquet(bkg_path)
        return sig, bkg

    # Fallback: generate synthetic data and cache to parquet
    rng = np.random.default_rng(RNG_SEED)
    sig = _generate_signal(n=15_000, rng=rng)
    bkg = _generate_background(n=60_000, rng=rng)

    DATA_DIR.mkdir(exist_ok=True)
    sig.to_parquet(sig_path, index=False)
    bkg.to_parquet(bkg_path, index=False)

    return sig, bkg


# ── Real ROOT loader (to activate when you have CERN Open Data files) ─────────
#
# import uproot, awkward as ak
#
# def load_root_data(sig_path: str, bkg_path: str):
#     def _read(path):
#         with uproot.open(path) as f:
#             tree = f["mini"]          # adjust tree name to your ntuple
#             arrays = tree.arrays(
#                 ["lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_charge",
#                  "lep_type", "met_et", "jet_n"],
#                 library="ak"
#             )
#         # ... reconstruct m_Z, mT_W, select 3-lepton events, build DataFrame
#         return df
#     return _read(sig_path), _read(bkg_path)
