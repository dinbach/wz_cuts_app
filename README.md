# WZ → 3ℓ ν  Cut Optimisation Tool

An interactive Streamlit application for university students to explore
**signal vs. background event selection** for the WZ → W(ℓν) Z(ℓℓ)
physics process using CERN Open Data.

---

## Physics context

The app targets the **WZ di-boson** process:

```
pp → WZ → (ℓν)(ℓℓ)
```

The signal signature is **3 isolated leptons + missing transverse energy**.
Students apply cuts on experimental variables and observe in real time how
the significance $S/\sqrt{B}$ (and other metrics) respond.

### Variables

| Variable | Symbol | Cut direction |
|---|---|---|
| Leading lepton pT | $p_{T1}$ | > minimum |
| Sub-leading lepton pT | $p_{T2}$ | > minimum |
| Third lepton pT | $p_{T3}$ | > minimum |
| Missing transverse energy | $E_T^{miss}$ | > minimum |
| W transverse mass | $m_T^W$ | > minimum |
| Z candidate mass | $m_Z$ | window |
| Lepton track isolation | Isolation | < maximum |
| Impact parameter significance | $\|d_0\|$ sig. | < maximum |

---

## Quickstart (local)

```bash
git clone <your-repo-url>
cd wz_cuts_app
pip install -r requirements.txt
streamlit run app.py
```

The first run generates synthetic signal/background data and saves it
to `data/signal.parquet` and `data/background.parquet`.

---

## Using real CERN Open Data ROOT files

1. Download the WZ MC signal and background ROOT ntuples from
   [opendata.cern.ch](https://opendata.cern.ch) (ATLAS Open Data, 13 TeV).
2. Pre-process them into a flat pandas DataFrame with the columns:
   `pT1, pT2, pT3, eta1, eta2, eta3, MET, mT_W, m_Z, isolation, d0_sig, weight`
3. Save as `data/signal.parquet` and `data/background.parquet`.
4. The app's `load_data()` function will pick them up automatically.

A skeleton `uproot` loader is provided at the bottom of `utils/data_loader.py`.

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repository to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set main file to `app.py`.
4. Done — the app is live at `https://<your-app>.streamlit.app`.

No server, no Docker, no cost.

---

## Project structure

```
wz_cuts_app/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
├── data/
│   ├── signal.parquet      # auto-generated on first run
│   └── background.parquet
└── utils/
    ├── data_loader.py      # data loading + synthetic generation
    ├── cuts.py             # cut definitions & application
    ├── significance.py     # S/sqrt(B), Asimov Z, ROC
    └── plotting.py         # matplotlib / mplhep figures
```

---

## Extending the app

- **Add variables**: add a new `CutDef` entry to `CUT_DEFS` in `cuts.py`
  and ensure the corresponding column exists in your DataFrame.
- **Add real data overlay**: load a data DataFrame in `data_loader.py`
  and pass it through `apply_cuts`; overlay on histograms in `plotting.py`.
- **Add a multivariate scan**: implement a 2D grid scan over two variables
  simultaneously and display as a heat map.
- **Add a cut table export**: use `st.download_button` to let students
  download their cut configuration as a CSV.
